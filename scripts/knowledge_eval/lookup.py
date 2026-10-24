"""Bibliographic lookup (Crossref, OpenAlex) for TASK-106 Phase A Task 6.

Design authority:
docs/superpowers/specs/2026-09-24-grounded-knowledge-note-pipeline-design.md
§5.3 -- "query Crossref and OpenAlex (free, no key)... Lookups are cached by
query hash." A polite `User-Agent`/`mailto` is sent per each API's usage
guidance.

This module does the network fetching and on-disk caching; the pure
decision logic (thresholds, year tolerance, uniqueness) lives in
`podcast_engine.knowledge.notes_v2.sources`, which this module feeds with
already-fetched candidate dicts. A fetch failure for either provider is
caught here and degrades to an empty candidate list for that provider --
never raised, never fails the run (design spec §5.3 / plan Task 6 Step 1:
"HTTP failure degrades to `as heard` without failing the run").
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import requests

CROSSREF_URL = "https://api.crossref.org/works"
OPENALEX_URL = "https://api.openalex.org/works"
DEFAULT_TIMEOUT_SECONDS = 15
DEFAULT_ROWS = 5

# Contact email for the polite pool / User-Agent, per Crossref's and
# OpenAlex's usage guidance. Configurable via the client so an operator can
# set their own contact; a generic User-Agent is still sent without one.
CONTACT_EMAIL_ENV_VAR = "KNOWLEDGE_EVAL_CONTACT_EMAIL"


def _polite_headers(contact_email: str | None) -> dict:
    agent = "PodcastWorkerKnowledgeEval/1.0 (Phase A offline evaluation harness)"
    if contact_email:
        agent += f" (mailto:{contact_email})"
    return {"User-Agent": agent}


def _crossref_query(surnames: list[str], year, topic_words: list[str]) -> str:
    return " ".join([*surnames, *topic_words]).strip()


def _default_crossref_transport(query: str, *, year, contact_email: str | None) -> dict:
    params: dict[str, Any] = {"query": query, "rows": DEFAULT_ROWS}
    if contact_email:
        params["mailto"] = contact_email
    response = requests.get(
        CROSSREF_URL,
        params=params,
        headers=_polite_headers(contact_email),
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def _default_openalex_transport(query: str, *, year, contact_email: str | None) -> dict:
    params: dict[str, Any] = {"search": query, "per_page": DEFAULT_ROWS}
    if contact_email:
        params["mailto"] = contact_email
    response = requests.get(
        OPENALEX_URL,
        params=params,
        headers=_polite_headers(contact_email),
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def _parse_crossref_candidates(payload: dict) -> list[dict]:
    items = (payload or {}).get("message", {}).get("items", [])
    candidates = []
    for item in items:
        title_list = item.get("title") or []
        title = title_list[0] if title_list else None
        authors = [
            author.get("family")
            for author in (item.get("author") or [])
            if author.get("family")
        ]
        year = None
        date_parts = (
            (item.get("issued") or {}).get("date-parts") or [[None]]
        )[0]
        if date_parts and date_parts[0]:
            year = date_parts[0]
        score = item.get("score")
        candidates.append({
            "title": title,
            "url": item.get("URL"),
            "doi": item.get("DOI"),
            "year": year,
            "authors": authors,
            "score": _normalize_crossref_score(score),
            "provider": "crossref",
        })
    return candidates


def _normalize_crossref_score(raw_score) -> float | None:
    """Crossref relevance scores are unbounded (often tens); map to a
    coarse 0-1 confidence band. Callers with no score are excluded upstream
    by `select_bibliographic_candidate`'s threshold check.
    """
    if not isinstance(raw_score, (int, float)):
        return None
    # Empirically Crossref scores above ~40 indicate a strong single-work
    # match for an author+title query; this is a conservative, tunable
    # starting normalization (plan Task 6 Step 3: tuned on the dev set).
    return min(1.0, raw_score / 40.0)


def _parse_openalex_candidates(payload: dict) -> list[dict]:
    results = (payload or {}).get("results", [])
    candidates = []
    for item in results:
        authors = [
            (authorship.get("author") or {}).get("display_name", "").split(" ")[-1]
            for authorship in (item.get("authorships") or [])
            if (authorship.get("author") or {}).get("display_name")
        ]
        doi = item.get("doi")
        if isinstance(doi, str):
            doi = doi.replace("https://doi.org/", "")
        relevance = item.get("relevance_score")
        candidates.append({
            "title": item.get("title") or item.get("display_name"),
            "url": item.get("id"),
            "doi": doi,
            "year": item.get("publication_year"),
            "authors": [a for a in authors if a],
            "score": _normalize_openalex_score(relevance),
            "provider": "openalex",
        })
    return candidates


def _normalize_openalex_score(raw_score) -> float | None:
    if not isinstance(raw_score, (int, float)):
        return None
    # OpenAlex relevance scores are also unbounded; same conservative
    # starting normalization as Crossref, tuned later on the dev set.
    return min(1.0, raw_score / 40.0)


@dataclass
class SourceLookupClient:
    """Cached, fault-tolerant Crossref + OpenAlex candidate fetcher.

    `crossref_transport`/`openalex_transport` default to real HTTPS GETs;
    tests inject a fake `Callable[[str], dict]` (or one that raises, to
    exercise the HTTP-failure degradation).
    """

    cache_dir: Path = field(default_factory=lambda: Path(".local/knowledge-eval/source-lookup-cache"))
    contact_email: str | None = None
    crossref_transport: Callable[..., dict] | None = None
    openalex_transport: Callable[..., dict] | None = None

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)
        if self.crossref_transport is None:
            self.crossref_transport = _default_crossref_transport
        if self.openalex_transport is None:
            self.openalex_transport = _default_openalex_transport
        self.fetch_count = 0

    def _cache_key(self, surnames: list[str], year, topic_words: list[str]) -> str:
        canonical = json.dumps(
            {"surnames": sorted(surnames), "year": year, "topic_words": sorted(topic_words)},
            sort_keys=True,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _cache_read(self, key: str) -> list[dict] | None:
        path = self._cache_path(key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            return None

    def _cache_write(self, key: str, candidates: list[dict]) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_path(key).write_text(
            json.dumps(candidates, ensure_ascii=False, sort_keys=True), encoding="utf-8"
        )

    def _safe_fetch(self, transport, query: str, year) -> list[dict]:
        try:
            payload = transport(query, year=year, contact_email=self.contact_email)
        except Exception:
            # Any transport failure (timeout, connection error, HTTP error,
            # malformed response) degrades to no candidates from this
            # provider rather than failing the run.
            return []
        return payload

    def candidates(self, *, surnames: list[str], year, topic_words: list[str]) -> list[dict]:
        """Merged Crossref + OpenAlex candidates for one query, cached by
        query hash. Never raises.
        """
        key = self._cache_key(surnames, year, topic_words)
        cached = self._cache_read(key)
        if cached is not None:
            return cached

        self.fetch_count += 1
        query = _crossref_query(surnames, year, topic_words)

        crossref_payload = self._safe_fetch(self.crossref_transport, query, year)
        crossref_candidates = (
            _parse_crossref_candidates(crossref_payload) if isinstance(crossref_payload, dict) else []
        )

        openalex_payload = self._safe_fetch(self.openalex_transport, query, year)
        openalex_candidates = (
            _parse_openalex_candidates(openalex_payload) if isinstance(openalex_payload, dict) else []
        )

        merged = crossref_candidates + openalex_candidates
        self._cache_write(key, merged)
        return merged


# ---------------------------------------------------------------------------
# Operator overrides (PodcastOps corrections, keyed by (episode_key, quote hash))
# ---------------------------------------------------------------------------


def load_overrides(path: Path | str) -> dict:
    """Load the local operator-override file: `{episode_key: {quote_hash:
    {"title", "url", "doi"}}}`. A missing file is simply no overrides.
    """
    path = Path(path)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}
