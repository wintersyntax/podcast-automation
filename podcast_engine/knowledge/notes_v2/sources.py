"""Source resolution for TASK-106 Phase A Task 6 (pure decision logic).

Design authority:
docs/superpowers/specs/2026-09-24-grounded-knowledge-note-pipeline-design.md
§5.3 (source resolution). For each `source_mention`/`study_description`
item, in order of precedence:

1. An operator override keyed by `(episode_key, quote hash)` wins over
   everything -> `corrected`.
2. A show-notes fuzzy match above threshold -> `show_notes`.
3. A single Crossref/OpenAlex candidate above threshold, whose year is
   within +/-1 and at least one author surname matches -> `auto_matched`.
4. Otherwise -> `as_heard`.

Everything here is pure: no network, no file I/O, no model call. HTTP
fetching, caching, and override-file loading live in
`scripts/knowledge_eval/lookup.py`, which feeds already-fetched
`bibliographic_candidates` into `resolve_source_mention` -- a lookup
failure there degrades to an empty candidate list, which this module
naturally resolves to `as_heard` without ever raising.

Lookups are never treated as claim verification (design spec §5.3): a
resolved reference only changes how the source is *displayed*, never
whether the item it belongs to was accepted by Task 3's `validate_items`.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from difflib import SequenceMatcher

STATUS_SHOW_NOTES = "show_notes"       # rendered "✔︎ show notes"
STATUS_AUTO_MATCHED = "auto_matched"   # rendered "✔︎ auto-matched"
STATUS_AS_HEARD = "as_heard"           # rendered "as heard"
STATUS_CORRECTED = "corrected"         # rendered "✎ corrected"

# Conservative starting thresholds (design spec §5.3 / plan Task 6 Step 3:
# "Thresholds start conservative and are tuned on the dev set in Task 12").
SHOW_NOTES_THRESHOLD = 0.6
BIBLIOGRAPHIC_THRESHOLD = 0.6
YEAR_TOLERANCE = 1

_URL_RE = re.compile(r"https?://\S+")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")

# Words too generic to use as a "topic word" query term.
_STOPWORDS = frozenset({
    "the", "a", "an", "of", "and", "or", "in", "on", "for", "to", "with",
    "study", "trial", "review", "meta", "analysis", "effect", "effects",
})


@dataclass(frozen=True)
class SourceResolution:
    """How one `source_mention`/`study_description` item's reference should
    be rendered in the note's "Sources mentioned" section."""

    status: str  # one of the STATUS_* constants
    title: str | None
    url: str | None
    doi: str | None


def quote_hash(quote: str) -> str:
    """Stable key for an item's quote, used for the operator-override file."""
    return hashlib.sha256(quote.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Query-term extraction (pure text processing over a source_mention payload)
# ---------------------------------------------------------------------------


def extract_surnames(authors_as_heard: str | None) -> list[str]:
    """Best-effort surnames from a heard author string ("Smith and Jones",
    "J. Smith, K. Jones et al."). Returns lowercase surnames, deduplicated,
    order preserved.
    """
    if not authors_as_heard:
        return []
    text = re.sub(r"\bet al\.?\b", "", authors_as_heard, flags=re.IGNORECASE)
    parts = re.split(r",|\band\b|&", text, flags=re.IGNORECASE)
    surnames: list[str] = []
    for part in parts:
        words = _WORD_RE.findall(part)
        if not words:
            continue
        surname = words[-1].lower()
        if surname and surname not in surnames:
            surnames.append(surname)
    return surnames


def extract_topic_words(title_as_heard: str | None) -> list[str]:
    """Non-stopword lowercase words from a heard title, for a bibliographic
    query and for show-notes fuzzy matching.
    """
    if not title_as_heard:
        return []
    words = [w.lower() for w in _WORD_RE.findall(title_as_heard)]
    return [w for w in words if w not in _STOPWORDS and len(w) > 2]


def _fuzzy_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _extract_url(text: str) -> str | None:
    match = _URL_RE.search(text)
    if not match:
        return None
    return match.group(0).rstrip(").,;")


def _candidate_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# 1. Show notes
# ---------------------------------------------------------------------------


_WORD_MATCH_THRESHOLD = 0.75
_YEAR_RE = re.compile(r"\b\d{4}\b")


def _query_words(source_mention: dict) -> list[str]:
    words = list(extract_surnames(source_mention.get("authors_as_heard")))
    year = source_mention.get("year_as_heard")
    if year:
        words.append(str(year))
    words.extend(extract_topic_words(source_mention.get("title_as_heard")))
    return words


def _line_words(line: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(line)] + _YEAR_RE.findall(line)


def _word_matches(query_word: str, line_words: list[str]) -> bool:
    for word in line_words:
        if query_word == word:
            return True
        if len(query_word) > 3 and _fuzzy_ratio(query_word, word) >= _WORD_MATCH_THRESHOLD:
            return True
    return False


def match_show_notes(
    source_mention: dict,
    description: str | None,
    *,
    threshold: float = SHOW_NOTES_THRESHOLD,
) -> SourceResolution | None:
    """Fuzzy-match authors/year/title words from `source_mention` against
    the episode description/show-notes text, one line at a time.

    Scored by the fraction of query words (surnames, year, non-stopword
    title words) found in a line -- exactly, or within
    `_WORD_MATCH_THRESHOLD` character-similarity for a word longer than 3
    characters, so a misheard/ASR-mangled name ("Nikolaitis" for
    "Nikolaidis") still matches. A whole-string ratio against a long,
    unrelated show-notes line would be diluted by everything else on that
    line; per-word matching is not. Returns a `show_notes` resolution for
    the best-scoring line when it is at or above `threshold`, else None.
    """
    if not description:
        return None

    query_words = _query_words(source_mention)
    if not query_words:
        return None

    best_score = 0.0
    best_line: str | None = None
    for line in _candidate_lines(description):
        line_words = _line_words(line)
        if not line_words:
            continue
        matched = sum(1 for word in query_words if _word_matches(word, line_words))
        score = matched / len(query_words)
        if score > best_score:
            best_score = score
            best_line = line

    if best_line is not None and best_score >= threshold:
        return SourceResolution(
            status=STATUS_SHOW_NOTES,
            title=best_line,
            url=_extract_url(best_line),
            doi=None,
        )
    return None


# ---------------------------------------------------------------------------
# 2. Bibliographic lookup (candidates already fetched by lookup.py)
# ---------------------------------------------------------------------------


def _dedupe_candidates(candidates: list[dict]) -> list[dict]:
    """Collapse the same work reported by both Crossref and OpenAlex (same
    DOI, case-insensitive) into one candidate -- keeping the first seen --
    before uniqueness is judged.
    """
    seen_dois: set[str] = set()
    deduped: list[dict] = []
    for candidate in candidates:
        doi = candidate.get("doi")
        key = doi.lower() if isinstance(doi, str) and doi else None
        if key is not None:
            if key in seen_dois:
                continue
            seen_dois.add(key)
        deduped.append(candidate)
    return deduped


def select_bibliographic_candidate(
    source_mention: dict,
    candidates: list[dict],
    *,
    threshold: float = BIBLIOGRAPHIC_THRESHOLD,
    year_tolerance: int = YEAR_TOLERANCE,
) -> SourceResolution | None:
    """Accept a bibliographic candidate only when, after deduplication,
    exactly one candidate scores above `threshold`, has a year within
    `year_tolerance` of the heard year (when either is known), and shares at
    least one author surname with the heard authors (when heard authors are
    known). Otherwise None (falls through to `as_heard`).
    """
    heard_year = source_mention.get("year_as_heard")
    heard_surnames = {s.lower() for s in extract_surnames(source_mention.get("authors_as_heard"))}

    qualifying: list[dict] = []
    for candidate in _dedupe_candidates(candidates):
        score = candidate.get("score")
        if not isinstance(score, (int, float)) or score < threshold:
            continue

        candidate_year = candidate.get("year")
        if heard_year is not None and candidate_year is not None:
            try:
                if abs(int(candidate_year) - int(heard_year)) > year_tolerance:
                    continue
            except (TypeError, ValueError):
                continue

        candidate_surnames = {
            str(name).lower() for name in (candidate.get("authors") or [])
        }
        if heard_surnames and not (heard_surnames & candidate_surnames):
            continue

        qualifying.append(candidate)

    if len(qualifying) != 1:
        return None

    winner = qualifying[0]
    return SourceResolution(
        status=STATUS_AUTO_MATCHED,
        title=winner.get("title"),
        url=winner.get("url"),
        doi=winner.get("doi"),
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def resolve_source_mention(
    source_mention: dict,
    *,
    quote: str,
    episode_key: str,
    description: str | None,
    bibliographic_candidates: list[dict] | None = None,
    overrides: dict | None = None,
) -> SourceResolution:
    """Resolve one `source_mention`/`study_description` item's reference.

    `bibliographic_candidates` is whatever `scripts.knowledge_eval.lookup`
    already fetched (possibly `[]` on an HTTP failure -- that degrades here
    to `as_heard`, never an exception). `overrides` is the loaded operator-
    override file, `{episode_key: {quote_hash: {"title", "url", "doi"}}}`.
    """
    overrides = overrides or {}
    override = overrides.get(episode_key, {}).get(quote_hash(quote))
    if override is not None:
        return SourceResolution(
            status=STATUS_CORRECTED,
            title=override.get("title"),
            url=override.get("url"),
            doi=override.get("doi"),
        )

    show_notes_match = match_show_notes(source_mention, description)
    if show_notes_match is not None:
        return show_notes_match

    bibliographic_match = select_bibliographic_candidate(
        source_mention, bibliographic_candidates or []
    )
    if bibliographic_match is not None:
        return bibliographic_match

    return SourceResolution(
        status=STATUS_AS_HEARD,
        title=source_mention.get("title_as_heard"),
        url=None,
        doi=None,
    )
