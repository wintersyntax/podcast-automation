"""Shadow-only terminology retrieval for compiler observability.

Phase A deliberately runs *after* transcript/resolver/Human Review decisions are
complete. Candidate terminology may therefore be measured against real diffs,
but it cannot affect resolver eligibility, merge actions, review routing, or the
canonical transcript. The output is telemetry only.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import hashlib
import json
from typing import Any, Iterable

from compiler.terminology import TERMINOLOGY_REGISTRY, TERMINOLOGY_REGISTRY_SHA256
from compiler.transcript import TERMINOLOGY_RETRIEVAL_SCHEMA_VERSION

from .terminology_candidates import (
    ALLOWED_CANDIDATE_CATEGORIES,
    CandidateRegistryStore,
    normalize_candidate_text,
    validate_candidate_registry,
)


TERMINOLOGY_RETRIEVAL_MODE = "shadow"
TERMINOLOGY_RETRIEVAL_TOP_K = 3
TERMINOLOGY_RETRIEVAL_MIN_SIMILARITY = 0.82
TERMINOLOGY_RETRIEVAL_MAX_CANDIDATES = 5_000

_NON_TERMINOLOGY_CATEGORIES = {"protocol_number", "unit", "negation"}


@dataclass(frozen=True)
class RetrievalTerm:
    canonical: str
    category: str
    forms: tuple[str, ...]
    source: str
    provenance: tuple[str, ...]
    external_sources: tuple[str, ...]


def _shadow_payload(*, registry_status: str) -> dict[str, Any]:
    return {
        "schema_version": TERMINOLOGY_RETRIEVAL_SCHEMA_VERSION,
        "mode": TERMINOLOGY_RETRIEVAL_MODE,
        "stage": "post_decision_observability",
        "authoritative": False,
        "decision_effect": "none",
        "resolver_visibility": "none",
        "review_routing_effect": "none",
        "top_k": TERMINOLOGY_RETRIEVAL_TOP_K,
        "minimum_lexical_similarity": TERMINOLOGY_RETRIEVAL_MIN_SIMILARITY,
        "active_registry_sha256": TERMINOLOGY_REGISTRY_SHA256,
        "candidate_registry_status": registry_status,
        "candidate_registry_sha256": None,
        "candidate_registry_updated_at": None,
        "active_term_count": len(TERMINOLOGY_REGISTRY),
        "candidate_term_count": 0,
        "retrieval_universe_count": len(TERMINOLOGY_REGISTRY),
        "scanned_difference_count": 0,
        "retrieved_difference_count": 0,
        "candidate_hit_count": 0,
        "signals": [],
    }


def disabled_terminology_retrieval() -> dict[str, Any]:
    payload = _shadow_payload(registry_status="not_attached")
    payload["mode"] = "disabled"
    return payload


def _snapshot_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _dedupe_forms(values: Iterable[str]) -> tuple[str, ...]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value.strip():
            continue
        cleaned = " ".join(value.split())
        key = normalize_candidate_text(cleaned)
        if key in seen:
            continue
        seen.add(key)
        output.append(cleaned)
    return tuple(output)


def _primary_category(entry: dict[str, Any]) -> str | None:
    categories = entry.get("suggested_categories")
    if not isinstance(categories, dict) or not categories:
        return None
    valid = {
        category: count
        for category, count in categories.items()
        if category in ALLOWED_CANDIDATE_CATEGORIES
        and isinstance(count, int)
        and count > 0
    }
    if not valid:
        return None
    return max(valid, key=lambda category: (valid[category], category))


def _external_sources(entry: dict[str, Any]) -> tuple[str, ...]:
    values: set[str] = set()
    for sample in entry.get("evidence_samples", []):
        if not isinstance(sample, dict) or sample.get("provenance") != "external-curated":
            continue
        source = sample.get("external_source")
        if isinstance(source, str) and source.strip():
            values.add(source.strip())
    return tuple(sorted(values, key=str.casefold))


def _retrieval_universe(
    candidate_registry: dict[str, Any] | None,
) -> tuple[list[RetrievalTerm], dict[str, Any]]:
    terms: dict[str, RetrievalTerm] = {}
    for entry in TERMINOLOGY_REGISTRY:
        key = normalize_candidate_text(entry.canonical)
        terms[key] = RetrievalTerm(
            canonical=entry.canonical,
            category=entry.category,
            forms=_dedupe_forms((entry.canonical, *entry.aliases)),
            source="active",
            provenance=("active-registry",),
            external_sources=(),
        )

    metadata = {
        "status": "missing",
        "sha256": None,
        "updated_at": None,
        "candidate_count": 0,
    }
    if candidate_registry is None:
        return list(terms.values()), metadata

    validate_candidate_registry(candidate_registry)
    entries = [
        entry
        for _, entry in sorted(candidate_registry["candidates"].items())
        if isinstance(entry, dict) and entry.get("status") == "candidate"
    ]
    metadata = {
        "status": "available",
        "sha256": _snapshot_sha256(candidate_registry),
        "updated_at": candidate_registry.get("updated_at"),
        "candidate_count": len(entries),
    }
    if len(entries) > TERMINOLOGY_RETRIEVAL_MAX_CANDIDATES:
        metadata["status"] = "too_large"
        return list(terms.values()), metadata

    for entry in entries:
        category = _primary_category(entry)
        canonical = entry.get("canonical")
        if category is None or not isinstance(canonical, str):
            continue
        key = normalize_candidate_text(canonical)
        if key in terms:
            # Active policy always owns an overlapping canonical form.
            continue
        provenance_counts = entry.get("provenance_counts", {})
        provenance = tuple(
            sorted(
                (
                    name
                    for name, count in provenance_counts.items()
                    if isinstance(name, str) and isinstance(count, int) and count > 0
                ),
                key=str.casefold,
            )
        )
        terms[key] = RetrievalTerm(
            canonical=canonical,
            category=category,
            forms=_dedupe_forms((canonical, *entry.get("observed_forms", []))),
            source="candidate",
            provenance=provenance,
            external_sources=_external_sources(entry),
        )
    return list(terms.values()), metadata


def _focus_words(difference: Any, source: str) -> str:
    # Context is deliberate here: a one-token ASR error can occur inside a
    # multi-word term (``lengthened partals`` -> ``lengthened partials``). The
    # lexical scorer below searches term-sized windows, so surrounding context
    # helps recover the phrase without treating the whole sentence as similar.
    context = getattr(difference, f"{source}_context", "")
    if isinstance(context, str) and context.strip():
        return context
    value = getattr(difference, f"{source}_text", "")
    if isinstance(value, str) and value.strip():
        return value
    changed = getattr(difference, f"changed_{source}_words", None)
    if isinstance(changed, list) and changed:
        return " ".join(str(word) for word in changed if str(word).strip())
    return ""


def _lexical_similarity(focus: str, form: str) -> float:
    if not focus.strip() or not form.strip():
        return 0.0
    focus_words = normalize_candidate_text(focus).split()
    form_words = normalize_candidate_text(form).split()
    if not focus_words or not form_words:
        return 0.0

    best = SequenceMatcher(
        None,
        " ".join(focus_words),
        " ".join(form_words),
        autojunk=False,
    ).ratio()
    minimum = max(1, len(form_words) - 1)
    maximum = min(len(focus_words), len(form_words) + 1)
    for size in range(minimum, maximum + 1):
        for start in range(0, len(focus_words) - size + 1):
            score = SequenceMatcher(
                None,
                " ".join(focus_words[start : start + size]),
                " ".join(form_words),
                autojunk=False,
            ).ratio()
            best = max(best, score)
    return round(best, 6)


def _category_allowed(difference_category: str, term_category: str) -> bool:
    if difference_category == "proper_name":
        return term_category in {"proper_name", "citation"}
    if difference_category == "citation":
        return term_category in {"citation", "proper_name"}
    if difference_category == "other":
        return term_category in ALLOWED_CANDIDATE_CATEGORIES
    return term_category == difference_category


def _episode_confirmed_forms(result: Any) -> set[str]:
    memory = getattr(result, "episode_local_memory", None)
    if not isinstance(memory, dict) or memory.get("mode") != "shadow":
        return set()
    confirmed: set[str] = set()
    for signal in memory.get("signals", []):
        if not isinstance(signal, dict):
            continue
        for form in signal.get("forms", []):
            if not isinstance(form, dict) or form.get("episode_confirmed") is not True:
                continue
            text = form.get("text")
            if isinstance(text, str) and text.strip():
                confirmed.add(normalize_candidate_text(text))
    return confirmed


def build_terminology_retrieval_shadow(
    result: Any,
    candidate_registry: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build top-3 lexical terminology telemetry without decision authority."""

    universe, metadata = _retrieval_universe(candidate_registry)
    payload = _shadow_payload(registry_status=metadata["status"])
    payload["candidate_registry_sha256"] = metadata["sha256"]
    payload["candidate_registry_updated_at"] = metadata["updated_at"]
    payload["candidate_term_count"] = metadata["candidate_count"]
    payload["retrieval_universe_count"] = len(universe)

    # A future oversized registry must not trigger an unbounded O(diff * terms)
    # production scan. Shadow telemetry simply abstains until retrieval is
    # redesigned/indexed for that scale.
    if metadata["status"] == "too_large":
        payload["abstention_reason"] = "candidate_registry_exceeds_phase_a_bound"
        return payload

    episode_confirmed = _episode_confirmed_forms(result)
    signals: list[dict[str, Any]] = []
    scanned = 0
    retrieved = 0
    total_hits = 0

    for difference in getattr(result, "differences", []):
        category = getattr(difference, "resolver_category", "other")
        if category in _NON_TERMINOLOGY_CATEGORIES:
            continue
        apple_focus = _focus_words(difference, "apple")
        whisper_focus = _focus_words(difference, "whisper")
        if not apple_focus.strip() and not whisper_focus.strip():
            continue
        scanned += 1

        ranked: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        for term in universe:
            if not _category_allowed(category, term.category):
                continue
            best_form = term.canonical
            apple_score = 0.0
            whisper_score = 0.0
            best_score = 0.0
            for form in term.forms:
                current_apple = _lexical_similarity(apple_focus, form)
                current_whisper = _lexical_similarity(whisper_focus, form)
                current = max(current_apple, current_whisper)
                if current > best_score:
                    best_score = current
                    best_form = form
                    apple_score = current_apple
                    whisper_score = current_whisper
            if best_score < TERMINOLOGY_RETRIEVAL_MIN_SIMILARITY:
                continue
            normalized_forms = {normalize_candidate_text(form) for form in term.forms}
            item = {
                "canonical": term.canonical,
                "category": term.category,
                "source": term.source,
                "matched_form": best_form,
                "lexical_similarity": best_score,
                "apple_similarity": apple_score,
                "whisper_similarity": whisper_score,
                "two_sided_lexical_support": min(apple_score, whisper_score) >= TERMINOLOGY_RETRIEVAL_MIN_SIMILARITY,
                "episode_local_confirmed": bool(normalized_forms & episode_confirmed),
                "provenance": list(term.provenance),
                "external_sources": list(term.external_sources),
            }
            rank_key = (
                -best_score,
                -min(apple_score, whisper_score),
                0 if term.source == "active" else 1,
                term.canonical.casefold(),
            )
            ranked.append((rank_key, item))

        ranked.sort(key=lambda pair: pair[0])
        candidates = [item for _, item in ranked[:TERMINOLOGY_RETRIEVAL_TOP_K]]
        if candidates:
            retrieved += 1
            total_hits += len(candidates)
        # Keep domain/proper-name/citation misses visible for denominator
        # telemetry, while avoiding report bloat from unrelated low-value diffs.
        if candidates or category in ALLOWED_CANDIDATE_CATEGORIES:
            signals.append(
                {
                    "difference_id": getattr(difference, "id", None),
                    "resolver_category": category,
                    "severity": getattr(difference, "severity", None),
                    "candidate_count": len(candidates),
                    "candidates": candidates,
                }
            )

    payload["scanned_difference_count"] = scanned
    payload["retrieved_difference_count"] = retrieved
    payload["candidate_hit_count"] = total_hits
    payload["signals"] = signals
    return payload


def attach_terminology_retrieval_shadow_best_effort(
    result: Any,
    *,
    store: CandidateRegistryStore | None = None,
) -> dict[str, Any]:
    """Attach post-decision telemetry; candidate-store failure never blocks work."""

    try:
        registry = (store or CandidateRegistryStore()).read_snapshot()
        payload = build_terminology_retrieval_shadow(result, registry)
    except Exception as error:  # Shadow diagnostics must never affect canonical work.
        payload = _shadow_payload(registry_status="unavailable")
        payload["error_type"] = type(error).__name__
        print(f"WARNING: terminology retrieval shadow unavailable: {error}")
    result.terminology_retrieval = payload
    return payload
