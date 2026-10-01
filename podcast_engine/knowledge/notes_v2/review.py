"""Review verdicts, failure policy, and the bounded fix-excerpt window for
TASK-106 Phase A Task 8 (pure decision logic).

Design authority:
docs/superpowers/specs/2026-09-24-grounded-knowledge-note-pipeline-design.md
§5.6 (review), §5.6.1 (`condition_dropped` re-extraction routing), §5.6.2
(targeted fix with a bounded transcript window), §8.1 (rejected items,
footer line), §8.2 (units failing review after fix rounds).

Everything here is pure: no network, no file I/O, no model call. The
model-calling orchestration (one review call, the `condition_dropped` ->
re-extraction routing via Task 5's `reextract_section`, the bounded
targeted-fix rounds) lives in `scripts/knowledge_eval/review.py`, mirroring
this package's relationship to `scripts/knowledge_eval/extract.py` and
`scripts/knowledge_eval/composition.py`.

**Review scope.** Design spec §5.6 reviews "every bullet, TL;DR sentence,
and takeaway" -- not `research_discussed`/`also_discussed`/`follow_up`,
which Task 7's composition validation does check for citation/grounding but
which §5.6 does not name as reviewed units. `enumerate_review_units` follows
that narrower scope deliberately.

**Failure-policy scope (§8.2).** The design spec's removal wording covers
only "bullets" ("publish without the failing bullets"); a TL;DR sentence,
a protocol-associated bullet, or more than 3 ordinary bullets force a hold.
It does not explicitly say what happens to a failing *takeaway* that is
neither of those. This module treats a failing takeaway the same as a
failing TL;DR sentence -- it forces a hold rather than being silently
dropped -- since removal is only ever defined for bullets and the design's
general posture is to fail closed on an undefined case, not to guess.

**Fix-excerpt window size (§5.6.2).** The phase-A plan's Task 8 Step 1
attributes the fixed window size to "TASK-076's evidence-window pattern";
the design spec's §5.6.2 attributes the same mechanism to "TASK-075['s]
revision step" -- the two documents disagree on the task number. This
implementation follows the design spec's attribution: TASK-075's bounded
transcript-evidence-span mechanism
(`podcast_engine.knowledge.summary_review_evidence.TRANSCRIPT_SPAN_MAX_WORDS
= 48`) bounds transcript context handed to a *model*, matching Task 8's use
case, whereas TASK-076's `compiler.transcript.MAX_HUMAN_EDIT_EXPANSION_WORDS`
(3 words/side) bounds a *human* review-UI edit affordance -- a different use
case, and too small to be useful model context. See the Task 8
implementation note in the phase-A plan.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

REVIEW_SCHEMA_VERSION = "note-review-v1"
FIX_SCHEMA_VERSION = "note-fix-v1"

# ---------------------------------------------------------------------------
# Verdicts (design spec §5.6)
# ---------------------------------------------------------------------------

VERDICTS = frozenset({
    "supported",
    "unsupported_claim",
    "epistemic_drift",
    "causal_overstatement",
    "recommendation_drift",
    "condition_dropped",
    "scope_drift",
})

REASON_UNKNOWN_KEY = "unknown_key"
REASON_MISSING_KEY = "missing_key"
REASON_UNKNOWN_UNIT_ID = "unknown_unit_id"
REASON_MISSING_VERDICT = "missing_verdict"
REASON_DUPLICATE_VERDICT = "duplicate_verdict"
REASON_INVALID_VERDICT = "invalid_verdict"

_VERDICT_ENTRY_KEYS = frozenset({"unit_id", "verdict", "reason"})
_TOP_LEVEL_REVIEW_KEYS = frozenset({"verdicts"})

REVIEW_JSON_SCHEMA: dict[str, Any] = {
    "name": "note_review_v1",
    "schema": {
        "type": "object",
        "required": ["verdicts"],
        "properties": {
            "verdicts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["unit_id", "verdict", "reason"],
                    "properties": {
                        "unit_id": {"type": "string"},
                        "verdict": {"type": "string", "enum": sorted(VERDICTS)},
                        "reason": {"type": ["string", "null"]},
                    },
                },
            },
        },
    },
}


@dataclass(frozen=True)
class ReviewError:
    code: str
    detail: str


@dataclass(frozen=True)
class ReviewValidationResult:
    valid: bool
    verdicts_by_unit_id: dict[str, dict] = field(default_factory=dict)
    errors: list[ReviewError] = field(default_factory=list)


def enumerate_review_units(composition: dict) -> dict[str, dict]:
    """`unit_id -> {"text": ..., "item_ids": [...]}` for exactly the unit
    kinds design spec §5.6 reviews: TL;DR entries, bullets, and takeaways.
    `unit_id` paths match the ones `validate_composition` uses in its error
    messages (`"tldr[0]"`, `"sections[0].bullets[1]"`, `"takeaways[2]"`).
    """
    units: dict[str, dict] = {}
    for i, entry in enumerate(composition.get("tldr") or []):
        units[f"tldr[{i}]"] = {"text": entry.get("text"), "item_ids": list(entry.get("item_ids") or [])}
    for si, section in enumerate(composition.get("sections") or []):
        for bi, bullet in enumerate(section.get("bullets") or []):
            units[f"sections[{si}].bullets[{bi}]"] = {
                "text": bullet.get("text"),
                "item_ids": list(bullet.get("item_ids") or []),
            }
    for i, entry in enumerate(composition.get("takeaways") or []):
        units[f"takeaways[{i}]"] = {"text": entry.get("text"), "item_ids": list(entry.get("item_ids") or [])}
    return units


def validate_review(review: dict, units_by_id: dict[str, dict]) -> ReviewValidationResult:
    """Validate one `note-review-v1` payload: exactly one verdict per unit
    in `units_by_id` (design spec §5.6: "Python verifies that exactly one
    verdict exists per unit"), never raises.
    """
    errors: list[ReviewError] = []

    if not isinstance(review, dict):
        return ReviewValidationResult(valid=False, errors=[ReviewError(REASON_MISSING_KEY, "review is not an object")])

    for key in review:
        if key not in _TOP_LEVEL_REVIEW_KEYS:
            errors.append(ReviewError(REASON_UNKNOWN_KEY, f"review.{key} is not a recognized field"))
    if "verdicts" not in review:
        errors.append(ReviewError(REASON_MISSING_KEY, "review.verdicts is required"))
        return ReviewValidationResult(valid=False, errors=errors)

    raw_verdicts = review.get("verdicts")
    seen: dict[str, int] = {}
    verdicts_by_unit_id: dict[str, dict] = {}

    if not isinstance(raw_verdicts, list):
        errors.append(ReviewError(REASON_MISSING_KEY, "review.verdicts is not an array"))
    else:
        for i, entry in enumerate(raw_verdicts):
            path = f"verdicts[{i}]"
            if not isinstance(entry, dict):
                errors.append(ReviewError(REASON_MISSING_KEY, f"{path} is not an object"))
                continue
            for key in entry:
                if key not in _VERDICT_ENTRY_KEYS:
                    errors.append(ReviewError(REASON_UNKNOWN_KEY, f"{path}.{key} is not a recognized field"))
            for key in _VERDICT_ENTRY_KEYS:
                if key not in entry:
                    errors.append(ReviewError(REASON_MISSING_KEY, f"{path}.{key} is required"))

            unit_id = entry.get("unit_id")
            verdict = entry.get("verdict")
            if isinstance(unit_id, str):
                seen[unit_id] = seen.get(unit_id, 0) + 1
                if unit_id not in units_by_id:
                    errors.append(ReviewError(REASON_UNKNOWN_UNIT_ID, f"{path} references unknown unit_id {unit_id!r}"))
            if verdict not in VERDICTS:
                errors.append(ReviewError(REASON_INVALID_VERDICT, f"{path} has invalid verdict {verdict!r}"))
            if isinstance(unit_id, str) and unit_id in units_by_id and verdict in VERDICTS:
                verdicts_by_unit_id[unit_id] = {"verdict": verdict, "reason": entry.get("reason")}

    for unit_id, count in seen.items():
        if count > 1:
            errors.append(ReviewError(REASON_DUPLICATE_VERDICT, f"unit_id {unit_id!r} has {count} verdicts"))

    for unit_id in units_by_id:
        if unit_id not in seen:
            errors.append(ReviewError(REASON_MISSING_VERDICT, f"unit {unit_id!r} has no verdict"))

    return ReviewValidationResult(valid=not errors, verdicts_by_unit_id=verdicts_by_unit_id if not errors else {}, errors=errors)


# ---------------------------------------------------------------------------
# Bounded fix excerpt (design spec §5.6.2)
# ---------------------------------------------------------------------------

# See the module docstring for why this reuses TASK-075's evidence-span
# bound rather than TASK-076's human-edit-expansion bound.
FIX_EXCERPT_WINDOW_WORDS = 48

_WORD_SPAN_RE = re.compile(r"\S+")


def _char_offset_n_words_before(text: str, position: int, n: int) -> int:
    """Character offset in `text` that starts the n-th word before
    `position` (or 0 if fewer than n words precede it)."""
    words = list(_WORD_SPAN_RE.finditer(text, 0, position))
    if not words:
        return 0
    if len(words) <= n:
        return words[0].start()
    return words[-n].start()


def _char_offset_n_words_after(text: str, position: int, n: int) -> int:
    """Character offset in `text` that ends the n-th word after `position`
    (or len(text) if fewer than n words follow it, or n <= 0)."""
    if n <= 0:
        return position
    words = list(_WORD_SPAN_RE.finditer(text, position))
    if not words or len(words) <= n:
        return len(text)
    return words[n - 1].end()


def build_fix_excerpt(
    transcript: str,
    *,
    quote_start: int,
    quote_end: int,
    window_words: int = FIX_EXCERPT_WINDOW_WORDS,
) -> dict:
    """Bounded exact-offset transcript excerpt around `[quote_start,
    quote_end)` -- up to `window_words` words on each side, never the full
    transcript (design spec §5.6.2: "scoped to this single already-flagged
    span, never the full transcript"). This is the only point in the whole
    pipeline where composition (via the targeted fix) sees transcript text,
    and only for a unit whose `condition_dropped` verdict survived the
    §5.6.1 re-extraction attempt.

    Raises `ValueError` when `window_words` exceeds `FIX_EXCERPT_WINDOW_WORDS`
    -- an oversized request is rejected here, before it ever reaches a
    prompt (design spec / plan Task 8 Step 1).
    """
    if window_words > FIX_EXCERPT_WINDOW_WORDS:
        raise ValueError(
            f"fix excerpt window ({window_words} words/side) exceeds the bounded "
            f"maximum of {FIX_EXCERPT_WINDOW_WORDS} words/side"
        )
    if window_words < 0:
        raise ValueError("fix excerpt window must be a non-negative word count")
    if not (0 <= quote_start <= quote_end <= len(transcript)):
        raise ValueError("quote span is out of bounds for this transcript")

    excerpt_start = _char_offset_n_words_before(transcript, quote_start, window_words)
    excerpt_end = _char_offset_n_words_after(transcript, quote_end, window_words)

    return {
        "excerpt_start": excerpt_start,
        "excerpt_end": excerpt_end,
        "text": transcript[excerpt_start:excerpt_end],
        "quote_start": quote_start,
        "quote_end": quote_end,
        "window_words": window_words,
    }


# ---------------------------------------------------------------------------
# Fix output schema (design spec §5.6.2: "only those units")
# ---------------------------------------------------------------------------

FIX_JSON_SCHEMA: dict[str, Any] = {
    "name": "note_fix_v1",
    "schema": {
        "type": "object",
        "required": ["fixed_units"],
        "properties": {
            "fixed_units": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["unit_id", "text", "item_ids"],
                    "properties": {
                        "unit_id": {"type": "string"},
                        "text": {"type": "string"},
                        "item_ids": {"type": "array", "items": {"type": "string"}},
                        "primary_item_id": {"type": ["string", "null"]},
                        "asserts": {"type": ["boolean", "null"]},
                    },
                },
            },
        },
    },
}


def splice_fixed_units(composition: dict, fixed_units_by_id: dict[str, dict]) -> dict:
    """Return a new composition with each `unit_id` in `fixed_units_by_id`
    replaced in place by its fixed `text`/`item_ids` (and, for a bullet,
    `primary_item_id`/`asserts` when given -- otherwise the bullet's
    existing values are kept). Pure data manipulation; never calls a model.
    Unknown `unit_id`s in `fixed_units_by_id` are ignored (the caller only
    ever passes ids from `enumerate_review_units`).
    """
    import copy

    result = copy.deepcopy(composition)

    for i, entry in enumerate(result.get("tldr") or []):
        fixed = fixed_units_by_id.get(f"tldr[{i}]")
        if fixed is not None:
            entry["text"] = fixed["text"]
            entry["item_ids"] = list(fixed["item_ids"])

    for si, section in enumerate(result.get("sections") or []):
        for bi, bullet in enumerate(section.get("bullets") or []):
            fixed = fixed_units_by_id.get(f"sections[{si}].bullets[{bi}]")
            if fixed is not None:
                bullet["text"] = fixed["text"]
                bullet["item_ids"] = list(fixed["item_ids"])
                if fixed.get("primary_item_id") is not None:
                    bullet["primary_item_id"] = fixed["primary_item_id"]
                elif bullet.get("primary_item_id") not in fixed["item_ids"]:
                    bullet["primary_item_id"] = None
                if fixed.get("asserts") is not None:
                    bullet["asserts"] = fixed["asserts"]

    for i, entry in enumerate(result.get("takeaways") or []):
        fixed = fixed_units_by_id.get(f"takeaways[{i}]")
        if fixed is not None:
            entry["text"] = fixed["text"]
            entry["item_ids"] = list(fixed["item_ids"])

    return result


# ---------------------------------------------------------------------------
# Failure policy (design spec §8.1 footer line, §8.2 hold/drop decision)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReviewDecision:
    action: str  # "publish" | "hold"
    removed_unit_ids: list[str] = field(default_factory=list)
    hold_reason: str | None = None
    footer_line: str | None = None


def _is_tldr_or_takeaway(unit_id: str) -> bool:
    return unit_id.startswith("tldr[") or unit_id.startswith("takeaways[")


def _cites_a_protocol_item(unit: dict, items_by_id: dict[str, dict]) -> bool:
    for item_id in unit.get("item_ids", []):
        item = items_by_id.get(item_id)
        if item is not None and item.get("kind") == "protocol":
            return True
    return False


def decide(
    final_verdicts: dict[str, str],
    *,
    units_by_id: dict[str, dict],
    items_by_id: dict[str, dict],
    rejected_high_value_item_ids: list[str] | None = None,
) -> ReviewDecision:
    """The design spec §8.2 decision, after all fix rounds are exhausted.

    Hold the whole run when a failing unit is a TL;DR sentence or a
    takeaway (see the module docstring), when a failing bullet is
    associated with a `protocol`-kind item, or when more than 3 ordinary
    bullets are still failing. Otherwise publish with the failing
    (non-protocol) bullets removed, and attach the §8.1 footer line when a
    high-value item stayed rejected.
    """
    failing = [unit_id for unit_id, verdict in final_verdicts.items() if verdict != "supported"]

    tldr_or_takeaway_failures = [uid for uid in failing if _is_tldr_or_takeaway(uid)]
    remaining = [uid for uid in failing if uid not in tldr_or_takeaway_failures]
    protocol_bullet_failures = [
        uid for uid in remaining if _cites_a_protocol_item(units_by_id.get(uid, {}), items_by_id)
    ]
    plain_bullet_failures = [uid for uid in remaining if uid not in protocol_bullet_failures]

    if tldr_or_takeaway_failures or protocol_bullet_failures or len(plain_bullet_failures) > 3:
        reasons = []
        if tldr_or_takeaway_failures:
            reasons.append(f"{len(tldr_or_takeaway_failures)} TL;DR/takeaway unit(s) failed review")
        if protocol_bullet_failures:
            reasons.append(f"{len(protocol_bullet_failures)} protocol bullet(s) failed review")
        if len(plain_bullet_failures) > 3:
            reasons.append(f"{len(plain_bullet_failures)} bullets failed review (more than 3)")
        return ReviewDecision(action="hold", hold_reason="; ".join(reasons))

    footer_line = None
    if rejected_high_value_item_ids:
        footer_line = f"{len(rejected_high_value_item_ids)} key point(s) could not be verified — see PodcastOps."

    return ReviewDecision(action="publish", removed_unit_ids=list(plain_bullet_failures), footer_line=footer_line)
