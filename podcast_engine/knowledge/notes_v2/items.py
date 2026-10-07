"""Item schema validation for TASK-106 Phase A Task 3 (`knowledge-items-v1`).

Design authority:
docs/superpowers/specs/2026-09-24-grounded-knowledge-note-pipeline-design.md
Sections 5.1 (extraction schema), 5.2 (item validation), 5.2.1
(low-condition-confidence heuristic), and 6 (deterministic validation
summary).

Every check here is pure Python: no model call, no network, no GCS access.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal

from .quantities import numeric_claim_tokens, parse_number, quantity_in_quote

# ---------------------------------------------------------------------------
# Schema (design spec §5.1)
# ---------------------------------------------------------------------------

ITEMS_SCHEMA_VERSION = "knowledge-items-v1"

REQUIRED_KEYS = frozenset({
    "local_id", "kind", "statement", "quote", "quote_occurrence",
    "quantities", "negated", "evidence_basis", "hedged", "conditions",
    "scope", "value", "protocol", "source_mention", "rationale_for",
})

KNOWN_KINDS = frozenset({
    "claim", "recommendation", "protocol", "mechanism", "rationale",
    "caveat", "study_description", "source_mention", "side_topic",
    "follow_up",
})
CONDITIONS_REQUIRED_KINDS = frozenset({"claim", "recommendation", "protocol"})
KNOWN_VALUES = frozenset({"high", "normal", "low"})
KNOWN_SCOPES = frozenset({"core", "life_support", "sport_philosophy", "out_of_scope"})
KNOWN_EVIDENCE_BASIS = frozenset({
    "research", "coaching_experience", "personal_experience", "opinion",
})

NONE_STATED = "none_stated"

# Reason codes (design spec §6 "on failure" + the item-schema RED tests):
# every rejection carries exactly one of these, and `missing_conditions` is
# always distinct from the generic `missing_key`/`invalid_enum` buckets.
REASON_UNKNOWN_KEY = "unknown_key"
REASON_MISSING_KEY = "missing_key"
REASON_INVALID_ENUM = "invalid_enum"
REASON_MISSING_CONDITIONS = "missing_conditions"
REASON_QUOTE_NOT_FOUND = "quote_not_found"
REASON_QUANTITY_NOT_IN_QUOTE = "quantity_not_in_quote"
REASON_STATEMENT_QUANTITY_UNDECLARED = "statement_quantity_undeclared"
REASON_NEGATION_MISMATCH = "negation_mismatch"
REASON_NOTHING_RELEVANT_INCONSISTENT = "nothing_relevant_inconsistent"


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ValidatedItem:
    """An accepted (in-scope) item, with its resolved transcript position."""

    item_id: str
    section_id: str
    kind: str
    conditions: str | None
    quote: str
    quote_start: int  # offset within the section's own text
    quote_end: int
    global_start: int  # offset within the full compiled transcript
    global_end: int
    payload: dict = field(repr=False)


@dataclass(frozen=True)
class ExcludedItem:
    """An out-of-scope item: stored, never composed or rendered."""

    item_id: str
    section_id: str
    kind: str
    payload: dict = field(repr=False)


@dataclass(frozen=True)
class RejectedItem:
    """A schema/validation failure. Always carries a reason code (§6)."""

    section_id: str
    local_id: str | None
    reason: str
    detail: str


@dataclass(frozen=True)
class ItemValidationResult:
    accepted: list[ValidatedItem]
    rejected: list[RejectedItem]
    excluded: list[ExcludedItem]


# ---------------------------------------------------------------------------
# Quote location (design spec §5.2: "line-ending normalization only")
# ---------------------------------------------------------------------------

def _normalize_line_endings(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def locate_quote(section_text: str, quote: str, occurrence: int) -> tuple[int, int] | None:
    """Return the (start, end) span of the `occurrence`-th match of `quote`
    inside `section_text`, or None if it does not select an exact span.

    `occurrence` is 1-indexed, matching the extraction schema's
    `quote_occurrence` field. Only line-ending differences are normalized;
    whitespace and case are compared exactly.
    """
    if occurrence < 1 or not quote:
        return None
    text = _normalize_line_endings(section_text)
    needle = _normalize_line_endings(quote)
    start = -1
    for _ in range(occurrence):
        start = text.find(needle, start + 1)
        if start == -1:
            return None
    return (start, start + len(needle))


# ---------------------------------------------------------------------------
# Negation agreement (design spec §5.2: "negation token within the quote")
# ---------------------------------------------------------------------------

_NEGATION_TOKENS = frozenset({
    "not", "never", "no", "none", "without", "avoid", "avoiding",
    "don't", "doesn't", "didn't", "won't", "wouldn't", "shouldn't",
    "couldn't", "can't", "cannot", "isn't", "aren't", "wasn't", "weren't",
})
_WORD_TOKEN_RE = re.compile(r"[A-Za-z']+")


def quote_has_negation(quote: str) -> bool:
    tokens = _WORD_TOKEN_RE.findall(quote.lower())
    return any(token in _NEGATION_TOKENS for token in tokens)


# ---------------------------------------------------------------------------
# Stable IDs (`secNN-iNN`)
# ---------------------------------------------------------------------------

def _section_number_token(section_id: str) -> str:
    digits = "".join(character for character in section_id if character.isdigit())
    return f"sec{digits}" if digits else section_id


def _local_id_token(local_id: object, fallback_index: int) -> str:
    if isinstance(local_id, str) and local_id:
        digits = "".join(character for character in local_id if character.isdigit())
        if digits:
            return f"i{digits.zfill(2)}"
    return f"i{fallback_index:02d}"


def _stable_item_id(section_id: str, local_id: object, fallback_index: int) -> str:
    return f"{_section_number_token(section_id)}-{_local_id_token(local_id, fallback_index)}"


# ---------------------------------------------------------------------------
# Per-item validation
# ---------------------------------------------------------------------------

def _reject(section_id: str, local_id: object, reason: str, detail: str) -> RejectedItem:
    local_id_value = local_id if isinstance(local_id, str) else None
    return RejectedItem(section_id=section_id, local_id=local_id_value, reason=reason, detail=detail)


def _validate_one_item(
    section_id: str,
    section_text: str,
    section_start: int,
    item: object,
    fallback_index: int,
) -> ValidatedItem | ExcludedItem | RejectedItem:
    if not isinstance(item, dict):
        return _reject(section_id, None, REASON_MISSING_KEY, "item is not an object")

    local_id = item.get("local_id")

    unknown_keys = set(item.keys()) - REQUIRED_KEYS
    if unknown_keys:
        return _reject(
            section_id, local_id, REASON_UNKNOWN_KEY,
            f"unknown keys: {sorted(unknown_keys)}",
        )

    missing_keys = REQUIRED_KEYS - set(item.keys()) - {"conditions"}
    if missing_keys:
        return _reject(
            section_id, local_id, REASON_MISSING_KEY,
            f"missing keys: {sorted(missing_keys)}",
        )

    kind = item.get("kind")
    if kind not in KNOWN_KINDS:
        return _reject(section_id, local_id, REASON_INVALID_ENUM, f"unknown kind: {kind!r}")

    value = item.get("value")
    if value not in KNOWN_VALUES:
        return _reject(section_id, local_id, REASON_INVALID_ENUM, f"unknown value: {value!r}")

    scope = item.get("scope")
    if scope not in KNOWN_SCOPES:
        return _reject(section_id, local_id, REASON_INVALID_ENUM, f"unknown scope: {scope!r}")

    evidence_basis = item.get("evidence_basis")
    if evidence_basis not in KNOWN_EVIDENCE_BASIS:
        return _reject(
            section_id, local_id, REASON_INVALID_ENUM,
            f"unknown evidence_basis: {evidence_basis!r}",
        )

    hedged = item.get("hedged")
    if not isinstance(hedged, bool):
        return _reject(section_id, local_id, REASON_INVALID_ENUM, "hedged must be a boolean")

    # `conditions` requirement (design spec §5.1/§5.2): a dedicated reason
    # code, distinct from the generic schema-violation buckets above, for
    # both a missing key and an explicit null on a kind that requires it.
    conditions_present = "conditions" in item
    conditions = item.get("conditions")
    if kind in CONDITIONS_REQUIRED_KINDS:
        if not conditions_present or conditions is None:
            return _reject(
                section_id, local_id, REASON_MISSING_CONDITIONS,
                f"conditions is required (real text or {NONE_STATED!r}) for kind {kind!r}",
            )
        if not isinstance(conditions, str) or not conditions.strip():
            return _reject(
                section_id, local_id, REASON_MISSING_CONDITIONS,
                "conditions must be a non-empty string",
            )
    elif conditions_present and conditions is not None and not isinstance(conditions, str):
        return _reject(section_id, local_id, REASON_INVALID_ENUM, "conditions must be a string or null")

    quote = item.get("quote")
    occurrence = item.get("quote_occurrence")
    if not isinstance(quote, str) or not quote:
        return _reject(section_id, local_id, REASON_QUOTE_NOT_FOUND, "quote is empty or not a string")
    if not isinstance(occurrence, int) or isinstance(occurrence, bool) or occurrence < 1:
        return _reject(
            section_id, local_id, REASON_QUOTE_NOT_FOUND,
            "quote_occurrence must be a positive integer",
        )
    span = locate_quote(section_text, quote, occurrence)
    if span is None:
        return _reject(
            section_id, local_id, REASON_QUOTE_NOT_FOUND,
            "quote/quote_occurrence does not select an exact span inside the section",
        )
    quote_start, quote_end = span

    quantities = item.get("quantities")
    if not isinstance(quantities, list):
        return _reject(section_id, local_id, REASON_INVALID_ENUM, "quantities must be a list")
    for quantity in quantities:
        if not isinstance(quantity, dict):
            return _reject(section_id, local_id, REASON_QUANTITY_NOT_IN_QUOTE, "quantity entry is not an object")
        quantity_value = quantity.get("value")
        quantity_value_high = quantity.get("value_high")
        if not quantity_in_quote(quantity_value, quantity_value_high, quote):
            return _reject(
                section_id, local_id, REASON_QUANTITY_NOT_IN_QUOTE,
                f"quantity value={quantity_value!r} value_high={quantity_value_high!r} not found in quote",
            )

    # Statement-quantity fidelity (2026-09-27, Task 12 Step 6 dev-tuning): the
    # complementary direction to the quantity-in-quote check above. That
    # check confirms every *declared* quantity is backed by the quote; this
    # one confirms every number the item's own `statement` plainly asserts is
    # backed by a *declared* quantity. Real dev-set extraction produced items
    # whose `quote`/`statement` stated a number ("half a set") that never
    # made it into `quantities` -- composition then correctly refused to cite
    # a number absent from the item's own quantities array (design spec
    # §5.5), so the note silently dropped a fact the transcript plainly
    # stated. Rejecting the item here lets the existing bounded
    # re-extraction/rejection machinery (Task 5) try again instead.
    #
    # Reuses `numeric_claim_tokens`, the same tokenizer composition.py's own
    # placeholder-smuggling check uses, so both checks stay in lockstep
    # (structural words like "one"/"two"/ordinals, and vague words like
    # "several", are not flagged here either -- see that function's
    # docstring). A token that `parse_number` can't resolve to a specific
    # value (shouldn't happen for anything `numeric_claim_tokens` returns,
    # but defends against future drift between the two functions) is
    # skipped rather than treated as a violation.
    statement = item.get("statement")
    if isinstance(statement, str):
        declared_values: set = set()
        for quantity in quantities:
            if not isinstance(quantity, dict):
                continue
            for key in ("value", "value_high"):
                raw = quantity.get(key)
                if raw is not None:
                    declared_values.add(Decimal(str(raw)))
        for token in numeric_claim_tokens(statement):
            claimed_value = parse_number(token)
            if claimed_value is None:
                continue
            if claimed_value not in declared_values:
                return _reject(
                    section_id, local_id, REASON_STATEMENT_QUANTITY_UNDECLARED,
                    f"statement asserts {token!r} but no quantities entry declares it",
                )

    negated = item.get("negated")
    if not isinstance(negated, bool):
        return _reject(section_id, local_id, REASON_INVALID_ENUM, "negated must be a boolean")
    if negated != quote_has_negation(quote):
        return _reject(
            section_id, local_id, REASON_NEGATION_MISMATCH,
            "negated flag disagrees with negation tokens present in the quote",
        )

    item_id = _stable_item_id(section_id, local_id, fallback_index)

    if scope == "out_of_scope":
        return ExcludedItem(item_id=item_id, section_id=section_id, kind=kind, payload=item)

    return ValidatedItem(
        item_id=item_id,
        section_id=section_id,
        kind=kind,
        conditions=conditions if conditions_present else None,
        quote=quote,
        quote_start=quote_start,
        quote_end=quote_end,
        global_start=section_start + quote_start,
        global_end=section_start + quote_end,
        payload=item,
    )


# ---------------------------------------------------------------------------
# Overlap merge (design spec §5.2: "duplicates from section overlap ...
# merged into one item")
# ---------------------------------------------------------------------------

def _merge_overlapping(accepted: list[ValidatedItem]) -> list[ValidatedItem]:
    ordered = sorted(accepted, key=lambda candidate: (candidate.global_start, candidate.global_end))
    merged: list[ValidatedItem] = []
    for candidate in ordered:
        overlaps_kept = any(
            candidate.global_start < kept.global_end and kept.global_start < candidate.global_end
            for kept in merged
        )
        if not overlaps_kept:
            merged.append(candidate)
    return merged


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def validate_items(sections: list[dict], extractions: list[dict]) -> ItemValidationResult:
    """Validate every section's `knowledge-items-v1` extraction output.

    `sections` is the `transcript-sections-v1` output (each with
    `section_id`, `start`, `end`, `text`). `extractions` is one
    `knowledge-items-v1` payload per section, matched by `section_id`.

    Returns accepted items (overlap duplicates merged into one, per §5.2),
    rejected items with reason codes, and excluded (out-of-scope) items.
    """
    extraction_by_section = {
        extraction.get("section_id"): extraction
        for extraction in extractions
        if isinstance(extraction, dict)
    }

    all_accepted: list[ValidatedItem] = []
    all_rejected: list[RejectedItem] = []
    all_excluded: list[ExcludedItem] = []

    for section in sections:
        section_id = section["section_id"]
        section_text = section["text"]
        section_start = section.get("start", 0)
        extraction = extraction_by_section.get(section_id)
        if extraction is None:
            continue

        nothing_relevant = extraction.get("nothing_relevant", False)
        nothing_relevant_reason = extraction.get("nothing_relevant_reason")
        items = extraction.get("items", [])
        if not isinstance(items, list):
            items = []

        if nothing_relevant:
            in_scope_present = any(
                isinstance(entry, dict) and entry.get("scope") != "out_of_scope"
                for entry in items
            )
            valid_reason = isinstance(nothing_relevant_reason, str) and nothing_relevant_reason.strip()
            if not valid_reason or in_scope_present:
                all_rejected.append(_reject(
                    section_id, None, REASON_NOTHING_RELEVANT_INCONSISTENT,
                    "nothing_relevant=true requires a non-empty reason and zero in-scope items",
                ))
            continue

        for index, item in enumerate(items, start=1):
            outcome = _validate_one_item(section_id, section_text, section_start, item, index)
            if isinstance(outcome, RejectedItem):
                all_rejected.append(outcome)
            elif isinstance(outcome, ExcludedItem):
                all_excluded.append(outcome)
            else:
                all_accepted.append(outcome)

    return ItemValidationResult(
        accepted=_merge_overlapping(all_accepted),
        rejected=all_rejected,
        excluded=all_excluded,
    )


# ---------------------------------------------------------------------------
# Low-condition-confidence heuristic (design spec §5.2.1)
# ---------------------------------------------------------------------------

LOW_CONFIDENCE_KINDS = frozenset({"protocol", "recommendation"})

_HEDGE_PHRASES = (
    "usually", "typically", "generally", "often", "sometimes",
    "for most people", "for some people", "for beginners", "for someone",
    "if you're new", "if you are new", "depending on", "unless", "only if",
)

_PROXIMITY_WINDOW_CHARS = 200


def flag_low_condition_confidence(
    *,
    kind: str,
    conditions: str | None,
    quote_start: int,
    quote_end: int,
    section_text: str,
) -> bool:
    """Pure-Python heuristic (design spec §5.2.1): applied after
    `validate_items` accepts an item, never during validation, and never a
    model call. Never changes acceptance/rejection -- it is review/PodcastOps
    metadata only (§13).
    """
    if kind not in LOW_CONFIDENCE_KINDS:
        return False
    if conditions != NONE_STATED:
        return False

    window_start = max(0, quote_start - _PROXIMITY_WINDOW_CHARS)
    window_end = min(len(section_text), quote_end + _PROXIMITY_WINDOW_CHARS)
    # Exclude the item's own quote: only a hedge word the item does not
    # itself cite counts (§5.2.1).
    surrounding_text = section_text[window_start:quote_start] + section_text[quote_end:window_end]
    lowered = surrounding_text.lower()
    return any(phrase in lowered for phrase in _HEDGE_PHRASES)
