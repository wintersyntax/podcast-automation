"""Composition validation for TASK-106 Phase A Task 7 (pure decision logic).

Design authority:
docs/superpowers/specs/2026-09-24-grounded-knowledge-note-pipeline-design.md
§5.4 (composition), §5.5 (composition validation), §4 (note format v2),
§2 (locked decisions).

Everything here is pure: no network, no file I/O, no model call -- the
`note-composition-v1` shape and its validation rules, mirroring how
`items.py` holds the pure `knowledge-items-v1` schema and `validate_items`.
The model-calling orchestration (building the request, the one-retry-then-
hold policy) lives in `scripts/knowledge_eval/composition.py`, which imports
`COMPOSITION_JSON_SCHEMA` and `validate_composition` from here -- the same
relationship `scripts/knowledge_eval/extract.py` has with this package's
`items.py`.

Composition never sees the transcript (§2 locked decision): it receives only
verified in-scope items and writes bullet text that may not introduce a
number, unit, or fact absent from the cited items. Any quantity is written
as a `{q:<item_id>:<n>}` placeholder, never a literal digit or number word;
Python later replaces the placeholder with the rendered SI form (Task 9's
renderer). This module validates that the placeholder mechanism was
followed and that citations are real; it never performs the SI conversion
or substitution itself.

Deliberate interface decision: the design spec's §5.4 example JSON shows
`"tldr": ["sentence", "..."]` -- plain strings. But §5.5 explicitly requires
"every ... TL;DR sentence ... cites >= 1 existing verified item", which is
unenforceable against a bare string. Every other list in the same JSON shape
(`research_discussed`, `also_discussed`, `takeaways`, `follow_up`) is
already `[{"text": ..., "item_ids": [...]}, ...]`, so this module requires
`tldr` in that same shape too, for consistency and so §5.5's citation rule
is actually checkable. This is a considered deviation from the literal
example, not an oversight -- see the Task 7 implementation note in the
phase-A plan.

Similarly, while §5.5 names only bullets/TL;DR/takeaways for the "cites >= 1
item" and "no free numbers" rules, this module applies both checks uniformly
to every text-bearing unit (including `research_discussed`, `also_discussed`,
and `follow_up` entries), since those also carry `item_ids` and the same
grounding rationale applies; leaving them unchecked would allow ungrounded
free text through a side door.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .quantities import numeric_claim_tokens

COMPOSITION_SCHEMA_VERSION = "note-composition-v1"

# ---------------------------------------------------------------------------
# Schema (design spec §5.4, with the tldr-shape decision above)
# ---------------------------------------------------------------------------

_UNIT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["text", "item_ids"],
    "properties": {
        "text": {"type": "string"},
        "item_ids": {"type": "array", "items": {"type": "string"}},
    },
}

_BULLET_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["text", "item_ids", "primary_item_id", "asserts"],
    "properties": {
        "text": {"type": "string"},
        "item_ids": {"type": "array", "items": {"type": "string"}},
        "primary_item_id": {"type": ["string", "null"]},
        "asserts": {"type": "boolean"},
    },
}

_SECTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["title", "bullets", "protocol_item_ids"],
    "properties": {
        "title": {"type": "string"},
        "bullets": {"type": "array", "items": _BULLET_SCHEMA},
        "protocol_item_ids": {"type": "array", "items": {"type": "string"}},
    },
}

COMPOSITION_JSON_SCHEMA: dict[str, Any] = {
    "name": "note_composition_v1",
    "schema": {
        "type": "object",
        "required": [
            "tldr", "sections", "research_discussed", "also_discussed",
            "takeaways", "follow_up",
        ],
        "properties": {
            "tldr": {"type": "array", "items": _UNIT_SCHEMA},
            "sections": {"type": "array", "items": _SECTION_SCHEMA},
            "research_discussed": {"type": "array", "items": _UNIT_SCHEMA},
            "also_discussed": {"type": "array", "items": _UNIT_SCHEMA},
            "takeaways": {"type": "array", "items": _UNIT_SCHEMA},
            "follow_up": {"type": "array", "items": _UNIT_SCHEMA},
        },
    },
}

_TOP_LEVEL_KEYS = frozenset({
    "tldr", "sections", "research_discussed", "also_discussed", "takeaways", "follow_up",
})
_SECTION_KEYS = frozenset({"title", "bullets", "protocol_item_ids"})
_BULLET_KEYS = frozenset({"text", "item_ids", "primary_item_id", "asserts"})
_UNIT_KEYS = frozenset({"text", "item_ids"})

# Reason codes, mirroring items.py's REASON_* convention.
REASON_UNKNOWN_KEY = "unknown_key"
REASON_MISSING_KEY = "missing_key"
REASON_UNCITED_UNIT = "uncited_unit"
REASON_UNKNOWN_ITEM_ID = "unknown_item_id"
REASON_OUT_OF_SCOPE_CITED = "out_of_scope_cited"
REASON_FREE_NUMBER = "free_number_in_text"
REASON_INVALID_PLACEHOLDER = "invalid_placeholder"
REASON_HIGH_VALUE_NOT_CITED = "high_value_item_not_cited"
REASON_PROTOCOL_NOT_LISTED = "protocol_item_not_listed"
REASON_ALSO_DISCUSSED_SCOPE = "also_discussed_wrong_scope"


# ---------------------------------------------------------------------------
# Free-number detection (design spec §5.5)
# ---------------------------------------------------------------------------

_PLACEHOLDER_RE = re.compile(r"\{q:([A-Za-z0-9_\-]+):(\d+)\}")


def _free_number_violations(text: str) -> list[str]:
    """Digits or exact-quantity number words found outside `{q:...}`
    placeholders. Empty when the text is clean.

    Strips placeholders, then delegates the actual token scan to
    `quantities.numeric_claim_tokens` -- the same tokenizer items.py's
    statement-quantity-fidelity check uses, so the two checks (composition
    may not smuggle a number past a placeholder; an item's own statement may
    not assert a number its own `quantities` doesn't declare) can never
    drift out of sync with each other. See that function's docstring for the
    mixed-alphanumeric-identifier and hyphenated-compound handling.
    """
    stripped = _PLACEHOLDER_RE.sub(" ", text)
    return numeric_claim_tokens(stripped)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompositionError:
    code: str
    detail: str


@dataclass(frozen=True)
class CompositionValidationResult:
    valid: bool
    errors: list[CompositionError] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Validation (design spec §5.5)
# ---------------------------------------------------------------------------


def _check_keys(d: object, allowed: frozenset, required: frozenset, path: str, errors: list[CompositionError]) -> bool:
    """Reject unknown keys and report missing required keys. Returns False
    when `d` cannot be treated as a well-formed object at all (caller should
    not descend further)."""
    if not isinstance(d, dict):
        errors.append(CompositionError(REASON_MISSING_KEY, f"{path} is not an object"))
        return False
    ok = True
    for key in d:
        if key not in allowed:
            errors.append(CompositionError(REASON_UNKNOWN_KEY, f"{path}.{key} is not a recognized field"))
            ok = False
    for key in required:
        if key not in d:
            errors.append(CompositionError(REASON_MISSING_KEY, f"{path}.{key} is required"))
            ok = False
    return ok


def _validate_unit_text(
    text: object,
    item_ids: object,
    path: str,
    *,
    errors: list[CompositionError],
    items_by_id: dict[str, dict],
    cited_ids: set[str],
    require_citation: bool = True,
) -> list[str]:
    """Citation, out-of-scope, free-number, and placeholder checks shared by
    every text-bearing unit (tldr entry, bullet, research_discussed,
    also_discussed, takeaway, follow_up). Returns the item_ids that resolved
    to a real item (for callers that need them, e.g. also_discussed's
    scope check)."""
    if not isinstance(text, str):
        errors.append(CompositionError(REASON_MISSING_KEY, f"{path}.text is not a string"))
        text = ""
    if not isinstance(item_ids, list):
        errors.append(CompositionError(REASON_MISSING_KEY, f"{path}.item_ids is not an array"))
        item_ids = []

    if require_citation and not item_ids:
        errors.append(CompositionError(REASON_UNCITED_UNIT, f"{path} cites no items"))

    resolved_item_ids: list[str] = []
    for item_id in item_ids:
        item = items_by_id.get(item_id)
        if item is None:
            errors.append(CompositionError(REASON_UNKNOWN_ITEM_ID, f"{path} cites unknown item_id {item_id!r}"))
            continue
        resolved_item_ids.append(item_id)
        cited_ids.add(item_id)
        if item.get("scope") == "out_of_scope":
            errors.append(CompositionError(REASON_OUT_OF_SCOPE_CITED, f"{path} cites out-of-scope item {item_id!r}"))

    violations = _free_number_violations(text)
    if violations:
        errors.append(CompositionError(
            REASON_FREE_NUMBER,
            f"{path} contains free number(s) outside {{q:...}} placeholders: {violations!r}",
        ))

    for match in _PLACEHOLDER_RE.finditer(text):
        ref_item_id, index_str = match.group(1), match.group(2)
        index = int(index_str)
        item = items_by_id.get(ref_item_id)
        if item is None:
            errors.append(CompositionError(REASON_INVALID_PLACEHOLDER, f"{path} placeholder cites unknown item {ref_item_id!r}"))
            continue
        if ref_item_id not in item_ids:
            errors.append(CompositionError(
                REASON_INVALID_PLACEHOLDER,
                f"{path} placeholder cites item {ref_item_id!r} which is not in this unit's item_ids",
            ))
            continue
        quantities = item.get("quantities") or []
        if not (0 <= index < len(quantities)):
            errors.append(CompositionError(
                REASON_INVALID_PLACEHOLDER,
                f"{path} placeholder index {index} is out of range for item {ref_item_id!r} ({len(quantities)} quantities)",
            ))

    return resolved_item_ids


def validate_unit(text: str, item_ids: list[str], items_by_id: dict[str, dict]) -> list[CompositionError]:
    """Public wrapper around the same per-unit checks `validate_composition`
    applies to every bullet/tldr/takeaway/etc (citation, out-of-scope,
    free-number, and placeholder checks) -- for reuse by Task 8's targeted-
    fix output validation, which needs the identical rules applied to one
    unit at a time without a full composition payload around it.
    """
    errors: list[CompositionError] = []
    cited_ids: set[str] = set()
    _validate_unit_text(
        text, item_ids, "fixed_unit",
        errors=errors, items_by_id=items_by_id, cited_ids=cited_ids,
    )
    return errors


def validate_composition(composition: dict, items: list[dict]) -> CompositionValidationResult:
    """Validate one `note-composition-v1` payload against the items it may
    cite. Never raises -- every failure becomes a `CompositionError`, and the
    caller decides the retry/hold policy (design spec §5.5, §8.3).
    """
    errors: list[CompositionError] = []

    if not isinstance(composition, dict):
        return CompositionValidationResult(valid=False, errors=[CompositionError(REASON_MISSING_KEY, "composition is not an object")])

    items_by_id = {item["item_id"]: item for item in items if "item_id" in item}
    cited_ids: set[str] = set()
    protocol_listed: set[str] = set()

    _check_keys(composition, _TOP_LEVEL_KEYS, _TOP_LEVEL_KEYS, "composition", errors)

    def check_unit_list(raw: object, key: str, *, scope_check=None) -> None:
        if not isinstance(raw, list):
            if key in composition:
                errors.append(CompositionError(REASON_MISSING_KEY, f"composition.{key} is not an array"))
            return
        for i, entry in enumerate(raw):
            path = f"{key}[{i}]"
            if not _check_keys(entry, _UNIT_KEYS, _UNIT_KEYS, path, errors):
                continue
            resolved = _validate_unit_text(
                entry.get("text"), entry.get("item_ids"), path,
                errors=errors, items_by_id=items_by_id, cited_ids=cited_ids,
            )
            if scope_check is not None:
                for item_id in resolved:
                    scope_check(item_id, path)

    check_unit_list(composition.get("tldr"), "tldr")

    def _also_discussed_scope(item_id: str, path: str) -> None:
        item = items_by_id[item_id]
        if item.get("kind") != "side_topic" and item.get("value") != "low":
            errors.append(CompositionError(
                REASON_ALSO_DISCUSSED_SCOPE,
                f"{path} cites {item_id!r}, which is neither a side_topic item nor low-value",
            ))

    check_unit_list(composition.get("also_discussed"), "also_discussed", scope_check=_also_discussed_scope)
    check_unit_list(composition.get("research_discussed"), "research_discussed")
    check_unit_list(composition.get("takeaways"), "takeaways")
    check_unit_list(composition.get("follow_up"), "follow_up")

    sections = composition.get("sections")
    if isinstance(sections, list):
        for si, section in enumerate(sections):
            spath = f"sections[{si}]"
            if not _check_keys(section, _SECTION_KEYS, _SECTION_KEYS, spath, errors):
                continue

            bullets = section.get("bullets")
            if not isinstance(bullets, list):
                errors.append(CompositionError(REASON_MISSING_KEY, f"{spath}.bullets is not an array"))
                bullets = []
            for bi, bullet in enumerate(bullets):
                bpath = f"{spath}.bullets[{bi}]"
                if not _check_keys(bullet, _BULLET_KEYS, _BULLET_KEYS, bpath, errors):
                    continue
                item_ids = bullet.get("item_ids")
                _validate_unit_text(
                    bullet.get("text"), item_ids, bpath,
                    errors=errors, items_by_id=items_by_id, cited_ids=cited_ids,
                )
                primary = bullet.get("primary_item_id")
                if primary is not None and primary not in (item_ids or []):
                    errors.append(CompositionError(
                        REASON_UNKNOWN_ITEM_ID,
                        f"{bpath}.primary_item_id {primary!r} is not among this bullet's item_ids",
                    ))

            protocol_ids = section.get("protocol_item_ids")
            if isinstance(protocol_ids, list):
                for item_id in protocol_ids:
                    item = items_by_id.get(item_id)
                    if item is None:
                        errors.append(CompositionError(REASON_UNKNOWN_ITEM_ID, f"{spath}.protocol_item_ids cites unknown item_id {item_id!r}"))
                        continue
                    cited_ids.add(item_id)
                    protocol_listed.add(item_id)
                    if item.get("scope") == "out_of_scope":
                        errors.append(CompositionError(REASON_OUT_OF_SCOPE_CITED, f"{spath}.protocol_item_ids cites out-of-scope item {item_id!r}"))
            else:
                errors.append(CompositionError(REASON_MISSING_KEY, f"{spath}.protocol_item_ids is not an array"))
    elif "sections" in composition:
        errors.append(CompositionError(REASON_MISSING_KEY, "composition.sections is not an array"))

    for item_id, item in items_by_id.items():
        if item.get("scope") == "out_of_scope":
            continue
        if item.get("value") == "high" and item_id not in cited_ids:
            errors.append(CompositionError(REASON_HIGH_VALUE_NOT_CITED, f"high-value item {item_id!r} is not cited anywhere"))
        if item.get("kind") == "protocol" and item_id not in protocol_listed:
            errors.append(CompositionError(REASON_PROTOCOL_NOT_LISTED, f"protocol item {item_id!r} does not appear in any section's protocol_item_ids"))

    return CompositionValidationResult(valid=not errors, errors=errors)
