"""Review, targeted fixes, and the `condition_dropped` routing gate for
TASK-106 Phase A Task 8.

Design authority:
docs/superpowers/specs/2026-09-24-grounded-knowledge-note-pipeline-design.md
§5.6 (review), §5.6.1 (`condition_dropped` re-extraction routing), §5.6.2
(targeted fix with a bounded transcript window), §8.1 (rejected items,
footer line), §8.2 (units failing review after fix rounds).

The pure `note-review-v1`/`note-fix-v1` schemas, verdict validation, the
bounded fix-excerpt window, and the §8.2 decision policy live in
`podcast_engine.knowledge.notes_v2.review` (mirroring this package's
relationship to `notes_v2.items`/`notes_v2.composition`). This module holds
the model-calling orchestration: one review call, the `condition_dropped` ->
re-extraction routing via Task 5's `reextract_section`, and the bounded
(at most two) targeted-fix rounds -- each role (review, fix, extraction) is
injected as its own client, since design spec §9 pins each to a different
model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from podcast_engine.knowledge.notes_v2.composition import validate_unit
from podcast_engine.knowledge.notes_v2.review import (
    FIX_JSON_SCHEMA,
    REVIEW_JSON_SCHEMA,
    ReviewDecision,
    ReviewValidationResult,
    decide,
    enumerate_review_units,
    splice_fixed_units,
    validate_review,
)
from scripts.knowledge_eval.extract import EXTRACTION_JSON_SCHEMA, reextract_section

_REVIEW_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "knowledge_eval" / "review-v1.md"
_FIX_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "knowledge_eval" / "fix-v1.md"


def load_review_prompt_v1() -> str:
    return _REVIEW_PROMPT_PATH.read_text(encoding="utf-8")


def load_fix_prompt_v1() -> str:
    return _FIX_PROMPT_PATH.read_text(encoding="utf-8")


def _parse_json_content(response_payload: object) -> dict | None:
    if not isinstance(response_payload, dict):
        return None
    try:
        content = response_payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None
    if not isinstance(content, str):
        return None
    try:
        parsed = json.loads(content)
    except ValueError:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


# ---------------------------------------------------------------------------
# Review (no retry -- design spec §5.6: a missing/duplicate verdict, or
# unparseable output, holds the run directly)
# ---------------------------------------------------------------------------


def build_review_messages(
    *, units_by_id: dict[str, dict], items_by_id: dict[str, dict], episode_context: dict, prompt_text: str,
) -> list[dict]:
    """Deterministic messages for one review call. For each unit, the
    reviewer sees its text and the statements+quotes of the items it cites
    (design spec §5.6: "the text and the cited items' statements and
    quotes... not the full transcript") -- never the transcript.
    """
    system = {"role": "system", "content": prompt_text}
    units_payload = []
    for unit_id in sorted(units_by_id):
        unit = units_by_id[unit_id]
        cited_items = [
            {
                "item_id": item_id,
                "statement": items_by_id.get(item_id, {}).get("statement"),
                "quote": items_by_id.get(item_id, {}).get("quote"),
            }
            for item_id in unit.get("item_ids", [])
        ]
        units_payload.append({"unit_id": unit_id, "text": unit.get("text"), "cited_items": cited_items})
    user_payload: dict[str, Any] = {"units": units_payload, "episode_context": episode_context}
    user = {"role": "user", "content": json.dumps(user_payload, sort_keys=True, ensure_ascii=False)}
    return [system, user]


@dataclass(frozen=True)
class ReviewRunResult:
    review: dict | None
    validation: ReviewValidationResult | None
    verdicts_by_unit_id: dict[str, dict]
    held: bool
    failure_reason: str | None


def review_units(
    units_by_id: dict[str, dict],
    items_by_id: dict[str, dict],
    *,
    client,
    model: str,
    provider: str,
    episode_context: dict,
    prompt_text: str | None = None,
    schema: dict | None = REVIEW_JSON_SCHEMA,
) -> ReviewRunResult:
    """One review call over exactly `units_by_id` (which may be every unit
    in a composition, or a bounded re-review subset -- design spec: "only
    changed units are re-reviewed"). No retry.
    """
    prompt_text = prompt_text if prompt_text is not None else load_review_prompt_v1()
    messages = build_review_messages(units_by_id=units_by_id, items_by_id=items_by_id, episode_context=episode_context, prompt_text=prompt_text)
    result = client.call(model=model, provider=provider, messages=messages, schema=schema)
    review = _parse_json_content(result.response_payload)

    if review is None:
        return ReviewRunResult(review=None, validation=None, verdicts_by_unit_id={}, held=True, failure_reason="invalid_json_response")

    validation = validate_review(review, units_by_id)
    if not validation.valid:
        return ReviewRunResult(review=review, validation=validation, verdicts_by_unit_id={}, held=True, failure_reason="validation_failed")

    return ReviewRunResult(review=review, validation=validation, verdicts_by_unit_id=validation.verdicts_by_unit_id, held=False, failure_reason=None)


def review_composition(
    composition: dict,
    items_by_id: dict[str, dict],
    *,
    client,
    model: str,
    provider: str,
    episode_context: dict,
    prompt_text: str | None = None,
    schema: dict | None = REVIEW_JSON_SCHEMA,
) -> ReviewRunResult:
    """One review call over every TL;DR/bullet/takeaway unit in `composition`."""
    units_by_id = enumerate_review_units(composition)
    return review_units(units_by_id, items_by_id, client=client, model=model, provider=provider, episode_context=episode_context, prompt_text=prompt_text, schema=schema)


# ---------------------------------------------------------------------------
# condition_dropped routing (design spec §5.6.1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConditionDroppedRoutingResult:
    changed: bool
    new_item: dict | None
    new_extraction: dict | None


def route_condition_dropped(
    *,
    original_item: dict,
    section: dict,
    verdict_reason: str | None,
    client,
    model: str,
    provider: str,
    episode_context: dict,
    prompt_text: str | None = None,
    schema: dict | None = None,
) -> ConditionDroppedRoutingResult:
    """Exactly one bounded re-extraction attempt of `original_item`'s
    owning section, via Task 5's `reextract_section` (design spec §5.6.1).
    `changed=True` only when the re-extraction returns a materially
    different item for the same `local_id` -- a changed `conditions` or
    quote span. Never loops or retries itself, and never consumes a
    fix-round slot -- the caller decides what happens next.
    """
    rejected_items = [{
        "item": original_item,
        "reason": "condition_dropped",
        "detail": verdict_reason or "review flagged a dropped qualifying condition",
    }]
    new_extraction = reextract_section(
        section,
        client=client,
        model=model,
        provider=provider,
        episode_context=episode_context,
        rejected_items=rejected_items,
        prompt_text=prompt_text,
        schema=schema if schema is not None else EXTRACTION_JSON_SCHEMA,
    )
    if new_extraction is None:
        return ConditionDroppedRoutingResult(changed=False, new_item=None, new_extraction=None)

    new_items = new_extraction.get("items")
    if not isinstance(new_items, list):
        return ConditionDroppedRoutingResult(changed=False, new_item=None, new_extraction=new_extraction)

    new_item = next(
        (item for item in new_items if isinstance(item, dict) and item.get("local_id") == original_item.get("local_id")),
        None,
    )
    if new_item is None:
        return ConditionDroppedRoutingResult(changed=False, new_item=None, new_extraction=new_extraction)

    changed = (
        new_item.get("conditions") != original_item.get("conditions")
        or new_item.get("quote") != original_item.get("quote")
        or new_item.get("quote_occurrence") != original_item.get("quote_occurrence")
    )
    return ConditionDroppedRoutingResult(changed=changed, new_item=new_item if changed else None, new_extraction=new_extraction)


# ---------------------------------------------------------------------------
# Targeted fix (design spec §5.6.2)
# ---------------------------------------------------------------------------


def build_fix_messages(
    *,
    flagged_units: list[dict],  # [{"unit_id", "text", "item_ids", "reason"}]
    items_by_id: dict[str, dict],
    episode_context: dict,
    prompt_text: str,
    excerpts_by_unit_id: dict[str, dict] | None = None,
) -> list[dict]:
    """Deterministic messages for one targeted-fix call: only the flagged
    units, with the verdict reason, and -- only for a unit reaching this
    step because of an unresolved `condition_dropped` verdict -- one
    bounded transcript excerpt (design spec §5.6.2). Every other fix reason
    uses the existing items-only fix input.
    """
    system = {"role": "system", "content": prompt_text}
    excerpts_by_unit_id = excerpts_by_unit_id or {}
    units_payload = []
    for unit in sorted(flagged_units, key=lambda u: u["unit_id"]):
        cited_items = [
            {
                "item_id": item_id,
                "statement": items_by_id.get(item_id, {}).get("statement"),
                "quote": items_by_id.get(item_id, {}).get("quote"),
            }
            for item_id in unit.get("item_ids", [])
        ]
        entry: dict[str, Any] = {
            "unit_id": unit["unit_id"],
            "text": unit.get("text"),
            "item_ids": list(unit.get("item_ids", [])),
            "reason": unit.get("reason"),
            "cited_items": cited_items,
        }
        excerpt = excerpts_by_unit_id.get(unit["unit_id"])
        if excerpt is not None:
            entry["transcript_excerpt"] = excerpt["text"]
        units_payload.append(entry)
    user_payload: dict[str, Any] = {"flagged_units": units_payload, "episode_context": episode_context}
    user = {"role": "user", "content": json.dumps(user_payload, sort_keys=True, ensure_ascii=False)}
    return [system, user]


@dataclass(frozen=True)
class FixRunResult:
    fixed_units_by_id: dict[str, dict]
    invalid_unit_ids: list[str]
    failure_reason: str | None


def apply_fix_round(
    flagged_units: list[dict],
    *,
    items_by_id: dict[str, dict],
    client,
    model: str,
    provider: str,
    episode_context: dict,
    excerpts_by_unit_id: dict[str, dict] | None = None,
    prompt_text: str | None = None,
    schema: dict | None = FIX_JSON_SCHEMA,
) -> FixRunResult:
    """One targeted-fix call for exactly `flagged_units`. A fixed unit that
    still fails the same citation/free-number checks `validate_composition`
    applies (via `validate_unit`) is reported in `invalid_unit_ids` rather
    than spliced in -- it keeps its prior text and stays counted as failing.
    """
    prompt_text = prompt_text if prompt_text is not None else load_fix_prompt_v1()
    messages = build_fix_messages(
        flagged_units=flagged_units, items_by_id=items_by_id, episode_context=episode_context,
        prompt_text=prompt_text, excerpts_by_unit_id=excerpts_by_unit_id,
    )
    result = client.call(model=model, provider=provider, messages=messages, schema=schema)
    parsed = _parse_json_content(result.response_payload)
    flagged_ids = {u["unit_id"] for u in flagged_units}

    if parsed is None or not isinstance(parsed.get("fixed_units"), list):
        return FixRunResult(fixed_units_by_id={}, invalid_unit_ids=sorted(flagged_ids), failure_reason="invalid_json_response")

    fixed_units_by_id: dict[str, dict] = {}
    invalid_unit_ids: list[str] = []
    for entry in parsed["fixed_units"]:
        if not isinstance(entry, dict):
            continue
        unit_id = entry.get("unit_id")
        if unit_id not in flagged_ids:
            continue
        text = entry.get("text")
        item_ids = entry.get("item_ids")
        if not isinstance(text, str) or not isinstance(item_ids, list):
            invalid_unit_ids.append(unit_id)
            continue
        errors = validate_unit(text, item_ids, items_by_id)
        if errors:
            invalid_unit_ids.append(unit_id)
            continue
        fixed_units_by_id[unit_id] = {
            "text": text,
            "item_ids": item_ids,
            "primary_item_id": entry.get("primary_item_id"),
            "asserts": entry.get("asserts"),
        }

    for unit_id in flagged_ids:
        if unit_id not in fixed_units_by_id and unit_id not in invalid_unit_ids:
            invalid_unit_ids.append(unit_id)

    return FixRunResult(fixed_units_by_id=fixed_units_by_id, invalid_unit_ids=invalid_unit_ids, failure_reason=None)


# ---------------------------------------------------------------------------
# Orchestration: review -> condition_dropped routing (unbounded-by-round) ->
# up to two ordinary fix rounds -> the §8.2 decision.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReviewFixRunResult:
    composition: dict
    final_verdicts: dict[str, str]
    rounds_used: int
    decision: ReviewDecision | None
    held: bool
    failure_reason: str | None


def run_review_and_fix(
    composition: dict,
    items_by_id: dict[str, dict],
    *,
    review_client,
    review_model: str,
    review_provider: str,
    fix_client,
    fix_model: str,
    fix_provider: str,
    episode_context: dict,
    extraction_client=None,
    extraction_model: str | None = None,
    extraction_provider: str | None = None,
    sections_by_id: dict[str, dict] | None = None,
    review_prompt_text: str | None = None,
    fix_prompt_text: str | None = None,
    extract_prompt_text: str | None = None,
    max_fix_rounds: int = 2,
    rejected_high_value_item_ids: list[str] | None = None,
) -> ReviewFixRunResult:
    """The whole Task 8 pipeline for one composed note. Items in
    `items_by_id` that may be routed through `condition_dropped` must
    include `"local_id"` and `"section_id"` (the raw extraction fields);
    `sections_by_id` maps that `section_id` to `{"section_id", "text"}` for
    `reextract_section`.
    """
    sections_by_id = sections_by_id or {}
    items_by_id = dict(items_by_id)

    initial_review = review_composition(
        composition, items_by_id, client=review_client, model=review_model, provider=review_provider,
        episode_context=episode_context, prompt_text=review_prompt_text,
    )
    if initial_review.held:
        return ReviewFixRunResult(composition=composition, final_verdicts={}, rounds_used=0, decision=None, held=True, failure_reason=initial_review.failure_reason)

    current_composition = composition
    units_by_id = enumerate_review_units(current_composition)
    verdicts = {uid: v["verdict"] for uid, v in initial_review.verdicts_by_unit_id.items()}
    reasons = {uid: v["reason"] for uid, v in initial_review.verdicts_by_unit_id.items()}

    # --- condition_dropped routing: never consumes a fix-round slot ---
    for unit_id, verdict in list(verdicts.items()):
        if verdict != "condition_dropped" or extraction_client is None:
            continue
        unit = units_by_id[unit_id]
        if len(unit["item_ids"]) != 1:
            continue  # ambiguous which item to re-extract; falls through to an ordinary fix
        item_id = unit["item_ids"][0]
        original_item = items_by_id.get(item_id)
        if not original_item or "local_id" not in original_item or "section_id" not in original_item:
            continue
        section = sections_by_id.get(original_item["section_id"])
        if section is None:
            continue

        routing = route_condition_dropped(
            original_item=original_item, section=section, verdict_reason=reasons.get(unit_id),
            client=extraction_client, model=extraction_model, provider=extraction_provider,
            episode_context=episode_context, prompt_text=extract_prompt_text,
        )
        if not routing.changed or routing.new_item is None:
            continue  # falls through to the ordinary targeted-fix path below

        items_by_id[item_id] = {
            **original_item,
            "conditions": routing.new_item.get("conditions"),
            "quote": routing.new_item.get("quote"),
            "quote_occurrence": routing.new_item.get("quote_occurrence"),
        }
        fix_result = apply_fix_round(
            [{"unit_id": unit_id, "text": unit["text"], "item_ids": unit["item_ids"], "reason": "condition recovered by re-extraction"}],
            items_by_id=items_by_id, client=fix_client, model=fix_model, provider=fix_provider,
            episode_context=episode_context, prompt_text=fix_prompt_text,
        )
        if unit_id not in fix_result.fixed_units_by_id:
            continue
        current_composition = splice_fixed_units(current_composition, fix_result.fixed_units_by_id)
        units_by_id = enumerate_review_units(current_composition)
        re_review = review_units(
            {unit_id: units_by_id[unit_id]}, items_by_id, client=review_client, model=review_model,
            provider=review_provider, episode_context=episode_context, prompt_text=review_prompt_text,
        )
        if not re_review.held and unit_id in re_review.verdicts_by_unit_id:
            verdicts[unit_id] = re_review.verdicts_by_unit_id[unit_id]["verdict"]
            reasons[unit_id] = re_review.verdicts_by_unit_id[unit_id]["reason"]
        # Otherwise the unit keeps its condition_dropped verdict and falls
        # through to the ordinary targeted-fix path below.

    # --- ordinary targeted-fix rounds (bounded to max_fix_rounds) ---
    rounds_used = 0
    while rounds_used < max_fix_rounds:
        failing_ids = [uid for uid, v in verdicts.items() if v != "supported"]
        if not failing_ids:
            break
        rounds_used += 1

        flagged = [
            {"unit_id": uid, "text": units_by_id[uid]["text"], "item_ids": units_by_id[uid]["item_ids"], "reason": reasons.get(uid)}
            for uid in failing_ids
        ]
        fix_result = apply_fix_round(
            flagged, items_by_id=items_by_id, client=fix_client, model=fix_model, provider=fix_provider,
            episode_context=episode_context, prompt_text=fix_prompt_text,
        )
        if not fix_result.fixed_units_by_id:
            break

        current_composition = splice_fixed_units(current_composition, fix_result.fixed_units_by_id)
        units_by_id = enumerate_review_units(current_composition)
        changed_ids = list(fix_result.fixed_units_by_id.keys())

        re_review = review_units(
            {uid: units_by_id[uid] for uid in changed_ids}, items_by_id, client=review_client, model=review_model,
            provider=review_provider, episode_context=episode_context, prompt_text=review_prompt_text,
        )
        if re_review.held:
            continue  # changed units keep their prior (still-failing) verdicts
        for unit_id in changed_ids:
            if unit_id in re_review.verdicts_by_unit_id:
                verdicts[unit_id] = re_review.verdicts_by_unit_id[unit_id]["verdict"]
                reasons[unit_id] = re_review.verdicts_by_unit_id[unit_id]["reason"]

    decision = decide(
        verdicts, units_by_id=units_by_id, items_by_id=items_by_id,
        rejected_high_value_item_ids=rejected_high_value_item_ids,
    )
    return ReviewFixRunResult(
        composition=current_composition,
        final_verdicts=verdicts,
        rounds_used=rounds_used,
        decision=decision,
        held=decision.action == "hold",
        failure_reason=decision.hold_reason if decision.action == "hold" else None,
    )
