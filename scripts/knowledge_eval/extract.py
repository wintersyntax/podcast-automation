"""Extraction stage for TASK-106 Phase A Task 5.

Design authority:
docs/superpowers/specs/2026-09-24-grounded-knowledge-note-pipeline-design.md
§5.1 (extraction, `knowledge-items-v1`), §2 (locked decisions), §8.1
(rejected-item re-extraction policy).

One OpenRouter call per section, through
`scripts.knowledge_eval.openrouter_eval.OpenRouterEvalClient` (or any object
with the same `.call(*, model, provider, messages, schema) -> CallResult`
shape -- tests inject a fake). Every response is parsed and re-validated by
`podcast_engine.knowledge.notes_v2.items.validate_items` regardless of
whether the served model claims structured-output support (§9: "the response
is always parsed and validated by Python regardless"). This module never
patches over what the model returned: in particular, it never defaults a
missing/`null` `conditions` field to `"none_stated"` itself -- that stays a
Task 3 validation rejection, the same class of failure as unparseable JSON.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from podcast_engine.knowledge.notes_v2.items import ITEMS_SCHEMA_VERSION, ItemValidationResult, validate_items

EXTRACTION_SCHEMA_VERSION = ITEMS_SCHEMA_VERSION  # "knowledge-items-v1"

# Task 12 Step 3 tuning (2026-09-26) compared extract-v1.md, extract-v2.md,
# and extract-v3.md on `openai/gpt-6-luna` and decided extract-v2.md as the
# production extraction prompt (v2: 73.2% avg dev coverage vs v1's 45.0%,
# zero `wrong` answers; v3 regressed and was not adopted -- see
# docs/superpowers/plans/2026-09-24-grounded-knowledge-note-pipeline-phase-a.md
# Task 12 Step 3). `_PROMPT_PATH` is this module's *default* prompt for every
# caller that does not pass its own `prompt_text` override (Task 12 Step 3's
# whole point is comparing overrides against this default) -- it must always
# point at whichever prompt is *currently decided as production*, the same
# way `composition.py`/`review.py`/`grade.py`/`questions.py`/`gap_check.py`
# each keep exactly one live prompt file. A 2026-09-26 audit found this had
# silently gone stale at extract-v1.md after the v2 decision above, because
# nothing re-pointed it when v2 was adopted -- multiple downstream dev-tuning
# measurements (Task 12 Step 5's full retest, Task 12 Step 6's baseline and
# re-test) were run against the superseded v1 prompt by every driver script
# that built an extraction `RoleConfig` without an explicit `prompt_text`
# override, entirely unnoticed because no test asserted this default matched
# the decided version (added below: `test_default_extraction_prompt_matches_the_decided_production_version`).
# Update this path (and that test's expected file) the next time a new
# extraction prompt version is decided.
_PROMPT_PATH = (
    Path(__file__).resolve().parents[2] / "prompts" / "knowledge_eval" / "extract-v2.md"
)

_GAP_FILL_PROMPT_PATH = (
    Path(__file__).resolve().parents[2] / "prompts" / "knowledge_eval" / "gap-fill-v1.md"
)

EXTRACTION_JSON_SCHEMA: dict[str, Any] = {
    "name": "knowledge_items_v1",
    "schema": {
        "type": "object",
        "required": ["section_id", "nothing_relevant", "nothing_relevant_reason", "items"],
        "additionalProperties": False,
        "properties": {
            "section_id": {"type": "string"},
            "nothing_relevant": {"type": "boolean"},
            "nothing_relevant_reason": {"type": ["string", "null"]},
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": [
                        "local_id", "kind", "statement", "quote", "quote_occurrence",
                        "quantities", "negated", "evidence_basis", "hedged", "conditions",
                        "scope", "value", "protocol", "source_mention", "rationale_for",
                    ],
                    "additionalProperties": False,
                    "properties": {
                        "local_id": {"type": "string"},
                        "kind": {
                            "type": "string",
                            "enum": [
                                "claim", "recommendation", "protocol", "mechanism",
                                "rationale", "caveat", "study_description",
                                "source_mention", "side_topic", "follow_up",
                            ],
                        },
                        "statement": {"type": "string"},
                        "quote": {"type": "string"},
                        "quote_occurrence": {"type": "integer"},
                        "quantities": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "required": ["value", "value_high", "unit_as_spoken"],
                                "additionalProperties": False,
                                "properties": {
                                    "value": {"type": "number"},
                                    "value_high": {"type": ["number", "null"]},
                                    "unit_as_spoken": {"type": "string"},
                                },
                            },
                        },
                        "negated": {"type": "boolean"},
                        # anyOf, not `type: [string, null]` + `enum` (with None in the enum
                        # list) -- a real live call against anthropic/claude-haiku-4.5 (Task
                        # 12 Step 2 candidate probe) rejected the old shape outright: "Enum
                        # value 'research' does not match declared type ['string', 'null']".
                        # `anyOf` is the form every provider's strict-schema compiler accepts
                        # for a nullable enum.
                        "evidence_basis": {
                            "anyOf": [
                                {
                                    "type": "string",
                                    "enum": ["research", "coaching_experience", "personal_experience", "opinion"],
                                },
                                {"type": "null"},
                            ],
                        },
                        "hedged": {"type": "boolean"},
                        "conditions": {"type": ["string", "null"]},
                        "scope": {
                            "type": "string",
                            "enum": ["core", "life_support", "sport_philosophy", "out_of_scope"],
                        },
                        "value": {"type": "string", "enum": ["high", "normal", "low"]},
                        # `protocol`/`source_mention` were a bare `{"type": ["object", "null"]}`
                        # with no declared shape -- never validated by items.py (an opaque
                        # pass-through dict), but a generic untyped object is rejected by
                        # strict-schema providers (openai/gpt-6-luna: "'additionalProperties'
                        # is required to be supplied and to be false", found on the same
                        # Task 12 Step 2 probe). Both now spell out the exact shape
                        # prompts/knowledge_eval/extract-v1.md already documents.
                        "protocol": {
                            "anyOf": [
                                {
                                    "type": "object",
                                    "required": ["what", "dose", "when_for_whom", "caveat"],
                                    "additionalProperties": False,
                                    "properties": {
                                        "what": {"type": ["string", "null"]},
                                        "dose": {"type": ["string", "null"]},
                                        "when_for_whom": {"type": ["string", "null"]},
                                        "caveat": {"type": ["string", "null"]},
                                    },
                                },
                                {"type": "null"},
                            ],
                        },
                        "source_mention": {
                            "anyOf": [
                                {
                                    "type": "object",
                                    "required": ["authors_as_heard", "year_as_heard", "title_as_heard", "type"],
                                    "additionalProperties": False,
                                    "properties": {
                                        "authors_as_heard": {"type": ["string", "null"]},
                                        "year_as_heard": {"type": ["string", "null"]},
                                        "title_as_heard": {"type": ["string", "null"]},
                                        "type": {"type": ["string", "null"]},
                                    },
                                },
                                {"type": "null"},
                            ],
                        },
                        "rationale_for": {"type": ["string", "null"]},
                    },
                },
            },
        },
    },
}


class ExtractionClient(Protocol):
    """The subset of `OpenRouterEvalClient` this module depends on."""

    def call(
        self, *, model: str, provider: str, messages: list[dict], schema: dict | None
    ) -> Any:  # returns something with .response_payload
        ...


# ---------------------------------------------------------------------------
# Prompt / cache-identity helpers
# ---------------------------------------------------------------------------


def load_prompt_v1() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


def load_gap_fill_prompt_v1() -> str:
    return _GAP_FILL_PROMPT_PATH.read_text(encoding="utf-8")


def prompt_hash(prompt_text: str) -> str:
    return hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()


def section_hash(section_text: str) -> str:
    return hashlib.sha256(section_text.encode("utf-8")).hexdigest()


def build_messages(
    *,
    section: dict,
    episode_context: dict,
    prompt_text: str,
    rejected_items: list[dict] | None = None,
    gap_hints: list[dict] | None = None,
) -> list[dict]:
    """Deterministic messages for one section: identical
    (section, episode_context, prompt_text, rejected_items, gap_hints) always
    build byte-identical messages, so the client's request cache naturally
    keys on exactly (section hash, model, prompt hash, schema version, gap
    hints) -- no run-varying content (timestamps, uuids) is ever added here.

    `rejected_items`, when given, is the bounded re-extraction context
    (design spec §8.1): each entry carries the prior item payload the
    validator rejected plus why, so the model sees exactly what it must
    correct -- `{"item": <raw item payload>, "reason": <reason code>,
    "detail": <text>}`.

    `gap_hints`, when given, is the bounded gap-check re-extraction context
    (design spec §5.7's missing-item mode, Task 12 Step 5): each entry is a
    validated missing-item gap -- `{"quote": <exact transcript text>,
    "reason": <one sentence>}` -- found by a separate gap-check pass and
    confirmed by Python to actually locate in this section's text. Distinct
    from `rejected_items`: a rejection describes something the model already
    extracted and got wrong; a gap hint describes real section content the
    model never extracted at all.
    """
    system = {"role": "system", "content": prompt_text}
    user_payload: dict[str, Any] = {
        "section_id": section["section_id"],
        "section_text": section["text"],
        "episode_context": episode_context,
    }
    if rejected_items:
        user_payload["previous_rejections"] = list(rejected_items)
    if gap_hints:
        user_payload["gap_hints"] = list(gap_hints)
    user = {"role": "user", "content": json.dumps(user_payload, sort_keys=True, ensure_ascii=False)}
    return [system, user]


def build_gap_fill_messages(
    *,
    section: dict,
    episode_context: dict,
    gap_hints: list[dict],
    accepted_items: list[dict],
    prompt_text: str,
) -> list[dict]:
    """Deterministic messages for a gap-fill-only call (Task 12 Step 5
    strategy revision, 2026-09-26): unlike `build_messages`'s `gap_hints`
    path (a full-section re-extraction), this asks the model for ONLY the
    new items covering `gap_hints` -- it never re-decides already-accepted
    content. `accepted_items` (reduced to `statement`/`quote`, matching
    `gap_check.build_gap_check_messages`'s own reduction) lets the model
    recognize when a gap's content is already substantively covered under
    different wording and skip it, rather than emit a near-duplicate.

    Identical (section, episode_context, gap_hints, accepted_items,
    prompt_text) always builds byte-identical messages, matching every
    other message builder in this module's cache-friendliness contract.
    """
    system = {"role": "system", "content": prompt_text}
    user_payload: dict[str, Any] = {
        "section_id": section["section_id"],
        "section_text": section["text"],
        "episode_context": episode_context,
        "gap_hints": list(gap_hints),
        "already_accepted_items": [
            {"statement": item.get("statement"), "quote": item.get("quote")}
            for item in accepted_items
        ],
    }
    user = {"role": "user", "content": json.dumps(user_payload, sort_keys=True, ensure_ascii=False)}
    return [system, user]


def _parse_extraction_content(response_payload: object) -> dict | None:
    """Parse the model's completion as `knowledge-items-v1` JSON.

    Never raises: any shape/parse failure returns None, and the caller
    records the section as failed with no hidden retry (design spec §8.3).
    """
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
# Results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FailedSection:
    """A section whose model response could not be parsed as
    `knowledge-items-v1` JSON. Never auto-retried by this module."""

    section_id: str
    reason: str


@dataclass(frozen=True)
class ExtractionRunResult:
    extractions: dict[str, dict] = field(repr=False)  # section_id -> raw knowledge-items-v1 dict
    failed_sections: list[FailedSection] = field(default_factory=list)
    reextracted_section_ids: list[str] = field(default_factory=list)
    validation: ItemValidationResult = None


# ---------------------------------------------------------------------------
# Bounded single-section re-extraction (design spec §8.1; also used by
# Task 8's condition_dropped review routing)
# ---------------------------------------------------------------------------


def reextract_section(
    section: dict,
    *,
    client: ExtractionClient,
    model: str,
    provider: str,
    episode_context: dict,
    rejected_items: list[dict] | None = None,
    gap_hints: list[dict] | None = None,
    prompt_text: str | None = None,
    schema: dict | None = EXTRACTION_JSON_SCHEMA,
) -> dict | None:
    """One bounded re-extraction of exactly this section, carrying whichever
    of two independent context kinds the caller has: prior rejected item(s)
    plus why each was rejected (design spec §8.1, `rejected_items`,
    `[{"item": <raw item payload>, "reason": <code>, "detail": <text>}, ...]`),
    and/or validated gap-check gap(s) (design spec §5.7's missing-item mode,
    Task 12 Step 5, `gap_hints`, `[{"quote": <exact text>, "reason": <text>},
    ...]`). Either, both, or neither may be given -- an empty/`None` value
    for one omits its key from the request entirely (see `build_messages`).
    Cached and journaled like any other extraction call, via `client.call`.
    This function never loops or retries itself -- the caller decides
    whether a further attempt is warranted (design spec: exactly one bounded
    re-extraction per rejection/gap context).
    """
    prompt_text = prompt_text if prompt_text is not None else load_prompt_v1()
    messages = build_messages(
        section=section,
        episode_context=episode_context,
        prompt_text=prompt_text,
        rejected_items=rejected_items,
        gap_hints=gap_hints,
    )
    result = client.call(model=model, provider=provider, messages=messages, schema=schema)
    return _parse_extraction_content(result.response_payload)


def extract_gap_fill_items(
    section: dict,
    *,
    client: ExtractionClient,
    model: str,
    provider: str,
    episode_context: dict,
    gap_hints: list[dict],
    accepted_items: list[dict],
    prompt_text: str | None = None,
    schema: dict | None = EXTRACTION_JSON_SCHEMA,
) -> dict | None:
    """One bounded gap-fill call: returns a `knowledge-items-v1` payload
    whose `items` are meant to be ONLY the new items covering `gap_hints` --
    never a full section re-extraction (Task 12 Step 5 strategy revision,
    2026-09-26: the original "re-extract the whole section" approach
    measured a net coverage regression -- see
    docs/knowledge-eval/task-12-step5-gap-check-openai-gpt-6-luna.md --
    because each re-extraction is an independent model call that can fail to
    reproduce a previously-good item, and that risk compounded across every
    refilled section). `None` on any parse/shape failure, matching every
    other bounded call in this module. This function makes exactly one call
    and never merges its result into anything -- the caller (`gap_check.py`)
    does the merge into the section's existing extraction.
    """
    prompt_text = prompt_text if prompt_text is not None else load_gap_fill_prompt_v1()
    messages = build_gap_fill_messages(
        section=section, episode_context=episode_context, gap_hints=gap_hints,
        accepted_items=accepted_items, prompt_text=prompt_text,
    )
    result = client.call(model=model, provider=provider, messages=messages, schema=schema)
    return _parse_extraction_content(result.response_payload)


# ---------------------------------------------------------------------------
# Whole-episode extraction
# ---------------------------------------------------------------------------


def _high_value_rejected_items_by_section(
    extractions: dict[str, dict], validation: ItemValidationResult
) -> dict[str, list[dict]]:
    """section_id -> rejected-item context entries (`{"item", "reason",
    "detail"}`), limited to sections with at least one rejected item whose
    raw payload was `"value": "high"` (design spec §8.1: only a *high*-value
    rejection triggers re-extraction).
    """
    context_by_section: dict[str, list[dict]] = {}
    for rejected in validation.rejected:
        extraction = extractions.get(rejected.section_id)
        if not extraction:
            continue
        raw_items = extraction.get("items")
        if not isinstance(raw_items, list):
            continue
        raw_item = next(
            (
                item
                for item in raw_items
                if isinstance(item, dict) and item.get("local_id") == rejected.local_id
            ),
            None,
        )
        if raw_item is not None and raw_item.get("value") == "high":
            context_by_section.setdefault(rejected.section_id, []).append(
                {"item": raw_item, "reason": rejected.reason, "detail": rejected.detail}
            )
    return context_by_section


def extract_episode(
    sections: list[dict],
    *,
    client: ExtractionClient,
    model: str,
    provider: str,
    episode_context: dict,
    prompt_text: str | None = None,
    schema: dict | None = EXTRACTION_JSON_SCHEMA,
) -> ExtractionRunResult:
    """Extract every section (one call each), validate everything (Task 3),
    then give each section with a rejected *high*-value item exactly one
    re-extraction carrying the rejection reasons (design spec §8.1). A
    section still rejected after that one attempt stays rejected and is
    counted -- no further attempts.
    """
    prompt_text = prompt_text if prompt_text is not None else load_prompt_v1()

    extractions: dict[str, dict] = {}
    failed_sections: list[FailedSection] = []

    for section in sections:
        messages = build_messages(
            section=section, episode_context=episode_context, prompt_text=prompt_text
        )
        result = client.call(model=model, provider=provider, messages=messages, schema=schema)
        parsed = _parse_extraction_content(result.response_payload)
        if parsed is None or parsed.get("section_id") != section["section_id"]:
            failed_sections.append(
                FailedSection(section_id=section["section_id"], reason="invalid_json_response")
            )
            continue
        extractions[section["section_id"]] = parsed

    validation = validate_items(sections, list(extractions.values()))

    reextracted_section_ids: list[str] = []
    sections_by_id = {section["section_id"]: section for section in sections}
    high_value_rejections = _high_value_rejected_items_by_section(extractions, validation)

    for section_id, rejected_items in high_value_rejections.items():
        section = sections_by_id.get(section_id)
        if section is None:
            continue
        reextracted_section_ids.append(section_id)
        new_extraction = reextract_section(
            section,
            client=client,
            model=model,
            provider=provider,
            episode_context=episode_context,
            rejected_items=rejected_items,
            prompt_text=prompt_text,
            schema=schema,
        )
        if new_extraction is not None and new_extraction.get("section_id") == section_id:
            extractions[section_id] = new_extraction
        else:
            failed_sections.append(
                FailedSection(section_id=section_id, reason="invalid_json_response_on_reextraction")
            )
            extractions.pop(section_id, None)

    validation = validate_items(sections, list(extractions.values()))

    return ExtractionRunResult(
        extractions=extractions,
        failed_sections=failed_sections,
        reextracted_section_ids=reextracted_section_ids,
        validation=validation,
    )
