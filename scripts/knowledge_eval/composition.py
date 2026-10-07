"""Composition stage for TASK-106 Phase A Task 7.

Design authority:
docs/superpowers/specs/2026-09-24-grounded-knowledge-note-pipeline-design.md
§5.4 (composition, `note-composition-v1`), §5.5 (composition validation),
§2 (locked decisions).

One OpenRouter call for the whole note, through
`scripts.knowledge_eval.openrouter_eval.OpenRouterEvalClient` (or any object
with the same `.call(*, model, provider, messages, schema) -> CallResult`
shape -- tests inject a fake), mirroring Task 5's `extract.py` relationship
to `openrouter_eval.py`. The pure `note-composition-v1` schema and its
validation rules live in `podcast_engine.knowledge.notes_v2.composition`
(mirroring `items.py`'s relationship to `extract.py`); this module only
builds the request, parses the response, and applies the bounded
retry-then-hold policy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from podcast_engine.knowledge.notes_v2.composition import (
    COMPOSITION_JSON_SCHEMA,
    COMPOSITION_SCHEMA_VERSION,
    CompositionValidationResult,
    validate_composition,
)

_PROMPT_PATH = (
    Path(__file__).resolve().parents[2] / "prompts" / "knowledge_eval" / "compose-v1.md"
)


class CompositionClient(Protocol):
    """The subset of `OpenRouterEvalClient` this module depends on (same
    shape as Task 5's `ExtractionClient`)."""

    def call(
        self, *, model: str, provider: str, messages: list[dict], schema: dict | None
    ) -> Any:  # returns something with .response_payload
        ...


def load_prompt_v1() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


def build_messages(
    *,
    items: list[dict],
    episode_context: dict,
    prompt_text: str,
    previous_errors: list[str] | None = None,
) -> list[dict]:
    """Deterministic messages for one composition call: identical
    (items, episode_context, prompt_text, previous_errors) always build
    byte-identical messages, so the client's own request cache keys on them
    naturally. Items are sorted by `item_id` so caller-supplied ordering
    never changes the built request.
    """
    system = {"role": "system", "content": prompt_text}
    sorted_items = sorted((dict(item) for item in items), key=lambda item: item.get("item_id", ""))
    user_payload: dict[str, Any] = {
        "items": sorted_items,
        "episode_context": episode_context,
    }
    if previous_errors:
        user_payload["previous_errors"] = list(previous_errors)
    user = {"role": "user", "content": json.dumps(user_payload, sort_keys=True, ensure_ascii=False)}
    return [system, user]


def _parse_composition_content(response_payload: object) -> dict | None:
    """Parse the model's completion as `note-composition-v1` JSON. Never
    raises: any shape/parse failure returns None.
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
# Orchestration: one call, one bounded retry with the exact error list, then
# hold (design spec §5.5: "A validation failure admits one bounded
# composition retry with the exact error list; a second failure holds the
# run."). Unparseable JSON is treated the same way -- one retry naming the
# problem, then hold (design spec §8.3 lists "invalid or missing model
# output after the bounded retry" as a hold condition).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompositionRunResult:
    composition: dict | None
    validation: CompositionValidationResult | None
    retried: bool
    held: bool
    failure_reason: str | None


def _attempt(*, client: CompositionClient, model: str, provider: str, messages: list[dict], schema: dict | None) -> dict | None:
    result = client.call(model=model, provider=provider, messages=messages, schema=schema)
    return _parse_composition_content(result.response_payload)


def compose_note(
    items: list[dict],
    *,
    client: CompositionClient,
    model: str,
    provider: str,
    episode_context: dict,
    prompt_text: str | None = None,
    schema: dict | None = COMPOSITION_JSON_SCHEMA,
) -> CompositionRunResult:
    """Compose the whole note in one call, validate it, and give it exactly
    one bounded retry carrying the exact validation-error list when the
    first attempt fails -- never more than one retry (design spec §5.5).
    """
    prompt_text = prompt_text if prompt_text is not None else load_prompt_v1()
    items = list(items)

    messages = build_messages(items=items, episode_context=episode_context, prompt_text=prompt_text)
    composition = _attempt(client=client, model=model, provider=provider, messages=messages, schema=schema)

    if composition is not None:
        validation = validate_composition(composition, items)
        if validation.valid:
            return CompositionRunResult(composition=composition, validation=validation, retried=False, held=False, failure_reason=None)
        error_strings = [f"{e.code}: {e.detail}" for e in validation.errors]
    else:
        validation = None
        error_strings = ["invalid_json_response: the previous response was not valid JSON matching note-composition-v1"]

    retry_messages = build_messages(
        items=items, episode_context=episode_context, prompt_text=prompt_text, previous_errors=error_strings,
    )
    retry_composition = _attempt(client=client, model=model, provider=provider, messages=retry_messages, schema=schema)

    if retry_composition is None:
        return CompositionRunResult(composition=None, validation=validation, retried=True, held=True, failure_reason="invalid_json_response")

    retry_validation = validate_composition(retry_composition, items)
    if not retry_validation.valid:
        return CompositionRunResult(composition=retry_composition, validation=retry_validation, retried=True, held=True, failure_reason="validation_failed_after_retry")

    return CompositionRunResult(composition=retry_composition, validation=retry_validation, retried=True, held=False, failure_reason=None)
