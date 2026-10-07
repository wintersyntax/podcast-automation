"""Structured-output contract for single-pass knowledge notes (TASK-118).

The writer model returns one JSON object matching ``NOTE_RESPONSE_SCHEMA``.
The schema is sent as a strict ``json_schema`` response format, but a
provider's structured-output support is never trusted: ``validate_note``
re-checks the whole shape in Python and returns every problem it finds.
"""

from __future__ import annotations

import copy
from typing import Any

from ..models import METADATA_RESPONSE_SCHEMA

NOTE_SCHEMA_VERSION = "knowledge-note-v3"

BASIS_VALUES = (
    "research",
    "coaching_experience",
    "personal_experience",
    "opinion",
    "none",
)

_S = {"type": "string"}
_B = {"type": "boolean"}
_BASIS = {"type": "string", "enum": list(BASIS_VALUES)}


def _obj(properties: dict) -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(properties),
        "properties": properties,
    }


def _arr(items: dict, **bounds: int) -> dict:
    return {"type": "array", "items": items, **bounds}


BULLET_SCHEMA = _obj({"text": _S, "basis": _BASIS, "hedged": _B, "anchor": _S})
PROTOCOL_SCHEMA = _obj({
    "name": _S,
    "what": _S,
    "dose_parameters": _S,
    "when_for_whom": _S,
    "caveat": _S,
    "basis": _BASIS,
    "hedged": _B,
    "anchor": _S,
})
STUDY_SCHEMA = _obj({
    "authors_year": _S,
    "design": _S,
    "sample": _S,
    "duration": _S,
    "result": _S,
    "host_comment": _S,
    "basis": _BASIS,
    "hedged": _B,
    "anchor": _S,
})


def _without_item_bounds(schema: dict) -> dict:
    """Drop ``maxItems``: not every provider's strict mode accepts it, and
    ``metadata.normalize_metadata`` already bounds these lists in Python."""

    result = copy.deepcopy(schema)
    result.pop("maxItems", None)
    return result


_METADATA_PROPERTIES = {
    key: _without_item_bounds(value)
    for key, value in METADATA_RESPONSE_SCHEMA["properties"].items()
}

NOTE_RESPONSE_SCHEMA: dict[str, Any] = _obj({
    "tldr": _arr(_S),
    "sections": _arr(_obj({
        "title": _S,
        "bottom_line": _S,
        "bullets": _arr(BULLET_SCHEMA),
        "protocols": _arr(PROTOCOL_SCHEMA),
    })),
    "research_discussed": _arr(STUDY_SCHEMA),
    "numbers": _arr(_obj({"topic": _S, "value": _S, "basis": _BASIS})),
    "takeaways": _arr(_S),
    "sources_mentioned": _arr(_obj({"reference_as_heard": _S, "anchor": _S})),
    "also_discussed": _arr(_obj({"text": _S, "anchor": _S})),
    "follow_up": _arr(_S),
    "topics": _METADATA_PROPERTIES["topics"],
    "people": _METADATA_PROPERTIES["people"],
    "existing_tags": _METADATA_PROPERTIES["existing_tags"],
    "new_tag_candidates": _METADATA_PROPERTIES["new_tag_candidates"],
})

RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "podcast_knowledge_note_v3",
        "strict": True,
        "schema": NOTE_RESPONSE_SCHEMA,
    },
}


def _check(value: Any, schema: dict, path: str, errors: list[str]) -> None:
    kind = schema.get("type")
    if kind == "object":
        if not isinstance(value, dict):
            errors.append(f"{path}: expected object")
            return
        properties = schema["properties"]
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}.{key}: missing")
        for key in value:
            if key not in properties:
                errors.append(f"{path}.{key}: unexpected field")
        for key, sub in properties.items():
            if key in value:
                _check(value[key], sub, f"{path}.{key}", errors)
    elif kind == "array":
        if not isinstance(value, list):
            errors.append(f"{path}: expected array")
            return
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(f"{path}: more than {schema['maxItems']} items")
        for index, item in enumerate(value):
            _check(item, schema["items"], f"{path}[{index}]", errors)
    elif kind == "string":
        if not isinstance(value, str):
            errors.append(f"{path}: expected string")
        elif "enum" in schema and value not in schema["enum"]:
            errors.append(f"{path}: {value!r} is not one of {schema['enum']}")
    elif kind == "boolean":
        if not isinstance(value, bool):
            errors.append(f"{path}: expected boolean")


def validate_note(payload: Any) -> list[str]:
    """Return every structural problem in a writer response (empty = valid).

    Beyond the JSON schema: the note needs 1-6 TL;DR sentences, at least one
    section, and a non-empty text and anchor on every anchored unit, because
    downstream checks and rendering rely on them.
    """

    errors: list[str] = []
    _check(payload, NOTE_RESPONSE_SCHEMA, "note", errors)
    if errors:
        return errors
    if not 1 <= len(payload["tldr"]) <= 6:
        errors.append("note.tldr: expected 1-6 sentences")
    if not payload["sections"]:
        errors.append("note.sections: at least one section is required")
    for s_index, section in enumerate(payload["sections"]):
        if not section["title"].strip():
            errors.append(f"note.sections[{s_index}].title: empty")
        for b_index, bullet in enumerate(section["bullets"]):
            path = f"note.sections[{s_index}].bullets[{b_index}]"
            if not bullet["text"].strip():
                errors.append(f"{path}.text: empty")
            if not bullet["anchor"].strip():
                errors.append(f"{path}.anchor: empty")
    return errors
