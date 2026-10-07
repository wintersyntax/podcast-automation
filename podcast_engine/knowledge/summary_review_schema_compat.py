"""Provider-compatible JSON-schema projection for summary-review requests."""

from __future__ import annotations


_UNSUPPORTED_OPENROUTER_STRUCTURED_OUTPUT_KEYWORDS = frozenset(
    {"minLength", "maxLength", "maxItems"}
)


def _is_openai_nullable_type_union(value: dict) -> bool:
    """True for the OpenAI Structured-Outputs nullable idiom on this node.

    OpenAI's strict Structured Outputs mode represents a nullable field as
    ``{"type": [X, "null"]}``, optionally with ``"enum": [...values..., None]``
    when the non-null branch is also enum-constrained. That exact array-type
    shape is OpenAI-specific: OpenRouter's other backends for the same
    logical model (observed live for ``anthropic/claude-haiku-4.5``, routed
    through both Amazon Bedrock and Anthropic direct) reject it with
    HTTP 400. It is not part of standard JSON Schema, whether or not an
    ``enum`` constraint is also present.
    """

    type_value = value.get("type")
    if not (isinstance(type_value, list) and "null" in type_value):
        return False
    non_null_types = [item for item in type_value if item != "null"]
    if len(non_null_types) != 1:
        return False
    if "enum" in value:
        enum_value = value["enum"]
        if not isinstance(enum_value, list) or None not in enum_value:
            return False
    return True


def _project_nullable_type_union(value: dict) -> dict:
    """Rewrite an OpenAI-only nullable type-array node into portable ``anyOf``.

    ``{"type": [X, "null"], "enum": [...values..., None]}`` becomes
    ``{"anyOf": [{"type": X, "enum": [...values without None...]}, {"type": "null"}]}``.
    A freeform nullable field with no ``enum`` constraint projects the same
    way, simply without an ``enum`` key on the non-null branch. This
    preserves the exact same nullable semantics (the field may still be
    JSON ``null``, and any enum constraint on non-null values is preserved)
    using a form that portable JSON Schema consumers, including Anthropic's
    own tool-use schema validation, accept.
    """

    type_value = value["type"]
    non_null_type = next(item for item in type_value if item != "null")
    enum_value = value.get("enum")

    non_null_branch: dict = {"type": non_null_type}
    if isinstance(enum_value, list):
        non_null_branch["enum"] = [item for item in enum_value if item is not None]

    remainder = {
        key: item
        for key, item in value.items()
        if key not in ("type", "enum")
    }

    projected = dict(remainder)
    projected["anyOf"] = [non_null_branch, {"type": "null"}]
    return projected


def project_openrouter_json_schema(schema: dict, *, strict_mode: bool = True) -> dict:
    """Return a provider-compatible deep copy without weakening Python validation.

    ``strict_mode`` selects the wire shape for nullable fields:

    - ``True`` (default): keep the OpenAI Structured-Outputs nullable
      type-array idiom (``"type": [X, "null"]``) as-is. This is required for
      OpenAI's ``strict: true`` Structured Outputs contract and is the shape
      already proven to work end to end against ``openai/gpt-5-mini``.
    - ``False``: rewrite that same idiom into the portable ``anyOf`` form via
      :func:`_project_nullable_type_union`, for non-OpenAI candidates whose
      backends reject the OpenAI-only array-type shape.
    """

    if not isinstance(schema, dict):
        raise TypeError("Summary review JSON schema must be an object")

    def project(value):
        if isinstance(value, dict):
            filtered = {
                key: project(item)
                for key, item in value.items()
                if key not in _UNSUPPORTED_OPENROUTER_STRUCTURED_OUTPUT_KEYWORDS
            }
            if not strict_mode and _is_openai_nullable_type_union(filtered):
                return _project_nullable_type_union(filtered)
            return filtered
        if isinstance(value, list):
            return [project(item) for item in value]
        return value

    return project(schema)
