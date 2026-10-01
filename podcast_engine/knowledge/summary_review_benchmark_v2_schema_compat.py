"""Provider-compatible wire schemas for Summary Review Benchmark v2."""

from __future__ import annotations

from functools import wraps

from .summary_review_schema_compat import project_openrouter_json_schema


def _model_supports_openai_strict_structured_outputs(model_id: object) -> bool:
    """True only for OpenAI-family model slugs on OpenRouter.

    OpenRouter's ``openai/...`` model slugs are served by OpenAI's own API
    and accept its ``strict: true`` Structured Outputs contract, including
    the ``"type": [X, "null"]`` nullable-type-array idiom. Every other
    backend observed for this benchmark (Amazon Bedrock and Anthropic direct,
    both routed for ``anthropic/claude-haiku-4.5``; also Qwen candidates)
    either rejects that OpenAI-only shape outright (HTTP 400) or silently
    ignores the schema constraint, so they must receive the portable
    ``anyOf``-based projection instead.
    """

    return isinstance(model_id, str) and model_id.startswith("openai/")


def _project_benchmark_payload(payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise TypeError("Benchmark v2 transport payload must be an object")

    projected = dict(payload)
    response_format = payload.get("response_format")
    if not isinstance(response_format, dict):
        return projected
    json_schema = response_format.get("json_schema")
    if not isinstance(json_schema, dict):
        return projected
    schema = json_schema.get("schema")
    if not isinstance(schema, dict):
        return projected

    strict_mode = _model_supports_openai_strict_structured_outputs(
        payload.get("model")
    )

    projected_response_format = dict(response_format)
    projected_json_schema = dict(json_schema)
    projected_json_schema["schema"] = project_openrouter_json_schema(
        schema, strict_mode=strict_mode
    )
    if not strict_mode:
        # The `strict` flag is an OpenAI Structured-Outputs-only contract
        # keyword. It is meaningless (and, per this benchmark's live 400s,
        # sometimes actively rejected) for other backends, so it is dropped
        # rather than sent as a no-op.
        projected_json_schema.pop("strict", None)
    projected_response_format["json_schema"] = projected_json_schema
    projected["response_format"] = projected_response_format
    return projected


def install_benchmark_v2_schema_compat(namespace: dict) -> None:
    """Project only benchmark wire schemas before paid-operator hardening wraps them."""

    original_execute_case = namespace["execute_case"]

    @wraps(original_execute_case)
    def compatible_execute_case(*args, **kwargs):
        transport = kwargs.get("transport")
        if transport is None:
            raise ValueError("Benchmark v2 transport must be supplied by keyword")
        if not callable(transport):
            return original_execute_case(*args, **kwargs)

        def compatible_transport(payload: dict):
            return transport(_project_benchmark_payload(payload))

        compatible_kwargs = dict(kwargs)
        compatible_kwargs["transport"] = compatible_transport
        return original_execute_case(*args, **compatible_kwargs)

    namespace["execute_case"] = compatible_execute_case
