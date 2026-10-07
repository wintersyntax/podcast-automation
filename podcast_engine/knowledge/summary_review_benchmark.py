"""Evaluation-only OpenRouter transport and pure helpers for reviewer benchmarks."""

from __future__ import annotations

import hashlib
import json
import os
import string
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import requests

from .summary_review_contract_v1 import (
    MAX_CONTRACT_ATTEMPTS,
    SUMMARY_REVIEW_RESPONSE_SCHEMA,
    accepted_final_markdown,
    aggregate_completion_metadata,
    build_contract_repair_payload,
    parse_review_content,
    validate_review_result,
    validate_review_shape,
)


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
REVIEW_TIMEOUT_SECONDS = 120
REVIEW_MAX_ATTEMPTS = 2
ERROR_BODY_LIMIT = 2000
VALIDATION_MESSAGE_LIMIT = 500


class ReviewContractError(ValueError):
    """Carry a shape-valid reviewer result across semantic validation failure."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        review_result: dict | None,
        completion_metadata: dict,
        elapsed_ms: int,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.review_result = review_result
        self.completion_metadata = completion_metadata
        self.elapsed_ms = elapsed_ms


def _validation_code(error: ValueError) -> str:
    message = str(error)
    exact = {
        "pass status requires zero issues": "pass_issues_not_empty",
        "pass status requires null final Markdown": "pass_final_must_be_null",
        "revised status requires at least one issue": "revised_issues_empty",
        "revised status requires non-empty final Markdown": "revised_final_missing",
        "revised status requires changed final Markdown": "revised_final_unchanged",
        "fail status requires null final Markdown": "fail_final_must_be_null",
    }
    if message in exact:
        return exact[message]
    if message.startswith("Summary review final Markdown"):
        return "final_markdown_invalid"
    if "draft_excerpt" in message:
        return "draft_evidence_invalid"
    if "transcript_evidence" in message:
        return "transcript_evidence_invalid"
    return "semantic_contract_failed"


def build_review_payload(
    model_id: str,
    system_prompt: str,
    frozen_inputs: dict,
    config: dict,
) -> dict:
    """Build one exact-model benchmark request from the four frozen inputs."""

    required_inputs = {
        "episode",
        "podcast_profile",
        "compiled_transcript",
        "draft_summary",
    }
    if set(frozen_inputs) != required_inputs:
        raise ValueError("Summary review benchmark inputs do not match the frozen contract")

    payload = {
        "model": model_id,
        "temperature": config["temperature"],
        "max_tokens": config["max_tokens"],
        "provider": dict(config["provider"]),
        "usage": {"include": True},
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "podcast_summary_review",
                "strict": True,
                "schema": SUMMARY_REVIEW_RESPONSE_SCHEMA,
            },
        },
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps(frozen_inputs, ensure_ascii=False),
            },
        ],
    }

    overrides = config.get("model_parameter_overrides", {}).get(model_id, {})
    if not isinstance(overrides, dict):
        raise ValueError("Benchmark model parameter overrides must be objects")

    allowed_override_keys = {
        "temperature",
        "max_tokens",
        "max_completion_tokens",
    }
    unexpected = set(overrides) - allowed_override_keys
    if unexpected:
        raise ValueError(
            "Unsupported benchmark model parameter override keys: "
            + ", ".join(sorted(unexpected))
        )

    for key, value in overrides.items():
        if value is None:
            payload.pop(key, None)
        else:
            payload[key] = value

    return payload


def _request_text_values(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _request_text_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from _request_text_values(item)


def _sanitize_error_body(response: requests.Response, payload: dict, api_key: str) -> str:
    try:
        body = response.text
    except (AttributeError, UnicodeDecodeError):
        return "<unavailable>"

    redactions = {
        api_key,
        f"Bearer {api_key}",
        "PODCAST_SUMMARY_REVIEW_API_KEY",
        "Authorization",
    }
    redactions.update(value for value in _request_text_values(payload) if value)
    for value in sorted(redactions, key=len, reverse=True):
        body = body.replace(value, "[REDACTED]")
    return body[:ERROR_BODY_LIMIT]


def _http_error_with_body(
    error: requests.HTTPError,
    payload: dict,
    api_key: str,
) -> RuntimeError:
    response = error.response
    if response is None:
        return RuntimeError("OpenRouter summary review request failed with an HTTP error")
    body = _sanitize_error_body(response, payload, api_key)
    return RuntimeError(
        "OpenRouter summary review request failed "
        f"(HTTP {response.status_code}): {body}"
    )


def post_review_request(payload: dict):
    """Post a benchmark request with bounded retries and secret-safe errors."""

    api_key = os.getenv("PODCAST_SUMMARY_REVIEW_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing PODCAST_SUMMARY_REVIEW_API_KEY for summary review benchmark"
        )

    last_error = None
    for attempt in range(1, REVIEW_MAX_ATTEMPTS + 1):
        try:
            response = requests.post(
                OPENROUTER_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "X-OpenRouter-Metadata": "enabled",
                },
                json=payload,
                timeout=REVIEW_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            return response
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as error:
            last_error = error
            status = (
                error.response.status_code
                if isinstance(error, requests.HTTPError) and error.response is not None
                else None
            )
            retryable = (
                isinstance(error, (requests.Timeout, requests.ConnectionError))
                or status == 429
                or (status is not None and 500 <= status < 600)
            )
            if not retryable or attempt >= REVIEW_MAX_ATTEMPTS:
                if isinstance(error, requests.HTTPError):
                    raise _http_error_with_body(error, payload, api_key) from error
                raise
            time.sleep(2)

    raise RuntimeError("OpenRouter summary review request failed") from last_error


def completion_metadata(response_payload: dict) -> dict:
    """Return only actual OpenRouter identity and usage values that were supplied."""

    if not isinstance(response_payload, dict):
        return {}

    metadata: dict = {}
    completion_id = response_payload.get("id")
    served_model = response_payload.get("model")
    if isinstance(completion_id, str) and completion_id:
        metadata["completion_id"] = completion_id
    if isinstance(served_model, str) and served_model:
        metadata["served_model"] = served_model

    openrouter_metadata = response_payload.get("openrouter_metadata")
    endpoints = (
        openrouter_metadata.get("endpoints", {}).get("available", [])
        if isinstance(openrouter_metadata, dict)
        else []
    )
    if isinstance(endpoints, list):
        for endpoint in endpoints:
            if isinstance(endpoint, dict) and endpoint.get("selected") is True:
                provider = endpoint.get("provider")
                if isinstance(provider, str) and provider:
                    metadata["served_provider"] = provider
                break

    usage = response_payload.get("usage")
    if isinstance(usage, dict):
        for field in ("prompt_tokens", "completion_tokens", "total_tokens", "cost"):
            value = usage.get(field)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                metadata[field] = value

    return metadata


def execute_review_preflight(
    model_id: str,
    system_prompt: str,
    frozen_inputs: dict,
    config: dict,
) -> dict:
    """Verify a model can produce one fully grounded revised result."""

    result = execute_review(model_id, system_prompt, frozen_inputs, config)
    review_result = result["review_result"]
    if review_result.get("status") != "revised":
        raise ValueError("Summary review preflight requires revised status")
    final = review_result.get("final_markdown")
    if not isinstance(final, str) or "definitely works" in final.casefold():
        raise ValueError("Summary review preflight failed to correct the seeded claim")
    return result


def execute_review(
    model_id: str,
    system_prompt: str,
    frozen_inputs: dict,
    config: dict,
) -> dict:
    """Execute and strictly validate a review with one bounded repair turn."""

    payload = build_review_payload(model_id, system_prompt, frozen_inputs, config)
    started = time.monotonic()
    metadata_attempts: list[dict] = []

    for attempt in range(1, MAX_CONTRACT_ATTEMPTS + 1):
        response = post_review_request(payload)
        content: str | None = None
        review_result: dict | None = None
        try:
            response_payload = response.json()
        except (TypeError, ValueError) as error:
            failure_code = "invalid_response_json"
            failure_message = "Summary review response is not valid JSON"
            failure = error
        else:
            metadata_attempts.append(completion_metadata(response_payload))
            try:
                content = response_payload["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as error:
                failure_code = "missing_review_content"
                failure_message = "Summary review response lacks review content"
                failure = error
            else:
                try:
                    review_result = parse_review_content(content)
                except (TypeError, ValueError) as error:
                    failure_code = "invalid_review_json"
                    failure_message = "Summary review content is not valid contract JSON"
                    failure = error
                else:
                    try:
                        validate_review_shape(review_result)
                    except (TypeError, ValueError) as error:
                        failure_code = "invalid_review_schema"
                        failure_message = "Summary review content failed schema validation"
                        failure = error
                    else:
                        try:
                            validate_review_result(
                                review_result,
                                frozen_inputs["draft_summary"],
                                frozen_inputs["compiled_transcript"],
                            )
                        except ValueError as error:
                            failure_code = _validation_code(error)
                            failure_message = str(error)[:VALIDATION_MESSAGE_LIMIT]
                            failure = error
                        else:
                            elapsed_ms = round((time.monotonic() - started) * 1000)
                            return {
                                "review_result": review_result,
                                "completion_metadata": aggregate_completion_metadata(
                                    metadata_attempts,
                                    attempt_count=attempt,
                                ),
                                "elapsed_ms": elapsed_ms,
                            }

        if attempt < MAX_CONTRACT_ATTEMPTS:
            payload = build_contract_repair_payload(
                payload,
                content,
                failure_code,
            )
            continue

        elapsed_ms = round((time.monotonic() - started) * 1000)
        raise ReviewContractError(
            failure_code,
            failure_message,
            review_result=review_result,
            completion_metadata=aggregate_completion_metadata(
                metadata_attempts,
                attempt_count=attempt,
            ),
            elapsed_ms=elapsed_ms,
        ) from failure

    raise AssertionError("unreachable summary review contract loop")


def build_blind_plan(models: list[str], runs_per_model: int, rng) -> dict:
    """Assign anonymous labels and shuffle the exact benchmark call order."""

    if not isinstance(models, list) or not models or len(models) > len(string.ascii_uppercase):
        raise ValueError("Benchmark models must be a non-empty bounded list")
    if len(set(models)) != len(models):
        raise ValueError("Benchmark model IDs must be unique")
    if not isinstance(runs_per_model, int) or isinstance(runs_per_model, bool) or runs_per_model < 1:
        raise ValueError("runs_per_model must be a positive integer")

    shuffled_models = list(models)
    rng.shuffle(shuffled_models)
    labels = list(string.ascii_uppercase[: len(shuffled_models)])
    model_map = dict(zip(labels, shuffled_models, strict=True))
    calls = [
        {
            "run_id": f"{label}{run_number}",
            "label": label,
            "run_number": run_number,
            "model_id": model_map[label],
        }
        for label in labels
        for run_number in range(1, runs_per_model + 1)
    ]
    rng.shuffle(calls)
    return {"model_map": model_map, "calls": calls}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_verified_fixture(fixture_dir: Path, config: dict) -> dict:
    """Load and re-hash the immutable local benchmark fixture before any call."""

    fixture_dir = Path(fixture_dir)
    manifest_path = fixture_dir / "fixture.json"
    transcript_path = fixture_dir / "compiled-transcript.txt"
    draft_path = fixture_dir / "draft.md"
    for path in (manifest_path, transcript_path, draft_path):
        if not path.is_file():
            raise ValueError(f"Benchmark fixture is missing required file: {path.name}")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Benchmark fixture manifest is unreadable") from exc
    if not isinstance(manifest, dict):
        raise ValueError("Benchmark fixture manifest must be an object")

    transcript_bytes = transcript_path.read_bytes()
    draft_bytes = draft_path.read_bytes()
    transcript_sha = _sha256(transcript_bytes)
    draft_sha = _sha256(draft_bytes)
    if manifest.get("compiled_transcript_sha256") != transcript_sha:
        raise ValueError("Benchmark fixture compiled transcript hash does not match manifest")
    if config.get("expected_compiled_sha256") != transcript_sha:
        raise ValueError("Benchmark fixture compiled transcript hash does not match config")
    if manifest.get("draft_sha256") != draft_sha:
        raise ValueError("Benchmark fixture draft hash does not match manifest")
    if manifest.get("benchmark_id") != config.get("benchmark_id"):
        raise ValueError("Benchmark fixture ID does not match config")
    if manifest.get("episode_key") != config.get("episode_key"):
        raise ValueError("Benchmark fixture episode key does not match config")
    if manifest.get("draft_summary_preset_version") != config.get("draft_summary_preset_version"):
        raise ValueError("Benchmark fixture draft preset version does not match config")
    if manifest.get("draft_summary_prompt_sha256") != config.get("draft_summary_prompt_sha256"):
        raise ValueError("Benchmark fixture draft prompt identity does not match config")

    try:
        transcript = transcript_bytes.decode("utf-8")
        draft = draft_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Benchmark fixture text must be valid UTF-8") from exc

    episode = manifest.get("episode")
    profile = manifest.get("podcast_profile")
    if not isinstance(episode, dict) or not isinstance(profile, dict):
        raise ValueError("Benchmark fixture episode/profile context is invalid")

    return {
        "manifest": manifest,
        "compiled_transcript_sha256": transcript_sha,
        "draft_sha256": draft_sha,
        "frozen_inputs": {
            "episode": episode,
            "podcast_profile": profile,
            "compiled_transcript": transcript,
            "draft_summary": draft,
        },
    }


def machine_gate(review_result: dict, draft: str) -> dict:
    """Apply deterministic evaluator-safe checks to one validated AI result."""

    status = review_result.get("status") if isinstance(review_result, dict) else None
    publishable = status in {"pass", "revised"}
    final = accepted_final_markdown(review_result, draft) if publishable else None
    final_text = final if isinstance(final, str) else ""
    no_unsupported_99 = publishable and "99%" not in final_text
    no_concept2 = publishable and "concept2" not in final_text.casefold()

    failure_code = None
    if status == "fail":
        failure_code = "reviewer_fail"
    elif not publishable:
        failure_code = "contract_failed"
    elif not no_unsupported_99:
        failure_code = "fixture_99_survived"
    elif not no_concept2:
        failure_code = "unauthorized_term_normalization"

    return {
        "request_succeeded": True,
        "contract_valid": status in {"pass", "revised", "fail"},
        "publishable_status": publishable,
        "fixture_assertions": {
            "unsupported_99_removed": no_unsupported_99,
            "unauthorized_concept2_absent": no_concept2,
        },
        "machine_pass": failure_code is None,
        "failure_code": failure_code,
    }


def failed_machine_gate(failure_code: str, *, request_succeeded: bool) -> dict:
    """Represent a request/parse/contract failure without leaking private error text."""

    return {
        "request_succeeded": request_succeeded,
        "contract_valid": False,
        "publishable_status": False,
        "fixture_assertions": {
            "unsupported_99_removed": False,
            "unauthorized_concept2_absent": False,
        },
        "machine_pass": False,
        "failure_code": failure_code,
    }
