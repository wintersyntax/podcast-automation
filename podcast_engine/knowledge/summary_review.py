"""Production OpenRouter requests for independent grounded summary review."""

from __future__ import annotations

import copy
from decimal import Decimal
import json
import os
import time
from collections.abc import Iterator
from typing import Any
from uuid import uuid4

import requests

from ..ai_budget import (
    STAGE_SUMMARY_REVIEW,
    mark_budget_attempt_uncertain,
    reserve_budget_batch,
    settle_budget_attempt,
)
from ..ai_pricing import (
    bounded_completion_tokens,
    derive_text_reservation_usd,
    resolve_preset_model_and_pricing,
    with_provider_price_ceiling,
)
from ..episode_contract import now_iso
from ..preset_provenance import PresetProvenance, fetch_current_designated_preset
from .models import DEFAULT_SUMMARY_REVIEW_PRESET, SUMMARY_REVIEW_PRESET_ENV
from .summary import (
    _WORST_CASE_EPISODE,
    episode_context,
    podcast_profile,
    worst_case_transcript_text,
)
from .summary_review_contract import (
    MAX_APPROVED_REPLACEMENT_TEXT_CHARS,
    MAX_ISSUES,
    MAX_REPAIR_TURNS_PER_CASE,
    MAX_RESOLUTION_CHARS,
    SUMMARY_REVIEW_AUDIT_SCHEMA,
    SUMMARY_REVIEW_EDIT_SCHEMA,
    accepted_final_markdown,
    aggregate_completion_metadata,
    build_audit_repair_payload,
    build_edit_obligations,
    build_edit_repair_payload,
    parse_review_content,
    slugify_repair_detail_code,
    validate_audit_result,
    validate_edit_result,
)
from .summary_review_evidence import (
    MAX_RISK_ITEMS,
    build_review_context,
    validate_review_context,
)


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
REVIEW_TIMEOUT_SECONDS = 120
REVIEW_MAX_ATTEMPTS = 2
MAX_MODEL_CALLS_PER_CASE = 3
ERROR_BODY_LIMIT = 2000


class SummaryReviewTransportError(RuntimeError):
    """Secret-safe transport failure with the number of physical calls consumed."""

    def __init__(self, message: str, *, attempts_used: int) -> None:
        super().__init__(message)
        self.attempts_used = attempts_used


class SummaryReviewValidationError(ValueError):
    """A bounded reviewer failure with safe attempt provenance."""

    def __init__(
        self,
        message: str,
        *,
        stage: str,
        code: str,
        completion_metadata: dict | None = None,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.code = code
        self.completion_metadata = dict(completion_metadata or {})


def active_summary_review_preset() -> str | None:
    """Return the explicitly activated production reviewer preset, or None."""

    configured = os.getenv(SUMMARY_REVIEW_PRESET_ENV)
    if configured is None or not configured.strip():
        return None

    api_key = os.getenv("PODCAST_SUMMARY_REVIEW_API_KEY")
    if api_key is None or not api_key.strip():
        raise RuntimeError(
            f"{SUMMARY_REVIEW_PRESET_ENV} is configured but "
            "PODCAST_SUMMARY_REVIEW_API_KEY is missing or blank"
        )

    return configured.strip()


def summary_review_preset() -> str:
    return os.getenv(SUMMARY_REVIEW_PRESET_ENV, DEFAULT_SUMMARY_REVIEW_PRESET).strip()


def _phase_payload(
    *,
    schema_name: str,
    schema: dict,
    content: dict,
    provenance: PresetProvenance | None = None,
) -> dict:
    """Build a direct request, preferring one verified current preset snapshot.

    TASK-076 Task 7: production V2 review (``generate``) always passes a
    verified ``provenance`` -- the exact resolved config is the base and
    the verified system prompt is placed before the user payload, never a
    mutable ``@preset/<slug>`` alias, so the physical request always
    matches the snapshot the episode-AI-budget reservation was derived
    from.

    ``provenance`` is optional only for the separate, pre-existing
    Benchmark v2 tournament harness (podcast_engine.knowledge.
    summary_review_benchmark_v2), which reuses this payload shape as a
    template and then substitutes its own candidate model/config and
    tracks spend through its own independent journal -- it never
    participates in the episode AI budget ledger this task adds, and is
    slated for replacement by TASK-075, so it keeps its pre-Task-7 mutable
    -alias payload shape unchanged rather than being pulled into scope
    here.
    """

    if provenance is None:
        payload: dict = {
            "model": f"@preset/{summary_review_preset()}",
        }
    else:
        if (
            not provenance.verified
            or not isinstance(provenance.config, dict)
            or not isinstance(provenance.system_prompt, str)
        ):
            raise RuntimeError("Verified OpenRouter preset provenance is required")
        payload = copy.deepcopy(provenance.config)

    messages = (
        [{"role": "user", "content": json.dumps(content, ensure_ascii=False)}]
        if provenance is None
        else [
            {"role": "system", "content": provenance.system_prompt},
            {"role": "user", "content": json.dumps(content, ensure_ascii=False)},
        ]
    )
    payload.update(
        {
            "usage": {"include": True},
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": schema,
                },
            },
            "messages": messages,
        }
    )
    return payload


def openrouter_audit_payload(
    episode: dict,
    transcript: str,
    draft: str,
    review_context: dict,
    *,
    provenance: PresetProvenance | None = None,
) -> dict:
    """Build the isolated V2 AUDIT request from grounded sources plus derived context."""

    validate_review_context(review_context, transcript, draft)
    return _phase_payload(
        schema_name="podcast_summary_review_audit",
        schema=SUMMARY_REVIEW_AUDIT_SCHEMA,
        content={
            "episode": episode_context(episode),
            "podcast_profile": podcast_profile(episode),
            "compiled_transcript": transcript,
            "draft_summary": draft,
            "python_review_context": review_context,
        },
        provenance=provenance,
    )


def openrouter_edit_payload(
    episode: dict,
    transcript: str,
    draft: str,
    review_context: dict,
    validated_audit: dict,
    edit_obligations: list[dict],
    *,
    provenance: PresetProvenance | None = None,
) -> dict:
    """Build the isolated V2 EDIT request after Python validates the audit."""

    validate_review_context(review_context, transcript, draft)
    return _phase_payload(
        schema_name="podcast_summary_review_edit",
        schema=SUMMARY_REVIEW_EDIT_SCHEMA,
        content={
            "episode": episode_context(episode),
            "podcast_profile": podcast_profile(episode),
            "compiled_transcript": transcript,
            "draft_summary": draft,
            "python_review_context": review_context,
            "validated_audit": validated_audit,
            "edit_obligations": edit_obligations,
        },
        provenance=provenance,
    )


# TASK-076 Task 5: worst-case bound for episode-AI-budget downstream-
# reserve planning. Up to MAX_MODEL_CALLS_PER_CASE model calls can occur
# for one case (AUDIT + EDIT + up to MAX_REPAIR_TURNS_PER_CASE repair
# turns). The EDIT payload is the largest of the three shapes -- it
# layers validated_audit and edit_obligations on top of the exact same
# transcript/draft/python_review_context the AUDIT call also sends, and a
# repair payload (see build_audit_repair_payload/build_edit_repair_payload)
# is narrower still (bounded issue lists, not the full audit). Bounding
# every one of the case's calls at the EDIT payload's own worst-case size
# is therefore conservative, not a guess.
WORST_CASE_MODEL_CALLS_PER_CASE = MAX_MODEL_CALLS_PER_CASE


def _worst_case_validated_audit() -> dict:
    """A validated_audit dict maxed out against SUMMARY_REVIEW_AUDIT_SCHEMA's
    own declared bounds (MAX_RISK_ITEMS risk assessments, MAX_ISSUES
    additional issues, each field at its own schema maxLength) -- a
    provable schema bound, not a guessed size. Never sent to a provider;
    used only to measure a worst-case EDIT request size."""

    risk_assessments = [
        {
            "risk_id": f"R{index:04d}",
            "disposition": "issue",
            "issue_type": "unsupported_claim",
            "severity": "material",
            "draft_block_ids": [f"B{index:04d}"],
            "transcript_span_ids": [f"S{index:04d}"],
            "resolution": "x" * MAX_RESOLUTION_CHARS,
            "approved_replacement_text": "x" * MAX_APPROVED_REPLACEMENT_TEXT_CHARS,
        }
        for index in range(MAX_RISK_ITEMS)
    ]
    additional_issues = [
        {
            "issue_type": "other",
            "severity": "material",
            "draft_block_ids": [f"B{index:04d}"],
            "transcript_span_ids": [f"S{index:04d}"],
            "resolution": "x" * MAX_RESOLUTION_CHARS,
            "approved_replacement_text": "x" * MAX_APPROVED_REPLACEMENT_TEXT_CHARS,
        }
        for index in range(MAX_ISSUES)
    ]
    return {
        "status": "revised",
        "risk_assessments": risk_assessments,
        "additional_issues": additional_issues,
    }


def worst_case_openrouter_payload(provenance: PresetProvenance) -> dict:
    """Build the largest-shaped summary-review request this stage can
    actually send (the EDIT phase, see WORST_CASE_MODEL_CALLS_PER_CASE),
    at the architecture's documented maximum transcript size and a
    schema-maxed validated_audit/edit_obligations, from the given resolved
    preset snapshot, so podcast_engine.ai_pricing can derive a conservative
    per-call reservation bound for episode-AI-budget downstream-reserve
    planning (Task 5). The caller multiplies the resulting per-call bound
    by WORST_CASE_MODEL_CALLS_PER_CASE for the whole case's worst-case
    reservation. Never sent to a provider -- this is measurement input
    only."""

    transcript = worst_case_transcript_text()
    draft = transcript
    review_context = build_review_context(transcript, draft)
    validated_audit = _worst_case_validated_audit()
    edit_obligations = build_edit_obligations(validated_audit)
    return openrouter_edit_payload(
        _WORST_CASE_EPISODE,
        transcript,
        draft,
        review_context,
        validated_audit,
        edit_obligations,
        provenance=provenance,
    )


def _request_text_values(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
        stripped = value.strip()
        if stripped.startswith(("{", "[")):
            try:
                nested = json.loads(stripped)
            except json.JSONDecodeError:
                nested = None
            if isinstance(nested, (dict, list)):
                yield from _request_text_values(nested)
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


def _transport_failure_message(
    error: Exception,
    payload: dict,
    api_key: str,
) -> str:
    if isinstance(error, requests.HTTPError) and error.response is not None:
        body = _sanitize_error_body(error.response, payload, api_key)
        return (
            "OpenRouter summary review request failed "
            f"(HTTP {error.response.status_code}): {body}"
        )
    if isinstance(error, requests.Timeout):
        return "OpenRouter summary review request timed out"
    if isinstance(error, requests.ConnectionError):
        return "OpenRouter summary review request failed to connect"
    return "OpenRouter summary review request failed"


def _extract_actual_usd(response_payload: object) -> Decimal | None:
    """Extract trustworthy provider-reported cost from an OpenRouter response.

    Missing, non-numeric, negative, or otherwise untrustworthy cost data
    returns None rather than guessing -- the caller settles the attempt as
    ``uncertain`` (still charged against the episode budget) instead of
    releasing it optimistically. See ``settle_budget_attempt``.
    """

    if not isinstance(response_payload, dict):
        return None

    usage = response_payload.get("usage")
    if not isinstance(usage, dict):
        return None

    cost = usage.get("cost")
    if isinstance(cost, bool) or not isinstance(cost, (int, float)):
        return None

    try:
        value = Decimal(str(cost))
    except (ArithmeticError, ValueError):
        return None

    if not value.is_finite() or value < 0:
        return None

    return value


def _budget_uncertain_reason(error: Exception) -> str:
    """Bounded, secret-safe reason for marking one reservation uncertain."""

    status = (
        error.response.status_code
        if isinstance(error, requests.HTTPError) and error.response is not None
        else None
    )
    reason = type(error).__name__
    if status is not None:
        reason = f"{reason}:{status}"
    return reason


def post_review_openrouter(
    payload: dict,
    *,
    max_attempts: int | None = None,
    episode_key: str,
    source_fingerprint: str,
    provenance: PresetProvenance,
    pricing_transport=None,
    pricing_now=None,
):
    """Post one reviewer phase with the dedicated workload key and bounded retry.

    TASK-076 Task 7: every physical attempt here (including each internal
    retry) independently reserves episode AI budget immediately before the
    request is sent, then settles or marks the reservation uncertain based
    on that attempt's own outcome -- the same contract Task 6 established
    for transcript review/triage. A retry always requires a fresh
    reservation; an earlier attempt's reservation, once marked uncertain,
    is never optimistically released.
    """

    api_key = os.getenv("PODCAST_SUMMARY_REVIEW_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing PODCAST_SUMMARY_REVIEW_API_KEY for summary review requests"
        )

    if (
        not isinstance(episode_key, str)
        or not episode_key
        or not isinstance(source_fingerprint, str)
        or not source_fingerprint
    ):
        raise RuntimeError(
            "episode_key and source_fingerprint are required to reserve "
            "episode AI budget before a summary review provider call"
        )

    attempts_allowed = REVIEW_MAX_ATTEMPTS if max_attempts is None else max_attempts
    if (
        not isinstance(attempts_allowed, int)
        or isinstance(attempts_allowed, bool)
        or attempts_allowed < 1
        or attempts_allowed > REVIEW_MAX_ATTEMPTS
    ):
        raise ValueError("Summary review transport attempt limit is invalid")

    resolved_preset = resolve_preset_model_and_pricing(
        provenance,
        api_key=api_key,
        transport=pricing_transport,
        now=pricing_now,
    )

    payload = with_provider_price_ceiling(payload, resolved_preset.pricing_bound)
    request_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    max_completion_tokens = bounded_completion_tokens(payload)

    reserved_usd = derive_text_reservation_usd(
        request_bytes=request_bytes,
        system_prompt=resolved_preset.system_prompt,
        max_completion_tokens=max_completion_tokens,
        prompt_usd_per_token=resolved_preset.pricing_bound.prompt_usd_per_token,
        completion_usd_per_token=resolved_preset.pricing_bound.completion_usd_per_token,
        context_length=resolved_preset.pricing_bound.context_length,
    )

    last_error: Exception | None = None
    for attempt in range(1, attempts_allowed + 1):
        attempt_id = f"{STAGE_SUMMARY_REVIEW}-{uuid4().hex}"
        reserve_budget_batch(
            episode_key,
            source_fingerprint,
            [
                {
                    "attempt_id": attempt_id,
                    "stage": STAGE_SUMMARY_REVIEW,
                    "reserved_usd": reserved_usd,
                    "third_asr": False,
                }
            ],
        )
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
            response.__dict__["_summary_review_http_attempt_count"] = attempt

            try:
                response_payload = response.json()
            except ValueError:
                response_payload = None

            settle_budget_attempt(
                episode_key,
                source_fingerprint,
                attempt_id,
                actual_usd=_extract_actual_usd(response_payload),
            )

            return response
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as error:
            last_error = error

            mark_budget_attempt_uncertain(
                episode_key,
                source_fingerprint,
                attempt_id,
                reason=_budget_uncertain_reason(error),
            )

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
            if not retryable or attempt >= attempts_allowed:
                raise SummaryReviewTransportError(
                    _transport_failure_message(error, payload, api_key),
                    attempts_used=attempt,
                ) from error
            time.sleep(2)

    raise SummaryReviewTransportError(
        "OpenRouter summary review request failed",
        attempts_used=attempts_allowed,
    ) from last_error


def _completion_metadata(response_payload: dict) -> dict:
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


def _require_served_identity(
    metadata: dict,
    expected_identity: tuple[str, str] | None,
) -> tuple[str, str]:
    served_model = metadata.get("served_model")
    served_provider = metadata.get("served_provider")
    if not isinstance(served_model, str) or not served_model.strip():
        raise ValueError("Summary reviewer response lacks an independent served model")
    if served_model.casefold().startswith("google/gemini"):
        raise ValueError("Summary reviewer did not use an independent served model")
    if not isinstance(served_provider, str) or not served_provider.strip():
        raise ValueError("Summary reviewer response lacks served provider identity")

    identity = (served_model, served_provider)
    if expected_identity is not None and identity != expected_identity:
        raise ValueError("Summary reviewer served identity changed within one review case")
    return identity


def _case_completion_metadata(attempts: list[dict], *, physical_call_count: int) -> dict:
    aggregate = aggregate_completion_metadata(
        attempts,
        attempt_count=physical_call_count,
    )
    aggregate["attempts"] = [dict(item) for item in attempts]
    return aggregate


def _physical_attempts_used(response) -> int:
    value = response.__dict__.get("_summary_review_http_attempt_count", 1)
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 1 else 1


def generate(
    episode: dict,
    transcript: str,
    draft: str,
    review_context: dict,
    *,
    episode_key: str,
    source_fingerprint: str,
    pricing_transport=None,
    pricing_now=None,
) -> dict:
    """Run V2 AUDIT then EDIT with one shared repair turn and fail-closed identity checks."""

    validate_review_context(review_context, transcript, draft)

    api_key = os.getenv("PODCAST_SUMMARY_REVIEW_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing PODCAST_SUMMARY_REVIEW_API_KEY for summary review requests"
        )

    provenance = fetch_current_designated_preset(
        summary_review_preset(), api_key=api_key, verified_at=now_iso()
    )
    if not provenance.verified:
        raise RuntimeError(
            f"Summary review preset could not be resolved: {provenance.status} "
            f"({provenance.reason or 'unknown'})"
        )

    completion_attempts: list[dict] = []
    physical_call_count = 0
    repair_turns = 0
    expected_identity: tuple[str, str] | None = None

    def request_phase(payload: dict, *, phase: str, phase_attempt: int) -> str:
        nonlocal physical_call_count, expected_identity

        remaining = MAX_MODEL_CALLS_PER_CASE - physical_call_count
        if remaining < 1:
            raise SummaryReviewValidationError(
                "Summary reviewer exhausted the three-call case ceiling",
                stage=phase,
                code="review_call_budget_exhausted",
                completion_metadata=_case_completion_metadata(
                    completion_attempts,
                    physical_call_count=physical_call_count,
                ),
            )

        try:
            response = post_review_openrouter(
                payload,
                max_attempts=min(REVIEW_MAX_ATTEMPTS, remaining),
                episode_key=episode_key,
                source_fingerprint=source_fingerprint,
                provenance=provenance,
                pricing_transport=pricing_transport,
                pricing_now=pricing_now,
            )
        except SummaryReviewTransportError as error:
            physical_call_count += error.attempts_used
            raise SummaryReviewValidationError(
                str(error),
                stage="request",
                code="request_failed",
                completion_metadata=_case_completion_metadata(
                    completion_attempts,
                    physical_call_count=physical_call_count,
                ),
            ) from error

        physical_call_count += _physical_attempts_used(response)
        try:
            response_payload = response.json()
        except (TypeError, ValueError) as error:
            raise ValueError("invalid_response_json") from error
        if not isinstance(response_payload, dict):
            raise ValueError("invalid_response_json")

        metadata = _completion_metadata(response_payload)
        completion_attempts.append(
            {
                "phase": phase,
                "phase_attempt": phase_attempt,
                **metadata,
            }
        )
        try:
            identity = _require_served_identity(metadata, expected_identity)
        except ValueError as error:
            raise SummaryReviewValidationError(
                str(error),
                stage="request",
                code="independent_or_stable_served_identity_required",
                completion_metadata=_case_completion_metadata(
                    completion_attempts,
                    physical_call_count=physical_call_count,
                ),
            ) from error
        if expected_identity is None:
            expected_identity = identity

        try:
            content = response_payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise ValueError("missing_review_content") from error
        if not isinstance(content, str):
            raise ValueError("missing_review_content")
        return content

    audit_payload = openrouter_audit_payload(
        episode, transcript, draft, review_context, provenance=provenance
    )
    audit_attempt = 1
    audit_content: str | None = None

    while True:
        try:
            audit_content = request_phase(
                audit_payload,
                phase="audit",
                phase_attempt=audit_attempt,
            )
            audit_result = parse_review_content(audit_content)
            validated_audit = validate_audit_result(
                audit_result,
                transcript,
                review_context,
            )
        except SummaryReviewValidationError:
            raise
        except (TypeError, ValueError) as error:
            failure_code = (
                str(error)
                if str(error) in {"invalid_response_json", "missing_review_content"}
                else "invalid_audit_contract"
            )
            if repair_turns >= MAX_REPAIR_TURNS_PER_CASE:
                raise SummaryReviewValidationError(
                    "Summary reviewer AUDIT failed strict validation",
                    stage="audit",
                    code=failure_code,
                    completion_metadata=_case_completion_metadata(
                        completion_attempts,
                        physical_call_count=physical_call_count,
                    ),
                ) from error
            if physical_call_count >= MAX_MODEL_CALLS_PER_CASE:
                raise SummaryReviewValidationError(
                    "Summary reviewer AUDIT failed with no call budget left for repair",
                    stage="audit",
                    code="review_call_budget_exhausted",
                    completion_metadata=_case_completion_metadata(
                        completion_attempts,
                        physical_call_count=physical_call_count,
                    ),
                ) from error
            repair_turns += 1
            audit_attempt += 1
            repair_detail_code = (
                failure_code
                if failure_code in {"invalid_response_json", "missing_review_content"}
                else slugify_repair_detail_code(str(error))
            )
            audit_payload = build_audit_repair_payload(
                audit_payload,
                audit_content,
                repair_detail_code,
            )
            continue
        break

    if validated_audit["status"] == "fail":
        return {
            "audit_result": validated_audit,
            "edit_result": None,
            "accepted_final_markdown": None,
            "completion_metadata": _case_completion_metadata(
                completion_attempts,
                physical_call_count=physical_call_count,
            ),
            "review_context": review_context,
        }

    edit_obligations = build_edit_obligations(validated_audit)
    edit_payload = openrouter_edit_payload(
        episode,
        transcript,
        draft,
        review_context,
        validated_audit,
        edit_obligations,
        provenance=provenance,
    )
    edit_attempt = 1
    edit_content: str | None = None

    while True:
        try:
            edit_content = request_phase(
                edit_payload,
                phase="edit",
                phase_attempt=edit_attempt,
            )
            edit_result = parse_review_content(edit_content)
            validated_edit = validate_edit_result(
                edit_result,
                validated_audit,
                draft,
                transcript,
                review_context,
            )
        except SummaryReviewValidationError:
            raise
        except (TypeError, ValueError) as error:
            failure_code = (
                str(error)
                if str(error) in {"invalid_response_json", "missing_review_content"}
                else "invalid_edit_contract"
            )
            if repair_turns >= MAX_REPAIR_TURNS_PER_CASE:
                raise SummaryReviewValidationError(
                    "Summary reviewer EDIT failed strict validation",
                    stage="edit",
                    code=failure_code,
                    completion_metadata=_case_completion_metadata(
                        completion_attempts,
                        physical_call_count=physical_call_count,
                    ),
                ) from error
            if physical_call_count >= MAX_MODEL_CALLS_PER_CASE:
                raise SummaryReviewValidationError(
                    "Summary reviewer EDIT failed with no call budget left for repair",
                    stage="edit",
                    code="review_call_budget_exhausted",
                    completion_metadata=_case_completion_metadata(
                        completion_attempts,
                        physical_call_count=physical_call_count,
                    ),
                ) from error
            repair_turns += 1
            edit_attempt += 1
            repair_detail_code = (
                failure_code
                if failure_code in {"invalid_response_json", "missing_review_content"}
                else slugify_repair_detail_code(str(error))
            )
            edit_payload = build_edit_repair_payload(
                edit_payload,
                edit_content,
                repair_detail_code,
            )
            continue
        break

    return {
        "audit_result": validated_audit,
        "edit_result": validated_edit,
        "accepted_final_markdown": accepted_final_markdown(
            validated_audit,
            validated_edit,
            draft,
        ),
        "completion_metadata": _case_completion_metadata(
            completion_attempts,
            physical_call_count=physical_call_count,
        ),
        "review_context": review_context,
    }
