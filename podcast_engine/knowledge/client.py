"""Shared OpenRouter client for knowledge-note requests."""

from __future__ import annotations

from decimal import Decimal
import json
import os
import time
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import requests
from dotenv import load_dotenv

if TYPE_CHECKING:
    from ..preset_provenance import PresetProvenance


load_dotenv()


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
KNOWLEDGE_TIMEOUT_SECONDS = 120
KNOWLEDGE_MAX_ATTEMPTS = 2
ERROR_BODY_LIMIT = 2000


def _request_text_values(value: Any) -> Iterator[str]:
    """Yield request text only for redacting an echoed OpenRouter response."""

    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _request_text_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from _request_text_values(item)


def _sanitize_error_body(response: requests.Response, payload: dict, api_key: str) -> str:
    """Return a bounded body without secrets or request-derived content."""

    try:
        body = response.text
    except (AttributeError, UnicodeDecodeError):
        return "<unavailable>"

    redactions = {
        api_key,
        f"Bearer {api_key}",
        "PODCAST_KNOWLEDGE_API_KEY",
        "Authorization",
    }
    redactions.update(value for value in _request_text_values(payload) if value)
    for value in sorted(redactions, key=len, reverse=True):
        body = body.replace(value, "[REDACTED]")
    return body[:ERROR_BODY_LIMIT]


def _http_error_with_body(error: requests.HTTPError, payload: dict, api_key: str) -> RuntimeError:
    response = error.response
    if response is None:
        return RuntimeError("OpenRouter knowledge request failed with an HTTP error")
    body = _sanitize_error_body(response, payload, api_key)
    return RuntimeError(
        "OpenRouter knowledge request failed "
        f"(HTTP {response.status_code}): {body}"
    )


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


def post_openrouter(
    payload: dict,
    *,
    episode_key: str,
    source_fingerprint: str,
    stage: str,
    provenance: "PresetProvenance",
    pricing_transport=None,
    pricing_now=None,
    timeout_seconds: float = KNOWLEDGE_TIMEOUT_SECONDS,
):
    """Post a knowledge request, retrying only transient OpenRouter failures.

    TASK-076 Task 7: every physical attempt here independently reserves
    episode AI budget immediately before the request is sent, then settles
    or marks the reservation uncertain based on that attempt's own outcome
    -- the same contract Task 6 established for transcript review/triage.
    A retry always requires a fresh reservation; an earlier attempt's
    reservation, once marked uncertain, is never optimistically released.
    """

    # Imported locally, not at module level: podcast_engine.ai_budget pulls
    # in google-cloud-storage transitively, and podcast_engine.knowledge.
    # summary (which imports this module's post_openrouter) must stay
    # importable without that dependency. See
    # test_knowledge_import_boundaries.py.
    from ..ai_budget import (
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

    api_key = os.getenv("PODCAST_KNOWLEDGE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing PODCAST_KNOWLEDGE_API_KEY for knowledge requests")

    if (
        not isinstance(episode_key, str)
        or not episode_key
        or not isinstance(source_fingerprint, str)
        or not source_fingerprint
    ):
        raise RuntimeError(
            "episode_key and source_fingerprint are required to reserve "
            "episode AI budget before a knowledge provider call"
        )

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

    last_error = None
    for attempt in range(1, KNOWLEDGE_MAX_ATTEMPTS + 1):
        attempt_id = f"{stage}-{uuid4().hex}"
        reserve_budget_batch(
            episode_key,
            source_fingerprint,
            [
                {
                    "attempt_id": attempt_id,
                    "stage": stage,
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
                },
                json=payload,
                timeout=timeout_seconds,
            )
            response.raise_for_status()

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
            if not retryable or attempt >= KNOWLEDGE_MAX_ATTEMPTS:
                if isinstance(error, requests.HTTPError):
                    raise _http_error_with_body(error, payload, api_key) from error
                raise
            time.sleep(2)

    raise RuntimeError("OpenRouter knowledge request failed") from last_error
