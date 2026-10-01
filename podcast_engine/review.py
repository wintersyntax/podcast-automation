"""Small, source-grounded AI review for compiler differences."""

from __future__ import annotations

import copy
from decimal import Decimal
import hashlib
import json
import os
import time
from typing import Callable
from uuid import uuid4

import requests

from compiler.transcript import validate_resolver_response

from .ai_budget import (
    STAGE_RESOLVER,
    STAGE_TRIAGE,
    mark_budget_attempt_uncertain,
    reserve_budget_batch,
    settle_budget_attempt,
)
from .ai_pricing import (
    bounded_completion_tokens,
    derive_text_reservation_usd,
    resolve_preset_model_and_pricing,
    with_provider_price_ceiling,
)
from .preset_provenance import PresetProvenance


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

REVIEW_PRESET = os.environ.get(
    "PODCAST_TRANSCRIPT_REVIEW_PRESET",
    "podcast-transcript-reviewer",
)

REVIEW_TIMEOUT_SECONDS = 120
REVIEW_MAX_ATTEMPTS = 2
TRIAGE_CHUNK_SIZE = 10

# TASK-076: triage gets its own dedicated completion ceiling instead of
# inheriting the transcript-reviewer preset's reasoning-enabled max_tokens
# (8192). TRIAGE_OUTPUT_MAX_TOKENS covers the schema's provable worst case for one
# full TRIAGE_CHUNK_SIZE-item chunk (id + longest recommendation/confidence
# enum values + a full TRIAGE_REASON_MAX_LENGTH reason per item, UTF-8 byte
# length as the token upper bound), which is 2442 bytes; the ~15% margin absorbs
# JSON string-escaping overhead for non-ASCII "reason" text. See
# docs/superpowers/specs/2026-09-16-human-review-evidence-assisted-adjudication-design.md
# "Task 3 worst-case cost-estimate finding".
#
# TASK-123: the served Gemini reviewer endpoint rejects a disabled-reasoning
# request with HTTP 400 ("Reasoning is mandatory for this endpoint and cannot
# be disabled"), which made every triage chunk unavailable. Reasoning tokens
# count against max_tokens even when reasoning.exclude hides them, so triage
# now requests an explicit bounded reasoning budget and adds exactly that
# budget to the output ceiling. The completion ceiling therefore stays
# deterministic (and far below the inherited 8192) without asking the
# provider to turn reasoning off.
TRIAGE_OUTPUT_MAX_TOKENS = 2816
TRIAGE_REASONING_MAX_TOKENS = 1024
TRIAGE_MAX_COMPLETION_TOKENS = TRIAGE_OUTPUT_MAX_TOKENS + TRIAGE_REASONING_MAX_TOKENS


def _response_payload(content: str) -> dict:
    """Parse a JSON-only OpenRouter response, tolerating Markdown fences."""

    content = content.strip()

    if content.startswith("```"):
        content = content.split("\n", 1)[-1]
        content = content.rsplit("```", 1)[0].strip()

    parsed = json.loads(content)

    if not isinstance(parsed, dict):
        raise ValueError(
            "Transcript review response must be a JSON object"
        )

    return parsed


def _complete_resolution(
    batch: dict,
    resolution: dict,
) -> dict[str, list[dict]]:
    """Ensure every resolver input is explicitly accounted for."""

    accepted = [
        dict(item)
        for item in resolution.get("accepted", [])
        if isinstance(item, dict)
    ]

    review = [
        dict(item)
        for item in resolution.get("review", [])
        if isinstance(item, dict)
    ]

    accepted_ids = {
        item.get("id")
        for item in accepted
        if isinstance(item.get("id"), int)
    }

    reviewed_ids = {
        item.get("id")
        for item in review
        if isinstance(item.get("id"), int)
    }

    for item in batch.get(
        "batch",
        {},
    ).get(
        "diff_items",
        [],
    ):
        difference_id = (
            item.get("id")
            if isinstance(item, dict)
            else None
        )

        if not isinstance(
            difference_id,
            int,
        ):
            continue

        if (
            difference_id in accepted_ids
            or difference_id in reviewed_ids
        ):
            continue

        review.append(
            {
                "id": difference_id,
                "reason": "missing_resolver_response",
            }
        )

    outcomes = [
        dict(item)
        for item in resolution.get("outcomes", [])
        if isinstance(item, dict)
    ]
    outcome_ids = {
        item.get("id")
        for item in outcomes
        if isinstance(item.get("id"), int)
    }
    for item in batch.get("batch", {}).get("diff_items", []):
        difference_id = item.get("id") if isinstance(item, dict) else None
        if isinstance(difference_id, int) and difference_id not in outcome_ids:
            outcomes.append(
                {
                    "id": difference_id,
                    "status": "missing_response",
                    "reason": "missing_resolver_response",
                }
            )

    return {
        "accepted": accepted,
        "review": review,
        "outcomes": outcomes,
    }


def _resolver_outcomes(batch: dict, response: dict, validated: dict) -> list[dict]:
    """Record whether the model abstained or Python rejected a submitted choice."""

    reviewed = {
        item.get("id"): item.get("reason", "needs_human_review")
        for item in validated.get("review", [])
        if isinstance(item, dict) and isinstance(item.get("id"), int)
    }
    accepted = {
        item.get("id")
        for item in validated.get("accepted", [])
        if isinstance(item, dict) and isinstance(item.get("id"), int)
    }
    submitted = {
        item.get("id"): item
        for item in response.get("resolutions", [])
        if isinstance(item, dict) and isinstance(item.get("id"), int)
    }
    outcomes = []
    for item in batch.get("batch", {}).get("diff_items", []):
        difference_id = item.get("id") if isinstance(item, dict) else None
        if not isinstance(difference_id, int):
            continue
        resolution = submitted.get(difference_id)
        if difference_id in accepted:
            outcomes.append({"id": difference_id, "status": "accepted"})
        elif resolution is None:
            outcomes.append(
                {
                    "id": difference_id,
                    "status": "missing_response",
                    "reason": "missing_resolver_response",
                }
            )
        elif (
            resolution.get("resolved_value") == "unclear"
            or resolution.get("flag_for_human") is True
            or resolution.get("confidence") != "high"
        ):
            outcomes.append(
                {
                    "id": difference_id,
                    "status": "abstained",
                    "reason": reviewed.get(difference_id, "needs_human_review"),
                }
            )
        else:
            outcomes.append(
                {
                    "id": difference_id,
                    "status": "rejected_by_python",
                    "reason": reviewed.get(difference_id, "needs_human_review"),
                }
            )
    return outcomes


def _strict_response_schema(
    source_schema: dict,
) -> dict:
    """Convert the compiler schema into a strict structured-output schema."""

    schema = copy.deepcopy(
        source_schema
    )

    schema[
        "additionalProperties"
    ] = False

    resolutions = schema[
        "properties"
    ][
        "resolutions"
    ]

    item_schema = resolutions[
        "items"
    ]

    item_schema[
        "additionalProperties"
    ] = False

    required = list(
        item_schema.get(
            "required",
            [],
        )
    )

    if "note" not in required:
        required.append(
            "note"
        )

    item_schema[
        "required"
    ] = required

    return schema


def _openrouter_payload(
    batch: dict,
    provenance: PresetProvenance,
) -> dict:
    """Build a direct request from one verified immutable preset version.

    The exact version config is the base. The resolver's existing transient
    fields shallow-override it, matching documented preset merge precedence.
    The verified system prompt is placed before the existing user payload.
    """

    if (
        not provenance.verified
        or not isinstance(provenance.config, dict)
        or not isinstance(provenance.system_prompt, str)
    ):
        raise RuntimeError("Verified OpenRouter preset provenance is required")

    strict_schema = (
        _strict_response_schema(
            batch[
                "response_schema"
            ]
        )
    )

    # OpenRouter preset application shallow-merges request fields over the
    # preset config. These are precisely the current request-only fields.
    payload = copy.deepcopy(provenance.config)
    payload.update({
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "transcript_review",
                "strict": True,
                "schema": strict_schema,
            },
        },
        "messages": [
            {
                "role": "system",
                "content": provenance.system_prompt,
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "batch": (
                            batch["batch"]
                        ),
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        # TASK-076: request OpenRouter's actual per-call cost so budget
        # settlement can use trustworthy provider-reported usage.cost
        # instead of only ever falling back to "uncertain".
        "usage": {"include": True},
    })
    return payload


def _triage_openrouter_payload(
    batch: dict,
    provenance: PresetProvenance,
) -> dict:
    """Build an advisory-only request under verified preset authority."""

    if (
        not provenance.verified
        or not isinstance(provenance.config, dict)
        or not isinstance(provenance.system_prompt, str)
    ):
        raise RuntimeError("Verified OpenRouter preset provenance is required")

    payload = copy.deepcopy(provenance.config)
    payload.update(
        {
            # TASK-076: triage-only overrides -- see TRIAGE_MAX_COMPLETION_TOKENS.
            # These are request-only fields layered over the locked preset
            # config, the same shallow-merge precedence already used for
            # response_format/messages; the verified model and system prompt
            # are untouched.
            "reasoning": {
                "max_tokens": TRIAGE_REASONING_MAX_TOKENS,
                "exclude": True,
            },
            "max_tokens": TRIAGE_MAX_COMPLETION_TOKENS,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "transcript_review_triage",
                    "strict": True,
                    "schema": copy.deepcopy(batch["response_schema"]),
                },
            },
            "messages": [
                {
                    "role": "system",
                    "content": provenance.system_prompt,
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "instruction": (
                                "Advisory triage only. Recommend exactly one "
                                "supplied source when supported; otherwise use "
                                "likely_custom or needs_audio. Never rewrite "
                                "source text and abstain when uncertain."
                            ),
                            "batch": batch["batch"],
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            # TASK-076: see the matching note in _openrouter_payload.
            "usage": {"include": True},
        }
    )
    return payload


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
        if (
            isinstance(error, requests.HTTPError)
            and error.response is not None
        )
        else None
    )
    reason = type(error).__name__
    if status is not None:
        reason = f"{reason}:{status}"
    return reason


def _post_openrouter(
    payload: dict,
    *,
    episode_key: str | None,
    source_fingerprint: str | None,
    stage: str,
    provenance: PresetProvenance,
    pricing_transport: Callable[[str, str], list] | None = None,
    pricing_now: Callable[[], str] | None = None,
):
    """Call OpenRouter with one retry for transient failures.

    TASK-076: every physical attempt here independently reserves episode
    AI budget immediately before the request is sent, then settles or
    marks the reservation uncertain based on that attempt's own outcome.
    A retry always requires a fresh reservation (a new attempt_id); an
    earlier attempt's reservation, once marked uncertain, is never
    optimistically released and keeps consuming the episode's budget
    until reconciled. See docs/superpowers/specs/2026-09-16-human-review-
    evidence-assisted-adjudication-design.md ("Atomic admission",
    "Settlement and uncertain outcomes").
    """

    api_key = os.getenv(
        "PODCAST_TRANSCRIPT_REVIEW_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "Missing PODCAST_TRANSCRIPT_REVIEW_API_KEY for transcript review"
        )

    if (
        not isinstance(episode_key, str)
        or not episode_key
        or not isinstance(source_fingerprint, str)
        or not source_fingerprint
    ):
        raise RuntimeError(
            "episode_key and source_fingerprint are required to reserve "
            "episode AI budget before a transcript review provider call"
        )

    resolved_preset = resolve_preset_model_and_pricing(
        provenance,
        api_key=api_key,
        transport=pricing_transport,
        now=pricing_now,
    )

    payload = with_provider_price_ceiling(payload, resolved_preset.pricing_bound)
    request_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    # The max_tokens this physical request will actually send is read from
    # the outbound payload itself, not re-derived from the resolved
    # preset's own config: triage deliberately overrides max_tokens to a
    # tighter, schema-provable ceiling (see TRIAGE_MAX_COMPLETION_TOKENS)
    # via the same request-only shallow-merge already used for
    # response_format/messages, so the raw preset config value would
    # over-reserve (safe direction, but not the tight bound this design
    # requires). Reading it from payload keeps the reservation an exact
    # match for what is actually about to be sent, for every stage.
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

    for attempt in range(
        1,
        REVIEW_MAX_ATTEMPTS + 1,
    ):
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
                    "Authorization": (
                        f"Bearer {api_key}"
                    ),
                    "Content-Type": (
                        "application/json"
                    ),
                    "X-OpenRouter-Metadata": "enabled",
                },
                json=payload,
                timeout=(
                    REVIEW_TIMEOUT_SECONDS
                ),
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

        except (
            requests.Timeout,
            requests.ConnectionError,
            requests.HTTPError,
        ) as error:
            last_error = error

            mark_budget_attempt_uncertain(
                episode_key,
                source_fingerprint,
                attempt_id,
                reason=_budget_uncertain_reason(error),
            )

            status = (
                error.response.status_code
                if (
                    isinstance(
                        error,
                        requests.HTTPError,
                    )
                    and error.response
                    is not None
                )
                else None
            )

            retryable = (
                isinstance(
                    error,
                    (
                        requests.Timeout,
                        requests.ConnectionError,
                    ),
                )
                or status == 429
                or (
                    status is not None
                    and 500 <= status < 600
                )
            )

            if (
                not retryable
                or attempt
                >= REVIEW_MAX_ATTEMPTS
            ):
                raise

            time.sleep(
                2
            )

    raise RuntimeError(
        "OpenRouter transcript review failed"
    ) from last_error


def resolve_compiler_batch(
    batch: dict,
    provenance: PresetProvenance | None = None,
    *,
    episode_key: str | None = None,
    source_fingerprint: str | None = None,
    pricing_transport=None,
) -> dict[str, list[dict]]:
    """Resolve only the compiler's compact high-value review batch."""

    if not batch[
        "batch"
    ][
        "diff_items"
    ]:
        return {
            "accepted": [],
            "review": [],
        }

    if provenance is None or not provenance.verified:
        raise RuntimeError("Verified OpenRouter preset provenance is required")

    response = _post_openrouter(
        _openrouter_payload(
            batch,
            provenance,
        ),
        episode_key=episode_key,
        source_fingerprint=source_fingerprint,
        stage=STAGE_RESOLVER,
        provenance=provenance,
        pricing_transport=pricing_transport,
    )

    response_payload = response.json()
    content = response_payload[
        "choices"
    ][0][
        "message"
    ][
        "content"
    ]

    resolver_response = _response_payload(content)
    validated = validate_resolver_response(batch, resolver_response)
    validated["outcomes"] = _resolver_outcomes(
        batch,
        resolver_response,
        validated,
    )

    complete = _complete_resolution(
        batch,
        validated,
    )
    completion_metadata = _completion_metadata(response_payload)
    if completion_metadata:
        complete["completion_metadata"] = completion_metadata
    return complete


def _validated_triage(
    batch: dict,
    response: dict,
) -> dict[str, list[dict]]:
    """Validate advisory output without creating resolver acceptance."""

    items = {
        item["id"]: item
        for item in batch.get("batch", {}).get("diff_items", [])
        if isinstance(item, dict) and isinstance(item.get("id"), int)
    }
    submitted = (
        response.get("triage", [])
        if isinstance(response, dict)
        else []
    )
    if not isinstance(submitted, list):
        submitted = []

    by_id: dict[int, dict] = {}
    duplicate_ids: set[int] = set()

    for raw in submitted:
        if not isinstance(raw, dict):
            continue

        difference_id = raw.get("id")
        if (
            not isinstance(difference_id, int)
            or difference_id not in items
        ):
            continue

        if difference_id in by_id:
            duplicate_ids.add(difference_id)
            continue

        by_id[difference_id] = raw

    allowed = {
        "recommend_apple",
        "recommend_whisper",
        "likely_custom",
        "needs_audio",
    }
    recommendations: list[dict] = []
    outcomes: list[dict] = []

    for difference_id, item in items.items():
        raw = by_id.get(difference_id)

        if raw is None:
            outcomes.append(
                {
                    "id": difference_id,
                    "status": "missing_response",
                    "recommendation": "needs_audio",
                    "reason": "missing_triage_response",
                }
            )
            continue

        recommendation = raw.get("recommendation")
        confidence = raw.get("confidence")
        reason = raw.get("reason")
        valid_reason = (
            isinstance(reason, str)
            and 1 <= len(reason.strip()) <= 400
        )

        if (
            difference_id in duplicate_ids
            or recommendation not in allowed
            or confidence not in {"high", "medium", "low"}
            or not valid_reason
        ):
            outcomes.append(
                {
                    "id": difference_id,
                    "status": "invalid_recommendation",
                    "recommendation": "needs_audio",
                    "reason": "invalid_triage_recommendation",
                }
            )
            continue

        advisory = {
            "id": difference_id,
            "recommendation": recommendation,
            "confidence": confidence,
            "reason": reason.strip(),
        }

        if recommendation == "recommend_apple":
            advisory.update(
                {
                    "source": "apple",
                    "text": item["apple_text"],
                }
            )
        elif recommendation == "recommend_whisper":
            advisory.update(
                {
                    "source": "whisper",
                    "text": item["whisper_text"],
                }
            )

        recommendations.append(advisory)
        outcomes.append(
            {
                "id": difference_id,
                "status": "advisory",
                "recommendation": recommendation,
                "confidence": confidence,
                "reason": reason.strip(),
            }
        )

    return {
        "accepted": [],
        "recommendations": recommendations,
        "outcomes": outcomes,
    }


class TriageResponseParseError(ValueError):
    """Safe diagnostic wrapper for malformed advisory-triage responses."""

    def __init__(self, diagnostics: dict):
        self.diagnostics = dict(diagnostics)

        ordered_fields = (
            "finish_reason",
            "completion_id",
            "served_model",
            "served_provider",
            "content_chars",
            "content_sha256",
        )
        details = [
            f"{field}={self.diagnostics[field]}"
            for field in ordered_fields
            if field in self.diagnostics
        ]

        super().__init__(
            " ".join(
                [
                    "triage_response_invalid_json",
                    *details,
                ]
            )
        )


def _triage_response_parse_diagnostics(
    response_payload: dict,
    content,
) -> dict:
    """Return non-sensitive response-envelope evidence for parse failures."""

    diagnostics = _completion_metadata(response_payload)

    choices = (
        response_payload.get("choices", [])
        if isinstance(response_payload, dict)
        else []
    )
    first_choice = (
        choices[0]
        if isinstance(choices, list)
        and choices
        and isinstance(choices[0], dict)
        else {}
    )
    finish_reason = first_choice.get("finish_reason")
    if isinstance(finish_reason, str) and finish_reason:
        diagnostics["finish_reason"] = finish_reason

    if isinstance(content, str):
        diagnostics["content_chars"] = len(content)
        diagnostics["content_sha256"] = hashlib.sha256(
            content.encode("utf-8")
        ).hexdigest()

    return diagnostics


def _triage_single_batch(
    batch: dict,
    provenance: PresetProvenance,
    *,
    episode_key: str | None = None,
    source_fingerprint: str | None = None,
    pricing_transport=None,
) -> dict:
    """Run one bounded advisory-triage request."""

    response = _post_openrouter(
        _triage_openrouter_payload(
            batch,
            provenance,
        ),
        episode_key=episode_key,
        source_fingerprint=source_fingerprint,
        stage=STAGE_TRIAGE,
        provenance=provenance,
        pricing_transport=pricing_transport,
    )
    response_payload = response.json()
    content = response_payload["choices"][0]["message"]["content"]

    if not isinstance(content, str):
        raise TriageResponseParseError(
            _triage_response_parse_diagnostics(
                response_payload,
                content,
            )
        )

    try:
        triage_response = _response_payload(content)
    except ValueError as error:
        raise TriageResponseParseError(
            _triage_response_parse_diagnostics(
                response_payload,
                content,
            )
        ) from error

    result = _validated_triage(batch, triage_response)

    completion_metadata = _completion_metadata(response_payload)
    if completion_metadata:
        result["completion_metadata"] = completion_metadata

    return result


def _triage_chunk_failure_metadata(
    error: Exception,
    item_ids: list[int],
) -> dict:
    """Keep only bounded non-sensitive evidence for one failed triage chunk."""

    failure = {
        "item_ids": list(item_ids),
        "error_type": type(error).__name__,
    }
    # TASK-123: keep only the numerical HTTP status of a rejected request so a
    # deterministic request incompatibility is diagnosable without retaining
    # the provider body, headers, credentials or transcript content.
    response = getattr(error, "response", None)
    status_code = getattr(response, "status_code", None)
    if (
        isinstance(status_code, int)
        and not isinstance(status_code, bool)
        and 100 <= status_code <= 599
    ):
        failure["http_status"] = status_code
    diagnostics = getattr(error, "diagnostics", None)
    if not isinstance(diagnostics, dict):
        return failure

    for field in (
        "finish_reason",
        "completion_id",
        "served_model",
        "served_provider",
    ):
        value = diagnostics.get(field)
        if isinstance(value, str) and value:
            failure[field] = value

    content_chars = diagnostics.get("content_chars")
    if (
        isinstance(content_chars, int)
        and not isinstance(content_chars, bool)
        and content_chars >= 0
    ):
        failure["content_chars"] = content_chars

    content_sha256 = diagnostics.get("content_sha256")
    if (
        isinstance(content_sha256, str)
        and len(content_sha256) == 64
        and all(character in "0123456789abcdef" for character in content_sha256)
    ):
        failure["content_sha256"] = content_sha256

    return failure


def triage_compiler_batch(
    batch: dict,
    provenance: PresetProvenance | None = None,
    *,
    episode_key: str | None = None,
    source_fingerprint: str | None = None,
    pricing_transport=None,
) -> dict[str, list[dict]]:
    """Run non-authoritative advisory triage for pending review cards."""

    items = batch.get("batch", {}).get("diff_items", [])
    if not items:
        return {
            "accepted": [],
            "recommendations": [],
            "outcomes": [],
        }

    if provenance is None or not provenance.verified:
        raise RuntimeError("Verified OpenRouter preset provenance is required")

    chunk_size = TRIAGE_CHUNK_SIZE
    if (
        not isinstance(chunk_size, int)
        or isinstance(chunk_size, bool)
        or chunk_size <= 0
    ):
        raise RuntimeError("TRIAGE_CHUNK_SIZE must be a positive integer")

    # Preserve the existing single-request behavior for already-small batches,
    # including the current fail-closed exception path and its diagnostics.
    if len(items) <= chunk_size:
        return _triage_single_batch(
            batch,
            provenance,
            episode_key=episode_key,
            source_fingerprint=source_fingerprint,
            pricing_transport=pricing_transport,
        )

    merged = {
        "accepted": [],
        "recommendations": [],
        "outcomes": [],
    }
    failures: list[dict] = []
    completion_chunks: list[dict] = []

    for start in range(0, len(items), chunk_size):
        chunk_items = items[start : start + chunk_size]
        chunk_batch = copy.deepcopy(batch)
        chunk_batch["batch"]["diff_items"] = copy.deepcopy(chunk_items)
        item_ids = [
            item["id"]
            for item in chunk_items
            if isinstance(item, dict) and isinstance(item.get("id"), int)
        ]

        try:
            chunk_result = _triage_single_batch(
                chunk_batch,
                provenance,
                episode_key=episode_key,
                source_fingerprint=source_fingerprint,
                pricing_transport=pricing_transport,
            )
        except Exception as error:
            merged["outcomes"].extend(
                {
                    "id": difference_id,
                    "status": "unavailable",
                    "recommendation": "needs_audio",
                    "reason": "triage_unavailable",
                }
                for difference_id in item_ids
            )
            failures.append(
                _triage_chunk_failure_metadata(
                    error,
                    item_ids,
                )
            )
            continue

        merged["recommendations"].extend(
            dict(item)
            for item in chunk_result.get("recommendations", [])
            if isinstance(item, dict)
        )
        merged["outcomes"].extend(
            dict(item)
            for item in chunk_result.get("outcomes", [])
            if isinstance(item, dict)
        )

        completion_metadata = chunk_result.get("completion_metadata")
        if isinstance(completion_metadata, dict) and completion_metadata:
            completion_chunks.append(
                {
                    "item_ids": list(item_ids),
                    **completion_metadata,
                }
            )

    if completion_chunks:
        merged["completion_metadata"] = {
            "chunks": completion_chunks,
        }
    if failures:
        merged["failure_metadata"] = failures

    return merged


def _completion_metadata(response_payload: dict) -> dict:
    """Keep only documented served metadata that OpenRouter actually returned."""

    metadata: dict = {}
    completion_id = response_payload.get("id") if isinstance(response_payload, dict) else None
    served_model = response_payload.get("model") if isinstance(response_payload, dict) else None
    if isinstance(completion_id, str) and completion_id:
        metadata["completion_id"] = completion_id
    if isinstance(served_model, str) and served_model:
        metadata["served_model"] = served_model

    openrouter_metadata = response_payload.get("openrouter_metadata") if isinstance(response_payload, dict) else None
    endpoints = openrouter_metadata.get("endpoints", {}).get("available", []) if isinstance(openrouter_metadata, dict) else []
    if isinstance(endpoints, list):
        for endpoint in endpoints:
            if isinstance(endpoint, dict) and endpoint.get("selected") is True:
                provider = endpoint.get("provider")
                if isinstance(provider, str) and provider:
                    metadata["served_provider"] = provider
                break
    return metadata
