"""TASK-076 exact preset pricing and conservative reservation authority.

No paid request may be sent unless Python can derive a conservative
upper-bound reservation from current trusted pricing evidence and the
request's bounded dimensions. This module resolves that pricing evidence
(never from the generated docs/openrouter-presets.md inspection snapshot,
which is inspection material only and can be stale relative to the live
provider by design) and computes the conservative USD upper bound.

All money math uses Decimal; float, bool, NaN, Infinity, and negative
admission inputs are rejected outright rather than silently coerced.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from decimal import ROUND_CEILING, Decimal, InvalidOperation
import hashlib
import math
from typing import Callable

import requests

from .preset_provenance import (
    PresetProvenance,
    canonical_json_bytes,
)


OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
# TASK-076 Task 9: duration-priced audio/STT models (e.g. openai/gpt-transcribe)
# are absent from the unfiltered catalog entirely -- live-confirmed
# 2026-09-21, 446 unfiltered entries and zero matches (including zero fuzzy
# "transcri"/"whisper" matches) versus 21 entries and an exact match once
# filtered. Audio pricing resolution must always use this dedicated URL.
OPENROUTER_AUDIO_MODELS_URL = "https://openrouter.ai/api/v1/models?output_modalities=transcription"
PRICING_TIMEOUT_SECONDS = 15


class PricingResolutionError(ValueError):
    """Raised when current trusted pricing evidence cannot be derived.

    Missing, stale, malformed, ambiguous, or incompatible pricing evidence
    fails closed with this error before any provider call is considered.
    """


def _safe_reason(value: object) -> str:
    return str(value).replace("\n", " ")[:160]


@dataclass(frozen=True)
class ModelPricing:
    """Current trusted pricing evidence for one exact model."""

    model_id: str
    prompt_usd_per_token: Decimal
    completion_usd_per_token: Decimal
    context_length: int
    captured_at: str
    source_api: str
    evidence_sha256: str


@dataclass(frozen=True)
class AudioPricing:
    """Current trusted pricing evidence for one duration-priced audio model.

    TASK-076 Task 9: OpenRouter reuses the text-pricing "prompt" JSON field
    name to carry USD-per-second for duration-priced audio/STT models; this
    dataclass gives that value its own honest name rather than propagating
    the misleading "prompt" label into budget/reservation code. There is no
    context_length/completion-token analogue for duration pricing.
    """

    model_id: str
    usd_per_second: Decimal
    captured_at: str
    source_api: str
    evidence_sha256: str


@dataclass(frozen=True)
class ConservativeTextPricingBound:
    """The worst-case per-token prices and tightest context length across
    every model a preset's request could be routed to.

    When a preset config declares a single model, this is that model's own
    pricing. When it declares a fallback list of candidate models (an
    OpenRouter ``models`` array), the bound is conservative across the
    whole set: the maximum prompt/completion price (since routing could
    land on the more expensive candidate) and the minimum context length
    (the least headroom actually available across candidates).
    """

    model_ids: tuple[str, ...]
    prompt_usd_per_token: Decimal
    completion_usd_per_token: Decimal
    context_length: int
    pricings: tuple[ModelPricing, ...]


@dataclass(frozen=True)
class ResolvedPreset:
    """One verified/current preset snapshot bound to its pricing evidence.

    The config/system_prompt here are the exact same values the physical
    request must use: pricing is never resolved against a re-fetched (and
    possibly different) mutable ``@preset/<slug>`` snapshot.
    """

    slug: str | None
    preset_id: str | None
    version_id: str | None
    version: int | None
    config: dict
    system_prompt: str
    config_digest: str | None
    system_prompt_digest: str | None
    pricing_bound: ConservativeTextPricingBound


def bounded_completion_tokens(payload: dict) -> int:
    """Use the physical request's single, positive output-token ceiling."""

    legacy = payload.get("max_tokens")
    modern = payload.get("max_completion_tokens")
    if modern is not None and legacy is not None and modern != legacy:
        raise RuntimeError("Conflicting max_tokens and max_completion_tokens ceilings")
    ceiling = modern if modern is not None else legacy
    if not isinstance(ceiling, int) or isinstance(ceiling, bool) or ceiling <= 0:
        raise RuntimeError("Physical request has no valid bounded completion-token ceiling")
    return ceiling


def with_provider_price_ceiling(
    payload: dict, pricing_bound: ConservativeTextPricingBound
) -> dict:
    """Copy one physical request and retain only provider routes at reserved rates.

    OpenRouter's ``provider.max_price`` uses USD per million tokens. This is
    a routing filter alongside the ledger reservation, not a replacement for
    the per-request upper-bound cost calculation.
    """

    if not isinstance(payload, dict):
        raise PricingResolutionError("Physical request payload must be an object")
    result = copy.deepcopy(payload)
    provider = result.get("provider", {})
    if not isinstance(provider, dict):
        raise PricingResolutionError("Physical request provider policy must be an object")
    existing = provider.get("max_price", {})
    if not isinstance(existing, dict):
        raise PricingResolutionError("Physical request max_price must be an object")

    max_price = dict(existing)
    for side, per_token in (
        ("prompt", pricing_bound.prompt_usd_per_token),
        ("completion", pricing_bound.completion_usd_per_token),
    ):
        ceiling = per_token * Decimal(1_000_000)
        if side in existing:
            old = existing[side]
            if isinstance(old, bool) or not isinstance(old, (int, float)):
                raise PricingResolutionError(f"Invalid provider.max_price.{side}")
            old_decimal = Decimal(str(old))
            if not old_decimal.is_finite() or old_decimal < 0:
                raise PricingResolutionError(f"Invalid provider.max_price.{side}")
            ceiling = min(ceiling, old_decimal)
        numeric = float(ceiling)
        if not math.isfinite(numeric):
            raise PricingResolutionError(f"Unrepresentable provider.max_price.{side}")
        if Decimal(str(numeric)) > ceiling:
            numeric = math.nextafter(numeric, -math.inf)
        max_price[side] = numeric

    provider["max_price"] = max_price
    result["provider"] = provider
    return result


def decimal_from_admission_input(value: object, *, label: str) -> Decimal:
    """Convert one externally-controlled numeric admission input to Decimal.

    Money math never uses binary floating point. A caller must already
    hold a Decimal (or plain int) — float and bool are rejected outright,
    since a float admission input can silently encode an imprecise cost.
    """

    if isinstance(value, bool):
        raise ValueError(f"{label} must not be a boolean")
    if isinstance(value, float):
        raise ValueError(f"{label} must not be a float; use Decimal(str(value))")
    if isinstance(value, int):
        value = Decimal(value)
    if not isinstance(value, Decimal):
        raise ValueError(f"{label} must be a Decimal")
    if not value.is_finite():
        raise ValueError(f"{label} must be finite (not NaN/Infinity)")
    if value < 0:
        raise ValueError(f"{label} must not be negative")
    return value


def _parse_decimal_price(value: object, *, field: str, model_id: str) -> Decimal:
    if not isinstance(value, str) or not value:
        raise PricingResolutionError(
            f"{model_id}: pricing.{field} must be a non-empty decimal string"
        )
    try:
        price = Decimal(value)
    except InvalidOperation as error:
        raise PricingResolutionError(
            f"{model_id}: pricing.{field} is not a valid decimal"
        ) from error
    try:
        return decimal_from_admission_input(price, label=f"{model_id} pricing.{field}")
    except ValueError as error:
        raise PricingResolutionError(_safe_reason(error)) from error


def _default_models_transport(url: str, api_key: str) -> list:
    response = requests.get(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=PRICING_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        raise PricingResolutionError("OpenRouter models catalog response is malformed")
    return data


def resolve_model_pricing(
    model_id: str,
    *,
    api_key: str,
    transport: Callable[[str, str], list] | None = None,
    now: Callable[[], str] | None = None,
) -> ModelPricing:
    """Resolve current trusted OpenRouter pricing evidence for one model.

    This always performs (or delegates to an injected transport standing in
    for) a live fetch against OpenRouter's models catalog. It never reads
    the generated docs/openrouter-presets.md inspection snapshot, which can
    be stale relative to the live provider by design; production admission
    must derive pricing evidence from a live current-pricing fetch (or an
    explicitly frozen test fixture), never that file.
    """

    if not isinstance(model_id, str) or not model_id:
        raise PricingResolutionError("model_id must be a non-empty string")

    fetch = transport if transport is not None else _default_models_transport
    try:
        catalog = fetch(OPENROUTER_MODELS_URL, api_key)
    except PricingResolutionError:
        raise
    except requests.HTTPError as error:
        raise PricingResolutionError(
            f"OpenRouter models catalog request failed: {_safe_reason(error)}"
        ) from error
    except (requests.Timeout, requests.ConnectionError) as error:
        raise PricingResolutionError(
            f"OpenRouter models catalog is unavailable: {_safe_reason(error)}"
        ) from error
    except requests.RequestException as error:
        raise PricingResolutionError(
            f"OpenRouter models catalog request failed: {_safe_reason(error)}"
        ) from error

    if not isinstance(catalog, list):
        raise PricingResolutionError("OpenRouter models catalog response is malformed")

    matches = [
        entry
        for entry in catalog
        if isinstance(entry, dict) and entry.get("id") == model_id
    ]
    if not matches:
        raise PricingResolutionError(
            f"{model_id} is not present in the current OpenRouter models catalog"
        )
    if len(matches) > 1:
        raise PricingResolutionError(
            f"{model_id} matched more than one current OpenRouter models catalog entry"
        )
    entry = matches[0]

    pricing = entry.get("pricing")
    if not isinstance(pricing, dict):
        raise PricingResolutionError(f"{model_id}: pricing evidence is missing")
    prompt_price = _parse_decimal_price(pricing.get("prompt"), field="prompt", model_id=model_id)
    completion_price = _parse_decimal_price(
        pricing.get("completion"), field="completion", model_id=model_id
    )

    context_length = entry.get("context_length")
    if (
        not isinstance(context_length, int)
        or isinstance(context_length, bool)
        or context_length <= 0
    ):
        raise PricingResolutionError(f"{model_id}: context_length is missing or invalid")

    captured_at = (now or _iso_now)()
    evidence_sha256 = "sha256:" + hashlib.sha256(canonical_json_bytes(entry)).hexdigest()

    return ModelPricing(
        model_id=model_id,
        prompt_usd_per_token=prompt_price,
        completion_usd_per_token=completion_price,
        context_length=context_length,
        captured_at=captured_at,
        source_api=OPENROUTER_MODELS_URL,
        evidence_sha256=evidence_sha256,
    )


def resolve_audio_model_pricing(
    model_id: str,
    *,
    api_key: str,
    transport: Callable[[str, str], list] | None = None,
    now: Callable[[], str] | None = None,
) -> AudioPricing:
    """Resolve current trusted duration-pricing evidence for one audio model.

    TASK-076 Task 9: always performs (or delegates to an injected transport
    standing in for) a live fetch against OPENROUTER_AUDIO_MODELS_URL --
    never the unfiltered catalog resolve_model_pricing uses, which does not
    list these models at all -- and never the generated
    docs/openrouter-presets.md inspection snapshot.
    """

    if not isinstance(model_id, str) or not model_id:
        raise PricingResolutionError("model_id must be a non-empty string")

    fetch = transport if transport is not None else _default_models_transport
    try:
        catalog = fetch(OPENROUTER_AUDIO_MODELS_URL, api_key)
    except PricingResolutionError:
        raise
    except requests.HTTPError as error:
        raise PricingResolutionError(
            f"OpenRouter audio models catalog request failed: {_safe_reason(error)}"
        ) from error
    except (requests.Timeout, requests.ConnectionError) as error:
        raise PricingResolutionError(
            f"OpenRouter audio models catalog is unavailable: {_safe_reason(error)}"
        ) from error
    except requests.RequestException as error:
        raise PricingResolutionError(
            f"OpenRouter audio models catalog request failed: {_safe_reason(error)}"
        ) from error

    if not isinstance(catalog, list):
        raise PricingResolutionError("OpenRouter audio models catalog response is malformed")

    matches = [
        entry
        for entry in catalog
        if isinstance(entry, dict) and entry.get("id") == model_id
    ]
    if not matches:
        raise PricingResolutionError(
            f"{model_id} is not present in the current OpenRouter audio models catalog"
        )
    if len(matches) > 1:
        raise PricingResolutionError(
            f"{model_id} matched more than one current OpenRouter audio models catalog entry"
        )
    entry = matches[0]

    pricing = entry.get("pricing")
    if not isinstance(pricing, dict):
        raise PricingResolutionError(f"{model_id}: pricing evidence is missing")
    # OpenRouter carries duration pricing under the same "prompt" field name
    # used for text token pricing; see AudioPricing's docstring.
    usd_per_second = _parse_decimal_price(pricing.get("prompt"), field="prompt", model_id=model_id)
    if usd_per_second <= 0:
        raise PricingResolutionError(
            f"{model_id}: pricing.prompt (USD/second) must be positive"
        )

    captured_at = (now or _iso_now)()
    evidence_sha256 = "sha256:" + hashlib.sha256(canonical_json_bytes(entry)).hexdigest()

    return AudioPricing(
        model_id=model_id,
        usd_per_second=usd_per_second,
        captured_at=captured_at,
        source_api=OPENROUTER_AUDIO_MODELS_URL,
        evidence_sha256=evidence_sha256,
    )


def _iso_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _preset_candidate_model_ids(config: dict) -> tuple[str, ...]:
    single = config.get("model")
    plural = config.get("models")
    if single is not None and (not isinstance(single, str) or not single):
        return ()
    if plural is not None and (
        not isinstance(plural, list)
        or not plural
        or not all(isinstance(model_id, str) and model_id for model_id in plural)
    ):
        return ()
    return tuple(dict.fromkeys(([single] if single is not None else []) + (plural or [])))


def resolve_preset_model_and_pricing(
    preset: PresetProvenance,
    *,
    api_key: str,
    transport: Callable[[str, str], list] | None = None,
    now: Callable[[], str] | None = None,
) -> ResolvedPreset:
    """Bind one verified/current preset snapshot to its pricing evidence.

    ``preset`` must already be a status="verified" PresetProvenance from
    either verify_transcript_reviewer (locked exact version) or
    fetch_current_designated_preset (production-safe current mutable
    snapshot). This function never re-fetches or re-resolves the preset
    itself — it takes no preset-fetching transport at all, only a pricing
    transport — so the physical request always uses the exact same
    model/config/system-prompt snapshot the pricing bound was derived from,
    never a later, possibly different mutable ``@preset/<slug>`` alias.
    """

    if not isinstance(preset, PresetProvenance) or not preset.verified:
        raise ValueError("preset must be a verified PresetProvenance")
    if not isinstance(preset.config, dict):
        raise ValueError("preset config is missing")
    if not isinstance(preset.system_prompt, str):
        raise ValueError("preset system_prompt is missing")

    model_ids = _preset_candidate_model_ids(preset.config)
    if not model_ids:
        raise ValueError("preset config declares no candidate model(s)")

    pricings = tuple(
        resolve_model_pricing(model_id, api_key=api_key, transport=transport, now=now)
        for model_id in model_ids
    )

    pricing_bound = ConservativeTextPricingBound(
        model_ids=model_ids,
        prompt_usd_per_token=max(p.prompt_usd_per_token for p in pricings),
        completion_usd_per_token=max(p.completion_usd_per_token for p in pricings),
        context_length=min(p.context_length for p in pricings),
        pricings=pricings,
    )

    return ResolvedPreset(
        slug=preset.slug,
        preset_id=preset.preset_id,
        version_id=preset.version_id,
        version=preset.version,
        config=preset.config,
        system_prompt=preset.system_prompt,
        config_digest=preset.config_digest,
        system_prompt_digest=preset.system_prompt_digest,
        pricing_bound=pricing_bound,
    )


def derive_text_reservation_usd(
    *,
    request_bytes: bytes,
    system_prompt: str,
    max_completion_tokens: int,
    prompt_usd_per_token: Decimal,
    completion_usd_per_token: Decimal,
    context_length: int,
) -> Decimal:
    """Conservative upper-bound USD reservation for one token-priced request.

    The prompt-token upper bound is the UTF-8 byte length of the exact
    outbound request plus the exact resolved system prompt: a byte-fallback
    BPE tokenizer can never encode a UTF-8 byte sequence into more tokens
    than it has bytes, so byte length is a safe (never-under) token
    ceiling. This never calls a real tokenizer and never depends on network
    access. The completion side always uses the exact configured
    max-completion ceiling, never an expected/average estimate. A request
    whose bounded token total would exceed the model's context length is
    rejected outright rather than silently reservation-capped.
    """

    if not isinstance(request_bytes, (bytes, bytearray)):
        raise ValueError("request_bytes must be bytes")
    if not isinstance(system_prompt, str):
        raise ValueError("system_prompt must be a string")
    if (
        not isinstance(max_completion_tokens, int)
        or isinstance(max_completion_tokens, bool)
        or max_completion_tokens <= 0
    ):
        raise ValueError("max_completion_tokens must be a positive integer")
    if (
        not isinstance(context_length, int)
        or isinstance(context_length, bool)
        or context_length <= 0
    ):
        raise ValueError("context_length must be a positive integer")

    prompt_price = decimal_from_admission_input(
        prompt_usd_per_token, label="prompt_usd_per_token"
    )
    completion_price = decimal_from_admission_input(
        completion_usd_per_token, label="completion_usd_per_token"
    )

    prompt_token_upper_bound = len(request_bytes) + len(system_prompt.encode("utf-8"))
    total_token_upper_bound = prompt_token_upper_bound + max_completion_tokens
    if total_token_upper_bound > context_length:
        raise ValueError(
            "request plus max completion tokens exceeds the model's context "
            "length; not admissible under a bounded reservation"
        )

    return (
        Decimal(prompt_token_upper_bound) * prompt_price
        + Decimal(max_completion_tokens) * completion_price
    )


def _default_endpoints_transport(url: str, api_key: str) -> list:
    response = requests.get(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=PRICING_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    data = payload.get("data") if isinstance(payload, dict) else None
    endpoints = data.get("endpoints") if isinstance(data, dict) else None
    if not isinstance(endpoints, list):
        raise PricingResolutionError("OpenRouter model endpoints response is malformed")
    return endpoints


def resolve_audio_endpoint_pricing(
    model_id: str,
    *,
    provider: str,
    api_key: str,
    transport: Callable[[str, str], list] | None = None,
    now: Callable[[], str] | None = None,
) -> AudioPricing:
    """Resolve duration pricing for one audio model on one pinned provider.

    TASK-125: providers of the same audio model differ by several times in
    price (the Whisper benchmark was silently routed to one about 3.3x the
    catalog price), so a pinned-provider request must reserve against that
    provider's own current endpoint price, never the model-level catalog
    price. Exactly one endpoint whose provider tag matches must exist.
    """

    if not isinstance(model_id, str) or not model_id:
        raise PricingResolutionError("model_id must be a non-empty string")
    if not isinstance(provider, str) or not provider:
        raise PricingResolutionError("provider must be a non-empty string")

    url = f"https://openrouter.ai/api/v1/models/{model_id}/endpoints"
    fetch = transport if transport is not None else _default_endpoints_transport
    try:
        endpoints = fetch(url, api_key)
    except PricingResolutionError:
        raise
    except requests.RequestException as error:
        raise PricingResolutionError(
            f"OpenRouter model endpoints request failed: {_safe_reason(error)}"
        ) from error

    if not isinstance(endpoints, list):
        raise PricingResolutionError("OpenRouter model endpoints response is malformed")
    matches = [
        endpoint
        for endpoint in endpoints
        if isinstance(endpoint, dict)
        and isinstance(endpoint.get("tag"), str)
        and endpoint["tag"].split("/", 1)[0] == provider
    ]
    if len(matches) != 1:
        raise PricingResolutionError(
            f"{model_id}: expected exactly one {provider} endpoint, found {len(matches)}"
        )
    endpoint = matches[0]
    pricing = endpoint.get("pricing")
    if not isinstance(pricing, dict):
        raise PricingResolutionError(f"{model_id}/{provider}: pricing evidence is missing")
    usd_per_second = _parse_decimal_price(pricing.get("prompt"), field="prompt", model_id=model_id)
    if usd_per_second <= 0:
        raise PricingResolutionError(
            f"{model_id}/{provider}: pricing.prompt (USD/second) must be positive"
        )
    return AudioPricing(
        model_id=model_id,
        usd_per_second=usd_per_second,
        captured_at=(now or _iso_now)(),
        source_api=url,
        evidence_sha256="sha256:" + hashlib.sha256(canonical_json_bytes(endpoint)).hexdigest(),
    )


def derive_audio_reservation_usd(
    *,
    max_billable_seconds,
    usd_per_second,
) -> Decimal:
    """Conservative upper-bound USD reservation for one duration-priced request.

    Uses the maximum billable clip duration, per the design's reservation
    contract: the reservation must be an upper bound, not an
    expected/average estimate. Providers bill whole seconds, so a fractional
    duration is rounded up before pricing (TASK-126: a 15.6 s Third-ASR clip
    billed as 16 s exceeded its 15.6 s reservation and recorded a ledger
    integrity failure).
    """

    seconds = decimal_from_admission_input(max_billable_seconds, label="max_billable_seconds")
    price = decimal_from_admission_input(usd_per_second, label="usd_per_second")
    if seconds <= 0:
        raise ValueError("max_billable_seconds must be positive")

    return seconds.to_integral_value(rounding=ROUND_CEILING) * price


def resolve_stage_reservation_usd(
    *,
    resolved_preset: ResolvedPreset,
    worst_case_request_bytes: bytes,
) -> Decimal:
    """Conservative reservation for one stage's worst-case bounded request.

    TASK-076 Task 5: thin wrapper over derive_text_reservation_usd for
    episode-AI-budget downstream-reserve planning. Takes an
    already-resolved preset/pricing snapshot (see
    resolve_preset_model_and_pricing) and that stage's own worst-case
    outbound request bytes (see each knowledge-stage module's own
    worst_case_openrouter_payload builder), and extracts
    max_completion_tokens from the resolved preset's own config so
    callers never have to re-derive it by hand -- the exact same
    max_tokens value the physical request will actually send.
    """

    if not isinstance(resolved_preset, ResolvedPreset):
        raise ValueError("resolved_preset must be a ResolvedPreset")

    max_completion_tokens = resolved_preset.config.get("max_tokens")
    if (
        not isinstance(max_completion_tokens, int)
        or isinstance(max_completion_tokens, bool)
        or max_completion_tokens <= 0
    ):
        raise PricingResolutionError(
            "resolved preset config is missing a valid positive max_tokens"
        )

    return derive_text_reservation_usd(
        request_bytes=worst_case_request_bytes,
        system_prompt=resolved_preset.system_prompt,
        max_completion_tokens=max_completion_tokens,
        prompt_usd_per_token=resolved_preset.pricing_bound.prompt_usd_per_token,
        completion_usd_per_token=resolved_preset.pricing_bound.completion_usd_per_token,
        context_length=resolved_preset.pricing_bound.context_length,
    )
