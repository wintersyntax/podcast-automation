"""Durable human transcript review decisions and narrow audio evidence."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
import time
from typing import Callable, Iterable
from uuid import uuid4

from datetime import datetime, timedelta
from decimal import Decimal

from google.api_core.exceptions import GoogleAPICallError, NotFound, PreconditionFailed

from compiler.assisted_review import derive_assisted_state
from compiler.third_asr_window import anchored_third_asr_window
from compiler.review_tiers import derive_review_tier, review_tier_snapshot
from compiler.review_policy import (
    ASSISTED_REVIEW_POLICY_VERSION,
    assisted_routing,
    batch_recommendation,
)
from compiler.transcript import (
    MAX_HUMAN_EDIT_EXPANSION_WORDS,
    render_review_merge_preview,
    tokenize,
)

from .ai_budget import (
    _EPISODE_KEY,
    _SOURCE_FINGERPRINT,
    STAGE_THIRD_ASR,
    ensure_fresh_third_asr_budget_identity,
    mark_budget_attempt_uncertain,
    release_budget_attempt_pre_send,
    require_third_asr_budget_identity_reconciled,
    reserve_budget_batch,
    settle_budget_attempt,
)
from .ai_pricing import derive_audio_reservation_usd, resolve_audio_model_pricing
from .episode_contract import now_iso, paths_for
from .review_audio import (
    THIRD_ASR_FAILURE_PRE_SEND_RETRYABLE,
    THIRD_ASR_MODEL,
    THIRD_ASR_RETRYABLE_FAILURE_CLASSES,
    classify_third_asr_failure,
    review_clip_window,
    extract_audio_clip,
    third_asr_cache_key,
    transcribe_review_clip,
)
from .storage import get_bucket
from .worker_control import (
    WorkerRunRequestError,
    execution_matches_correlation,
    list_worker_executions,
    reconciled_execution_status,
    request_worker_run,
)


HUMAN_SOURCES = {"apple", "whisper", "third", "custom"}


# TASK-076 Task 9: generation-CAS per-item Third-ASR claims.
#
# The claim lives in its own GCS object per (episode generation, item),
# separate from both the episode budget ledger and the review record, so
# concurrent workers claiming different items never contend on the same
# blob. See docs/superpowers/specs/2026-09-16-human-review-evidence-
# assisted-adjudication-design.md, "Concurrency-safe Third ASR".

THIRD_ASR_CLAIM_TTL_SECONDS = 300
THIRD_ASR_CLAIM_MAX_CAS_ATTEMPTS = 3

# TASK-076 Task 9: one initial attempt plus at most one paid retry per
# item -- there is no unbounded exponential loop. See the design's
# "Retry and failure taxonomy".
THIRD_ASR_MAX_ATTEMPTS = 2
THIRD_ASR_RETRY_BASE_DELAY_SECONDS = 1.0
THIRD_ASR_RETRY_JITTER_SECONDS = 0.5


class ThirdAsrClaimError(RuntimeError):
    """Base error for Third-ASR per-item claim operations."""


class ThirdAsrInFlight(ThirdAsrClaimError):
    """An unexpired claim already owns this item.

    The caller must not send a duplicate provider request. This is raised
    whether or not the existing claim's cache_key matches the caller's:
    the item slot, not the cache key, is what must stay exclusive, since
    two physical requests racing for the same audio window is exactly
    what this claim exists to prevent.
    """

    def __init__(self, *, retry_after_seconds: float, cache_key: str):
        super().__init__("Third-ASR claim already in flight for this item")
        self.retry_after_seconds = retry_after_seconds
        self.cache_key = cache_key


class ThirdAsrClaimConflict(ThirdAsrClaimError):
    """Claim/finalize CAS attempts were exhausted by concurrent writers."""


def third_asr_claim_path(episode_key: str, source_fingerprint: str, difference_id: int) -> str:
    """Canonical GCS path for one item's Third-ASR claim object."""

    if not isinstance(episode_key, str) or not _EPISODE_KEY.fullmatch(episode_key):
        raise ValueError("episode_key must be a canonical 24-hex-character episode key")
    if not isinstance(source_fingerprint, str) or not _SOURCE_FINGERPRINT.fullmatch(
        source_fingerprint
    ):
        raise ValueError(
            "source_fingerprint must be a canonical 'sha256:<64-hex>' fingerprint"
        )
    if not isinstance(difference_id, int) or isinstance(difference_id, bool):
        raise ValueError("difference_id must be an int")
    digest = source_fingerprint.split(":", 1)[1]
    return f"episodes/{episode_key}/ai/third-asr-claims/{digest}/{difference_id}.json"


def _load_third_asr_claim_with_generation(
    episode_key: str, source_fingerprint: str, difference_id: int
) -> tuple[dict | None, int | None]:
    """Read current claim content and object generation.

    A generation of ``None`` means no claim object exists yet for this
    item -- the normal state before the first attempt or after release.
    """

    path = third_asr_claim_path(episode_key, source_fingerprint, difference_id)
    blob = get_bucket().blob(path)
    for _ in range(THIRD_ASR_CLAIM_MAX_CAS_ATTEMPTS):
        try:
            blob.reload()
        except NotFound:
            return None, None
        generation = int(blob.generation)
        try:
            content = blob.download_as_text(encoding="utf-8", if_generation_match=generation)
        except PreconditionFailed:
            continue
        payload = json.loads(content)
        if not isinstance(payload, dict):
            raise ValueError("Third-ASR claim object must be a JSON object")
        return payload, generation
    raise ThirdAsrClaimConflict(
        "Third-ASR claim object changed repeatedly while reading snapshot"
    )


def _save_third_asr_claim(
    episode_key: str,
    source_fingerprint: str,
    difference_id: int,
    claim: dict,
    *,
    if_generation_match: int | None,
) -> None:
    path = third_asr_claim_path(episode_key, source_fingerprint, difference_id)
    blob = get_bucket().blob(path)
    blob.upload_from_string(
        json.dumps(claim, ensure_ascii=False, indent=2) + "\n",
        content_type="application/json",
        if_generation_match=0 if if_generation_match is None else if_generation_match,
    )


def _claim_expiry_seconds_remaining(claim: dict, now: str) -> float:
    expires_at = datetime.fromisoformat(claim["expires_at"])
    current = datetime.fromisoformat(now)
    return (expires_at - current).total_seconds()


def acquire_third_asr_claim(
    episode_key: str,
    source_fingerprint: str,
    difference_id: int,
    *,
    cache_key: str,
    model: str,
    window: dict,
    budget_attempt_id: str,
    prepare_session_id: str | None = None,
    now: Callable[[], str] | None = None,
) -> dict:
    """Generation-pinned CAS acquire of one exclusive, TTL-bounded claim.

    Raises ``ThirdAsrInFlight`` without writing anything if an unexpired
    claim already owns this item. An expired claim is safely reclaimed
    via generation-matched CAS (never an unconditional overwrite).
    Bounded to ``THIRD_ASR_CLAIM_MAX_CAS_ATTEMPTS`` write attempts,
    matching the repository's existing bounded retry style (see
    ``ai_budget.MAX_BUDGET_CAS_ATTEMPTS``).
    """

    now_fn = now or now_iso
    for _ in range(THIRD_ASR_CLAIM_MAX_CAS_ATTEMPTS):
        existing, generation = _load_third_asr_claim_with_generation(
            episode_key, source_fingerprint, difference_id
        )
        current_now = now_fn()
        if existing is not None:
            remaining = _claim_expiry_seconds_remaining(existing, current_now)
            if remaining > 0:
                raise ThirdAsrInFlight(
                    retry_after_seconds=remaining, cache_key=existing.get("cache_key")
                )

        created = datetime.fromisoformat(current_now)
        expires = created + timedelta(seconds=THIRD_ASR_CLAIM_TTL_SECONDS)
        claim = {
            "difference_id": difference_id,
            "cache_key": cache_key,
            "request_id": uuid4().hex,
            "status": "in_flight",
            "created_at": current_now,
            "expires_at": expires.isoformat(),
            "model": model,
            "window": window,
            "prepare_session_id": prepare_session_id,
            "budget_attempt_id": budget_attempt_id,
        }
        try:
            _save_third_asr_claim(
                episode_key,
                source_fingerprint,
                difference_id,
                claim,
                if_generation_match=generation,
            )
        except PreconditionFailed:
            continue
        return claim

    raise ThirdAsrClaimConflict("Third-ASR claim changed repeatedly while acquiring it")


def release_third_asr_claim(
    episode_key: str, source_fingerprint: str, difference_id: int, *, request_id: str
) -> None:
    """Best-effort delete of a claim this exact ``request_id`` still owns.

    Only safe to call after a successful finalize or a definitively-not-
    sent pre-send failure; never for a provider-retryable/uncertain
    outcome, since that keeps the claim (and its budget reservation) live
    so a concurrent duplicate request still cannot slip through before
    the caller's own retry. A stale owner (superseded by its own retry,
    or reclaimed by someone else after expiry) never deletes the current
    owner's live claim.
    """

    path = third_asr_claim_path(episode_key, source_fingerprint, difference_id)
    blob = get_bucket().blob(path)
    try:
        blob.reload()
    except NotFound:
        return
    generation = int(blob.generation)
    try:
        content = blob.download_as_text(encoding="utf-8", if_generation_match=generation)
    except PreconditionFailed:
        return
    current = json.loads(content)
    if not isinstance(current, dict) or current.get("request_id") != request_id:
        return
    try:
        blob.delete(if_generation_match=generation)
    except (NotFound, PreconditionFailed):
        return


def _third_asr_retry_delay_seconds(*, random_fn: Callable[[], float] | None = None) -> float:
    """Small deterministic base delay plus bounded jitter, never unbounded."""

    fraction = (random_fn or random.random)()
    if not isinstance(fraction, (int, float)) or not (0.0 <= fraction <= 1.0):
        fraction = 0.0
    return THIRD_ASR_RETRY_BASE_DELAY_SECONDS + float(fraction) * THIRD_ASR_RETRY_JITTER_SECONDS


def _extract_third_asr_actual_usd(evidence: dict) -> Decimal | None:
    """Trustworthy provider-reported cost from third-ASR evidence, if any.

    ``parse_third_asr_response`` keeps ``cost``/``usage`` at the TOP level
    of its result (the OpenRouter STT response shape), unlike the nested
    ``usage.cost`` shape chat-completion stages use -- see
    ``podcast_engine.review._extract_actual_usd`` for that sibling. Missing
    or untrustworthy cost data returns ``None`` rather than guessing; the
    caller settles the attempt as ``uncertain`` instead.
    """

    if not isinstance(evidence, dict):
        return None
    cost = evidence.get("cost")
    if cost is None and isinstance(evidence.get("usage"), dict):
        cost = evidence["usage"].get("cost")
    if isinstance(cost, bool) or not isinstance(cost, (int, float)):
        return None
    try:
        value = Decimal(str(cost))
    except (ArithmeticError, ValueError):
        return None
    if not value.is_finite() or value < 0:
        return None
    return value


def _refresh_third_asr_claim_for_retry(
    episode_key: str,
    source_fingerprint: str,
    difference_id: int,
    *,
    budget_attempt_id: str,
    now: Callable[[], str] | None = None,
) -> dict:
    """Keep this worker's own claim exclusive across its bounded retry.

    A retry reuses the SAME claim object (same item slot stays owned by
    this call end-to-end) rather than releasing and re-acquiring, which
    would otherwise open a window for a concurrent duplicate request. Only
    ``request_id``, ``status``, timestamps, and ``budget_attempt_id``
    change; identity fields (cache_key/model/window/prepare_session_id)
    are preserved from the existing claim.
    """

    now_fn = now or now_iso
    for _ in range(THIRD_ASR_CLAIM_MAX_CAS_ATTEMPTS):
        existing, generation = _load_third_asr_claim_with_generation(
            episode_key, source_fingerprint, difference_id
        )
        if existing is None:
            raise ThirdAsrClaimConflict(
                "Third-ASR claim disappeared before its own retry could refresh it"
            )
        current_now = now_fn()
        created = datetime.fromisoformat(current_now)
        expires = created + timedelta(seconds=THIRD_ASR_CLAIM_TTL_SECONDS)
        refreshed = {
            **existing,
            "request_id": uuid4().hex,
            "status": "in_flight",
            "created_at": current_now,
            "expires_at": expires.isoformat(),
            "budget_attempt_id": budget_attempt_id,
        }
        try:
            _save_third_asr_claim(
                episode_key,
                source_fingerprint,
                difference_id,
                refreshed,
                if_generation_match=generation,
            )
        except PreconditionFailed:
            continue
        return refreshed

    raise ThirdAsrClaimConflict("Third-ASR claim changed repeatedly while refreshing it for retry")


def _finalize_third_asr_evidence(
    episode_key: str, difference_id: int, *, cache_key: str, evidence: dict
) -> dict:
    """Generation-verified write of successful evidence into the review record.

    Reloads the current record generation and re-derives this item's
    current cache key before persisting: stale evidence never attaches to
    a card whose episode input or clip window changed since the claim was
    acquired. A genuine identity drift silently skips persistence (the
    caller still received real, already-paid-for evidence) rather than
    raising -- only a repeated CAS generation conflict (an unrelated
    concurrent write) is retried, bounded to
    ``THIRD_ASR_CLAIM_MAX_CAS_ATTEMPTS`` attempts.
    """

    for _ in range(THIRD_ASR_CLAIM_MAX_CAS_ATTEMPTS):
        record, generation = load_review_record_with_generation(episode_key)
        try:
            item = _item(record, difference_id)
        except ValueError:
            return evidence

        window = review_clip_window(item)
        current_cache_key = third_asr_cache_key(
            input_fingerprint=record.get("input_fingerprint"),
            item_id=difference_id,
            window=window,
        )
        if current_cache_key != cache_key:
            return evidence

        updated = copy.deepcopy(record)
        for index, candidate in enumerate(updated.get("human_review", [])):
            if isinstance(candidate, dict) and candidate.get("id") == difference_id:
                updated["human_review"][index] = {**candidate, "third_asr": evidence}
                break
        try:
            save_review_record(episode_key, updated, if_generation_match=generation)
        except PreconditionFailed:
            continue
        return evidence

    raise ThirdAsrClaimConflict(
        "Review record changed repeatedly while finalizing Third-ASR evidence"
    )


RECOMPILE_IN_FLIGHT_STATUSES = {"starting", "started", "unknown"}
RECOMPILE_BLOCKING_STATUSES = RECOMPILE_IN_FLIGHT_STATUSES | {"completed"}


class RecompileAuditPersistenceError(RuntimeError):
    """The completed pipeline could not persist its terminal audit state."""


class RecompileLifecycleInvariantError(RuntimeError):
    """A correlated worker cannot prove its durable request identity."""


def review_queue_fingerprint(items: Iterable[dict]) -> str:
    """Return a stable identity for the decisions represented by review cards.

    This intentionally excludes timestamps, context, generated third-ASR
    evidence, and other incidental audit fields.  It captures only what a
    reviewer is being asked to decide, so it is safe to use in logs.
    """

    queue = []
    for item in items:
        if not isinstance(item, dict):
            continue
        focus = item.get("focus")
        queue.append(
            {
                "id": item.get("id"),
                "reason": item.get("reason"),
                "kind": item.get("kind"),
                "severity": item.get("severity"),
                "category": item.get("category"),
                "apple_text": item.get("apple_text"),
                "whisper_text": item.get("whisper_text"),
                "focus": {
                    "scope": focus.get("scope", "full"),
                    "apple_text": focus.get("apple_text"),
                    "whisper_text": focus.get("whisper_text"),
                }
                if isinstance(focus, dict)
                else None,
            }
        )

    encoded = json.dumps(
        sorted(
            queue,
            key=lambda candidate: json.dumps(
                candidate,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        ),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def review_generation_fingerprint(
    input_fingerprint: str | None, items: Iterable[dict]
) -> str:
    """Return the durable identity of one complete human-review generation."""

    encoded = json.dumps(
        {
            "input_fingerprint": input_fingerprint,
            "queue_fingerprint": review_queue_fingerprint(items),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def review_record_path(episode_key: str) -> str:
    return paths_for(episode_key)["resolver_record"]


def load_review_record(episode_key: str) -> dict:
    """Load one resolver audit record or report a review setup problem."""

    blob = get_bucket().blob(review_record_path(episode_key))
    if not blob.exists():
        raise FileNotFoundError(f"No resolver audit record for episode {episode_key}")
    payload = json.loads(blob.download_as_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Resolver audit record must be a JSON object")
    return payload


def load_review_record_with_generation(episode_key: str) -> tuple[dict, int]:
    """Read content and generation from one GCS object snapshot.

    The download is conditional on the generation returned by ``reload``. If
    another writer changes the object in between, GCS rejects the read and the
    next attempt starts from new metadata rather than pairing stale JSON with a
    newer generation.
    """

    blob = get_bucket().blob(review_record_path(episode_key))
    if not blob.exists():
        raise FileNotFoundError(f"No resolver audit record for episode {episode_key}")
    for _ in range(3):
        blob.reload()
        generation = int(blob.generation)
        try:
            content = blob.download_as_text(
                encoding="utf-8",
                if_generation_match=generation,
            )
        except PreconditionFailed:
            continue
        payload = json.loads(content)
        if not isinstance(payload, dict):
            raise ValueError("Resolver audit record must be a JSON object")
        return payload, generation
    raise RuntimeError("Review record changed repeatedly while reading snapshot")


def save_review_record(
    episode_key: str, record: dict, *, if_generation_match: int | None = None
) -> dict:
    """Persist only the review audit record; AI provenance is never replaced."""

    blob = get_bucket().blob(review_record_path(episode_key))
    kwargs = {"content_type": "application/json"}
    if if_generation_match is not None:
        kwargs["if_generation_match"] = if_generation_match
    blob.upload_from_string(json.dumps(record, ensure_ascii=False, indent=2) + "\n", **kwargs)
    return record


def pending_review_items(record: dict) -> list[dict]:
    """Return the remaining cards in a stable ID order."""

    return sorted(
        (copy.deepcopy(item) for item in record.get("human_review", []) if isinstance(item, dict)),
        key=lambda item: item.get("id", 0),
    )


def project_assisted_review_items(record: dict) -> list[dict]:
    """Project routing provenance, and for admitted cards deterministic
    assisted analysis, onto every pending review card.

    TASK-076 Task 10: every pending card gets routing/eligibility
    provenance (compiler.assisted_review.derive_assisted_state defers to
    assisted_routing() for admission); an admitted card additionally gets
    the derived assisted state (audio_pending / machine_supported_* /
    evidence_conflict / ambiguous_audio / neither_source_supported /
    audio_unavailable). This is pure, read-only projection -- it
    recomputes from the persisted card content (including any third_asr
    evidence or prepare-session placeholder already on the card) and
    never mutates the review record or performs I/O of its own. See
    docs/superpowers/specs/2026-09-16-human-review-evidence-assisted-
    adjudication-design.md, "Derived analysis and policy identity".
    """

    return [
        {**item, "assisted_review": derive_assisted_state(item)}
        for item in pending_review_items(record)
    ]


# TASK-076 Task 10: whole-selection prepare-session lease.
#
# A session lives in its own GCS object per (episode generation,
# session), separate from the episode budget ledger and from per-item
# Third-ASR claims, mirroring the same generation-CAS convention. See
# docs/superpowers/specs/2026-09-16-human-review-evidence-assisted-
# adjudication-design.md, "Prepare-assisted budget behavior".

PREPARE_SESSION_TTL_SECONDS = 600
PREPARE_SESSION_MAX_ITEMS = 40
PREPARE_SESSION_ITEM_STATES = frozenset(
    {"queued", "in_flight", "retrying", "prepared", "failed"}
)
_PREPARE_SESSION_ITEM_TRANSITIONS = {
    "queued": {"in_flight"},
    "in_flight": {"retrying", "prepared", "failed"},
    "retrying": {"in_flight", "prepared", "failed"},
}


class PrepareSessionError(RuntimeError):
    """Base error for whole-selection assisted-preparation sessions."""


class PrepareSessionConflict(PrepareSessionError):
    """Session claim/finalize CAS attempts were exhausted by concurrent writers."""


def prepare_session_path(episode_key: str, source_fingerprint: str, session_id: str) -> str:
    """Canonical GCS path for one whole-selection prepare-session lease."""

    if not isinstance(episode_key, str) or not _EPISODE_KEY.fullmatch(episode_key):
        raise ValueError("episode_key must be a canonical 24-hex-character episode key")
    if not isinstance(source_fingerprint, str) or not _SOURCE_FINGERPRINT.fullmatch(
        source_fingerprint
    ):
        raise ValueError(
            "source_fingerprint must be a canonical 'sha256:<64-hex>' fingerprint"
        )
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("session_id must be a non-empty string")
    digest = source_fingerprint.split(":", 1)[1]
    return f"episodes/{episode_key}/ai/prepare-sessions/{digest}/{session_id}.json"


def _load_prepare_session_with_generation(
    episode_key: str, source_fingerprint: str, session_id: str
) -> tuple[dict | None, int | None]:
    path = prepare_session_path(episode_key, source_fingerprint, session_id)
    blob = get_bucket().blob(path)
    for _ in range(THIRD_ASR_CLAIM_MAX_CAS_ATTEMPTS):
        try:
            blob.reload()
        except NotFound:
            return None, None
        generation = int(blob.generation)
        try:
            content = blob.download_as_text(encoding="utf-8", if_generation_match=generation)
        except PreconditionFailed:
            continue
        payload = json.loads(content)
        if not isinstance(payload, dict):
            raise ValueError("Prepare-session object must be a JSON object")
        return payload, generation
    raise PrepareSessionConflict(
        "Prepare-session object changed repeatedly while reading snapshot"
    )


def load_prepare_session(episode_key: str, source_fingerprint: str, session_id: str) -> dict | None:
    """Read one prepare-session snapshot, or None if it does not exist.

    Thin public read wrapper over the same private generation-pinned
    loader begin_assisted_preparation/advance_prepare_session_item use.
    TASK-076 Task 12: the web layer's canonical progress-reload endpoint
    and per-item preparation only need the current session content, not
    its object generation (every mutation already goes through the
    dedicated CAS-guarded helpers above).
    """

    session, _generation = _load_prepare_session_with_generation(
        episode_key, source_fingerprint, session_id
    )
    return session


def _save_prepare_session(
    episode_key: str,
    source_fingerprint: str,
    session_id: str,
    session: dict,
    *,
    if_generation_match: int | None,
) -> None:
    path = prepare_session_path(episode_key, source_fingerprint, session_id)
    blob = get_bucket().blob(path)
    blob.upload_from_string(
        json.dumps(session, ensure_ascii=False, indent=2) + "\n",
        content_type="application/json",
        if_generation_match=0 if if_generation_match is None else if_generation_match,
    )


def _prepare_session_expiry_seconds_remaining(session: dict, now: str) -> float:
    expires_at = datetime.fromisoformat(session["expires_at"])
    current = datetime.fromisoformat(now)
    return (expires_at - current).total_seconds()


def _prepare_session_matches_request(
    session: dict, *, expected_review_generation_fingerprint: str, selected_ids: list[int]
) -> bool:
    return (
        session.get("policy_version") == ASSISTED_REVIEW_POLICY_VERSION
        and session.get("expected_review_generation_fingerprint")
        == expected_review_generation_fingerprint
        and session.get("selected_ids") == sorted(selected_ids)
    )


def begin_assisted_preparation(
    episode_key: str,
    source_fingerprint: str,
    selected_ids: list[int],
    *,
    expected_review_generation_fingerprint: str,
    session_id: str | None = None,
    api_key: str | None = None,
    pricing_transport: Callable[[str, str], list] | None = None,
    pricing_now: Callable[[], str] | None = None,
    now: Callable[[], str] | None = None,
    required_downstream_reserve_usd: Decimal = Decimal("0"),
) -> dict:
    """Atomically admit one whole-selection Third-ASR preparation batch.

    Prices the entire uncached subset of ``selected_ids`` from one shared
    pricing resolution, then reserves every required attempt in a single
    atomic ``reserve_budget_batch`` call: the whole selection is admitted
    or none of it is, and a cache hit costs nothing. Returns one
    prepare-session lease (``PREPARE_SESSION_TTL_SECONDS`` from creation)
    with a stable per-item budget-attempt token that later per-item
    preparation must present rather than inventing an out-of-session
    paid attempt (see ``advance_prepare_session_item``).

    Calling again with the same ``session_id`` before it expires, for the
    exact same generation/policy/selection, is a safe no-op replay: the
    existing session is reloaded and returned unchanged, with zero new
    reservations or provider calls.
    """

    if not isinstance(selected_ids, list) or not selected_ids:
        raise ValueError("selected_ids must be a non-empty list")
    normalized_ids: list[int] = []
    seen_ids: set[int] = set()
    for candidate in selected_ids:
        if not isinstance(candidate, int) or isinstance(candidate, bool):
            raise ValueError("Every selected id must be an int")
        if candidate in seen_ids:
            raise ValueError(f"Duplicate selected id: {candidate!r}")
        seen_ids.add(candidate)
        normalized_ids.append(candidate)
    if len(normalized_ids) > PREPARE_SESSION_MAX_ITEMS:
        raise ValueError(
            f"selected_ids exceeds the {PREPARE_SESSION_MAX_ITEMS}-item preparation limit"
        )
    sorted_ids = sorted(normalized_ids)

    now_fn = now or now_iso
    preparation_epoch = "new"

    if session_id is not None:
        existing, generation = _load_prepare_session_with_generation(
            episode_key, source_fingerprint, session_id
        )
        if existing is not None:
            current_now = now_fn()
            remaining = _prepare_session_expiry_seconds_remaining(existing, current_now)
            if remaining > 0:
                if not _prepare_session_matches_request(
                    existing,
                    expected_review_generation_fingerprint=expected_review_generation_fingerprint,
                    selected_ids=normalized_ids,
                ):
                    raise PrepareSessionError(
                        f"session_id {session_id!r} already exists for a different "
                        "generation, policy, or selection"
                    )
                # Idempotent replay: identical request, no new spend.
                return existing
            # Delegate to the public reconcile entry point rather than
            # calling _reconcile_prepare_session_object directly: it
            # already re-loads and bounds its retry against a concurrent
            # reconciler's PreconditionFailed, which a bare direct call
            # here would leave uncaught.
            reconcile_expired_prepare_session(
                episode_key, source_fingerprint, session_id, now=now_fn
            )
            # This expired session instance's own (pre-reconcile) object
            # generation uniquely identifies this recovery cycle: stamping
            # it into the fresh attempt_ids below guarantees they never
            # collide with the just-released epoch's attempt_ids, while
            # staying identical across concurrent callers recovering the
            # exact same expired instance.
            preparation_epoch = str(generation)

    record, _ = load_review_record_with_generation(episode_key)
    if record.get("human_review_generation_fingerprint") != expected_review_generation_fingerprint:
        raise ValueError("Review generation changed; reload before preparing assisted evidence")

    items: dict[int, dict] = {}
    for difference_id in sorted_ids:
        item = _item(record, difference_id)
        routing = assisted_routing(item)
        if not routing["eligible"]:
            raise ValueError(
                f"Item {difference_id} is not an eligible assisted-review candidate: "
                f"{routing['reason_codes']}"
            )
        items[difference_id] = item

    windows = {
        difference_id: review_clip_window(item)
        for difference_id, item in items.items()
    }
    cache_keys = {
        difference_id: third_asr_cache_key(
            input_fingerprint=source_fingerprint, item_id=difference_id, window=window
        )
        for difference_id, window in windows.items()
    }

    uncached_ids = [
        difference_id
        for difference_id in sorted_ids
        if not (
            isinstance((existing_evidence := items[difference_id].get("third_asr")), dict)
            and existing_evidence.get("cache_key") == cache_keys[difference_id]
            and existing_evidence.get("text")
        )
    ]

    resolved_session_id = session_id or uuid4().hex

    def _attempt_id(difference_id: int) -> str:
        return f"prepare-{resolved_session_id}-{preparation_epoch}-{difference_id}"

    reserved_usd_by_id: dict[int, Decimal] = {}
    if uncached_ids:
        # The whole-selection hold itself is paid Third-ASR admission, so it
        # must use the same TASK-109 compatibility gate as direct execution.
        ensure_fresh_third_asr_budget_identity(
            episode_key, source_fingerprint, record.get("input_fingerprint")
        )
        resolved_api_key = api_key or os.getenv("PODCAST_REVIEW_ASR_API_KEY")
        if not resolved_api_key:
            raise RuntimeError("Missing PODCAST_REVIEW_ASR_API_KEY for third-ASR review")
        pricing = resolve_audio_model_pricing(
            THIRD_ASR_MODEL,
            api_key=resolved_api_key,
            transport=pricing_transport,
            now=pricing_now,
        )
        for difference_id in uncached_ids:
            reserved_usd_by_id[difference_id] = derive_audio_reservation_usd(
                max_billable_seconds=Decimal(str(windows[difference_id]["duration"])),
                usd_per_second=pricing.usd_per_second,
            )

        reservations = [
            {
                "attempt_id": _attempt_id(difference_id),
                "stage": STAGE_THIRD_ASR,
                "reserved_usd": reserved_usd_by_id[difference_id],
                "third_asr": True,
            }
            for difference_id in uncached_ids
        ]
        reserve_budget_batch(
            episode_key,
            source_fingerprint,
            reservations,
            required_downstream_reserve_usd=required_downstream_reserve_usd,
        )

    created_at = now_fn()
    expires_at = (
        datetime.fromisoformat(created_at) + timedelta(seconds=PREPARE_SESSION_TTL_SECONDS)
    ).isoformat()

    session_items = {}
    for difference_id in sorted_ids:
        cached = difference_id not in uncached_ids
        session_items[str(difference_id)] = {
            "state": "prepared" if cached else "queued",
            "cache_key": cache_keys[difference_id],
            "cached": cached,
            "attempt_id": (None if cached else _attempt_id(difference_id)),
            "reserved_usd": (
                None if cached else str(reserved_usd_by_id[difference_id])
            ),
        }

    session = {
        "session_id": resolved_session_id,
        "policy_version": ASSISTED_REVIEW_POLICY_VERSION,
        "expected_review_generation_fingerprint": expected_review_generation_fingerprint,
        "selected_ids": sorted_ids,
        "created_at": created_at,
        "expires_at": expires_at,
        "items": session_items,
    }

    # Bounded CAS on the final write itself, matching this file's other
    # generation-CAS loops: a concurrent writer (another caller recovering
    # the same expired session, or creating it for the first time) can
    # change the object between the lookahead read above and this write,
    # and that must retry rather than crash the whole preparation.
    for _ in range(THIRD_ASR_CLAIM_MAX_CAS_ATTEMPTS):
        write_generation = None
        if session_id is not None:
            _, write_generation = _load_prepare_session_with_generation(
                episode_key, source_fingerprint, session_id
            )
        try:
            _save_prepare_session(
                episode_key,
                source_fingerprint,
                resolved_session_id,
                session,
                if_generation_match=write_generation,
            )
        except PreconditionFailed:
            continue
        return session

    raise PrepareSessionConflict(
        "Prepare-session object changed repeatedly while creating it"
    )


def _reconcile_prepare_session_object(
    episode_key: str, source_fingerprint: str, session: dict, *, generation: int
) -> dict:
    """Release only provably pre-send (still-``queued``) reservations.

    An item already ``in_flight``/``retrying``/``prepared``/``failed`` is
    left untouched: its budget attempt was already settled, marked
    uncertain, or never made in the first place (cached), by whatever
    touched it -- this never optimistically releases money that may
    already be spent.
    """

    updated_items = {}
    changed = False
    for key, item in session.get("items", {}).items():
        if item.get("state") == "queued" and item.get("attempt_id"):
            release_budget_attempt_pre_send(
                episode_key, source_fingerprint, item["attempt_id"]
            )
            item = {**item, "state": "failed", "reason": "prepare_session_expired"}
            changed = True
        updated_items[key] = item

    if not changed:
        return session

    reconciled = {**session, "items": updated_items}
    _save_prepare_session(
        episode_key,
        source_fingerprint,
        session["session_id"],
        reconciled,
        if_generation_match=generation,
    )
    return reconciled


def reconcile_expired_prepare_session(
    episode_key: str, source_fingerprint: str, session_id: str, *, now: Callable[[], str] | None = None
) -> dict:
    """Idempotently release an expired session's unconsumed reservations.

    A non-expired session is returned unchanged. Bounded to
    ``THIRD_ASR_CLAIM_MAX_CAS_ATTEMPTS`` write attempts on an unrelated
    concurrent writer, matching the repository's existing bounded-retry
    convention.
    """

    now_fn = now or now_iso
    for _ in range(THIRD_ASR_CLAIM_MAX_CAS_ATTEMPTS):
        existing, generation = _load_prepare_session_with_generation(
            episode_key, source_fingerprint, session_id
        )
        if existing is None:
            raise PrepareSessionError(f"No prepare-session {session_id!r} found")
        remaining = _prepare_session_expiry_seconds_remaining(existing, now_fn())
        if remaining > 0:
            return existing
        try:
            return _reconcile_prepare_session_object(
                episode_key, source_fingerprint, existing, generation=generation
            )
        except PreconditionFailed:
            continue
    raise PrepareSessionConflict(
        "Prepare-session object changed repeatedly while reconciling expiry"
    )


def advance_prepare_session_item(
    episode_key: str,
    source_fingerprint: str,
    session_id: str,
    difference_id: int,
    *,
    state: str,
    attempt_id: str,
) -> dict:
    """Move one session item's lifecycle state forward, token-checked.

    ``attempt_id`` must exactly match the token ``begin_assisted_preparation``
    pre-authorized for this item: a caller cannot invent an out-of-session
    paid attempt, and a stale/superseded caller cannot silently move a
    session item it no longer owns. Bounded to
    ``THIRD_ASR_CLAIM_MAX_CAS_ATTEMPTS`` CAS attempts.
    """

    if state not in PREPARE_SESSION_ITEM_STATES:
        raise ValueError(f"Unknown prepare-session item state: {state!r}")

    for _ in range(THIRD_ASR_CLAIM_MAX_CAS_ATTEMPTS):
        session, generation = _load_prepare_session_with_generation(
            episode_key, source_fingerprint, session_id
        )
        if session is None:
            raise PrepareSessionError(f"No prepare-session {session_id!r} found")
        key = str(difference_id)
        item = session.get("items", {}).get(key)
        if item is None:
            raise ValueError(f"Item {difference_id} is not part of session {session_id!r}")
        if item.get("attempt_id") != attempt_id:
            raise PrepareSessionError(
                f"Item {difference_id} token does not match its pre-authorized attempt"
            )
        current_state = item.get("state")
        allowed = _PREPARE_SESSION_ITEM_TRANSITIONS.get(current_state, set())
        if state not in allowed:
            raise PrepareSessionError(
                f"Item {difference_id} cannot move from {current_state!r} to {state!r}"
            )

        updated_items = {**session["items"], key: {**item, "state": state}}
        updated_session = {**session, "items": updated_items}
        try:
            _save_prepare_session(
                episode_key,
                source_fingerprint,
                session_id,
                updated_session,
                if_generation_match=generation,
            )
        except PreconditionFailed:
            continue
        return updated_session

    raise PrepareSessionConflict(
        "Prepare-session object changed repeatedly while advancing an item"
    )


def _item(record: dict, difference_id: int) -> dict:
    for item in pending_review_items(record):
        if item.get("id") == difference_id:
            return item
    raise ValueError(f"Review item {difference_id} is not pending")


def _decision_text(item: dict, source: str, text: str | None) -> tuple[str, str]:
    """Validate explicit human text without pretending a third source is truth."""

    scope = item.get("focus", {}).get("scope", "full")
    expected = item.get("focus", {}).get(f"{source}_text") if scope == "partial" else item.get(f"{source}_text")
    if source in {"apple", "whisper"}:
        if not isinstance(expected, str):
            raise ValueError(f"Review item has no {source} text for the chosen scope")
        if text is not None and text.strip() != expected:
            raise ValueError(f"Use {source.title()} must preserve the supplied source text")
        return expected, scope
    if source == "third":
        evidence = item.get("third_asr")
        clip_text = evidence.get("text") if isinstance(evidence, dict) else None
        if not isinstance(clip_text, str) or not clip_text.strip():
            raise ValueError("Run third ASR before choosing its evidence")
        # TASK-123: the clip transcript covers the whole 12-15 s audio window;
        # only the anchored words that correspond to this card may replace
        # the card's span, or the surrounding sentences would be duplicated.
        window = anchored_third_asr_window(item)
        if window is None:
            raise ValueError(
                "Third ASR could not be aligned to this card; use Custom/Edit"
            )
        expected = window["text"]
        if text is not None and text != expected:
            raise ValueError("Use Third must preserve the persisted third-ASR evidence text")
        return expected, scope
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Custom/Edit requires non-empty replacement text")
    return text.strip(), scope


def _case_only_representation(
    source_text: str,
    representation_text: str | None,
) -> str | None:
    """Allow human-confirmed representation changes without changing speech truth."""

    if representation_text is None:
        return None
    if not isinstance(representation_text, str):
        raise ValueError("Representation/casing must be text")
    if representation_text == source_text:
        return None
    if (
        len(representation_text) != len(source_text)
        or representation_text.casefold() != source_text.casefold()
    ):
        raise ValueError(
            "Representation/casing may change case only; "
            "words, spacing, and punctuation must match the chosen source"
        )
    return representation_text



def _custom_edit_expansion_count(value: object, *, side: str) -> int:
    """Validate one reviewer-requested adjacent-word expansion count."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Custom/Edit {side} expansion must be whole words")
    if value < 0:
        raise ValueError(f"Custom/Edit {side} expansion must be whole words")
    if value > MAX_HUMAN_EDIT_EXPANSION_WORDS:
        raise ValueError(
            "Custom/Edit may expand by at most "
            f"{MAX_HUMAN_EDIT_EXPANSION_WORDS} words on either side"
        )
    return value


def _custom_edit_geometry(
    item: dict,
    *,
    expand_left_words: int = 0,
    expand_right_words: int = 0,
) -> dict:
    """Return the exact canonical-source slice a custom edit may replace."""

    left = _custom_edit_expansion_count(
        expand_left_words,
        side="left",
    )
    right = _custom_edit_expansion_count(
        expand_right_words,
        side="right",
    )

    geometry = item.get("merge_preview")
    if not isinstance(geometry, dict):
        raise ValueError("review preview geometry is unavailable")

    base_source = geometry.get("base_source")
    context_text = geometry.get("context_text")
    span_start = geometry.get("span_start")
    span_end = geometry.get("span_end")
    span_text = geometry.get("span_text")

    if (
        base_source not in {"apple", "whisper"}
        or not isinstance(context_text, str)
        or not isinstance(span_start, int)
        or not isinstance(span_end, int)
        or not isinstance(span_text, str)
        or span_start < 0
        or span_end < span_start
        or span_end > len(context_text)
        or context_text[span_start:span_end] != span_text
    ):
        raise ValueError(
            "review preview geometry no longer matches the canonical source span"
        )

    focus = item.get("focus")
    scope = (
        focus.get("scope", "full")
        if isinstance(focus, dict)
        else "full"
    )

    if left == 0 and right == 0 and scope == "partial":
        old_focus = focus.get(f"{base_source}_text")
        if not isinstance(old_focus, str) or not old_focus:
            raise ValueError("review preview focus geometry is unavailable")

        folded_span = span_text.casefold()
        folded_focus = old_focus.casefold()
        if folded_span.count(folded_focus) != 1:
            raise ValueError(
                "review preview focus does not map uniquely inside the base span"
            )

        focus_start = folded_span.index(folded_focus)
        focus_end = focus_start + len(old_focus)
        return {
            "base_source": base_source,
            "context_text": context_text,
            "span_start": span_start + focus_start,
            "span_end": span_start + focus_end,
            "replaced_text": context_text[
                span_start + focus_start : span_start + focus_end
            ],
            "expand_left_words": 0,
            "expand_right_words": 0,
        }

    if left == 0 and right == 0:
        return {
            "base_source": base_source,
            "context_text": context_text,
            "span_start": span_start,
            "span_end": span_end,
            "replaced_text": span_text,
            "expand_left_words": 0,
            "expand_right_words": 0,
        }

    if span_start == span_end:
        raise ValueError(
            "Custom/Edit range expansion requires a non-empty source span"
        )

    context_tokens = tokenize(context_text)
    start_matches = [
        index
        for index, token in enumerate(context_tokens)
        if token.start == span_start
    ]
    end_matches = [
        index + 1
        for index, token in enumerate(context_tokens)
        if token.end == span_end
    ]
    if len(start_matches) != 1 or len(end_matches) != 1:
        raise ValueError(
            "review preview span does not align to canonical source words"
        )

    start_word = start_matches[0]
    end_word = end_matches[0]
    expanded_start_word = start_word - left
    expanded_end_word = end_word + right
    if (
        expanded_start_word < 0
        or expanded_end_word > len(context_tokens)
        or expanded_start_word >= expanded_end_word
    ):
        raise ValueError(
            "Custom/Edit expansion exceeds the available canonical source context"
        )

    expanded_start = context_tokens[expanded_start_word].start
    expanded_end = context_tokens[expanded_end_word - 1].end
    return {
        "base_source": base_source,
        "context_text": context_text,
        "span_start": expanded_start,
        "span_end": expanded_end,
        "replaced_text": context_text[expanded_start:expanded_end],
        "expand_left_words": left,
        "expand_right_words": right,
    }


def preview_custom_edit_details(
    item: dict,
    text: str,
    *,
    expand_left_words: int = 0,
    expand_right_words: int = 0,
) -> dict:
    """Render and describe the exact bounded source edit before confirmation."""

    if not isinstance(text, str) or not text.strip():
        raise ValueError("Custom/Edit preview requires non-empty replacement text")

    geometry = _custom_edit_geometry(
        item,
        expand_left_words=expand_left_words,
        expand_right_words=expand_right_words,
    )
    merged_context = render_review_merge_preview(
        geometry["context_text"],
        span_start=geometry["span_start"],
        span_end=geometry["span_end"],
        replacement=text.strip(),
    )
    return {
        "merged_context": merged_context,
        "replaced_text": geometry["replaced_text"],
        "edit_base_source": geometry["base_source"],
        "expand_left_words": geometry["expand_left_words"],
        "expand_right_words": geometry["expand_right_words"],
    }


def preview_custom_edit(
    item: dict,
    text: str,
    *,
    expand_left_words: int = 0,
    expand_right_words: int = 0,
) -> str:
    """Render the exact bounded transcript context a Custom/Edit would create."""

    return preview_custom_edit_details(
        item,
        text,
        expand_left_words=expand_left_words,
        expand_right_words=expand_right_words,
    )["merged_context"]

def _routing_snapshot(item: dict) -> dict:
    """Bounded TASK-076 routing-provenance audit snapshot for one decision.

    Every persisted human decision -- Detailed Review, strict low-risk
    Batch, and Assisted -- carries the exact routing-policy version and
    exclusion reason codes recomputed at decision time (never a stored/
    cached routing result), so later corpus analysis can measure whether
    exclusion rules such as ``protected_proper_name`` or ``span_over_limit``
    are over-conservative. This is provenance only: it never chooses a
    source, alters transcript authority, or changes decision eligibility.
    """

    routing = assisted_routing(item)
    return {
        "policy_version": routing["policy_version"],
        "reason_codes": list(routing["reason_codes"]),
    }


def record_human_decision(
    episode_key: str,
    difference_id: int,
    *,
    source: str,
    text: str | None = None,
    representation_text: str | None = None,
    note: str | None = None,
    expand_left_words: int = 0,
    expand_right_words: int = 0,
) -> dict:
    """Move one card into an append-only human decision audit trail."""

    if source not in HUMAN_SOURCES:
        raise ValueError("Unsupported human review source")
    record = load_review_record(episode_key)
    item = _item(record, difference_id)
    chosen_text, scope = _decision_text(item, source, text)

    if representation_text is not None and source not in {"apple", "whisper"}:
        raise ValueError(
            "Representation/casing is available only for Apple or Whisper source choices"
        )
    representation = _case_only_representation(
        chosen_text,
        representation_text,
    )

    left = _custom_edit_expansion_count(
        expand_left_words,
        side="left",
    )
    right = _custom_edit_expansion_count(
        expand_right_words,
        side="right",
    )
    if source != "custom" and (left or right):
        raise ValueError(
            "Edit-range expansion is available only for Custom/Edit"
        )

    expansion_details = None
    if source == "custom" and (left or right):
        expansion_details = preview_custom_edit_details(
            item,
            chosen_text,
            expand_left_words=left,
            expand_right_words=right,
        )
        scope = "expanded"

    decision = {
        "id": difference_id,
        "chosen_source": source,
        "chosen_text": chosen_text,
        "scope": scope,
        "reviewed_by": "human",
        "reviewed_at": now_iso(),
        "note": note.strip() if isinstance(note, str) and note.strip() else None,
        "review_item": item,
        "routing_provenance": _routing_snapshot(item),
        "review_tier": review_tier_snapshot(item),
    }
    if representation is not None:
        decision["representation_text"] = representation
        decision["representation_kind"] = "case_only"
    if scope == "partial":
        decision["focus_apple_text"] = item["focus"]["apple_text"]
        decision["focus_whisper_text"] = item["focus"]["whisper_text"]
    elif scope == "expanded" and expansion_details is not None:
        decision.update(
            {
                "edit_base_source": expansion_details["edit_base_source"],
                "expand_left_words": expansion_details["expand_left_words"],
                "expand_right_words": expansion_details["expand_right_words"],
                "expected_base_text": expansion_details["replaced_text"],
            }
        )

    if item.get("third_asr") is not None:
        assisted = derive_assisted_state(item)
        if assisted["routing"]["eligible"] and assisted.get("state") != "audio_pending":
            decision["decision_support"] = _assisted_decision_support(item, assisted, source)

    decisions = [
        existing for existing in record.get("human_decisions", [])
        if isinstance(existing, dict) and existing.get("id") != difference_id
    ]
    decisions.append(decision)
    record["human_decisions"] = decisions
    record["human_review"] = [
        existing for existing in record.get("human_review", [])
        if isinstance(existing, dict) and existing.get("id") != difference_id
    ]
    record["human_review_queue_fingerprint"] = review_queue_fingerprint(
        record["human_review"]
    )
    record["human_review_updated_at"] = now_iso()
    save_review_record(episode_key, record)
    return decision


def _validated_batch_requests(decisions: object) -> list[dict]:
    """Validate the bounded source-choice request shape before storage access."""

    if not isinstance(decisions, list) or not decisions:
        raise ValueError("Batch decisions must be a non-empty list")

    validated: list[dict] = []
    seen_ids: set[int] = set()
    for request in decisions:
        if not isinstance(request, dict) or set(request) != {"id", "source"}:
            raise ValueError("Each batch decision must contain only id and source")

        difference_id = request.get("id")
        source = request.get("source")
        if (
            isinstance(difference_id, bool)
            or not isinstance(difference_id, int)
            or source not in {"apple", "whisper"}
        ):
            raise ValueError(
                "Batch decisions require an integer id and Apple/Whisper source"
            )
        if difference_id in seen_ids:
            raise ValueError("Batch decisions may not contain duplicate review IDs")

        seen_ids.add(difference_id)
        validated.append({"id": difference_id, "source": source})
    return validated


def _apply_batch_source_decision(
    record: dict,
    *,
    difference_id: int,
    source: str,
) -> None:
    """Apply one already-bounded source choice to an in-memory review record."""

    item = _item(record, difference_id)
    recommendation = batch_recommendation(item)
    chosen_text, scope = _decision_text(item, source, None)

    if (
        not isinstance(recommendation, dict)
        or recommendation.get("id") != difference_id
        or recommendation.get("source") != source
        or recommendation.get("text") != chosen_text
        or recommendation.get("reason")
        != "high_confidence_exact_source_low_risk"
    ):
        raise ValueError(
            f"Review item {difference_id} is not eligible for the requested batch source"
        )

    decision = {
        "id": difference_id,
        "chosen_source": source,
        "chosen_text": chosen_text,
        "scope": scope,
        "reviewed_by": "human",
        "reviewed_at": now_iso(),
        "note": None,
        "review_item": item,
        "routing_provenance": _routing_snapshot(item),
        "review_tier": review_tier_snapshot(item),
    }
    if scope == "partial":
        decision["focus_apple_text"] = item["focus"]["apple_text"]
        decision["focus_whisper_text"] = item["focus"]["whisper_text"]

    existing_decisions = [
        existing
        for existing in record.get("human_decisions", [])
        if isinstance(existing, dict) and existing.get("id") != difference_id
    ]
    existing_decisions.append(decision)
    record["human_decisions"] = existing_decisions
    record["human_review"] = [
        existing
        for existing in record.get("human_review", [])
        if isinstance(existing, dict) and existing.get("id") != difference_id
    ]
    record["human_review_queue_fingerprint"] = review_queue_fingerprint(
        record["human_review"]
    )
    record["human_review_updated_at"] = now_iso()


def _apply_human_decision_batch(record: dict, decisions: list[dict]) -> dict:
    """Validate and apply a complete batch only to a private record copy."""

    working = copy.deepcopy(record)
    for request in decisions:
        _apply_batch_source_decision(
            working,
            difference_id=request["id"],
            source=request["source"],
        )
    return working


def record_human_decision_batch(
    episode_key: str,
    decisions: object,
    *,
    expected_generation_fingerprint: str,
) -> dict:
    """Atomically persist eligible source choices as ordinary human decisions."""

    validated = _validated_batch_requests(decisions)
    if (
        not isinstance(expected_generation_fingerprint, str)
        or not expected_generation_fingerprint
    ):
        raise ValueError("Expected review generation is required")

    for _ in range(3):
        record, object_generation = load_review_record_with_generation(episode_key)
        if (
            record.get("human_review_generation_fingerprint")
            != expected_generation_fingerprint
        ):
            raise ValueError("Review generation changed; reload before batch approval")

        updated = _apply_human_decision_batch(record, validated)
        try:
            save_review_record(
                episode_key,
                updated,
                if_generation_match=object_generation,
            )
            return updated
        except PreconditionFailed:
            continue

    raise RuntimeError(
        "Review record changed repeatedly while recording batch decisions"
    )


# TASK-127: tier-A confirmation.
#
# The reviewer sends only card IDs. Python re-derives every card's tier from
# the current record and records the tier-A source proposal as an ordinary
# audited human source decision; any card that is no longer tier A, or whose
# proposal changed, rejects the whole confirmation.

TIER_A_MAX_ITEMS = 100


def _validated_tier_a_ids(ids: object) -> list[int]:
    if not isinstance(ids, list) or not ids:
        raise ValueError("Tier-A confirmation needs a non-empty list of review IDs")
    if len(ids) > TIER_A_MAX_ITEMS:
        raise ValueError(f"Tier-A confirmation is limited to {TIER_A_MAX_ITEMS} cards")
    validated: list[int] = []
    for difference_id in ids:
        if isinstance(difference_id, bool) or not isinstance(difference_id, int):
            raise ValueError("Tier-A confirmation IDs must be integers")
        if difference_id in validated:
            raise ValueError("Tier-A confirmation may not repeat a review ID")
        validated.append(difference_id)
    return validated


def _apply_tier_a_decision(record: dict, difference_id: int) -> None:
    item = _item(record, difference_id)
    tier = derive_review_tier(item)
    if tier["tier"] != "A" or tier["source"] not in {"apple", "whisper"}:
        raise ValueError(f"Review item {difference_id} is no longer in tier A")
    chosen_text, scope = _decision_text(item, tier["source"], None)
    if chosen_text != tier["text"]:
        raise ValueError(f"Review item {difference_id} tier-A proposal changed")

    decision = {
        "id": difference_id,
        "chosen_source": tier["source"],
        "chosen_text": chosen_text,
        "scope": scope,
        "reviewed_by": "human",
        "reviewed_at": now_iso(),
        "note": None,
        "review_item": item,
        "routing_provenance": _routing_snapshot(item),
        "review_tier": review_tier_snapshot(item),
    }
    if scope == "partial":
        decision["focus_apple_text"] = item["focus"]["apple_text"]
        decision["focus_whisper_text"] = item["focus"]["whisper_text"]

    record["human_decisions"] = [
        existing
        for existing in record.get("human_decisions", [])
        if isinstance(existing, dict) and existing.get("id") != difference_id
    ] + [decision]
    record["human_review"] = [
        existing
        for existing in record.get("human_review", [])
        if isinstance(existing, dict) and existing.get("id") != difference_id
    ]
    record["human_review_queue_fingerprint"] = review_queue_fingerprint(
        record["human_review"]
    )
    record["human_review_updated_at"] = now_iso()


def record_tier_a_decision_batch(
    episode_key: str,
    ids: object,
    *,
    expected_generation_fingerprint: str,
) -> dict:
    """Atomically record the human-confirmed tier-A proposals of the given cards."""

    validated = _validated_tier_a_ids(ids)
    if (
        not isinstance(expected_generation_fingerprint, str)
        or not expected_generation_fingerprint
    ):
        raise ValueError("Expected review generation is required")

    for _ in range(3):
        record, object_generation = load_review_record_with_generation(episode_key)
        if (
            record.get("human_review_generation_fingerprint")
            != expected_generation_fingerprint
        ):
            raise ValueError("Review generation changed; reload before confirming tier A")
        working = copy.deepcopy(record)
        for difference_id in validated:
            _apply_tier_a_decision(working, difference_id)
        try:
            save_review_record(
                episode_key,
                working,
                if_generation_match=object_generation,
            )
            return working
        except PreconditionFailed:
            continue

    raise RuntimeError(
        "Review record changed repeatedly while recording tier-A decisions"
    )


# TASK-133: decisions on cards the materiality filter grouped (control
# sample, "click one", proposals, and the settled cards the reviewer accepts
# with one explicit click). Every member is an ordinary audited human source
# decision that also snapshots the filter's grouping and reading.

MATERIALITY_DECISION_MAX_ITEMS = 200
MATERIALITY_CARD_MAX_SECONDS = 3600
MATERIALITY_NOTE_MAX_CHARS = 2000
MATERIALITY_REVIEW_LOG_MAX_ENTRIES = 200
_MATERIALITY_DECIDABLE_GROUPS = frozenset({"settled", "sample", "click_one", "proposal"})


def _validated_materiality_requests(decisions: object) -> list[dict]:
    if not isinstance(decisions, list) or not decisions:
        raise ValueError("Materiality decisions need a non-empty list")
    if len(decisions) > MATERIALITY_DECISION_MAX_ITEMS:
        raise ValueError(f"Materiality decisions are limited to {MATERIALITY_DECISION_MAX_ITEMS} cards")
    validated: list[dict] = []
    seen: set[int] = set()
    for request in decisions:
        if not isinstance(request, dict) or not {"id", "source"} <= set(request) <= {"id", "source", "seconds"}:
            raise ValueError("Each materiality decision needs an id and a source (and optionally seconds)")
        difference_id, source = request["id"], request["source"]
        seconds = request.get("seconds")
        if seconds is not None and (
            isinstance(seconds, bool) or not isinstance(seconds, (int, float))
            or not 0 <= seconds <= MATERIALITY_CARD_MAX_SECONDS
        ):
            raise ValueError("Materiality decision seconds must be a number of seconds")
        if isinstance(difference_id, bool) or not isinstance(difference_id, int):
            raise ValueError("Materiality decision IDs must be integers")
        if difference_id in seen:
            raise ValueError("Materiality decisions may not repeat a review ID")
        if source not in {"apple", "whisper"}:
            raise ValueError("Materiality decisions choose Apple or Whisper")
        seen.add(difference_id)
        validated.append({"id": difference_id, "source": source, "seconds": seconds})
    return validated


def _validated_review_session(session: object) -> dict:
    """Bounded audit context sent with a save: page start, list opened, a note."""

    if session is None:
        return {}
    if not isinstance(session, dict) or not set(session) <= {"started_at", "settled_list_opened", "note"}:
        raise ValueError("Review session must hold only started_at, settled_list_opened and note")
    started_at = session.get("started_at")
    opened = session.get("settled_list_opened")
    note = session.get("note")
    if started_at is not None and (not isinstance(started_at, str) or len(started_at) > 40):
        raise ValueError("Review session started_at must be a short timestamp")
    if opened is not None and not isinstance(opened, bool):
        raise ValueError("Review session settled_list_opened must be true or false")
    if note is not None and (not isinstance(note, str) or len(note) > MATERIALITY_NOTE_MAX_CHARS):
        raise ValueError(f"Review notes are limited to {MATERIALITY_NOTE_MAX_CHARS} characters")
    return {
        "started_at": started_at,
        "settled_list_opened": opened,
        "note": note.strip() if isinstance(note, str) and note.strip() else None,
    }


def _materiality_review_log_entry(decisions: list[dict], session: dict) -> dict:
    """One audit entry per save: what was decided, how often the filter was overruled, time, notes."""

    by_group: dict[str, int] = {}
    overruled: dict[str, int] = {}
    seconds = 0.0
    for decision in decisions:
        snapshot = decision["materiality"]
        group = snapshot["group"]
        by_group[group] = by_group.get(group, 0) + 1
        if snapshot["agrees_with_filter"] is False:
            overruled[group] = overruled.get(group, 0) + 1
        seconds += snapshot.get("seconds") or 0
    return {
        "at": now_iso(),
        "started_at": session.get("started_at"),
        "decisions": len(decisions),
        "by_group": by_group,
        "overruled_filter": overruled,
        "seconds": round(seconds),
        "settled_list_opened": session.get("settled_list_opened"),
        "note": session.get("note"),
    }


def _apply_materiality_decision(record: dict, request: dict, groups: dict) -> None:
    difference_id, source = request["id"], request["source"]
    item = _item(record, difference_id)
    group = groups.get(difference_id)
    if not isinstance(group, dict) or group.get("group") not in _MATERIALITY_DECIDABLE_GROUPS:
        raise ValueError(f"Review item {difference_id} is not in a materiality group")
    if group["group"] == "settled" and source != group.get("source"):
        raise ValueError(f"Review item {difference_id} settled reading changed; reload")
    chosen_text, scope = _decision_text(item, source, None)
    decision = {
        "id": difference_id,
        "chosen_source": source,
        "chosen_text": chosen_text,
        "scope": scope,
        "reviewed_by": "human",
        "reviewed_at": now_iso(),
        "note": None,
        "review_item": item,
        "routing_provenance": _routing_snapshot(item),
        "review_tier": review_tier_snapshot(item),
        "materiality": {
            "group": group["group"],
            "filter_source": group.get("source"),
            "step": group.get("step"),
            "reason": group.get("reason"),
            "agrees_with_filter": group.get("source") == source if group.get("source") else None,
            "seconds": request.get("seconds"),
        },
    }
    if scope == "partial":
        decision["focus_apple_text"] = item["focus"]["apple_text"]
        decision["focus_whisper_text"] = item["focus"]["whisper_text"]
    record["human_decisions"] = [
        existing
        for existing in record.get("human_decisions", [])
        if isinstance(existing, dict) and existing.get("id") != difference_id
    ] + [decision]
    record["human_review"] = [
        existing
        for existing in record.get("human_review", [])
        if isinstance(existing, dict) and existing.get("id") != difference_id
    ]
    return decision


def record_materiality_decision_batch(
    episode_key: str,
    decisions: object,
    *,
    expected_generation_fingerprint: str,
    session: object = None,
) -> dict:
    """Atomically record explicit human decisions on materiality-grouped cards.

    Python re-derives every card's group from its current evidence; a settled
    card is accepted only with the filter's own reading, and any card outside
    the groups (or a stale generation) rejects the whole request.
    """

    from .materiality_queue import materiality_groups

    validated = _validated_materiality_requests(decisions)
    review_session = _validated_review_session(session)
    if not isinstance(expected_generation_fingerprint, str) or not expected_generation_fingerprint:
        raise ValueError("Expected review generation is required")
    for _ in range(3):
        record, object_generation = load_review_record_with_generation(episode_key)
        if record.get("human_review_generation_fingerprint") != expected_generation_fingerprint:
            raise ValueError("Review generation changed; reload before confirming")
        working = copy.deepcopy(record)
        groups = materiality_groups(record, episode_key, pending_review_items(record))
        recorded = [_apply_materiality_decision(working, request, groups) for request in validated]
        log = [entry for entry in working.get("materiality_review_log", []) if isinstance(entry, dict)]
        log.append(_materiality_review_log_entry(recorded, review_session))
        working["materiality_review_log"] = log[-MATERIALITY_REVIEW_LOG_MAX_ENTRIES:]
        working["human_review_queue_fingerprint"] = review_queue_fingerprint(working["human_review"])
        working["human_review_updated_at"] = now_iso()
        try:
            save_review_record(episode_key, working, if_generation_match=object_generation)
            return working
        except PreconditionFailed:
            continue
    raise RuntimeError("Review record changed repeatedly while recording materiality decisions")


# TASK-076 Task 11: Assisted decision batch.
#
# Separate from the strict low-risk record_human_decision_batch above --
# this path does not weaken it. An Assisted decision may target any
# admitted (assisted_routing-eligible) card, not only cards whose
# high-confidence triage exactly matches the chosen source, and it never
# requires the human's choice to agree with the derived machine
# recommendation: an admitted card in any assisted state (including
# evidence_conflict, ambiguous_audio, or audio_pending) may be manually
# decided from the compact evidence table. See docs/superpowers/specs/
# 2026-09-16-human-review-evidence-assisted-adjudication-design.md,
# "Atomic assisted persistence" and "Decision and routing audit".

_ASSISTED_MACHINE_STATES = frozenset({"machine_supported_apple", "machine_supported_whisper"})


def _bounded_triage_support(raw_triage: object) -> dict | None:
    """Bounded triage identity for decision_support -- status/recommendation/
    confidence only, never the raw advisory text/source fields."""

    if not isinstance(raw_triage, dict):
        return None
    return {
        "status": raw_triage.get("status"),
        "recommendation": raw_triage.get("recommendation"),
        "confidence": raw_triage.get("confidence"),
    }


def _bounded_third_asr_support(evidence: object) -> dict | None:
    """Bounded Third-ASR identity for decision_support.

    Never stores the raw provider response or the raw transcript text --
    only cache/model/window identity plus a hash of the text. Any
    prepare-session/budget-attempt identity a future evidence-acquisition
    path stamps onto the persisted evidence is passed through only when
    actually present, never fabricated.
    """

    if not isinstance(evidence, dict) or not isinstance(evidence.get("text"), str):
        return None
    support = {
        "cache_key": evidence.get("cache_key"),
        "model": evidence.get("model"),
        "window": evidence.get("window"),
        "text_hash": hashlib.sha256(evidence["text"].encode("utf-8")).hexdigest(),
    }
    for optional_field in ("prepare_session_id", "budget_attempt_id"):
        if optional_field in evidence:
            support[optional_field] = evidence[optional_field]
    return support


def _compiler_suggestion_source(suggestion: object) -> str | None:
    """Bounded compiler-suggestion identity for decision_support -- source only."""

    if not isinstance(suggestion, dict):
        return None
    source = suggestion.get("source")
    return source if isinstance(source, str) else None


def _assisted_decision_support(item: dict, assisted: dict, source: str) -> dict:
    """Build the same bounded Assisted audit for either decision route."""

    machine_state = assisted.get("state")
    recommended_source = assisted.get("recommendation")
    if machine_state in _ASSISTED_MACHINE_STATES:
        relation = "confirmed_machine" if source == recommended_source else "human_override"
    else:
        relation = "manual_from_unresolved"

    return {
        "mode": "assisted",
        "policy_version": ASSISTED_REVIEW_POLICY_VERSION,
        "machine_state": machine_state,
        "recommended_source": recommended_source,
        "relation": relation,
        "routing": _routing_snapshot(item),
        "match_evidence": assisted.get("matches"),
        "margin": assisted.get("margin"),
        "triage": _bounded_triage_support(item.get("triage")),
        "third_asr": _bounded_third_asr_support(item.get("third_asr")),
        "compiler_suggestion_source": _compiler_suggestion_source(item.get("suggestion")),
    }


def _apply_assisted_batch_decision(
    record: dict,
    *,
    difference_id: int,
    source: str,
) -> None:
    """Apply one bounded Assisted Apple/Whisper choice to an in-memory review record."""

    item = _item(record, difference_id)
    assisted = derive_assisted_state(item)
    routing = assisted["routing"]
    if not routing["eligible"]:
        raise ValueError(
            f"Review item {difference_id} is not an assisted candidate"
        )
    if routing["policy_version"] != ASSISTED_REVIEW_POLICY_VERSION:
        raise ValueError(
            f"Review item {difference_id} routing policy version is stale"
        )

    chosen_text, scope = _decision_text(item, source, None)

    routing_snapshot = _routing_snapshot(item)
    decision_support = _assisted_decision_support(item, assisted, source)

    decision = {
        "id": difference_id,
        "chosen_source": source,
        "chosen_text": chosen_text,
        "scope": scope,
        "reviewed_by": "human",
        "reviewed_at": now_iso(),
        "note": None,
        "review_item": item,
        "routing_provenance": routing_snapshot,
        "review_tier": review_tier_snapshot(item),
        "decision_support": decision_support,
    }
    if scope == "partial":
        decision["focus_apple_text"] = item["focus"]["apple_text"]
        decision["focus_whisper_text"] = item["focus"]["whisper_text"]

    existing_decisions = [
        existing
        for existing in record.get("human_decisions", [])
        if isinstance(existing, dict) and existing.get("id") != difference_id
    ]
    existing_decisions.append(decision)
    record["human_decisions"] = existing_decisions
    record["human_review"] = [
        existing
        for existing in record.get("human_review", [])
        if isinstance(existing, dict) and existing.get("id") != difference_id
    ]
    record["human_review_queue_fingerprint"] = review_queue_fingerprint(
        record["human_review"]
    )
    record["human_review_updated_at"] = now_iso()


def _apply_assisted_human_decision_batch(record: dict, decisions: list[dict]) -> dict:
    """Validate and apply a complete Assisted batch only to a private record copy."""

    working = copy.deepcopy(record)
    for request in decisions:
        _apply_assisted_batch_decision(
            working,
            difference_id=request["id"],
            source=request["source"],
        )
    return working


def record_assisted_human_decision_batch(
    episode_key: str,
    decisions: object,
    *,
    expected_generation_fingerprint: str,
    assisted_policy_version: str,
) -> dict:
    """Atomically persist Assisted Apple/Whisper decisions with full audit support.

    Omitted/deferred cards are simply never submitted here and remain
    pending -- there is no explicit "defer" source. Any invalid member
    (not still an assisted candidate, a stale/wrong assisted policy
    version, a tampered request shape, a duplicate id, or an id that is
    no longer pending) rejects the whole batch atomically before any
    write. Agreement with the derived machine recommendation is never
    required.
    """

    validated = _validated_batch_requests(decisions)
    if (
        not isinstance(expected_generation_fingerprint, str)
        or not expected_generation_fingerprint
    ):
        raise ValueError("Expected review generation is required")
    if assisted_policy_version != ASSISTED_REVIEW_POLICY_VERSION:
        raise ValueError("Assisted review policy version is stale; reload before deciding")

    for _ in range(3):
        record, object_generation = load_review_record_with_generation(episode_key)
        if (
            record.get("human_review_generation_fingerprint")
            != expected_generation_fingerprint
        ):
            raise ValueError("Review generation changed; reload before batch approval")

        updated = _apply_assisted_human_decision_batch(record, validated)
        try:
            save_review_record(
                episode_key,
                updated,
                if_generation_match=object_generation,
            )
            return updated
        except PreconditionFailed:
            continue

    raise RuntimeError(
        "Review record changed repeatedly while recording assisted batch decisions"
    )


def _decision_review_item_matches(decision: dict, item: dict) -> bool:
    """Require the original human-reviewed source/focus text to be unchanged."""

    reviewed_item = decision.get("review_item")
    if not isinstance(reviewed_item, dict):
        return False
    for key in ("id", "apple_text", "whisper_text"):
        if reviewed_item.get(key) != item.get(key):
            return False
    reviewed_focus = reviewed_item.get("focus", {})
    current_focus = item.get("focus", {})
    return (
        reviewed_focus.get("scope", "full") == current_focus.get("scope", "full")
        and reviewed_focus.get("apple_text") == current_focus.get("apple_text")
        and reviewed_focus.get("whisper_text") == current_focus.get("whisper_text")
    )


def validated_human_resolutions(
    record: dict,
    review_items: Iterable[dict],
    *,
    require_current_item_evidence: bool = False,
) -> list[dict]:
    """Convert audited human choices to compiler resolutions conservatively.

    A decision only applies if its ID is still present in the freshly rebuilt
    review set. Source choices remain exact; third/custom text is marked human
    so the compiler cannot mistake it for an automatic source decision.
    """

    current = {item.get("id"): item for item in review_items if isinstance(item, dict)}
    resolutions: list[dict] = []
    for decision in record.get("human_decisions", []):
        if not isinstance(decision, dict) or decision.get("reviewed_by") != "human":
            continue
        difference_id = decision.get("id")
        item = current.get(difference_id)
        source = decision.get("chosen_source")
        text = decision.get("chosen_text")
        # An empty Apple or Whisper span is a valid, explicit human choice for
        # an omission conflict. Only non-string values are malformed here.
        if (
            item is None
            or source not in HUMAN_SOURCES
            or not isinstance(text, str)
        ):
            continue
        if require_current_item_evidence and not _decision_review_item_matches(decision, item):
            continue

        decision_scope = decision.get("scope")
        current_scope = item.get("focus", {}).get("scope", "full")
        if decision_scope == "expanded":
            if source != "custom":
                continue
            try:
                details = preview_custom_edit_details(
                    item,
                    text,
                    expand_left_words=decision.get("expand_left_words"),
                    expand_right_words=decision.get("expand_right_words"),
                )
            except ValueError:
                continue
            if (
                details["edit_base_source"] != decision.get("edit_base_source")
                or details["replaced_text"] != decision.get("expected_base_text")
                or details["expand_left_words"] != decision.get("expand_left_words")
                or details["expand_right_words"] != decision.get("expand_right_words")
            ):
                continue
            resolutions.append(
                {
                    "id": difference_id,
                    "source": "human",
                    "text": text,
                    "scope": "expanded",
                    "reviewed_by": "human",
                    "edit_base_source": details["edit_base_source"],
                    "expand_left_words": details["expand_left_words"],
                    "expand_right_words": details["expand_right_words"],
                    "expected_base_text": details["replaced_text"],
                }
            )
            continue

        if decision_scope != current_scope:
            continue

        resolution_text = text
        if source in {"apple", "whisper"}:
            expected = (
                item.get("focus", {}).get(source + "_text")
                if current_scope == "partial"
                else item.get(source + "_text")
            )
            if text != expected:
                continue

            representation = decision.get("representation_text")
            if representation is not None:
                if decision.get("representation_kind") != "case_only":
                    continue
                try:
                    representation = _case_only_representation(text, representation)
                except ValueError:
                    continue
                if representation is not None:
                    resolution_source = "human"
                    resolution_text = representation
                else:
                    resolution_source = source
            else:
                resolution_source = source
        else:
            resolution_source = "human"
        resolution = {
            "id": difference_id,
            "source": resolution_source,
            "text": resolution_text,
            "scope": current_scope,
            "reviewed_by": "human",
        }
        if current_scope == "partial":
            resolution["focus_apple_text"] = item["focus"]["apple_text"]
            resolution["focus_whisper_text"] = item["focus"]["whisper_text"]
        resolutions.append(resolution)
    return resolutions

def _completed_review_generation(record: dict) -> str:
    """Find the immutable review-generation identity, including old records."""

    current = record.get("human_review_generation_fingerprint")
    if isinstance(current, str) and current:
        return current
    items = list(pending_review_items(record))
    items.extend(
        decision["review_item"]
        for decision in record.get("human_decisions", [])
        if isinstance(decision, dict) and isinstance(decision.get("review_item"), dict)
    )
    return review_generation_fingerprint(record.get("input_fingerprint"), items)


def _active_recompile_request(record: dict, generation: str) -> dict | None:
    for request in reversed(record.get("recompile_requests", [])):
        if (
            isinstance(request, dict)
            and request.get("review_generation_fingerprint") == generation
            and request.get("status") in RECOMPILE_BLOCKING_STATUSES
        ):
            return request
    return None


def _is_structured_recompile_request(request: object) -> bool:
    """Return whether a request has the durable identity needed for recovery."""

    return (
        isinstance(request, dict)
        and isinstance(request.get("request_id"), str)
        and bool(request["request_id"])
        and isinstance(request.get("review_generation_fingerprint"), str)
        and bool(request["review_generation_fingerprint"])
        and isinstance(request.get("status"), str)
    )


def link_recompile_result_generation(
    requests: list[dict],
    *,
    request_id: str,
    review_generation: str,
    result_generation: str,
) -> bool:
    """Link one in-flight request to the resolver generation it produced."""

    matches = [
        request
        for request in requests
        if _is_structured_recompile_request(request)
        and request["request_id"] == request_id
        and request["review_generation_fingerprint"] == review_generation
    ]
    if len(matches) != 1:
        raise RecompileLifecycleInvariantError(
            "Recompile request reservation was not found exactly once"
        )
    request = matches[0]
    if request["status"] not in RECOMPILE_IN_FLIGHT_STATUSES:
        raise RecompileLifecycleInvariantError(
            "Recompile request has an incompatible lifecycle status"
        )
    existing = request.get("result_review_generation_fingerprint")
    if existing is None:
        request["result_review_generation_fingerprint"] = result_generation
        return True
    if existing != result_generation:
        raise RecompileLifecycleInvariantError(
            "Recompile request has a conflicting result generation"
        )
    return False


def recompile_status_for_record(record: dict) -> str | None:
    """Return the UI status proven for this resolver generation only."""

    generation = record.get("human_review_generation_fingerprint")
    if not isinstance(generation, str) or not generation:
        return None
    requests = record.get("recompile_requests", [])
    if not isinstance(requests, list):
        return None
    has_current_request = False
    for request in reversed(requests):
        if not _is_structured_recompile_request(request):
            continue
        if request["review_generation_fingerprint"] != generation:
            continue
        has_current_request = True
        if request["status"] in RECOMPILE_BLOCKING_STATUSES:
            return request["status"]
    if has_current_request:
        return None
    for request in reversed(requests):
        if (
            _is_structured_recompile_request(request)
            and request["status"] == "completed"
            and request.get("result_review_generation_fingerprint") == generation
        ):
            return "completed"
    return None


def _update_recompile_request(
    episode_key: str, request_id: str, *, status: str, worker_request: dict | None = None
) -> dict:
    """Finalize one reserved request without replacing newer review data."""

    for _ in range(3):
        record, generation = load_review_record_with_generation(episode_key)
        for request in record.get("recompile_requests", []):
            if isinstance(request, dict) and request.get("request_id") == request_id:
                request["status"] = status
                request["updated_at"] = now_iso()
                if worker_request is not None:
                    request["operation"] = worker_request.get("operation")
                    request["attempts"] = worker_request.get("attempts")
                try:
                    save_review_record(
                        episode_key, record, if_generation_match=generation
                    )
                    return request
                except PreconditionFailed:
                    break
        else:
            raise RuntimeError("Recompile request reservation was not found")
    raise RuntimeError("Review record changed repeatedly while updating recompile request")


def complete_recompile_request(
    episode_key: str, *, request_id: str, review_generation: str
) -> dict:
    """Persist successful completion for this worker's exact recompile request.

    The worker carries these identifiers from the Cloud Run execution override.
    They intentionally identify the original review generation, which may no
    longer be the resolver record's current generation after recompilation.
    """

    try:
        for _ in range(3):
            record, object_generation = load_review_record_with_generation(episode_key)
            for request in record.get("recompile_requests", []):
                if not _is_structured_recompile_request(request):
                    continue
                if (
                    request["request_id"] != request_id
                    or request["review_generation_fingerprint"] != review_generation
                ):
                    continue
                if request["status"] == "completed":
                    return request
                if request["status"] not in RECOMPILE_IN_FLIGHT_STATUSES:
                    raise RecompileLifecycleInvariantError(
                        "Recompile request has an incompatible terminal status"
                    )
                request["status"] = "completed"
                request["completed_at"] = now_iso()
                if not isinstance(
                    request.get("result_review_generation_fingerprint"), str
                ) or not request["result_review_generation_fingerprint"]:
                    raise RecompileLifecycleInvariantError(
                        "Recompile request has no durable result generation"
                    )
                request["updated_at"] = now_iso()
                try:
                    save_review_record(
                        episode_key, record, if_generation_match=object_generation
                    )
                    return request
                except PreconditionFailed:
                    break
            else:
                raise RecompileLifecycleInvariantError(
                    "Recompile request reservation was not found"
                )
    except RecompileLifecycleInvariantError:
        raise
    except (GoogleAPICallError, RuntimeError) as error:
        raise RecompileAuditPersistenceError(
            "Review record changed repeatedly while completing recompile"
        ) from error
    raise RecompileAuditPersistenceError(
        "Review record changed repeatedly while completing recompile"
    )


def request_worker_recompile(episode_key: str, *, requested_by: str) -> dict:
    """Start one worker per completed review generation, durably and safely."""

    reservation = None
    for _ in range(3):
        record, object_generation = load_review_record_with_generation(episode_key)
        if pending_review_items(record):
            raise ValueError("All human-review cards must be decided before recompiling")
        review_generation = _completed_review_generation(record)
        existing = _active_recompile_request(record, review_generation)
        if existing is not None:
            return {**existing, "idempotent": True}
        record["human_review_generation_fingerprint"] = review_generation
        reservation = {
            "request_id": str(uuid4()),
            "review_generation_fingerprint": review_generation,
            "requested_at": now_iso(),
            "requested_by": requested_by,
            "status": "starting",
        }
        record.setdefault("recompile_requests", []).append(reservation)
        try:
            save_review_record(
                episode_key, record, if_generation_match=object_generation
            )
            break
        except PreconditionFailed:
            reservation = None
    if reservation is None:
        raise RuntimeError("Review record changed repeatedly while reserving recompile")

    try:
        worker_request = request_worker_run(
            requested_by=requested_by,
            episode_key=episode_key,
            correlation={
                "request_id": reservation["request_id"],
                "review_generation": reservation["review_generation_fingerprint"],
                "episode_key": episode_key,
            },
        )
    except WorkerRunRequestError as error:
        _update_recompile_request(
            episode_key,
            reservation["request_id"],
            status="failed" if error.definitely_not_accepted else "unknown",
        )
        raise
    except Exception:
        _update_recompile_request(
            episode_key, reservation["request_id"], status="unknown"
        )
        raise
    return _update_recompile_request(
        episode_key,
        reservation["request_id"],
        status="started",
        worker_request=worker_request,
    )


def _recompile_reconciliation_plan(
    record: dict,
    *,
    episode_key: str,
    review_generation: str | None,
    executions: Iterable[dict],
) -> dict:
    """Build a fail-closed reconciliation plan without changing the record."""

    candidates = [
        request
        for request in record.get("recompile_requests", [])
        if _is_structured_recompile_request(request)
        and request["status"] in RECOMPILE_IN_FLIGHT_STATUSES
        and (
            review_generation is None
            or request["review_generation_fingerprint"] == review_generation
        )
    ]
    if not candidates:
        result = {"status": "no_stuck_request"}
        if review_generation is not None:
            result["review_generation_fingerprint"] = review_generation
        return result
    if len(candidates) != 1:
        result = {
            "status": "invariant_violation_multiple_stuck_requests",
            "request_count": len(candidates),
        }
        if review_generation is not None:
            result["review_generation_fingerprint"] = review_generation
        return result
    request = candidates[0]
    target_generation = request["review_generation_fingerprint"]
    matches = [
        execution
        for execution in executions
        if execution_matches_correlation(
            execution,
            request_id=request["request_id"],
            review_generation=target_generation,
            episode_key=episode_key,
        )
    ]
    base = {
        "request_id": request["request_id"],
        "review_generation_fingerprint": target_generation,
    }
    if not matches:
        return {"status": "unresolved_no_exact_execution", **base}
    if len(matches) != 1:
        return {
            "status": "invariant_violation_multiple_exact_executions",
            "execution_count": len(matches),
            **base,
        }
    execution = matches[0]
    return {
        "status": "exact_execution_found",
        "new_request_status": reconciled_execution_status(execution),
        "execution_name": execution.get("name"),
        "execution_uid": execution.get("uid"),
        "execution": execution,
        **base,
    }


def reconcile_review_recompile(
    episode_key: str,
    *,
    apply: bool = False,
    review_generation: str | None = None,
    expected_result_generation: str | None = None,
    execution_loader: Callable[[], list[dict]] = list_worker_executions,
) -> dict:
    """Safely reconcile one stuck recompile request against Cloud Run metadata.

    ``apply=False`` is read-only.  An apply writes only the exact reserved
    request after a generation-pinned record read; a concurrent change restarts
    the operation from a fresh snapshot.
    """

    executions = execution_loader()
    for _ in range(3):
        record, object_generation = load_review_record_with_generation(episode_key)
        plan = _recompile_reconciliation_plan(
            record,
            episode_key=episode_key,
            review_generation=review_generation,
            executions=executions,
        )
        if plan.get("status") != "exact_execution_found":
            return {key: value for key, value in plan.items() if key != "execution"}
        request_id = plan["request_id"]
        target_request = next(
            (
                request
                for request in record.get("recompile_requests", [])
                if isinstance(request, dict)
                and request.get("request_id") == request_id
                and request.get("review_generation_fingerprint")
                == plan["review_generation_fingerprint"]
            ),
            None,
        )
        if target_request is None:
            return {
                "status": "invariant_violation_request_disappeared",
                "request_id": request_id,
            }
        result_generation_to_link = None
        if plan["new_request_status"] == "completed":
            result_generation = target_request.get(
                "result_review_generation_fingerprint"
            )
            if result_generation is None:
                if expected_result_generation is None:
                    return {
                        "status": "invariant_violation_missing_result_generation",
                        "request_id": request_id,
                        "review_generation_fingerprint": plan[
                            "review_generation_fingerprint"
                        ],
                    }
                current_generation = record.get(
                    "human_review_generation_fingerprint"
                )
                if current_generation != expected_result_generation:
                    return {
                        "status": "invariant_violation_expected_result_generation_mismatch",
                        "request_id": request_id,
                        "review_generation_fingerprint": plan[
                            "review_generation_fingerprint"
                        ],
                    }
                result_generation_to_link = expected_result_generation
            elif not isinstance(result_generation, str) or not result_generation:
                return {
                    "status": "invariant_violation_invalid_result_generation",
                    "request_id": request_id,
                    "review_generation_fingerprint": plan[
                        "review_generation_fingerprint"
                    ],
                }
            elif (
                expected_result_generation is not None
                and result_generation != expected_result_generation
            ):
                return {
                    "status": "invariant_violation_conflicting_result_generation",
                    "request_id": request_id,
                    "review_generation_fingerprint": plan[
                        "review_generation_fingerprint"
                    ],
                }
        if not apply:
            result = {key: value for key, value in plan.items() if key != "execution"}
            if result_generation_to_link is not None:
                result["transitional_result_generation_link"] = result_generation_to_link
            return result
        for request in record.get("recompile_requests", []):
            if request is target_request:
                request["status"] = plan["new_request_status"]
                if result_generation_to_link is not None:
                    request["result_review_generation_fingerprint"] = (
                        result_generation_to_link
                    )
                request["execution_name"] = plan.get("execution_name")
                request["execution_uid"] = plan.get("execution_uid")
                request["reconciled_at"] = now_iso()
                request["updated_at"] = now_iso()
                try:
                    save_review_record(
                        episode_key, record, if_generation_match=object_generation
                    )
                except PreconditionFailed:
                    break
                return {
                    key: value
                    for key, value in {**plan, "applied": True}.items()
                    if key != "execution"
                }
        else:
            return {
                "status": "invariant_violation_request_disappeared",
                "request_id": request_id,
            }
    raise RuntimeError("Review record changed repeatedly while reconciling recompile")


def _audio_source(episode: dict) -> str:
    """Return the durable RSS enclosure used for every review clip."""

    audio_url = episode.get("audio_url")
    if isinstance(audio_url, str) and audio_url.startswith(("https://", "http://")):
        return audio_url
    raise ValueError("Episode has no downloadable audio source")


def ensure_audio_clip(episode: dict, item: dict, *, fingerprint: str) -> tuple[Path, dict]:
    """Create one short ephemeral WAV clip for a review request.

    The caller must unlink the returned path after sending it or after third
    ASR completes. Review clips are never cached in GCS or the project tree.
    """

    window = review_clip_window(item)
    prefix = f"podcast-review-{episode['episode_key']}-{item['id']}-"
    handle, raw_path = tempfile.mkstemp(prefix=prefix, suffix=".wav")
    os.close(handle)
    destination = Path(raw_path)
    try:
        extract_audio_clip(_audio_source(episode), destination, window)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    return destination, window


def ensure_third_asr(
    episode: dict,
    difference_id: int,
    *,
    api_key: str | None = None,
    pricing_transport: Callable[[str, str], list] | None = None,
    pricing_now: Callable[[], str] | None = None,
    now: Callable[[], str] | None = None,
    sleep: Callable[[float], None] | None = None,
    random_fn: Callable[[], float] | None = None,
    prepare_session_id: str | None = None,
) -> dict:
    """Generation-safe, concurrency-safe, budget-safe third-ASR evidence.

    Replaces the earlier unsafe load/modify/save path with: a per-item
    generation-CAS claim (see ``acquire_third_asr_claim``), an
    independently-resolved conservative budget reservation from current
    trusted audio pricing, a bounded failure taxonomy with at most one
    paid retry using a small base delay plus bounded jitter, and a
    generation-verified finalize write. See docs/superpowers/specs/
    2026-09-16-human-review-evidence-assisted-adjudication-design.md,
    "Concurrency-safe Third ASR" and "Retry and failure taxonomy".

    Raises ``ThirdAsrInFlight`` (without spending anything) if another
    unexpired claim already owns this item; callers such as
    ``podcast_engine.review_web`` must treat that as a non-blocking
    "try again shortly" outcome, never as an error to swallow and retry
    internally here.

    ``prepare_session_id`` is audit identity only, passed straight
    through to ``acquire_third_asr_claim`` and, when the attempt
    succeeds, stamped onto the persisted evidence -- it never affects
    admission, reservation, settlement, or retry mechanics (this call
    always reserves and settles its own independent budget attempt
    regardless of whether a prepare-session originated it). Omitted for
    callers -- such as the Detailed Review "Run third ASR" button -- that
    have no prepare-session to attribute the request to; the field is
    never fabricated onto evidence that did not actually carry it.
    """

    episode_key = episode["episode_key"]
    now_fn = now or now_iso
    sleep_fn = sleep or time.sleep

    record, _ = load_review_record_with_generation(episode_key)
    evidence_fingerprint = record["input_fingerprint"]
    budget_fingerprint = record["source_fingerprint"]
    item = _item(record, difference_id)
    window = review_clip_window(item)
    cache_key = third_asr_cache_key(
        input_fingerprint=evidence_fingerprint, item_id=difference_id, window=window
    )

    existing = item.get("third_asr")
    if isinstance(existing, dict) and existing.get("cache_key") == cache_key and existing.get("text"):
        return existing

    # TASK-109: every paid Third-ASR path is fail-closed until historical
    # input-keyed budget ledgers are explicitly reconciled against this
    # source-generation ledger. Cache hits above remain free/read-only.
    ensure_fresh_third_asr_budget_identity(
        episode_key, budget_fingerprint, evidence_fingerprint
    )

    resolved_api_key = api_key or os.getenv("PODCAST_REVIEW_ASR_API_KEY")
    if not resolved_api_key:
        raise RuntimeError("Missing PODCAST_REVIEW_ASR_API_KEY for third-ASR review")

    pricing = resolve_audio_model_pricing(
        THIRD_ASR_MODEL, api_key=resolved_api_key, transport=pricing_transport, now=pricing_now
    )
    # decimal_from_admission_input rejects raw floats outright (money math
    # never uses binary floating point); review_clip_window's "duration" is a
    # plain float, so it is converted via its exact decimal string
    # representation, never coerced implicitly.
    reserved_usd = derive_audio_reservation_usd(
        max_billable_seconds=Decimal(str(window["duration"])),
        usd_per_second=pricing.usd_per_second,
    )

    claim: dict | None = None

    for attempt in range(1, THIRD_ASR_MAX_ATTEMPTS + 1):
        attempt_id = f"third_asr-{uuid4().hex}"
        reserve_budget_batch(
            episode_key,
            budget_fingerprint,
            [
                {
                    "attempt_id": attempt_id,
                    "stage": STAGE_THIRD_ASR,
                    "reserved_usd": reserved_usd,
                    "third_asr": True,
                }
            ],
        )

        if claim is None:
            try:
                claim = acquire_third_asr_claim(
                    episode_key,
                    evidence_fingerprint,
                    difference_id,
                    cache_key=cache_key,
                    model=THIRD_ASR_MODEL,
                    window=window,
                    budget_attempt_id=attempt_id,
                    prepare_session_id=prepare_session_id,
                    now=now_fn,
                )
            except ThirdAsrClaimError:
                release_budget_attempt_pre_send(episode_key, budget_fingerprint, attempt_id)
                raise
        else:
            claim = _refresh_third_asr_claim_for_retry(
                episode_key,
                evidence_fingerprint,
                difference_id,
                budget_attempt_id=attempt_id,
                now=now_fn,
            )

        clip = None
        try:
            clip, _ = ensure_audio_clip(episode, item, fingerprint=evidence_fingerprint)
            evidence = transcribe_review_clip(clip)
        except Exception as error:
            # Failure before the clip even exists is provably pre-send: no
            # network request to the third-ASR provider was ever possible.
            failure_class = (
                THIRD_ASR_FAILURE_PRE_SEND_RETRYABLE
                if clip is None
                else classify_third_asr_failure(error)
            )

            if failure_class == THIRD_ASR_FAILURE_PRE_SEND_RETRYABLE:
                release_budget_attempt_pre_send(episode_key, budget_fingerprint, attempt_id)
            else:
                mark_budget_attempt_uncertain(
                    episode_key, budget_fingerprint, attempt_id, reason=type(error).__name__
                )

            if failure_class in THIRD_ASR_RETRYABLE_FAILURE_CLASSES and attempt < THIRD_ASR_MAX_ATTEMPTS:
                sleep_fn(_third_asr_retry_delay_seconds(random_fn=random_fn))
                continue

            release_third_asr_claim(
                episode_key, evidence_fingerprint, difference_id, request_id=claim["request_id"]
            )
            raise
        finally:
            if clip is not None:
                clip.unlink(missing_ok=True)

        settle_budget_attempt(
            episode_key,
            budget_fingerprint,
            attempt_id,
            actual_usd=_extract_third_asr_actual_usd(evidence),
        )
        evidence = {
            **evidence,
            "cache_key": cache_key,
            "window": window,
            "generated_at": now_fn(),
        }
        if prepare_session_id is not None:
            evidence["prepare_session_id"] = prepare_session_id
        release_third_asr_claim(
            episode_key, evidence_fingerprint, difference_id, request_id=claim["request_id"]
        )
        return _finalize_third_asr_evidence(
            episode_key, difference_id, cache_key=cache_key, evidence=evidence
        )

    raise ThirdAsrClaimConflict(  # pragma: no cover -- every loop iteration returns or raises
        "Third-ASR attempt loop exited without a terminal outcome"
    )
