"""TASK-076 durable per-episode-generation AI budget ledger with CAS admission.

Every provider-billed AI request attributable to one immutable episode
source-generation (see ``podcast_engine.episode_generation``) must be
admitted through this ledger, at or below EPISODE_AI_HARD_CAP_USD total and
ASSISTED_THIRD_ASR_SUBCAP_USD for assisted Third ASR specifically (the
Third-ASR sub-cap shares the same total budget; it is not additive). A
request without a provable upper-bound reservation (see
``podcast_engine.ai_pricing``) is not admissible.

One ledger lives at ``episodes/<episode_key>/ai/budgets/<digest>.json``,
keyed by the exact source-generation fingerprint's hex digest. Money math
uses Decimal only; ``decimal_from_admission_input`` (imported from
``podcast_engine.ai_pricing``) rejects float/bool/NaN/Infinity/negative
inputs outright.

Mutations use the same bounded generation-CAS retry loop already
established for durable per-episode records (see
``podcast_engine.human_review.load_review_record_with_generation`` /
``save_review_record``): load the current object generation, mutate a
local copy, and write conditionally with ``if_generation_match``, retrying
up to MAX_BUDGET_CAS_ATTEMPTS times on a concurrent writer. Reservation,
prepare-session lease (Task 10), and per-item claim (Task 9) records live in
separate GCS objects; a reservation is written first and carries its own
bounded validity independent of whether any later session/claim object was
ever successfully written, so a crash after a reservation commits leaves a
reservation this module can independently reconcile (release), never a
silently permanent one.
"""

from __future__ import annotations

import copy
from decimal import Decimal
import json
import re
from typing import Callable, Iterable, Mapping

from google.api_core.exceptions import NotFound, PreconditionFailed

from .ai_pricing import (
    decimal_from_admission_input,
    resolve_preset_model_and_pricing,
    resolve_stage_reservation_usd,
)
from .episode_contract import now_iso
from .fingerprint_types import SourceFingerprint
from .preset_provenance import PresetProvenance, fetch_current_designated_preset
from .storage import get_bucket


AI_BUDGET_SCHEMA_VERSION = 1
AI_BUDGET_POLICY_VERSION = "episode-ai-budget-v1"

# TASK-076: raised from 0.35 to 0.40 per the Task 3 worst-case
# cost-estimate finding (resolver + triage + summary + metadata alone
# required ~$0.393 in the worst case even after tightening triage's
# reasoning/max_tokens). Raised again, 0.40 -> 0.70, 2026-09-17: a live
# re-check of summary/metadata's current designated-preset pricing (via
# the exact production resolve_required_downstream_reserve_usd path, zero
# paid calls) found their combined worst case had grown to $0.2033219
# (summary $0.0769864 + metadata $0.1263355), up from Task 3's $0.11322530
# -- most likely provider/preset pricing drift since Task 3, not
# re-investigated further. Combined with Task 3's still-unverified-since
# resolver+triage figures ($0.07296900 + $0.20647200), the live worst case
# for the four mandatory stages alone is already ~$0.483, over the old
# $0.40 cap even before Third ASR. 0.70 restores real headroom for both
# the mandatory stages and the separate Third-ASR subcap below, with
# margin to absorb further drift; verify against real episode-generation
# evidence and tighten or raise further if needed. See
# docs/superpowers/specs/2026-09-16-human-review-evidence-assisted-adjudication-design.md
# "Task 3 worst-case cost-estimate finding" and "Cap re-verification,
# 2026-09-17".
# Raised 0.70 -> 1.60, 2026-09-29 (TASK-118, user decision): the
# single-pass knowledge-note writer (anthropic/claude-sonnet-5.5, whole
# transcript, max_tokens 48000) actually costs about USD 0.22-0.50 per
# episode, but its per-attempt reservation is a byte-bound prompt plus the
# full max_tokens at the completion price -- about USD 0.64-0.72 for
# 14-24k-token transcripts -- which does not fit under 0.70 at all.
# 1.60 leaves room for one retried writer attempt after a failure: an
# uncertain attempt keeps its full reservation, so transcript stages
# (~0.08) + a timed-out attempt (~0.72) + its retry (~0.72) ~= 1.52.
# Normal actual spend stays around USD 0.45 per episode. See
# docs/superpowers/specs/2026-09-29-single-pass-knowledge-note-design.md
# "Budget".
EPISODE_AI_HARD_CAP_USD = Decimal("1.60")
ASSISTED_THIRD_ASR_SUBCAP_USD = Decimal("0.10")
MAX_BUDGET_CAS_ATTEMPTS = 3

THIRD_ASR_BUDGET_RECONCILIATION_SCHEMA_VERSION = 1
THIRD_ASR_BUDGET_IDENTITY_POLICY_VERSION = "third-asr-budget-identity-v1"
THIRD_ASR_BUDGET_RECONCILIATION_CLASSIFICATIONS = frozenset(
    {"legacy_current_generation", "previous_source_generation"}
)

_ATTEMPT_STATES = frozenset({"reserved", "settled", "released", "uncertain"})
_SETTLEABLE_STATES = frozenset({"reserved", "uncertain"})

# TASK-076 Task 5: canonical stage-id vocabulary for every episode-AI-budget
# attempt. transcript resolver/triage already run before assisted Third ASR
# "Prepare assisted evidence" is ever offered, so by the time
# required_downstream_reserve matters they are always already-settled, not
# pending -- but the vocabulary still names them for completeness and for
# any future caller that legitimately needs to protect a pre-resolver/
# triage budget check. third_asr is deliberately its own stage: it is
# optional assisted evidence, never a protected mandatory downstream stage
# (see _DOWNSTREAM_RESERVE_ELIGIBLE_STAGES below).
STAGE_RESOLVER = "resolver"
STAGE_TRIAGE = "triage"
STAGE_THIRD_ASR = "third_asr"
STAGE_SUMMARY = "summary"
STAGE_METADATA = "metadata"
STAGE_SUMMARY_REVIEW = "summary_review"
# TASK-118: the single-pass knowledge-note writer, which replaces
# summary + summary_review + metadata when PODCAST_KNOWLEDGE_WRITER_PRESET
# is configured.
STAGE_NOTE_WRITER = "note_writer"
# TASK-125: the Whisper source is transcribed through OpenRouter before the
# source generation exists, so its attempts live in a ledger keyed by the
# downloaded audio's sha256 fingerprint (same shape and cap semantics) rather
# than the Apple/Whisper source-generation fingerprint.
STAGE_WHISPER_TRANSCRIPTION = "whisper_transcription"

AI_BUDGET_STAGE_IDS = frozenset(
    {
        STAGE_WHISPER_TRANSCRIPTION,
        STAGE_RESOLVER,
        STAGE_TRIAGE,
        STAGE_THIRD_ASR,
        STAGE_SUMMARY,
        STAGE_METADATA,
        STAGE_SUMMARY_REVIEW,
        STAGE_NOTE_WRITER,
    }
)

# Stages required_downstream_reserve ever protects money for. Third ASR is
# optional assisted evidence, not mandatory downstream work, so it is never
# in this set even though it is a valid attempt stage elsewhere.
_DOWNSTREAM_RESERVE_ELIGIBLE_STAGES = frozenset(
    {
        STAGE_RESOLVER,
        STAGE_TRIAGE,
        STAGE_SUMMARY,
        STAGE_METADATA,
        STAGE_SUMMARY_REVIEW,
        STAGE_NOTE_WRITER,
    }
)

_ATTEMPT_STATES = frozenset({"reserved", "settled", "released", "uncertain"})
_SETTLEABLE_STATES = frozenset({"reserved", "uncertain"})

_EPISODE_KEY = re.compile(r"^[0-9a-f]{24}$")
_SOURCE_FINGERPRINT = re.compile(r"^sha256:[0-9a-f]{64}$")


class BudgetLedgerError(RuntimeError):
    """Base class for every fail-closed budget-ledger rejection."""


class BudgetLedgerIdentityError(BudgetLedgerError):
    """The ledger object's own recorded identity does not match the request."""


class BudgetIdentityReconciliationRequired(BudgetLedgerError):
    """Third-ASR spend cannot continue until historical ledger identity is reconciled."""


class BudgetAdmissionError(BudgetLedgerError):
    """A reservation cannot be admitted without exceeding a budget cap."""


class BudgetIntegrityError(BudgetLedgerError):
    """A settled actual cost exceeded its own reservation.

    Recorded durably on the ledger before this is raised. Once recorded, the
    ledger refuses every further reservation for this episode generation
    until a human reconciles it (there is no automatic recovery path).
    """


class BudgetConcurrencyError(BudgetLedgerError):
    """The ledger changed repeatedly across every CAS attempt."""


class DownstreamReserveError(BudgetLedgerError):
    """Stage state or a stage's own reservation bound is malformed.

    Also raised when a pending mandatory stage's reservation bound cannot
    be resolved at all (see resolve_required_downstream_reserve_usd): an
    unprovable downstream bound must fail closed, never be treated as
    zero, so an optional evidence purchase can never silently proceed
    without protecting money the pipeline will still need.
    """


def budget_ledger_path(episode_key: str, source_fingerprint: SourceFingerprint) -> str:
    """Return the canonical GCS path for one episode generation's ledger."""

    if not isinstance(episode_key, str) or not _EPISODE_KEY.fullmatch(episode_key):
        raise ValueError("episode_key must be a canonical 24-hex-character episode key")
    if not isinstance(source_fingerprint, str) or not _SOURCE_FINGERPRINT.fullmatch(
        source_fingerprint
    ):
        raise ValueError(
            "source_fingerprint must be a canonical 'sha256:<64-hex>' fingerprint"
        )
    digest = source_fingerprint.split(":", 1)[1]
    return f"episodes/{episode_key}/ai/budgets/{digest}.json"


def third_asr_budget_reconciliation_path(
    episode_key: str, source_fingerprint: str
) -> str:
    """Return the durable reconciliation marker path for one source generation."""

    budget_ledger_path(episode_key, source_fingerprint)
    digest = source_fingerprint.split(":", 1)[1]
    return (
        f"episodes/{episode_key}/ai/budget-identity-reconciliations/"
        f"{digest}.json"
    )


def _empty_ledger(episode_key: str, source_fingerprint: str) -> dict:
    return {
        "schema_version": AI_BUDGET_SCHEMA_VERSION,
        "policy_version": AI_BUDGET_POLICY_VERSION,
        "episode_key": episode_key,
        "source_fingerprint": source_fingerprint,
        "hard_cap_usd": str(EPISODE_AI_HARD_CAP_USD),
        "third_asr_subcap_usd": str(ASSISTED_THIRD_ASR_SUBCAP_USD),
        "created_at": now_iso(),
        "attempts": {},
    }


def _validate_ledger_shape(payload: object, episode_key: str, source_fingerprint: str) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("Budget ledger record must be a JSON object")
    if payload.get("schema_version") != AI_BUDGET_SCHEMA_VERSION:
        raise ValueError("Unsupported budget ledger schema_version")
    if payload.get("policy_version") != AI_BUDGET_POLICY_VERSION:
        raise ValueError("Unsupported budget ledger policy_version")
    if (
        payload.get("episode_key") != episode_key
        or payload.get("source_fingerprint") != source_fingerprint
    ):
        raise BudgetLedgerIdentityError(
            "Budget ledger object identity does not match the requested "
            "episode_key/source_fingerprint"
        )
    attempts = payload.get("attempts")
    if not isinstance(attempts, dict):
        raise ValueError("Budget ledger 'attempts' must be a JSON object")
    for attempt_id, attempt in attempts.items():
        if not isinstance(attempt, dict) or attempt.get("state") not in _ATTEMPT_STATES:
            raise ValueError(f"Malformed budget attempt record: {attempt_id!r}")
    return payload


def _load_ledger_with_generation(
    episode_key: str, source_fingerprint: str
) -> tuple[dict, int | None]:
    """Read current ledger content and object generation.

    A generation of ``None`` means no ledger object exists yet for this
    episode generation -- the normal state before the first reservation.
    """

    path = budget_ledger_path(episode_key, source_fingerprint)
    blob = get_bucket().blob(path)
    for _ in range(MAX_BUDGET_CAS_ATTEMPTS):
        try:
            blob.reload()
        except NotFound:
            return _empty_ledger(episode_key, source_fingerprint), None
        generation = int(blob.generation)
        try:
            content = blob.download_as_text(
                encoding="utf-8",
                if_generation_match=generation,
            )
        except PreconditionFailed:
            continue
        payload = json.loads(content)
        return _validate_ledger_shape(payload, episode_key, source_fingerprint), generation
    raise BudgetConcurrencyError(
        "Episode budget ledger changed repeatedly while reading snapshot"
    )


def _save_ledger(
    episode_key: str,
    source_fingerprint: str,
    ledger: dict,
    *,
    if_generation_match: int | None,
) -> None:
    path = budget_ledger_path(episode_key, source_fingerprint)
    blob = get_bucket().blob(path)
    blob.upload_from_string(
        json.dumps(ledger, ensure_ascii=False, indent=2) + "\n",
        content_type="application/json",
        if_generation_match=0 if if_generation_match is None else if_generation_match,
    )


def _existing_attempt(ledger: dict, attempt_id: str) -> dict:
    attempt = ledger["attempts"].get(attempt_id)
    if attempt is None:
        raise ValueError(f"Unknown budget attempt_id: {attempt_id!r}")
    return attempt


def _summary_from_ledger(ledger: dict) -> dict:
    """Pure aggregation of one loaded ledger's spend, by state and sub-cap.

    Kept separate from any I/O so admission math and ``budget_summary`` share
    one implementation.
    """

    settled = Decimal("0")
    uncertain = Decimal("0")
    live_reservations = Decimal("0")
    third_asr_settled = Decimal("0")
    third_asr_uncertain = Decimal("0")
    third_asr_live_reservations = Decimal("0")

    for attempt in ledger["attempts"].values():
        state = attempt["state"]
        third_asr = bool(attempt.get("third_asr", False))
        if state == "settled":
            amount = Decimal(attempt["settled_usd"])
            settled += amount
            if third_asr:
                third_asr_settled += amount
        elif state == "uncertain":
            amount = Decimal(attempt["reserved_usd"])
            uncertain += amount
            if third_asr:
                third_asr_uncertain += amount
        elif state == "reserved":
            amount = Decimal(attempt["reserved_usd"])
            live_reservations += amount
            if third_asr:
                third_asr_live_reservations += amount
        # "released" attempts contribute no spend.

    return {
        "settled_spend": settled,
        "uncertain_spend": uncertain,
        "live_reservations": live_reservations,
        "third_asr_settled": third_asr_settled,
        "third_asr_uncertain": third_asr_uncertain,
        "third_asr_live_reservations": third_asr_live_reservations,
        "integrity_failure": ledger.get("integrity_failure"),
    }


def _third_asr_consumption_from_ledger(ledger: dict) -> dict:
    """Return conservative Third-ASR budget consumption for compatibility work."""

    settled = Decimal("0")
    uncertain = Decimal("0")
    attempt_count = 0
    for attempt in ledger["attempts"].values():
        if not bool(attempt.get("third_asr", False)):
            continue
        state = attempt["state"]
        if state == "settled":
            settled += Decimal(attempt["settled_usd"])
            attempt_count += 1
        elif state in {"reserved", "uncertain"}:
            # A historical live reservation cannot be proven pre-send after
            # the fact, so compatibility reconciliation carries it forward
            # conservatively as uncertain spend.
            uncertain += Decimal(attempt["reserved_usd"])
            attempt_count += 1
        # Released attempts consume no cap and need no carry-forward.
    return {
        "settled_usd": settled,
        "uncertain_usd": uncertain,
        "total_usd": settled + uncertain,
        "attempt_count": attempt_count,
    }


def _read_budget_blob(blob, episode_key: str) -> tuple[dict, int]:
    """Read and validate one ledger discovered through the episode prefix."""

    blob.reload()
    generation = int(blob.generation)
    content = blob.download_as_text(
        encoding="utf-8",
        if_generation_match=generation,
    )
    payload = json.loads(content)
    fingerprint = payload.get("source_fingerprint") if isinstance(payload, dict) else None
    if not isinstance(fingerprint, str):
        raise BudgetLedgerIdentityError(
            f"Budget ledger {blob.name!r} has no source_fingerprint identity"
        )
    expected_path = budget_ledger_path(episode_key, fingerprint)
    if blob.name != expected_path:
        raise BudgetLedgerIdentityError(
            f"Budget ledger path {blob.name!r} does not match its recorded identity"
        )
    return _validate_ledger_shape(payload, episode_key, fingerprint), generation


def _third_asr_budget_candidate(ledger: dict, generation: int) -> dict:
    consumption = _third_asr_consumption_from_ledger(ledger)
    return {
        "source_fingerprint": ledger["source_fingerprint"],
        "generation": int(generation),
        "settled_usd": str(consumption["settled_usd"]),
        "uncertain_usd": str(consumption["uncertain_usd"]),
        "attempt_count": consumption["attempt_count"],
        "integrity_failure": copy.deepcopy(ledger.get("integrity_failure")),
    }


def _load_named_third_asr_budget_candidate(
    episode_key: str, source_fingerprint: str
) -> dict:
    """Load one exact candidate ledger without bucket listing authority."""

    path = budget_ledger_path(episode_key, source_fingerprint)
    blob = get_bucket().blob(path)
    try:
        ledger, generation = _read_budget_blob(blob, episode_key)
    except NotFound as error:
        raise BudgetIdentityReconciliationRequired(
            f"Reconciled Third-ASR ledger {source_fingerprint} is no longer present"
        ) from error
    return _third_asr_budget_candidate(ledger, generation)


def inventory_third_asr_budget_identities(
    episode_key: str, source_fingerprint: str
) -> dict:
    """Inventory non-canonical ledgers that still consume Third-ASR budget.

    TASK-109 deliberately does not infer whether another ledger belongs to a
    previous canonical source generation or to the historical bug that keyed
    direct Third-ASR spend by input_fingerprint. That distinction requires an
    explicit operator classification before reconciliation may write.
    """

    budget_ledger_path(episode_key, source_fingerprint)
    prefix = f"episodes/{episode_key}/ai/budgets/"
    candidates = []
    for blob in get_bucket().list_blobs(prefix=prefix):
        ledger, generation = _read_budget_blob(blob, episode_key)
        fingerprint = ledger["source_fingerprint"]
        if fingerprint == source_fingerprint:
            continue
        candidate = _third_asr_budget_candidate(ledger, generation)
        if Decimal(candidate["settled_usd"]) + Decimal(candidate["uncertain_usd"]) <= 0:
            continue
        candidates.append(candidate)
    candidates.sort(key=lambda item: item["source_fingerprint"])
    return {
        "episode_key": episode_key,
        "source_fingerprint": source_fingerprint,
        "candidates": candidates,
    }


def _load_third_asr_budget_reconciliation(
    episode_key: str, source_fingerprint: str
) -> tuple[dict | None, int | None]:
    path = third_asr_budget_reconciliation_path(episode_key, source_fingerprint)
    blob = get_bucket().blob(path)
    try:
        blob.reload()
    except NotFound:
        return None, None
    generation = int(blob.generation)
    content = blob.download_as_text(
        encoding="utf-8",
        if_generation_match=generation,
    )
    payload = json.loads(content)
    if not isinstance(payload, dict):
        raise ValueError("Third-ASR budget reconciliation marker must be a JSON object")
    if payload.get("schema_version") != THIRD_ASR_BUDGET_RECONCILIATION_SCHEMA_VERSION:
        raise ValueError("Unsupported Third-ASR budget reconciliation schema_version")
    if payload.get("policy_version") != THIRD_ASR_BUDGET_IDENTITY_POLICY_VERSION:
        raise ValueError("Unsupported Third-ASR budget reconciliation policy_version")
    if (
        payload.get("episode_key") != episode_key
        or payload.get("source_fingerprint") != source_fingerprint
    ):
        raise BudgetLedgerIdentityError(
            "Third-ASR budget reconciliation marker identity does not match request"
        )
    if not isinstance(payload.get("candidates"), list):
        raise ValueError("Third-ASR budget reconciliation candidates must be a list")
    return payload, generation


def _save_third_asr_budget_reconciliation(
    episode_key: str,
    source_fingerprint: str,
    payload: dict,
    *,
    if_generation_match: int | None,
) -> None:
    path = third_asr_budget_reconciliation_path(episode_key, source_fingerprint)
    get_bucket().blob(path).upload_from_string(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        content_type="application/json",
        if_generation_match=0 if if_generation_match is None else if_generation_match,
    )


def _legacy_reconciliation_attempt_id(source_fingerprint: str, kind: str) -> str:
    digest = source_fingerprint.split(":", 1)[1]
    return f"legacy-third-asr-{digest}-{kind}"


def _candidate_evidence(candidate: dict) -> dict:
    return {
        "source_fingerprint": candidate["source_fingerprint"],
        "generation": int(candidate["generation"]),
        "settled_usd": str(candidate["settled_usd"]),
        "uncertain_usd": str(candidate["uncertain_usd"]),
        "attempt_count": int(candidate["attempt_count"]),
        "integrity_failure": copy.deepcopy(candidate.get("integrity_failure")),
    }


def _reconciliation_inventory_matches(marker: dict, inventory: dict) -> bool:
    recorded = [
        {
            key: candidate[key]
            for key in (
                "source_fingerprint",
                "generation",
                "settled_usd",
                "uncertain_usd",
                "attempt_count",
                "integrity_failure",
            )
        }
        for candidate in marker.get("candidates", [])
    ]
    current = [_candidate_evidence(candidate) for candidate in inventory["candidates"]]
    return recorded == current


def _expected_legacy_attempt(
    candidate: dict, *, kind: str, reconciled_at: str
) -> dict | None:
    if kind == "settled":
        amount = Decimal(candidate["settled_usd"])
        if amount <= 0:
            return None
        return {
            "state": "settled",
            "stage": STAGE_THIRD_ASR,
            "third_asr": True,
            "reserved_usd": str(amount),
            "settled_usd": str(amount),
            "created_at": reconciled_at,
            "updated_at": reconciled_at,
            "reconciled_from_fingerprint": candidate["source_fingerprint"],
            "reconciled_from_generation": int(candidate["generation"]),
        }

    amount = Decimal(candidate["uncertain_usd"])
    if amount <= 0:
        return None
    return {
        "state": "uncertain",
        "stage": STAGE_THIRD_ASR,
        "third_asr": True,
        "reserved_usd": str(amount),
        "created_at": reconciled_at,
        "updated_at": reconciled_at,
        "uncertainty_reason": "legacy_budget_identity_reconciliation",
        "reconciled_from_fingerprint": candidate["source_fingerprint"],
        "reconciled_from_generation": int(candidate["generation"]),
    }


def _legacy_attempt_matches(recorded: dict, expected: dict) -> bool:
    keys = (
        "state",
        "stage",
        "third_asr",
        "reserved_usd",
        "settled_usd",
        "uncertainty_reason",
        "reconciled_from_fingerprint",
        "reconciled_from_generation",
    )
    return all(recorded.get(key) == expected.get(key) for key in keys)


def require_third_asr_budget_identity_reconciled(
    episode_key: str, source_fingerprint: str
) -> dict:
    """Fail closed unless TASK-109 compatibility evidence is still current.

    Runtime callers deliberately use named-object reads only. Full prefix
    inventory is reserved for the operator reconciliation command, which must
    run after pre-fix Third-ASR writers are quiesced.
    """

    canonical, _ = _load_ledger_with_generation(
        episode_key, source_fingerprint
    )
    marker, _ = _load_third_asr_budget_reconciliation(
        episode_key, source_fingerprint
    )
    if marker is None:
        raise BudgetIdentityReconciliationRequired(
            "Source-generation Third-ASR budget identity requires explicit "
            "operator reconciliation before paid admission"
        )

    for recorded_candidate in marker["candidates"]:
        current_candidate = _load_named_third_asr_budget_candidate(
            episode_key, recorded_candidate["source_fingerprint"]
        )
        if _candidate_evidence(current_candidate) != _candidate_evidence(
            recorded_candidate
        ):
            raise BudgetIdentityReconciliationRequired(
                "A reconciled Third-ASR budget ledger changed after reconciliation"
            )
    by_fingerprint = {
        candidate["source_fingerprint"]: candidate for candidate in marker["candidates"]
    }
    for fingerprint, candidate in by_fingerprint.items():
        classification = candidate.get("classification")
        if classification not in THIRD_ASR_BUDGET_RECONCILIATION_CLASSIFICATIONS:
            raise BudgetIdentityReconciliationRequired(
                f"Third-ASR ledger {fingerprint} has no valid reconciliation classification"
            )
        if classification != "legacy_current_generation":
            continue
        if candidate.get("integrity_failure") and not canonical.get("integrity_failure"):
            raise BudgetIdentityReconciliationRequired(
                "Canonical budget ledger no longer preserves a reconciled legacy integrity failure"
            )
        for kind in ("settled", "uncertain"):
            attempt_id = _legacy_reconciliation_attempt_id(fingerprint, kind)
            expected = _expected_legacy_attempt(
                candidate, kind=kind, reconciled_at=marker["reconciled_at"]
            )
            if expected is None:
                continue
            recorded = canonical["attempts"].get(attempt_id)
            if recorded is None or not _legacy_attempt_matches(recorded, expected):
                raise BudgetIdentityReconciliationRequired(
                    "Canonical budget ledger no longer proves reconciled legacy Third-ASR spend"
                )
    return marker


def reconcile_third_asr_budget_identity(
    episode_key: str,
    source_fingerprint: str,
    classifications: Mapping[str, str] | None = None,
    *,
    apply: bool = False,
) -> dict:
    """Plan or apply one explicit, idempotent TASK-109 compatibility repair.

    Every non-canonical ledger that still consumes Third-ASR cap must be
    classified explicitly. legacy_current_generation is carried into the
    canonical source ledger; previous_source_generation remains historical
    and separate. No classification is inferred from timestamps or names.
    """

    inventory = inventory_third_asr_budget_identities(episode_key, source_fingerprint)
    candidate_by_fingerprint = {
        candidate["source_fingerprint"]: candidate for candidate in inventory["candidates"]
    }
    supplied = dict(classifications or {})
    unknown = sorted(set(supplied) - set(candidate_by_fingerprint))
    invalid = sorted(
        fingerprint
        for fingerprint, classification in supplied.items()
        if classification not in THIRD_ASR_BUDGET_RECONCILIATION_CLASSIFICATIONS
    )
    unclassified = sorted(set(candidate_by_fingerprint) - set(supplied))
    if unknown or invalid:
        raise ValueError(
            "Third-ASR reconciliation classifications do not match inventory: "
            f"unknown={unknown}, invalid={invalid}"
        )

    plan_candidates = [
        {
            **_candidate_evidence(candidate),
            "classification": supplied.get(candidate["source_fingerprint"]),
        }
        for candidate in inventory["candidates"]
    ]
    plan = {
        "status": "ready" if not unclassified else "classification_required",
        "episode_key": episode_key,
        "source_fingerprint": source_fingerprint,
        "candidates": plan_candidates,
        "unclassified": unclassified,
        "apply": False,
    }
    if not apply:
        return plan
    if unclassified:
        raise BudgetIdentityReconciliationRequired(
            "Every consuming non-canonical Third-ASR ledger must be classified before apply"
        )

    existing_marker, marker_generation = _load_third_asr_budget_reconciliation(
        episode_key, source_fingerprint
    )
    if existing_marker is not None:
        if not _reconciliation_inventory_matches(existing_marker, inventory):
            raise BudgetIdentityReconciliationRequired(
                "Existing Third-ASR reconciliation marker no longer matches inventory"
            )
        existing_classifications = {
            candidate["source_fingerprint"]: candidate.get("classification")
            for candidate in existing_marker["candidates"]
        }
        if existing_classifications != supplied:
            raise BudgetIdentityReconciliationRequired(
                "Existing Third-ASR reconciliation classifications are immutable"
            )
        require_third_asr_budget_identity_reconciled(episode_key, source_fingerprint)
        return {**plan, "status": "already_reconciled", "apply": True}

    reconciled_at = now_iso()
    for _ in range(MAX_BUDGET_CAS_ATTEMPTS):
        canonical, canonical_generation = _load_ledger_with_generation(
            episode_key, source_fingerprint
        )
        updated = copy.deepcopy(canonical)
        changed = False
        legacy_integrity_failures = [
            {
                "source_fingerprint": candidate["source_fingerprint"],
                "generation": int(candidate["generation"]),
                "integrity_failure": copy.deepcopy(candidate["integrity_failure"]),
            }
            for candidate in plan_candidates
            if candidate["classification"] == "legacy_current_generation"
            and candidate.get("integrity_failure")
        ]
        if legacy_integrity_failures and not updated.get("integrity_failure"):
            updated["integrity_failure"] = {
                "reason": "legacy_budget_identity_reconciliation",
                "sources": legacy_integrity_failures,
                "detected_at": reconciled_at,
            }
            changed = True
        for candidate in plan_candidates:
            if candidate["classification"] != "legacy_current_generation":
                continue
            fingerprint = candidate["source_fingerprint"]
            for kind in ("settled", "uncertain"):
                attempt_id = _legacy_reconciliation_attempt_id(fingerprint, kind)
                expected = _expected_legacy_attempt(
                    candidate, kind=kind, reconciled_at=reconciled_at
                )
                if expected is None:
                    continue
                recorded = updated["attempts"].get(attempt_id)
                if recorded is None:
                    updated["attempts"][attempt_id] = expected
                    changed = True
                elif not _legacy_attempt_matches(recorded, expected):
                    raise BudgetLedgerIdentityError(
                        f"Legacy reconciliation attempt {attempt_id!r} conflicts with "
                        "the existing canonical ledger"
                    )
        if not changed:
            break
        try:
            _save_ledger(
                episode_key,
                source_fingerprint,
                updated,
                if_generation_match=canonical_generation,
            )
        except PreconditionFailed:
            continue
        break
    else:
        raise BudgetConcurrencyError(
            "Canonical budget ledger changed repeatedly during identity reconciliation"
        )

    final_inventory = inventory_third_asr_budget_identities(
        episode_key, source_fingerprint
    )
    if final_inventory != inventory:
        raise BudgetIdentityReconciliationRequired(
            "Third-ASR legacy budget inventory changed during reconciliation"
        )

    payload = {
        "schema_version": THIRD_ASR_BUDGET_RECONCILIATION_SCHEMA_VERSION,
        "policy_version": THIRD_ASR_BUDGET_IDENTITY_POLICY_VERSION,
        "episode_key": episode_key,
        "source_fingerprint": source_fingerprint,
        "reconciled_at": reconciled_at,
        "candidates": plan_candidates,
    }
    try:
        _save_third_asr_budget_reconciliation(
            episode_key,
            source_fingerprint,
            payload,
            if_generation_match=marker_generation,
        )
    except PreconditionFailed as error:
        raise BudgetConcurrencyError(
            "Third-ASR budget reconciliation marker changed concurrently"
        ) from error

    require_third_asr_budget_identity_reconciled(episode_key, source_fingerprint)
    return {**plan, "status": "reconciled", "apply": True}


def spend_breakdown(ledger: dict) -> dict:
    """Pure, JSON-safe per-stage spend view of one loaded ledger (TASK-118).

    ``settled_usd`` is provider-reported actual cost; ``uncertain_usd`` is
    reservation retained after an ambiguous outcome and still counts
    against the cap; ``reserved_usd`` is an in-flight hold. Released
    attempts count zero but are included in ``attempts``.
    """

    stages: dict[str, dict] = {}
    totals = {"settled": Decimal("0"), "uncertain": Decimal("0"), "reserved": Decimal("0")}
    for attempt in ledger["attempts"].values():
        entry = stages.setdefault(
            attempt["stage"],
            {"attempts": 0, "settled": Decimal("0"), "uncertain": Decimal("0"), "reserved": Decimal("0")},
        )
        entry["attempts"] += 1
        state = attempt["state"]
        if state == "settled":
            amount, key = Decimal(attempt["settled_usd"]), "settled"
        elif state in {"uncertain", "reserved"}:
            amount, key = Decimal(attempt["reserved_usd"]), state
        else:
            continue
        entry[key] += amount
        totals[key] += amount

    def usd(value: Decimal) -> str:
        return str(value)

    committed = totals["settled"] + totals["uncertain"] + totals["reserved"]
    return {
        "episode_key": ledger["episode_key"],
        "source_fingerprint": ledger["source_fingerprint"],
        "hard_cap_usd": str(ledger.get("hard_cap_usd", EPISODE_AI_HARD_CAP_USD)),
        "settled_usd": usd(totals["settled"]),
        "uncertain_usd": usd(totals["uncertain"]),
        "reserved_usd": usd(totals["reserved"]),
        "committed_usd": usd(committed),
        "stages": {
            stage: {
                "attempts": entry["attempts"],
                "settled_usd": usd(entry["settled"]),
                "uncertain_usd": usd(entry["uncertain"]),
                "reserved_usd": usd(entry["reserved"]),
            }
            for stage, entry in sorted(stages.items())
        },
    }


def episode_spend_summary(episode_key: str, source_fingerprint: SourceFingerprint) -> dict:
    """Read-only per-stage spend for one source generation (see spend_breakdown)."""

    ledger, _generation = _load_ledger_with_generation(episode_key, source_fingerprint)
    return spend_breakdown(ledger)


def budget_summary(episode_key: str, source_fingerprint: SourceFingerprint) -> dict:
    """Return the current aggregated spend/headroom view for one generation.

    Read-only: performs no CAS write. A cache hit or replay that never calls
    this or ``reserve_budget_batch`` creates no new spend reservation.
    """

    ledger, _generation = _load_ledger_with_generation(episode_key, source_fingerprint)
    totals = _summary_from_ledger(ledger)
    total_spend = (
        totals["settled_spend"] + totals["uncertain_spend"] + totals["live_reservations"]
    )
    third_asr_spend = (
        totals["third_asr_settled"]
        + totals["third_asr_uncertain"]
        + totals["third_asr_live_reservations"]
    )
    return {
        **totals,
        "hard_cap_usd": EPISODE_AI_HARD_CAP_USD,
        "third_asr_subcap_usd": ASSISTED_THIRD_ASR_SUBCAP_USD,
        "headroom_usd": EPISODE_AI_HARD_CAP_USD - total_spend,
        "third_asr_headroom_usd": ASSISTED_THIRD_ASR_SUBCAP_USD - third_asr_spend,
    }


def _validate_reservation_requests(reservations: Iterable[dict]) -> list[dict]:
    reservations = list(reservations)
    if not reservations:
        raise ValueError("reserve_budget_batch requires at least one reservation request")

    normalized: list[dict] = []
    seen_ids: set[str] = set()
    for entry in reservations:
        if not isinstance(entry, dict):
            raise ValueError("Each reservation request must be a dict")

        attempt_id = entry.get("attempt_id")
        if not isinstance(attempt_id, str) or not attempt_id:
            raise ValueError("Each reservation request needs a non-empty attempt_id")
        if attempt_id in seen_ids:
            raise ValueError(f"Duplicate attempt_id within one reservation batch: {attempt_id!r}")
        seen_ids.add(attempt_id)

        stage = entry.get("stage")
        if stage not in AI_BUDGET_STAGE_IDS:
            raise ValueError(
                f"{attempt_id}: stage must be one of {sorted(AI_BUDGET_STAGE_IDS)}"
            )

        requested_third_asr = entry.get("third_asr", False)
        if not isinstance(requested_third_asr, bool):
            raise ValueError(f"{attempt_id}: third_asr must be a boolean")
        third_asr = stage == STAGE_THIRD_ASR or requested_third_asr

        reserved_usd = decimal_from_admission_input(
            entry.get("reserved_usd"), label=f"{attempt_id} reserved_usd"
        )
        if reserved_usd <= 0:
            raise ValueError(f"{attempt_id}: reserved_usd must be positive")

        normalized.append(
            {
                "attempt_id": attempt_id,
                "stage": stage,
                "third_asr": third_asr,
                "reserved_usd": reserved_usd,
            }
        )
    return normalized


def _attempt_matches_request(attempt: dict, request: dict) -> bool:
    return (
        attempt["stage"] == request["stage"]
        and bool(attempt.get("third_asr", False)) == request["third_asr"]
        and Decimal(attempt["reserved_usd"]) == request["reserved_usd"]
    )


def _reservation_result(ledger: dict, requests: list[dict]) -> dict:
    return {
        "admitted": True,
        "attempts": {
            request["attempt_id"]: copy.deepcopy(ledger["attempts"][request["attempt_id"]])
            for request in requests
        },
    }


def reserve_budget_batch(
    episode_key: str,
    source_fingerprint: str,
    reservations: Iterable[dict],
    *,
    required_downstream_reserve_usd: Decimal = Decimal("0"),
) -> dict:
    """Atomically admit one whole batch of reservations, or none of them.

    Each entry in ``reservations`` is a dict with ``attempt_id`` (immutable,
    idempotent -- a replayed attempt_id already on the ledger is not
    re-reserved, matching a request-level retry or cache-hit replay that
    performs no new provider call), ``stage`` (a short string identity),
    ``reserved_usd`` (a conservative upper-bound Decimal from
    ``podcast_engine.ai_pricing``), and an optional ``third_asr`` bool.

    ``required_downstream_reserve_usd`` protects still-required metered
    stages that have not yet run (Task 5 computes this from those stages'
    own conservative reservation contracts; it defaults to zero here since
    Task 4 does not itself plan downstream reserve).

    Concurrent callers reserving against the same episode generation
    contend on one CAS loop: only a batch that still fits atomically may
    proceed, so four workers racing to reserve simultaneously can never
    jointly exceed either cap.
    """

    requests = _validate_reservation_requests(reservations)
    downstream_reserve = decimal_from_admission_input(
        required_downstream_reserve_usd, label="required_downstream_reserve_usd"
    )

    for _ in range(MAX_BUDGET_CAS_ATTEMPTS):
        ledger, generation = _load_ledger_with_generation(episode_key, source_fingerprint)

        if ledger.get("integrity_failure"):
            raise BudgetIntegrityError(
                "Episode budget ledger has a recorded integrity failure; "
                "no further paid calls are admissible until it is reconciled"
            )

        existing_attempts = ledger["attempts"]
        new_requests = []
        for request in requests:
            recorded = existing_attempts.get(request["attempt_id"])
            if recorded is None:
                new_requests.append(request)
                continue
            if not _attempt_matches_request(recorded, request):
                raise BudgetAdmissionError(
                    f"Reservation replay for {request['attempt_id']!r} does not "
                    "match the attempt already recorded on the ledger"
                )

        if not new_requests:
            # A pure replay of already-admitted attempt_ids: no new spend,
            # no write.
            return _reservation_result(ledger, requests)

        totals = _summary_from_ledger(ledger)
        new_total = sum((request["reserved_usd"] for request in new_requests), Decimal("0"))
        projected_total = (
            totals["settled_spend"]
            + totals["uncertain_spend"]
            + totals["live_reservations"]
            + new_total
            + downstream_reserve
        )
        if projected_total > EPISODE_AI_HARD_CAP_USD:
            raise BudgetAdmissionError(
                f"Reserving {new_total} would bring the episode generation to "
                f"{projected_total}, exceeding the {EPISODE_AI_HARD_CAP_USD} hard cap"
            )

        new_third_asr_total = sum(
            (request["reserved_usd"] for request in new_requests if request["third_asr"]),
            Decimal("0"),
        )
        if new_third_asr_total > 0:
            projected_third_asr = (
                totals["third_asr_settled"]
                + totals["third_asr_uncertain"]
                + totals["third_asr_live_reservations"]
                + new_third_asr_total
            )
            if projected_third_asr > ASSISTED_THIRD_ASR_SUBCAP_USD:
                raise BudgetAdmissionError(
                    f"Reserving {new_third_asr_total} would bring assisted Third-ASR "
                    f"spend to {projected_third_asr}, exceeding the "
                    f"{ASSISTED_THIRD_ASR_SUBCAP_USD} sub-cap"
                )

        now = now_iso()
        for request in new_requests:
            existing_attempts[request["attempt_id"]] = {
                "state": "reserved",
                "stage": request["stage"],
                "third_asr": request["third_asr"],
                "reserved_usd": str(request["reserved_usd"]),
                "created_at": now,
                "updated_at": now,
            }

        try:
            _save_ledger(episode_key, source_fingerprint, ledger, if_generation_match=generation)
        except PreconditionFailed:
            continue
        return _reservation_result(ledger, requests)

    raise BudgetConcurrencyError(
        "Episode budget ledger changed repeatedly while reserving a batch"
    )


def settle_budget_attempt(
    episode_key: str,
    source_fingerprint: str,
    attempt_id: str,
    *,
    actual_usd: Decimal | None,
    reason: str | None = None,
) -> dict:
    """Settle one reserved (or previously uncertain) attempt.

    ``actual_usd=None`` means the caller could not obtain trustworthy
    provider-reported cost (missing, malformed, or untied to this request);
    the attempt remains counted as ``uncertain`` rather than released
    optimistically. A trustworthy ``actual_usd`` exceeding the attempt's own
    reservation is a budget-integrity failure: it is durably recorded on the
    ledger and blocks every further reservation for this episode generation
    before BudgetIntegrityError is raised.
    """

    if actual_usd is not None:
        actual_usd = decimal_from_admission_input(actual_usd, label="actual_usd")

    for _ in range(MAX_BUDGET_CAS_ATTEMPTS):
        ledger, generation = _load_ledger_with_generation(episode_key, source_fingerprint)
        attempt = _existing_attempt(ledger, attempt_id)
        if attempt["state"] not in _SETTLEABLE_STATES:
            raise ValueError(
                f"Cannot settle attempt {attempt_id!r} in state {attempt['state']!r}"
            )

        reserved = Decimal(attempt["reserved_usd"])
        now = now_iso()
        integrity_violation = actual_usd is not None and actual_usd > reserved

        if actual_usd is None:
            attempt["state"] = "uncertain"
            attempt["updated_at"] = now
            if reason:
                attempt["reason"] = reason
        elif integrity_violation:
            attempt["state"] = "uncertain"
            attempt["updated_at"] = now
            ledger["integrity_failure"] = {
                "attempt_id": attempt_id,
                "reserved_usd": str(reserved),
                "actual_usd": str(actual_usd),
                "detected_at": now,
            }
        else:
            attempt["state"] = "settled"
            attempt["settled_usd"] = str(actual_usd)
            attempt["updated_at"] = now

        try:
            _save_ledger(episode_key, source_fingerprint, ledger, if_generation_match=generation)
        except PreconditionFailed:
            continue

        if integrity_violation:
            raise BudgetIntegrityError(
                f"Attempt {attempt_id!r} actual cost {actual_usd} exceeded its "
                f"reservation {reserved}; episode generation budget requires "
                "reconciliation"
            )
        return copy.deepcopy(attempt)

    raise BudgetConcurrencyError(
        "Episode budget ledger changed repeatedly while settling an attempt"
    )


WHISPER_INTEGRITY_RECONCILIABLE_STAGES = frozenset({STAGE_WHISPER_TRANSCRIPTION})


def inventory_budget_integrity_failures(episode_key: str) -> list[dict]:
    """List this episode's ledgers that carry an unreconciled integrity failure.

    Operator-only: it lists the episode's budget prefix, which production
    runtime code never does.
    """

    if not isinstance(episode_key, str) or not _EPISODE_KEY.fullmatch(episode_key):
        raise ValueError("episode_key must be a canonical 24-hex-character episode key")
    rows = []
    for blob in get_bucket().list_blobs(prefix=f"episodes/{episode_key}/ai/budgets/"):
        ledger, _generation = _read_budget_blob(blob, episode_key)
        failure = ledger.get("integrity_failure")
        if not failure:
            continue
        attempt = ledger["attempts"].get(failure.get("attempt_id"), {})
        rows.append(
            {
                "episode_key": episode_key,
                "source_fingerprint": ledger["source_fingerprint"],
                "attempt_id": failure.get("attempt_id"),
                "stage": attempt.get("stage"),
                "state": attempt.get("state"),
                "reserved_usd": failure.get("reserved_usd"),
                "actual_usd": failure.get("actual_usd"),
                "detected_at": failure.get("detected_at"),
            }
        )
    rows.sort(key=lambda row: row["source_fingerprint"])
    return rows


def reconcile_whisper_integrity_failure(
    episode_key: str,
    source_fingerprint: str,
    *,
    attempt_id: str,
    actual_usd: Decimal,
    reason: str,
) -> dict:
    """Settle one Whisper over-reservation at its recorded actual cost (TASK-125).

    A Whisper transcription ledger is keyed by the audio fingerprint, so an
    integrity failure there blocks only that episode's re-transcription. The
    operator must restate the exact recorded attempt and actual cost; any
    mismatch, another stage, or a non-uncertain attempt fails closed. The
    attempt becomes settled at the provider-reported actual cost, and the
    failure moves to an append-only ``integrity_reconciliations`` list with the
    operator's reason, so the overrun stays visible and counts in full.
    """

    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reconciliation requires a non-empty operator reason")
    stated_actual = decimal_from_admission_input(actual_usd, label="actual_usd")
    for _ in range(MAX_BUDGET_CAS_ATTEMPTS):
        ledger, generation = _load_ledger_with_generation(episode_key, source_fingerprint)
        failure = ledger.get("integrity_failure")
        if not failure:
            raise BudgetLedgerError("ledger has no unreconciled integrity failure")
        if failure.get("attempt_id") != attempt_id:
            raise BudgetLedgerError("stated attempt does not match the recorded integrity failure")
        if Decimal(str(failure.get("actual_usd"))) != stated_actual:
            raise BudgetLedgerError("stated actual cost does not match the recorded integrity failure")
        attempt = _existing_attempt(ledger, attempt_id)
        if attempt.get("stage") not in WHISPER_INTEGRITY_RECONCILIABLE_STAGES:
            raise BudgetLedgerError("only Whisper transcription integrity failures are reconcilable here")
        if attempt["state"] != "uncertain":
            raise BudgetLedgerError("the failed attempt is not in the uncertain state")

        now = now_iso()
        attempt["state"] = "settled"
        attempt["settled_usd"] = str(stated_actual)
        attempt["updated_at"] = now
        reconciliations = list(ledger.get("integrity_reconciliations") or [])
        reconciliations.append(
            {**copy.deepcopy(failure), "reconciled_at": now, "reason": reason.strip()}
        )
        ledger["integrity_reconciliations"] = reconciliations
        del ledger["integrity_failure"]
        try:
            _save_ledger(episode_key, source_fingerprint, ledger, if_generation_match=generation)
        except PreconditionFailed:
            continue
        return copy.deepcopy(attempt)
    raise BudgetConcurrencyError(
        "Episode budget ledger changed repeatedly while reconciling an integrity failure"
    )


def release_budget_attempt_pre_send(
    episode_key: str, source_fingerprint: str, attempt_id: str
) -> dict:
    """Release one reservation Python can prove was never sent to a provider.

    Requires only the ledger and this attempt_id -- no other object (a
    prepare-session lease, a per-item claim) needs to exist or have been
    written for this to succeed, so a reservation is always independently
    reconcilable rather than depending on some other record's fate.
    """

    for _ in range(MAX_BUDGET_CAS_ATTEMPTS):
        ledger, generation = _load_ledger_with_generation(episode_key, source_fingerprint)
        attempt = _existing_attempt(ledger, attempt_id)
        if attempt["state"] != "reserved":
            raise ValueError(
                f"Cannot pre-send release attempt {attempt_id!r} in state "
                f"{attempt['state']!r}"
            )
        attempt["state"] = "released"
        attempt["updated_at"] = now_iso()
        try:
            _save_ledger(episode_key, source_fingerprint, ledger, if_generation_match=generation)
        except PreconditionFailed:
            continue
        return copy.deepcopy(attempt)

    raise BudgetConcurrencyError(
        "Episode budget ledger changed repeatedly while releasing an attempt"
    )


def mark_budget_attempt_uncertain(
    episode_key: str, source_fingerprint: str, attempt_id: str, *, reason: str
) -> dict:
    """Mark one reserved attempt uncertain after a post-send ambiguous failure.

    A timeout, broken connection after send, or other state where provider
    execution cannot be disproved leaves the reservation counted, not
    released -- it continues consuming the cap until settled or reconciled.
    """

    if not isinstance(reason, str) or not reason:
        raise ValueError("reason is required to mark a budget attempt uncertain")

    for _ in range(MAX_BUDGET_CAS_ATTEMPTS):
        ledger, generation = _load_ledger_with_generation(episode_key, source_fingerprint)
        attempt = _existing_attempt(ledger, attempt_id)
        if attempt["state"] != "reserved":
            raise ValueError(
                f"Cannot mark attempt {attempt_id!r} uncertain from state "
                f"{attempt['state']!r}"
            )
        attempt["state"] = "uncertain"
        attempt["reason"] = reason
        attempt["updated_at"] = now_iso()
        try:
            _save_ledger(episode_key, source_fingerprint, ledger, if_generation_match=generation)
        except PreconditionFailed:
            continue
        return copy.deepcopy(attempt)

    raise BudgetConcurrencyError(
        "Episode budget ledger changed repeatedly while marking an attempt uncertain"
    )


def required_downstream_reserve(
    *,
    stage_pending: dict,
    stage_reservation_usd: dict,
) -> Decimal:
    """Sum the provable worst-case reservation for still-pending mandatory
    downstream AI stages -- the money optional assisted evidence (Third
    ASR) must never be allowed to consume.

    Pure: no I/O, no network side effects, and no internal memoization --
    every call recomputes from scratch against exactly the stage state it
    is given, so a recompile or a human mutation that flips a previously
    cached/settled stage back to pending is reflected correctly on the
    very next call with no stale state left over from an earlier one.

    ``stage_pending`` must supply a bool for every stage in
    _DOWNSTREAM_RESERVE_ELIGIBLE_STAGES (transcript resolver/triage,
    summary, metadata, summary review, note writer) -- no implicit
    default, and no ``third_asr`` key, since Third ASR is optional evidence, never a
    protected downstream stage. A stage that is not pending (whether it
    was never applicable, or already produced current output for this
    generation) contributes zero and needs no entry in
    ``stage_reservation_usd`` at all.

    A pending stage's own contribution must be supplied by the caller as
    an already-derived non-negative Decimal upper-bound reservation (see
    resolve_required_downstream_reserve_usd and each knowledge-stage
    module's own worst-case payload builder) -- this function never
    guesses a percentage of the cap or of remaining headroom for a stage
    whose real bound it was not given; a missing or malformed bound for a
    pending stage fails closed rather than silently treating it as zero.
    """

    if not isinstance(stage_pending, dict):
        raise DownstreamReserveError("stage_pending must be a dict")

    given_keys = set(stage_pending)
    missing = _DOWNSTREAM_RESERVE_ELIGIBLE_STAGES - given_keys
    if missing:
        raise DownstreamReserveError(
            f"stage_pending is missing required stage ids: {sorted(missing)}"
        )
    unexpected = given_keys - _DOWNSTREAM_RESERVE_ELIGIBLE_STAGES
    if unexpected:
        raise DownstreamReserveError(
            f"stage_pending contains stage ids that are never downstream-"
            f"reserve-eligible (third_asr is optional evidence, not "
            f"protected downstream work): {sorted(unexpected)}"
        )

    total = Decimal("0")
    for stage in _DOWNSTREAM_RESERVE_ELIGIBLE_STAGES:
        pending = stage_pending[stage]
        if not isinstance(pending, bool):
            raise DownstreamReserveError(
                f"stage_pending[{stage!r}] must be a bool"
            )
        if not pending:
            continue

        if stage not in stage_reservation_usd:
            raise DownstreamReserveError(
                f"stage {stage!r} is pending but no reservation bound was "
                "supplied in stage_reservation_usd"
            )
        bound = stage_reservation_usd[stage]
        if not isinstance(bound, Decimal) or isinstance(bound, bool):
            raise DownstreamReserveError(
                f"stage {stage!r} reservation bound must be a Decimal, "
                f"never a guessed float/percentage: got {type(bound).__name__}"
            )
        if bound < 0:
            raise DownstreamReserveError(
                f"stage {stage!r} reservation bound must be non-negative"
            )
        total += bound

    return total


def _resolve_stage_bound_or_fail_closed(
    slug: str,
    worst_case_payload_builder: Callable[[PresetProvenance], dict],
    *,
    api_key: str,
    transport: Callable[[str, str], list] | None,
    verified_at: str | None,
    now: Callable[[], str] | None,
    call_multiplier: int = 1,
) -> Decimal:
    """Resolve one stage's current preset once, then build ITS worst-case
    payload from that exact same resolved snapshot.

    TASK-076 Task 7: the real per-call payload for summary/metadata/summary
    review is built directly from a resolved PresetProvenance's own config
    (never the mutable ``@preset/<slug>`` alias) -- so a worst-case payload
    built without that same snapshot would measure a stale, shorter shape
    and under-reserve. ``worst_case_payload_builder`` is called with the
    preset this function already resolved, reusing the identical network
    round trip rather than requiring (or risking a mismatch with) a second,
    separately-resolved snapshot.
    """

    preset = fetch_current_designated_preset(slug, api_key=api_key, verified_at=verified_at)
    if not isinstance(preset, PresetProvenance) or not preset.verified:
        raise DownstreamReserveError(
            f"downstream stage preset {slug!r} could not be resolved with "
            "current trusted pricing evidence; failing closed rather than "
            "admitting an optional purchase against an unprovable reserve"
        )
    resolved = resolve_preset_model_and_pricing(
        preset, api_key=api_key, transport=transport, now=now
    )
    worst_case_payload = worst_case_payload_builder(preset)
    per_call = resolve_stage_reservation_usd(
        resolved_preset=resolved,
        worst_case_request_bytes=json.dumps(
            worst_case_payload, ensure_ascii=False
        ).encode("utf-8"),
    )
    return per_call * call_multiplier


def resolve_required_downstream_reserve_usd(
    *,
    stage_pending: Mapping[str, bool],
    api_key: str,
    transport: Callable[[str, str], list] | None = None,
    verified_at: str | None = None,
    now: Callable[[], str] | None = None,
) -> Decimal:
    """Network-resolving counterpart to required_downstream_reserve.

    Resolves live current pricing/preset evidence only for the stages this
    orchestrator knows how to bound -- summary, metadata, summary review
    and the TASK-118 note writer -- and only when the caller marks them
    pending. Transcript
    resolver and triage are always treated as already-settled here: by
    the time downstream-reserve protection matters (assisted Third ASR's
    "Prepare assisted evidence", during Human Review), resolver and
    triage have already run for this episode generation, so they are
    never pending at this call site. This function does not guess their
    bound if a future caller ever marks them pending; it fails closed
    instead (see the check below), same as an unresolvable preset.

    A stage inactive for this deployment (summary review with no
    SUMMARY_REVIEW_PRESET_ENV configured, or the note writer with no
    NOTE_WRITER_PRESET_ENV configured) contributes zero without any
    network call, exactly like an already-cached/settled stage -- both
    are simply "not pending" from the pure planner's point of view.

    Any resolution failure (unverified/stale/malformed preset or pricing
    evidence) raises DownstreamReserveError rather than silently
    contributing zero: an unprovable downstream bound must block the
    optional purchase it would otherwise fail to protect against, not be
    treated as "no money needed".
    """

    # TASK-076 Task 7: imported locally, not at module level -- these
    # knowledge modules import stage identifiers back from this module
    # (STAGE_SUMMARY/STAGE_METADATA/STAGE_SUMMARY_REVIEW), so a top-level
    # import here would be circular. This function is the only caller.
    from .knowledge import metadata as knowledge_metadata
    from .knowledge import summary as knowledge_summary
    from .knowledge import summary_review as knowledge_summary_review

    stage_pending = dict(stage_pending)
    if stage_pending.get(STAGE_RESOLVER) or stage_pending.get(STAGE_TRIAGE):
        raise DownstreamReserveError(
            "resolve_required_downstream_reserve_usd cannot resolve a "
            "pending resolver/triage bound; they must already be settled "
            "by the time downstream-reserve protection is checked"
        )
    stage_pending.setdefault(STAGE_RESOLVER, False)
    stage_pending.setdefault(STAGE_TRIAGE, False)

    stage_reservation_usd: dict[str, Decimal] = {}

    if stage_pending.get(STAGE_SUMMARY, False):
        stage_reservation_usd[STAGE_SUMMARY] = _resolve_stage_bound_or_fail_closed(
            knowledge_summary.summary_preset(),
            knowledge_summary.worst_case_openrouter_payload,
            api_key=api_key,
            transport=transport,
            verified_at=verified_at,
            now=now,
        )

    if stage_pending.get(STAGE_METADATA, False):
        stage_reservation_usd[STAGE_METADATA] = _resolve_stage_bound_or_fail_closed(
            knowledge_metadata.metadata_preset(),
            knowledge_metadata.worst_case_openrouter_payload,
            api_key=api_key,
            transport=transport,
            verified_at=verified_at,
            now=now,
        )

    if stage_pending.get(STAGE_SUMMARY_REVIEW, False):
        active_slug = knowledge_summary_review.active_summary_review_preset()
        if active_slug is None:
            # Not activated for this deployment: same as "not pending" for
            # the pure planner -- no network call, contributes zero.
            stage_pending[STAGE_SUMMARY_REVIEW] = False
        else:
            stage_reservation_usd[STAGE_SUMMARY_REVIEW] = (
                _resolve_stage_bound_or_fail_closed(
                    active_slug,
                    knowledge_summary_review.worst_case_openrouter_payload,
                    api_key=api_key,
                    transport=transport,
                    verified_at=verified_at,
                    now=now,
                    call_multiplier=knowledge_summary_review.WORST_CASE_MODEL_CALLS_PER_CASE,
                )
            )

    if stage_pending.get(STAGE_NOTE_WRITER, False):
        from .knowledge.notes_v3 import writer as knowledge_note_writer

        writer_slug = knowledge_note_writer.note_writer_preset()
        if writer_slug is None:
            # Not activated for this deployment: contributes zero, exactly
            # like an inactive summary reviewer above.
            stage_pending[STAGE_NOTE_WRITER] = False
        else:
            stage_reservation_usd[STAGE_NOTE_WRITER] = (
                _resolve_stage_bound_or_fail_closed(
                    writer_slug,
                    knowledge_note_writer.worst_case_openrouter_payload,
                    api_key=api_key,
                    transport=transport,
                    verified_at=verified_at,
                    now=now,
                )
            )

    return required_downstream_reserve(
        stage_pending=stage_pending, stage_reservation_usd=stage_reservation_usd
    )
