"""Crash-safe paid-attempt journal for Summary Review Benchmark v2."""

from __future__ import annotations

import hashlib
import json
import os
import fcntl
import stat
import tempfile
from contextlib import contextmanager
from decimal import Decimal, InvalidOperation
from functools import wraps
from pathlib import Path


JOURNAL_FORMAT = 2


def _sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Benchmark v2 execution journal value must be canonical JSON"
        ) from error


def _require_path(path: Path) -> Path:
    journal_path = Path(path)
    if journal_path.is_symlink():
        raise ValueError(
            "Benchmark v2 execution journal path must not be a symlink"
        )
    if journal_path.exists() and not journal_path.is_file():
        raise ValueError(
            "Benchmark v2 execution journal path must be a file"
        )
    return journal_path


@contextmanager
def _exclusive_journal_lock(path: Path):
    """Serialize one journal read-modify-write transaction across processes."""

    journal_path = _require_path(path)
    try:
        journal_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise ValueError(
            "Benchmark v2 execution journal directory is unavailable"
        ) from error

    lock_path = journal_path.with_name(journal_path.name + ".lock")
    if lock_path.is_symlink():
        raise ValueError(
            "Benchmark v2 execution journal lock path must not be a symlink"
        )

    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW

    descriptor = None
    locked = False
    try:
        descriptor = os.open(lock_path, flags, 0o600)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(
                "Benchmark v2 execution journal lock path must be a regular file"
            )
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        locked = True
        yield
    except OSError as error:
        raise ValueError(
            "Benchmark v2 execution journal lock is unavailable"
        ) from error
    finally:
        if descriptor is not None:
            if locked:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                except OSError:
                    pass
            os.close(descriptor)


def _locked_journal_mutation(function):
    @wraps(function)
    def wrapped(path: Path, *args, **kwargs):
        with _exclusive_journal_lock(Path(path)):
            return function(path, *args, **kwargs)

    return wrapped


def _parse_decimal(name: str, value: object) -> Decimal:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"Benchmark v2 execution journal {name} is invalid")
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise ValueError(
            f"Benchmark v2 execution journal {name} is invalid"
        ) from error
    if not parsed.is_finite() or parsed < 0:
        raise ValueError(
            f"Benchmark v2 execution journal {name} is invalid"
        )
    return parsed


def _validate_ordered_case_keys(ordered_case_keys: object) -> list[str]:
    if (
        not isinstance(ordered_case_keys, list)
        or not ordered_case_keys
        or any(
            not isinstance(case_key, str) or not case_key
            for case_key in ordered_case_keys
        )
        or len(set(ordered_case_keys)) != len(ordered_case_keys)
    ):
        raise ValueError(
            "Benchmark v2 execution journal requires canonical ordered case keys"
        )
    return list(ordered_case_keys)


def _validate_binding(benchmark_binding: object) -> dict:
    if not isinstance(benchmark_binding, dict) or not benchmark_binding:
        raise ValueError(
            "Benchmark v2 execution journal requires benchmark binding"
        )
    _canonical_json_bytes(benchmark_binding)
    return benchmark_binding


def _validate_attempt(attempt: object, ordered_case_keys: list[str]) -> dict:
    if not isinstance(attempt, dict):
        raise ValueError(
            "Benchmark v2 execution journal attempt must be an object"
        )

    case_key = attempt.get("case_key")
    phase = attempt.get("phase")
    attempt_number = attempt.get("attempt_number")
    request_sha256 = attempt.get("request_sha256")
    reservation_usd = attempt.get("reservation_usd")
    status = attempt.get("status")

    if (
        case_key not in ordered_case_keys
        or not isinstance(phase, str)
        or not phase
        or isinstance(attempt_number, bool)
        or not isinstance(attempt_number, int)
        or attempt_number < 1
        or not isinstance(request_sha256, str)
        or not request_sha256.startswith("sha256:")
        or len(request_sha256) != 71
        or status not in {
            "reserved",
            "completed",
            "cost_unknown",
            "cost_exceeds_reservation",
            "transport_failed",
        }
    ):
        raise ValueError(
            "Benchmark v2 execution journal attempt identity is invalid"
        )

    _parse_decimal("attempt reservation", reservation_usd)

    if status in {"completed", "cost_exceeds_reservation"}:
        response_sha256 = attempt.get("response_sha256")
        response_payload = attempt.get("response_payload")
        cost_usd = attempt.get("cost_usd")
        latency_ms = attempt.get("latency_ms")
        if (
            not isinstance(response_sha256, str)
            or not response_sha256.startswith("sha256:")
            or len(response_sha256) != 71
            or not isinstance(response_payload, dict)
            or isinstance(latency_ms, bool)
            or not isinstance(latency_ms, int)
            or latency_ms < 0
        ):
            raise ValueError(
                "Benchmark v2 costed journal attempt is malformed"
            )
        _parse_decimal("attempt cost", cost_usd)
        _canonical_json_bytes(response_payload)

    if status == "transport_failed":
        error_class = attempt.get("error_class")
        error_message = attempt.get("error_message")
        latency_ms = attempt.get("latency_ms")
        if (
            not isinstance(error_class, str)
            or not error_class
            or len(error_class) > 256
            or not isinstance(error_message, str)
            or len(error_message) > 2048
            or isinstance(latency_ms, bool)
            or not isinstance(latency_ms, int)
            or latency_ms < 0
        ):
            raise ValueError(
                "Benchmark v2 transport-failed journal attempt is malformed"
            )

    if status == "cost_unknown":
        response_sha256 = attempt.get("response_sha256")
        response_payload = attempt.get("response_payload")
        latency_ms = attempt.get("latency_ms")
        if (
            not isinstance(response_sha256, str)
            or not response_sha256.startswith("sha256:")
            or len(response_sha256) != 71
            or not isinstance(response_payload, dict)
            or isinstance(latency_ms, bool)
            or not isinstance(latency_ms, int)
            or latency_ms < 0
        ):
            raise ValueError(
                "Benchmark v2 cost-unknown journal attempt is malformed"
            )
        _canonical_json_bytes(response_payload)

    return attempt


def _validate_failed_case(
    failed_case: object,
    ordered_case_keys: list[str],
) -> dict:
    required_fields = {
        "case_key",
        "outcome",
        "stage",
        "failure_code",
        "attempt_count",
        "last_error_class",
        "last_error_message",
    }
    if not isinstance(failed_case, dict) or set(failed_case) != required_fields:
        raise ValueError(
            "Benchmark v2 execution journal failed case evidence is malformed"
        )

    case_key = failed_case["case_key"]
    stage = failed_case["stage"]
    failure_code = failed_case["failure_code"]
    attempt_count = failed_case["attempt_count"]
    error_class = failed_case["last_error_class"]
    error_message = failed_case["last_error_message"]
    if (
        case_key not in ordered_case_keys
        or failed_case["outcome"] != "model_failed"
        or stage not in {"audit", "edit"}
        or failure_code != f"{stage}_contract_exhausted"
        or isinstance(attempt_count, bool)
        or not isinstance(attempt_count, int)
        or attempt_count < 1
        or attempt_count > 3
        or not isinstance(error_class, str)
        or not error_class
        or len(error_class) > 256
        or not isinstance(error_message, str)
        or len(error_message) > 2048
    ):
        raise ValueError(
            "Benchmark v2 execution journal failed case evidence is invalid"
        )
    return failed_case


def _validate_state(
    state: object,
    *,
    benchmark_binding: dict | None = None,
    ordered_case_keys: list[str] | None = None,
) -> dict:
    if not isinstance(state, dict):
        raise ValueError(
            "Benchmark v2 execution journal must be a JSON object"
        )
    if state.get("journal_format") != JOURNAL_FORMAT:
        raise ValueError(
            "Benchmark v2 execution journal format is unsupported"
        )

    stored_binding = _validate_binding(state.get("benchmark_binding"))
    stored_case_keys = _validate_ordered_case_keys(
        state.get("ordered_case_keys")
    )

    if benchmark_binding is not None and stored_binding != benchmark_binding:
        raise ValueError(
            "Benchmark v2 execution journal benchmark binding conflicts"
        )
    if ordered_case_keys is not None and stored_case_keys != ordered_case_keys:
        raise ValueError(
            "Benchmark v2 execution journal ordered case keys conflict"
        )

    spent_usd = _parse_decimal("spent_usd", state.get("spent_usd"))
    completed_case_keys = state.get("completed_case_keys")
    if (
        not isinstance(completed_case_keys, list)
        or any(
            case_key not in stored_case_keys
            for case_key in completed_case_keys
        )
        or len(set(completed_case_keys)) != len(completed_case_keys)
    ):
        raise ValueError(
            "Benchmark v2 execution journal completed case state is invalid"
        )

    failed_case_keys = state.get("failed_case_keys")
    if (
        not isinstance(failed_case_keys, list)
        or any(case_key not in stored_case_keys for case_key in failed_case_keys)
        or len(set(failed_case_keys)) != len(failed_case_keys)
    ):
        raise ValueError(
            "Benchmark v2 execution journal failed case state is invalid"
        )
    if set(completed_case_keys) & set(failed_case_keys):
        raise ValueError(
            "Benchmark v2 execution journal completed and failed case keys overlap"
        )

    attempts = state.get("attempts")
    if not isinstance(attempts, list):
        raise ValueError(
            "Benchmark v2 execution journal attempts must be a list"
        )

    seen_attempts = set()
    completed_cost = Decimal("0")
    for attempt in attempts:
        _validate_attempt(attempt, stored_case_keys)
        identity = (
            attempt["case_key"],
            attempt["phase"],
            attempt["attempt_number"],
        )
        if identity in seen_attempts:
            raise ValueError(
                "Benchmark v2 execution journal attempt identity is duplicated"
            )
        seen_attempts.add(identity)
        if attempt["status"] in {
            "completed",
            "cost_exceeds_reservation",
        }:
            completed_cost += _parse_decimal(
                "attempt cost",
                attempt["cost_usd"],
            )

    if spent_usd != completed_cost:
        raise ValueError(
            "Benchmark v2 execution journal spent_usd disagrees with completed attempts"
        )

    completed_cases = state.get("completed_cases")
    if not isinstance(completed_cases, list):
        raise ValueError(
            "Benchmark v2 execution journal completed case evidence must be a list"
        )

    completed_evidence_keys = []
    for completed_case in completed_cases:
        case_key = (
            completed_case.get("case_key")
            if isinstance(completed_case, dict)
            else None
        )
        if (
            not isinstance(completed_case, dict)
            or set(completed_case)
            != {"case_key", "case_result", "execution_evidence"}
            or case_key not in completed_case_keys
            or not isinstance(completed_case.get("case_result"), dict)
            or not isinstance(completed_case.get("execution_evidence"), dict)
        ):
            raise ValueError(
                "Benchmark v2 execution journal completed case evidence is invalid"
            )
        completed_evidence_keys.append(case_key)
    if (
        len(set(completed_evidence_keys)) != len(completed_evidence_keys)
        or set(completed_evidence_keys) != set(completed_case_keys)
    ):
        raise ValueError(
            "Benchmark v2 execution journal completed case evidence is incomplete"
        )

    failed_cases = state.get("failed_cases")
    if not isinstance(failed_cases, list):
        raise ValueError(
            "Benchmark v2 execution journal failed case evidence must be a list"
        )
    failed_evidence_keys = []
    for failed_case in failed_cases:
        validated_failure = _validate_failed_case(
            failed_case,
            stored_case_keys,
        )
        case_key = validated_failure["case_key"]
        if case_key not in failed_case_keys:
            raise ValueError(
                "Benchmark v2 execution journal failed case evidence is invalid"
            )
        case_attempts = sorted(
            (
                attempt
                for attempt in attempts
                if attempt["case_key"] == case_key
            ),
            key=lambda attempt: attempt["attempt_number"],
        )
        if (
            len(case_attempts) != validated_failure["attempt_count"]
            or any(
                attempt["status"] != "completed"
                for attempt in case_attempts
            )
            or [attempt["attempt_number"] for attempt in case_attempts]
            != list(range(1, len(case_attempts) + 1))
            or case_attempts[-1]["phase"] != validated_failure["stage"]
        ):
            raise ValueError(
                "Benchmark v2 execution journal failed case attempts are invalid"
            )
        failed_evidence_keys.append(case_key)
    if (
        len(set(failed_evidence_keys)) != len(failed_evidence_keys)
        or set(failed_evidence_keys) != set(failed_case_keys)
    ):
        raise ValueError(
            "Benchmark v2 execution journal failed case evidence is incomplete"
        )

    return state


def _read_state(path: Path) -> dict:
    journal_path = _require_path(path)
    if not journal_path.exists():
        raise ValueError(
            "Benchmark v2 execution journal is unavailable"
        )
    try:
        journal_bytes = journal_path.read_bytes()
    except OSError as error:
        raise ValueError(
            "Benchmark v2 execution journal is unreadable"
        ) from error
    try:
        state = json.loads(journal_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            "Benchmark v2 execution journal must be valid UTF-8 JSON"
        ) from error
    if journal_bytes != _canonical_json_bytes(state):
        raise ValueError(
            "Benchmark v2 execution journal bytes must be canonical JSON"
        )
    return _validate_state(state)


def _fsync_parent_directory(path: Path) -> None:
    try:
        directory_fd = os.open(path.parent, os.O_RDONLY)
    except OSError:
        return
    try:
        try:
            os.fsync(directory_fd)
        except OSError:
            pass
    finally:
        os.close(directory_fd)


def _atomic_write(path: Path, state: dict) -> None:
    journal_path = _require_path(path)
    try:
        journal_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise ValueError(
            "Benchmark v2 execution journal directory is unavailable"
        ) from error

    if journal_path.is_symlink():
        raise ValueError(
            "Benchmark v2 execution journal path must not be a symlink"
        )

    journal_bytes = _canonical_json_bytes(state)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=journal_path.parent,
            prefix=f".{journal_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(journal_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, journal_path)
        temporary_path = None
        _fsync_parent_directory(journal_path)
    except OSError as error:
        raise ValueError(
            "Benchmark v2 execution journal could not be written atomically"
        ) from error
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except OSError:
                pass


@_locked_journal_mutation
def load_or_initialize_execution_journal(
    path: Path,
    *,
    benchmark_binding: dict,
    ordered_case_keys: list[str],
) -> dict:
    journal_path = _require_path(path)
    binding = _validate_binding(benchmark_binding)
    case_keys = _validate_ordered_case_keys(ordered_case_keys)

    if journal_path.exists():
        state = _read_state(journal_path)
        return _validate_state(
            state,
            benchmark_binding=binding,
            ordered_case_keys=case_keys,
        )

    state = {
        "journal_format": JOURNAL_FORMAT,
        "benchmark_binding": binding,
        "ordered_case_keys": case_keys,
        "spent_usd": "0.00",
        "completed_case_keys": [],
        "failed_case_keys": [],
        "attempts": [],
        "completed_cases": [],
        "failed_cases": [],
    }
    _validate_state(
        state,
        benchmark_binding=binding,
        ordered_case_keys=case_keys,
    )
    _atomic_write(journal_path, state)
    return state


def seed_execution_journal_from_retired_epoch(
    path: Path,
    *,
    retired_journal_bytes: bytes,
    benchmark_binding: dict,
    ordered_case_keys: list[str],
) -> dict:
    """Seed a fresh execution journal from a previously retired epoch's
    completed-case evidence, so a new epoch does not have to re-pay for
    cases that already passed before an unrelated case forced retirement.

    Refuses (raises ``ValueError``) rather than silently doing something
    partial or stale:

    - if a live journal already exists at ``path`` -- this never
      overwrites an active or already-seeded epoch; retire it first;
    - if ``retired_journal_bytes`` is not a well-formed, internally
      consistent execution journal (the exact bytes of a
      ``history/*/execution-state.json`` archive member);
    - if the retired journal's own ``benchmark_binding`` or
      ``ordered_case_keys`` do not exactly match the ones passed in --
      a stale or differently-configured archive is discarded whole,
      never partially applied.

    Both durable terminal outcomes -- the cases already in the retired
    journal's own ``completed_case_keys`` and those in its
    ``failed_case_keys`` -- are carried forward, together with just the
    attempts that back them. A ``model_failed`` case is resolved
    exactly like a completed one: it must never be recycled with a
    fresh attempt budget, so it moves forward as-is rather than being
    dropped. Only a case that never reached either terminal outcome
    (the one that actually forced retirement, an integrity failure) is
    dropped entirely, so it gets a full, untouched retry budget in the
    new epoch. ``spent_usd`` is recomputed from the retained attempts
    only -- honest bookkeeping that simply excludes the spend already
    written off by the epoch's own retirement.
    """
    journal_path = _require_path(path)
    if journal_path.exists() or journal_path.is_symlink():
        raise ValueError(
            "Benchmark v2 execution journal seed refuses an active journal"
        )

    binding = _validate_binding(benchmark_binding)
    case_keys = _validate_ordered_case_keys(ordered_case_keys)

    if not isinstance(retired_journal_bytes, bytes):
        raise ValueError(
            "Benchmark v2 retired execution journal must be bytes"
        )
    try:
        retired_state = json.loads(retired_journal_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            "Benchmark v2 retired execution journal must be valid UTF-8 JSON"
        ) from error
    if retired_journal_bytes != _canonical_json_bytes(retired_state):
        raise ValueError(
            "Benchmark v2 retired execution journal bytes must be canonical JSON"
        )
    retired_state = _validate_state(retired_state)

    if retired_state["benchmark_binding"] != binding:
        raise ValueError(
            "Benchmark v2 execution journal seed refuses a mismatched benchmark binding"
        )
    if retired_state["ordered_case_keys"] != case_keys:
        raise ValueError(
            "Benchmark v2 execution journal seed refuses mismatched ordered case keys"
        )

    completed_case_keys = list(retired_state["completed_case_keys"])
    completed_case_key_set = set(completed_case_keys)
    failed_case_keys = list(retired_state["failed_case_keys"])
    failed_case_key_set = set(failed_case_keys)
    resolved_case_key_set = completed_case_key_set | failed_case_key_set

    completed_cases = [
        dict(entry)
        for entry in retired_state["completed_cases"]
        if entry.get("case_key") in completed_case_key_set
    ]
    if {entry["case_key"] for entry in completed_cases} != completed_case_key_set:
        raise ValueError(
            "Benchmark v2 retired execution journal completed case evidence is incomplete"
        )

    failed_cases = [
        dict(entry)
        for entry in retired_state["failed_cases"]
        if entry.get("case_key") in failed_case_key_set
    ]
    if {entry["case_key"] for entry in failed_cases} != failed_case_key_set:
        raise ValueError(
            "Benchmark v2 retired execution journal failed case evidence is incomplete"
        )

    retained_attempts = [
        attempt
        for attempt in retired_state["attempts"]
        if attempt["case_key"] in resolved_case_key_set
    ]

    spent_usd = sum(
        (
            _parse_decimal("attempt cost", attempt["cost_usd"])
            for attempt in retained_attempts
            if attempt["status"] in {"completed", "cost_exceeds_reservation"}
        ),
        Decimal("0"),
    )

    state = {
        "journal_format": JOURNAL_FORMAT,
        "benchmark_binding": binding,
        "ordered_case_keys": case_keys,
        "spent_usd": format(spent_usd, "f") if spent_usd != 0 else "0.00",
        "completed_case_keys": completed_case_keys,
        "attempts": retained_attempts,
        "completed_cases": completed_cases,
        "failed_case_keys": failed_case_keys,
        "failed_cases": failed_cases,
    }
    _validate_state(
        state,
        benchmark_binding=binding,
        ordered_case_keys=case_keys,
    )
    _atomic_write(journal_path, state)
    return state


def _find_attempt(
    state: dict,
    *,
    case_key: str,
    phase: str,
    attempt_number: int,
) -> dict | None:
    matches = [
        attempt
        for attempt in state["attempts"]
        if attempt["case_key"] == case_key
        and attempt["phase"] == phase
        and attempt["attempt_number"] == attempt_number
    ]
    if len(matches) > 1:
        raise ValueError(
            "Benchmark v2 execution journal attempt identity is duplicated"
        )
    return matches[0] if matches else None


def _has_unresolved_attempt(state: dict) -> bool:
    return any(
        attempt["status"]
        in {
            "reserved",
            "cost_unknown",
            "cost_exceeds_reservation",
            "transport_failed",
        }
        for attempt in state["attempts"]
    )


@_locked_journal_mutation
def reserve_attempt_if_within_budget(
    path: Path,
    *,
    case_key: str,
    phase: str,
    attempt_number: int,
    request_bytes: bytes,
    reservation_usd: Decimal,
    budget_usd: Decimal,
) -> dict:
    state = _read_state(path)
    if (
        case_key not in state["ordered_case_keys"]
        or not isinstance(phase, str)
        or not phase
        or isinstance(attempt_number, bool)
        or not isinstance(attempt_number, int)
        or attempt_number < 1
        or not isinstance(request_bytes, bytes)
        or not isinstance(reservation_usd, Decimal)
        or not reservation_usd.is_finite()
        or reservation_usd < 0
        or not isinstance(budget_usd, Decimal)
        or not budget_usd.is_finite()
        or budget_usd <= 0
    ):
        raise ValueError(
            "Benchmark v2 execution journal reservation identity is invalid"
        )

    request_sha256 = _sha256_bytes(request_bytes)
    existing = _find_attempt(
        state,
        case_key=case_key,
        phase=phase,
        attempt_number=attempt_number,
    )
    if existing is not None:
        if existing["request_sha256"] != request_sha256:
            raise ValueError(
                "Benchmark v2 execution journal attempt request identity conflicts"
            )
        if existing["status"] in {
            "reserved",
            "cost_unknown",
            "transport_failed",
        }:
            raise ValueError(
                "Benchmark v2 execution journal has unresolved in-flight attempt"
            )
        raise ValueError(
            "Benchmark v2 execution journal attempt is already completed"
        )

    active_reservation_usd = sum(
        (
            _parse_decimal(
                "attempt reservation",
                attempt["reservation_usd"],
            )
            for attempt in state["attempts"]
            if attempt["status"] == "reserved"
        ),
        Decimal("0"),
    )
    committed_usd = (
        _parse_decimal("spent_usd", state["spent_usd"])
        + active_reservation_usd
    )
    if committed_usd > budget_usd:
        raise ValueError(
            "Benchmark execution journal committed liability exceeds the hard budget"
        )
    if _has_unresolved_attempt(state):
        raise ValueError(
            "Benchmark v2 execution journal has unresolved in-flight attempt"
        )
    if committed_usd + reservation_usd > budget_usd:
        raise ValueError(
            "Benchmark paid call would exceed the hard budget"
        )

    state["attempts"].append(
        {
            "case_key": case_key,
            "phase": phase,
            "attempt_number": attempt_number,
            "request_sha256": request_sha256,
            "reservation_usd": format(reservation_usd, "f"),
            "status": "reserved",
        }
    )
    _validate_state(state)
    _atomic_write(Path(path), state)
    return state


def _selected_provider(response_payload: dict) -> str | None:
    metadata = response_payload.get("openrouter_metadata")
    endpoints = (
        metadata.get("endpoints", {}).get("available", [])
        if isinstance(metadata, dict)
        else []
    )
    if not isinstance(endpoints, list):
        return None

    selected_endpoints = [
        endpoint
        for endpoint in endpoints
        if isinstance(endpoint, dict)
        and endpoint.get("selected") is True
    ]
    if len(selected_endpoints) != 1:
        return None

    provider = selected_endpoints[0].get("provider")
    return (
        provider
        if isinstance(provider, str) and provider
        else None
    )


@_locked_journal_mutation
def record_attempt_transport_failure(
    path: Path,
    *,
    case_key: str,
    phase: str,
    attempt_number: int,
    error_class: str,
    error_message: str,
    latency_ms: int,
) -> dict:
    state = _read_state(path)
    if (
        case_key not in state["ordered_case_keys"]
        or not isinstance(phase, str)
        or not phase
        or isinstance(attempt_number, bool)
        or not isinstance(attempt_number, int)
        or attempt_number < 1
        or not isinstance(error_class, str)
        or not error_class
        or len(error_class) > 256
        or not isinstance(error_message, str)
        or len(error_message) > 2048
        or isinstance(latency_ms, bool)
        or not isinstance(latency_ms, int)
        or latency_ms < 0
    ):
        raise ValueError(
            "Benchmark v2 execution journal transport failure is invalid"
        )

    attempt = _find_attempt(
        state,
        case_key=case_key,
        phase=phase,
        attempt_number=attempt_number,
    )
    if attempt is None:
        raise ValueError(
            "Benchmark v2 execution journal transport attempt is unavailable"
        )
    if attempt["status"] != "reserved":
        raise ValueError(
            "Benchmark v2 execution journal transport attempt is not reserved"
        )

    attempt.update(
        {
            "status": "transport_failed",
            "error_class": error_class,
            "error_message": error_message,
            "latency_ms": latency_ms,
        }
    )
    _validate_state(state)
    _atomic_write(Path(path), state)
    return state


@_locked_journal_mutation
def record_attempt_response(
    path: Path,
    *,
    case_key: str,
    attempt_number: int,
    response_bytes: bytes,
    response_payload: dict,
    latency_ms: int,
) -> dict:
    state = _read_state(path)
    if (
        case_key not in state["ordered_case_keys"]
        or isinstance(attempt_number, bool)
        or not isinstance(attempt_number, int)
        or attempt_number < 1
        or not isinstance(response_bytes, bytes)
        or not isinstance(response_payload, dict)
        or isinstance(latency_ms, bool)
        or not isinstance(latency_ms, int)
        or latency_ms < 0
    ):
        raise ValueError(
            "Benchmark v2 execution journal response checkpoint is invalid"
        )

    matches = [
        attempt
        for attempt in state["attempts"]
        if attempt["case_key"] == case_key
        and attempt["attempt_number"] == attempt_number
    ]
    if len(matches) != 1:
        raise ValueError(
            "Benchmark v2 execution journal response attempt is unavailable"
        )
    attempt = matches[0]
    if attempt["status"] != "reserved":
        raise ValueError(
            "Benchmark v2 execution journal response attempt is not reserved"
        )

    common_fields = {
        "response_sha256": _sha256_bytes(response_bytes),
        "response_payload": response_payload,
        "completion_id": response_payload.get("id"),
        "served_model": response_payload.get("model"),
        "served_provider": _selected_provider(response_payload),
        "usage": response_payload.get("usage"),
        "latency_ms": latency_ms,
    }

    usage = response_payload.get("usage")
    cost = usage.get("cost") if isinstance(usage, dict) else None
    try:
        cost_usd = _parse_decimal("completed-call cost", cost)
    except ValueError:
        attempt.update(common_fields)
        attempt["status"] = "cost_unknown"
        _validate_state(state)
        _atomic_write(Path(path), state)
        raise ValueError(
            "Benchmark completed-call cost telemetry is unavailable or invalid"
        )

    reservation_usd = _parse_decimal(
        "attempt reservation",
        attempt["reservation_usd"],
    )
    if cost_usd > reservation_usd:
        attempt.update(common_fields)
        attempt["cost_usd"] = format(cost_usd, "f")
        attempt["status"] = "cost_exceeds_reservation"
        spent_usd = (
            _parse_decimal("spent_usd", state["spent_usd"])
            + cost_usd
        )
        state["spent_usd"] = format(spent_usd, "f")
        _validate_state(state)
        _atomic_write(Path(path), state)
        raise ValueError(
            "Benchmark completed-call cost exceeds durable reservation"
        )

    attempt.update(common_fields)
    attempt["cost_usd"] = format(cost_usd, "f")
    attempt["status"] = "completed"
    spent_usd = _parse_decimal("spent_usd", state["spent_usd"]) + cost_usd
    state["spent_usd"] = format(spent_usd, "f")
    _validate_state(state)
    _atomic_write(Path(path), state)
    return state


def replay_completed_attempt(
    path: Path,
    *,
    case_key: str,
    phase: str,
    attempt_number: int,
    request_bytes: bytes,
) -> dict | None:
    state = _read_state(path)
    if (
        case_key not in state["ordered_case_keys"]
        or not isinstance(phase, str)
        or not phase
        or isinstance(attempt_number, bool)
        or not isinstance(attempt_number, int)
        or attempt_number < 1
        or not isinstance(request_bytes, bytes)
    ):
        raise ValueError(
            "Benchmark v2 execution journal replay identity is invalid"
        )

    request_sha256 = _sha256_bytes(request_bytes)
    existing = _find_attempt(
        state,
        case_key=case_key,
        phase=phase,
        attempt_number=attempt_number,
    )
    if existing is None:
        if _has_unresolved_attempt(state):
            raise ValueError(
                "Benchmark v2 execution journal has unresolved in-flight attempt"
            )
        return None

    if existing["request_sha256"] != request_sha256:
        raise ValueError(
            "Benchmark v2 execution journal attempt request identity conflicts"
        )
    if existing["status"] in {
        "reserved",
        "cost_unknown",
        "transport_failed",
    }:
        raise ValueError(
            "Benchmark v2 execution journal has unresolved in-flight attempt"
        )
    if existing["status"] != "completed":
        raise ValueError(
            "Benchmark v2 execution journal attempt status is invalid"
        )

    response_payload = existing.get("response_payload")
    if not isinstance(response_payload, dict):
        raise ValueError(
            "Benchmark v2 execution journal completed response is unavailable"
        )
    return response_payload


@_locked_journal_mutation
def record_case_completion(
    path: Path,
    *,
    case_key: str,
    case_result: dict,
    execution_evidence: dict,
) -> dict:
    state = _read_state(path)
    if (
        case_key not in state["ordered_case_keys"]
        or not isinstance(case_result, dict)
        or not isinstance(execution_evidence, dict)
    ):
        raise ValueError(
            "Benchmark v2 execution journal case completion is invalid"
        )
    if _has_unresolved_attempt(state):
        raise ValueError(
            "Benchmark v2 execution journal has unresolved in-flight attempt"
        )
    if case_key in state["completed_case_keys"]:
        raise ValueError(
            "Benchmark v2 execution journal case is already completed"
        )
    if case_key in state["failed_case_keys"]:
        raise ValueError(
            "Benchmark v2 execution journal case is already resolved as failed"
        )

    state["completed_cases"].append(
        {
            "case_key": case_key,
            "case_result": case_result,
            "execution_evidence": execution_evidence,
        }
    )
    state["completed_case_keys"].append(case_key)
    _validate_state(state)
    _atomic_write(Path(path), state)
    return state


@_locked_journal_mutation
def record_case_failure(
    path: Path,
    *,
    case_key: str,
    failure: dict,
) -> dict:
    state = _read_state(path)
    if case_key not in state["ordered_case_keys"] or not isinstance(failure, dict):
        raise ValueError(
            "Benchmark v2 execution journal case failure is invalid"
        )
    validated_failure = _validate_failed_case(
        failure,
        state["ordered_case_keys"],
    )
    if validated_failure["case_key"] != case_key:
        raise ValueError(
            "Benchmark v2 execution journal case failure identity conflicts"
        )
    if _has_unresolved_attempt(state):
        raise ValueError(
            "Benchmark v2 execution journal has unresolved in-flight attempt"
        )
    if case_key in state["completed_case_keys"]:
        raise ValueError(
            "Benchmark v2 execution journal case is already resolved as completed"
        )
    if case_key in state["failed_case_keys"]:
        matching = [
            item
            for item in state["failed_cases"]
            if item.get("case_key") == case_key
        ]
        if matching == [validated_failure]:
            return state
        raise ValueError(
            "Benchmark v2 execution journal case failure conflicts"
        )

    state["failed_cases"].append(dict(validated_failure))
    state["failed_case_keys"].append(case_key)
    _validate_state(state)
    _atomic_write(Path(path), state)
    return state
