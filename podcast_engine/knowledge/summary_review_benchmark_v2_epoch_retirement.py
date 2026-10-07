"""Zero-paid retirement of a partially spent Summary Review Benchmark v2 epoch."""

from __future__ import annotations

import hashlib
import os
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

from .summary_review_benchmark_v2_safety import (
    BENCHMARK_V2_BUDGET_USD,
    _canonical_json_bytes,
    _paid_runner_lock,
    _parse_canonical_json_bytes,
    _sha256_bytes,
    _validate_execution_journal_state,
    _workspace_root,
)


_TERMINAL_DECISION_LOCKS = (
    "qualification.lock.json",
    "qualification.no_winner.lock.json",
    "winner.lock.json",
    "no_winner.lock.json",
)
_RETIREMENT_ARCHIVE_MEMBERS = (
    "execution-state.json",
    "paid-authorization.json",
    "preflight-authorization.json",
    "preflight-result.json",
    "preflight",
)
_RETIREMENT_POINTER_FILENAME = "latest-partial-spend-retirement.json"
_RETIREMENT_COMPLETE_FILENAME = "retirement-complete.json"


def _require_decimal(value: object, *, label: str) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as error:
        raise ValueError(f"Benchmark v2 partial-epoch {label} is invalid") from error
    if not parsed.is_finite() or parsed < 0:
        raise ValueError(f"Benchmark v2 partial-epoch {label} is invalid")
    return parsed


def _require_regular_file(path: Path, *, label: str) -> bytes:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Benchmark v2 partial-epoch {label} is unavailable")
    try:
        return path.read_bytes()
    except OSError as error:
        raise ValueError(f"Benchmark v2 partial-epoch {label} is unreadable") from error


def _partial_epoch_details(journal_bytes: bytes) -> dict:
    journal = _parse_canonical_json_bytes(
        journal_bytes,
        label="partial-spend execution journal",
    )
    _validate_execution_journal_state(journal)

    spent_usd = _require_decimal(
        journal.get("spent_usd"),
        label="spent_usd",
    )
    hard_budget_usd = Decimal(BENCHMARK_V2_BUDGET_USD)
    completed_case_keys = journal.get("completed_case_keys")
    completed_cases = journal.get("completed_cases")
    attempts = journal.get("attempts")
    if (
        spent_usd <= 0
        or spent_usd >= hard_budget_usd
        or not isinstance(completed_case_keys, list)
        or not completed_case_keys
        or not isinstance(completed_cases, list)
        or not completed_cases
        or not isinstance(attempts, list)
    ):
        raise ValueError(
            "Benchmark v2 partial-epoch retirement requires a partially spent epoch"
        )

    completed_case_key_set = set(completed_case_keys)
    failed_case_keys = journal.get("failed_case_keys")
    if not isinstance(failed_case_keys, list):
        raise ValueError(
            "Benchmark v2 partial-epoch retirement requires a partially spent epoch"
        )
    resolved_case_key_set = completed_case_key_set | set(failed_case_keys)

    # Every attempt belongs either to a case that already reached one of the
    # two durable terminal outcomes (folded into ``completed_cases`` or
    # ``failed_cases`` -- fully accounted for there and never eligible for
    # retirement), or to the single case that halted the epoch on an
    # execution-integrity uncertainty. A case resolved as ``model_failed``
    # is terminal: it must never be treated as an orphan and recycled with
    # a fresh attempt budget, so it is excluded from orphan detection here
    # exactly like a completed case.
    orphaned_by_case: dict[str, list[dict]] = {}
    for attempt in attempts:
        case_key = attempt.get("case_key") if isinstance(attempt, dict) else None
        if case_key in resolved_case_key_set:
            continue
        orphaned_by_case.setdefault(case_key, []).append(attempt)

    if len(orphaned_by_case) != 1:
        raise ValueError(
            "Benchmark v2 partial-epoch retirement requires exactly one unresolved attempt"
        )
    ((_failed_case_key, failed_attempts),) = orphaned_by_case.items()

    # The only retirable shape is exactly one attempt for the halted case
    # that never reached a definitive provider response (a transport
    # failure or a crash before one arrived). There is no evidence to
    # review; the reservation it held is simply released. A case whose
    # attempts are all definitively costed but not yet resolved as
    # ``completed`` or ``model_failed`` is never retirable -- it is a
    # deterministic model-output outcome pending its own terminal
    # persistence (recovered by resuming paid execution, which replays the
    # saved response and converges on the same failed-case record without
    # another transport call), not an execution-integrity failure, and
    # retiring it here would let a resolved model failure be recycled with
    # a fresh attempt budget.
    if (
        len(failed_attempts) == 1
        and failed_attempts[0].get("status") in {"reserved", "transport_failed"}
    ):
        unresolved_attempt = failed_attempts[0]
        reservation_usd = _require_decimal(
            unresolved_attempt.get("reservation_usd"),
            label="unresolved reservation",
        )
        request_sha256 = unresolved_attempt.get("request_sha256")
        if (
            not isinstance(request_sha256, str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", request_sha256) is None
        ):
            raise ValueError(
                "Benchmark v2 partial-epoch unresolved request identity is invalid"
            )

        return {
            "failure_kind": "dangling_attempt",
            "journal": journal,
            "spent_usd": spent_usd,
            "completed_case_keys": completed_case_keys,
            "unresolved_attempt": unresolved_attempt,
            "reservation_usd": reservation_usd,
            "request_sha256": request_sha256,
        }

    raise ValueError(
        "Benchmark v2 partial-epoch retirement requires exactly one unresolved attempt"
    )


def _build_retirement_manifest(
    *,
    journal_bytes: bytes,
    archive_members: list[str],
) -> tuple[dict, bytes, dict]:
    if (
        not isinstance(archive_members, list)
        or len(archive_members) != len(set(archive_members))
        or any(name not in _RETIREMENT_ARCHIVE_MEMBERS for name in archive_members)
        or "execution-state.json" not in archive_members
        or "paid-authorization.json" not in archive_members
    ):
        raise ValueError("Benchmark v2 partial-epoch archive member set is invalid")

    details = _partial_epoch_details(journal_bytes)

    # ``_partial_epoch_details`` only ever returns the dangling-attempt
    # shape (or raises): a resolved ``model_failed`` case is terminal and
    # is never eligible for retirement, so there is no exhausted-case
    # manifest shape in the active format-2 path.
    unresolved_attempt = details["unresolved_attempt"]
    manifest = {
        "retirement_format": 1,
        "kind": "benchmark_v2_partial_spend_epoch_retirement",
        "hard_budget_usd": BENCHMARK_V2_BUDGET_USD,
        "next_epoch_budget_usd": BENCHMARK_V2_BUDGET_USD,
        "journal_sha256": _sha256_bytes(journal_bytes),
        "journal_spent_usd": format(details["spent_usd"], "f"),
        "completed_case_count": len(details["completed_case_keys"]),
        "archive_members": archive_members,
        "unresolved_attempt": {
            "case_key": unresolved_attempt.get("case_key"),
            "phase": unresolved_attempt.get("phase"),
            "attempt_number": unresolved_attempt.get("attempt_number"),
            "status": unresolved_attempt.get("status"),
            "request_sha256": details["request_sha256"],
            "reservation_usd": format(details["reservation_usd"], "f"),
        },
    }
    return manifest, _canonical_json_bytes(manifest), details


def _archive_root_for_material(
    history_root: Path,
    *,
    manifest_bytes: bytes,
    journal_bytes: bytes,
    paid_authorization_bytes: bytes,
) -> Path:
    archive_material = (
        manifest_bytes
        + b"\0"
        + journal_bytes
        + b"\0"
        + paid_authorization_bytes
    )
    return history_root / (
        "partial-spend-" + hashlib.sha256(archive_material).hexdigest()
    )


def _write_create_or_match(path: Path, data: bytes, *, label: str) -> None:
    if path.is_symlink():
        raise ValueError(f"Benchmark v2 partial-epoch {label} must not be a symlink")
    if path.exists():
        if not path.is_file():
            raise ValueError(f"Benchmark v2 partial-epoch {label} is invalid")
        try:
            existing = path.read_bytes()
        except OSError as error:
            raise ValueError(f"Benchmark v2 partial-epoch {label} is unreadable") from error
        if existing != data:
            raise ValueError(f"Benchmark v2 partial-epoch {label} conflicts")
        return
    try:
        with path.open("xb") as handle:
            handle.write(data)
    except FileExistsError:
        _write_create_or_match(path, data, label=label)
    except OSError as error:
        raise ValueError(f"Benchmark v2 partial-epoch {label} is unwritable") from error


def _write_latest_pointer(
    history_root: Path,
    *,
    archive_root: Path,
    manifest_bytes: bytes,
) -> None:
    try:
        history_root.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise ValueError("Benchmark v2 partial-epoch history is unavailable") from error
    if history_root.is_symlink() or not history_root.is_dir():
        raise ValueError("Benchmark v2 partial-epoch history path is invalid")

    pointer_path = history_root / _RETIREMENT_POINTER_FILENAME
    temp_path = history_root / f".{_RETIREMENT_POINTER_FILENAME}.tmp"
    if pointer_path.is_symlink() or temp_path.is_symlink():
        raise ValueError("Benchmark v2 partial-epoch retirement pointer is invalid")
    if pointer_path.exists() and not pointer_path.is_file():
        raise ValueError("Benchmark v2 partial-epoch retirement pointer is invalid")
    if temp_path.exists():
        if not temp_path.is_file():
            raise ValueError("Benchmark v2 partial-epoch retirement pointer is invalid")
        try:
            temp_path.unlink()
        except OSError as error:
            raise ValueError(
                "Benchmark v2 partial-epoch retirement pointer is unavailable"
            ) from error

    pointer = {
        "pointer_format": 1,
        "kind": "benchmark_v2_partial_spend_epoch_retirement_pointer",
        "archive_name": archive_root.name,
        "retirement_manifest_sha256": _sha256_bytes(manifest_bytes),
    }
    pointer_bytes = _canonical_json_bytes(pointer)
    try:
        with temp_path.open("xb") as handle:
            handle.write(pointer_bytes)
        os.replace(temp_path, pointer_path)
    except OSError as error:
        try:
            if temp_path.is_file() and not temp_path.is_symlink():
                temp_path.unlink()
        except OSError:
            pass
        raise ValueError(
            "Benchmark v2 partial-epoch retirement pointer could not be recorded"
        ) from error


def _read_latest_pointer(history_root: Path) -> tuple[Path, dict]:
    pointer_path = history_root / _RETIREMENT_POINTER_FILENAME
    pointer_bytes = _require_regular_file(
        pointer_path,
        label="retirement pointer",
    )
    pointer = _parse_canonical_json_bytes(
        pointer_bytes,
        label="partial-spend retirement pointer",
    )
    archive_name = pointer.get("archive_name")
    manifest_sha256 = pointer.get("retirement_manifest_sha256")
    if (
        set(pointer)
        != {
            "pointer_format",
            "kind",
            "archive_name",
            "retirement_manifest_sha256",
        }
        or pointer.get("pointer_format") != 1
        or pointer.get("kind")
        != "benchmark_v2_partial_spend_epoch_retirement_pointer"
        or not isinstance(archive_name, str)
        or re.fullmatch(r"partial-spend-[0-9a-f]{64}", archive_name) is None
        or not isinstance(manifest_sha256, str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", manifest_sha256) is None
    ):
        raise ValueError("Benchmark v2 partial-epoch retirement pointer is invalid")
    archive_root = history_root / archive_name
    if (
        archive_root.resolve(strict=False).parent
        != history_root.resolve(strict=False)
    ):
        raise ValueError("Benchmark v2 partial-epoch retirement pointer escapes history")
    return archive_root, pointer


def _member_shape_is_valid(path: Path, *, name: str) -> bool:
    if path.is_symlink():
        return False
    if name == "preflight":
        return path.is_dir()
    return path.is_file()


def _collect_archive_members(workspace: Path) -> list[str]:
    members = []
    for name in _RETIREMENT_ARCHIVE_MEMBERS:
        path = workspace / name
        if path.exists() or path.is_symlink():
            if not _member_shape_is_valid(path, name=name):
                raise ValueError(
                    f"Benchmark v2 partial-epoch retirement source is invalid: {name}"
                )
            members.append(name)
    if "execution-state.json" not in members or "paid-authorization.json" not in members:
        raise ValueError(
            "Benchmark v2 partial-epoch retirement requires journal and paid authorization"
        )
    return members


def _read_member_file_from_active_or_archive(
    *,
    workspace: Path,
    archive_root: Path,
    name: str,
    label: str,
) -> bytes:
    source = workspace / name
    destination = archive_root / name
    source_present = source.exists() or source.is_symlink()
    destination_present = destination.exists() or destination.is_symlink()
    if source_present and destination_present:
        raise ValueError(f"Benchmark v2 partial-epoch {label} exists in two locations")
    if source_present:
        return _require_regular_file(source, label=label)
    if destination_present:
        return _require_regular_file(destination, label=label)
    raise ValueError(f"Benchmark v2 partial-epoch {label} is unavailable")


def _ensure_archive_root(archive_root: Path, manifest_bytes: bytes) -> None:
    if archive_root.is_symlink():
        raise ValueError("Benchmark v2 partial-epoch archive must not be a symlink")
    if archive_root.exists():
        if not archive_root.is_dir():
            raise ValueError("Benchmark v2 partial-epoch archive path is invalid")
    else:
        try:
            archive_root.mkdir(parents=True, exist_ok=False)
        except OSError as error:
            raise ValueError(
                "Benchmark v2 partial-spend epoch could not be archived"
            ) from error
    _write_create_or_match(
        archive_root / "retirement-manifest.json",
        manifest_bytes,
        label="retirement manifest",
    )


def _move_archive_member(
    *,
    workspace: Path,
    archive_root: Path,
    name: str,
) -> None:
    source = workspace / name
    destination = archive_root / name
    source_present = source.exists() or source.is_symlink()
    destination_present = destination.exists() or destination.is_symlink()

    if source_present and destination_present:
        raise ValueError(
            f"Benchmark v2 partial-epoch retirement member conflicts: {name}"
        )
    if source_present:
        if not _member_shape_is_valid(source, name=name):
            raise ValueError(
                f"Benchmark v2 partial-epoch retirement source is invalid: {name}"
            )
        if destination.is_symlink():
            raise ValueError(
                f"Benchmark v2 partial-epoch retirement destination is invalid: {name}"
            )
        os.replace(source, destination)
        return
    if destination_present:
        if not _member_shape_is_valid(destination, name=name):
            raise ValueError(
                f"Benchmark v2 partial-epoch retirement destination is invalid: {name}"
            )
        return
    raise ValueError(
        f"Benchmark v2 partial-epoch retirement member is unavailable: {name}"
    )


def _completion_bytes(manifest_bytes: bytes) -> bytes:
    return _canonical_json_bytes(
        {
            "completion_format": 1,
            "kind": "benchmark_v2_partial_spend_epoch_retirement_complete",
            "retirement_manifest_sha256": _sha256_bytes(manifest_bytes),
        }
    )


def _result_from_manifest(
    *,
    status: str,
    archive_root: Path,
    manifest: dict,
    manifest_bytes: bytes,
) -> dict:
    result = {
        "status": status,
        "archive_path": archive_root,
        "journal_spent_usd": manifest["journal_spent_usd"],
        "next_epoch_budget_usd": manifest["next_epoch_budget_usd"],
        "retirement_manifest_sha256": _sha256_bytes(manifest_bytes),
    }
    unresolved_attempt = manifest["unresolved_attempt"]
    result["unresolved_reservation_usd"] = unresolved_attempt["reservation_usd"]
    return result


def _refuse_terminal_decision_locks(workspace: Path) -> None:
    decision_lock_dir = workspace / "locks"
    if decision_lock_dir.is_symlink():
        raise ValueError(
            "Benchmark v2 partial-epoch decision lock directory must not be a symlink"
        )
    if decision_lock_dir.exists():
        if not decision_lock_dir.is_dir():
            raise ValueError(
                "Benchmark v2 partial-epoch decision lock path must be a directory"
            )
        for filename in _TERMINAL_DECISION_LOCKS:
            if (decision_lock_dir / filename).exists():
                raise ValueError(
                    "Benchmark v2 partial-epoch retirement refuses an existing decision lock"
                )


def retire_benchmark_v2_partial_spend_epoch(*, repository_root: Path) -> dict:
    """Archive one failed partially spent epoch and reset the next epoch to $6.00."""

    repository_root = Path(repository_root)
    workspace = _workspace_root(repository_root)
    journal_path = workspace / "execution-state.json"
    history_root = workspace / "history"

    with _paid_runner_lock(repository_root):
        if workspace.is_symlink() or history_root.is_symlink():
            raise ValueError(
                "Benchmark v2 partial-epoch retirement workspace must not be a symlink"
            )
        _refuse_terminal_decision_locks(workspace)

        for evidence_name in ("provider-lock.json", "pricing-evidence.json"):
            _require_regular_file(
                workspace / evidence_name,
                label=evidence_name,
            )

        active_journal = journal_path.exists() or journal_path.is_symlink()
        if active_journal:
            journal_bytes = _require_regular_file(
                journal_path,
                label="execution journal",
            )
            paid_authorization_path = workspace / "paid-authorization.json"
            paid_authorization_bytes = _require_regular_file(
                paid_authorization_path,
                label="paid authorization",
            )
            _parse_canonical_json_bytes(
                paid_authorization_bytes,
                label="partial-spend paid authorization",
            )
            archive_members = _collect_archive_members(workspace)
            manifest, manifest_bytes, _details = _build_retirement_manifest(
                journal_bytes=journal_bytes,
                archive_members=archive_members,
            )
            archive_root = _archive_root_for_material(
                history_root,
                manifest_bytes=manifest_bytes,
                journal_bytes=journal_bytes,
                paid_authorization_bytes=paid_authorization_bytes,
            )
            _write_latest_pointer(
                history_root,
                archive_root=archive_root,
                manifest_bytes=manifest_bytes,
            )
        else:
            if not history_root.is_dir():
                raise ValueError(
                    "Benchmark v2 partial-epoch execution journal is unavailable"
                )
            archive_root, pointer = _read_latest_pointer(history_root)
            if archive_root.is_symlink() or not archive_root.is_dir():
                raise ValueError("Benchmark v2 partial-epoch retirement archive is unavailable")
            manifest_bytes = _require_regular_file(
                archive_root / "retirement-manifest.json",
                label="retirement manifest",
            )
            if _sha256_bytes(manifest_bytes) != pointer["retirement_manifest_sha256"]:
                raise ValueError("Benchmark v2 partial-epoch retirement manifest hash mismatch")
            manifest = _parse_canonical_json_bytes(
                manifest_bytes,
                label="partial-spend retirement manifest",
            )
            archive_members = manifest.get("archive_members")
            journal_bytes = _require_regular_file(
                archive_root / "execution-state.json",
                label="archived execution journal",
            )
            rebuilt_manifest, rebuilt_manifest_bytes, _details = _build_retirement_manifest(
                journal_bytes=journal_bytes,
                archive_members=archive_members,
            )
            if rebuilt_manifest_bytes != manifest_bytes or rebuilt_manifest != manifest:
                raise ValueError("Benchmark v2 partial-epoch retirement manifest is invalid")
            paid_authorization_bytes = _read_member_file_from_active_or_archive(
                workspace=workspace,
                archive_root=archive_root,
                name="paid-authorization.json",
                label="paid authorization",
            )
            _parse_canonical_json_bytes(
                paid_authorization_bytes,
                label="partial-spend paid authorization",
            )
            expected_archive_root = _archive_root_for_material(
                history_root,
                manifest_bytes=manifest_bytes,
                journal_bytes=journal_bytes,
                paid_authorization_bytes=paid_authorization_bytes,
            )
            if expected_archive_root != archive_root:
                raise ValueError("Benchmark v2 partial-epoch retirement archive identity mismatch")

            completion_path = archive_root / _RETIREMENT_COMPLETE_FILENAME
            if completion_path.exists() or completion_path.is_symlink():
                completion = _require_regular_file(
                    completion_path,
                    label="retirement completion marker",
                )
                if completion != _completion_bytes(manifest_bytes):
                    raise ValueError(
                        "Benchmark v2 partial-epoch retirement completion marker conflicts"
                    )
                for name in archive_members:
                    source = workspace / name
                    destination = archive_root / name
                    if source.exists() or source.is_symlink():
                        raise ValueError(
                            "Benchmark v2 completed partial-epoch retirement conflicts with active state"
                        )
                    if not _member_shape_is_valid(destination, name=name):
                        raise ValueError(
                            "Benchmark v2 completed partial-epoch retirement archive is incomplete"
                        )
                return _result_from_manifest(
                    status="already_retired",
                    archive_root=archive_root,
                    manifest=manifest,
                    manifest_bytes=manifest_bytes,
                )

        _ensure_archive_root(archive_root, manifest_bytes)
        try:
            for name in archive_members:
                _move_archive_member(
                    workspace=workspace,
                    archive_root=archive_root,
                    name=name,
                )
        except OSError as error:
            raise ValueError(
                "Benchmark v2 partial-spend epoch could not be archived"
            ) from error

        _write_create_or_match(
            archive_root / _RETIREMENT_COMPLETE_FILENAME,
            _completion_bytes(manifest_bytes),
            label="retirement completion marker",
        )
        return _result_from_manifest(
            status="retired",
            archive_root=archive_root,
            manifest=manifest,
            manifest_bytes=manifest_bytes,
        )


def install_partial_epoch_retirement(namespace: dict) -> None:
    """Expose the canonical retirement operation through the benchmark API."""

    namespace["retire_benchmark_v2_partial_spend_epoch"] = (
        retire_benchmark_v2_partial_spend_epoch
    )
