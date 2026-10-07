"""Archive a fully decided (WINNER/NO WINNER) Summary Review Benchmark v2 epoch.

``retire_benchmark_v2_partial_spend_epoch`` (summary_review_benchmark_v2_epoch_retirement.py)
only ever accepts an epoch with exactly one dangling attempt that never reached a
definitive provider response, and explicitly refuses when any terminal decision
lock already exists. ``reconcile_benchmark_v2_aborted_execution_epoch`` only ever
accepts a zero-spend journal with exactly one unresolved ``reserved``/``transport_failed``
attempt. Neither shape matches a genuinely completed epoch: every canonical case
already resolved to ``completed`` or ``model_failed``, real spend occurred, and a
terminal decision lock (``qualification.lock.json`` / ``qualification.no_winner.lock.json``
plus ``winner.lock.json`` / ``no_winner.lock.json``) already exists. That epoch is
finished, not broken -- but until now there was no explicit, audited operator path
to archive it and reset the next epoch's budget. Archiving a decided epoch is a
deliberate act (the operator has looked at the decision and chosen to try again,
typically after a targeted code repair), never an automatic retry.

A dual-track epoch (TASK-052) reaches its own, differently-shaped terminal
decision: ``locks/qualification.designated-tracks.lock.json`` plus, for every
*eligible* designated candidate, its own terminal
``locks/geoffrey/<slug>/{winner,no_winner}.lock.json``. An epoch where no
designated candidate is eligible is decided vacuously -- no Geoffrey attempt
was ever possible. This module recognizes both terminal shapes and refuses
either being present without the other, so a dual-track epoch that is only
partway decided (an eligible candidate with no terminal Geoffrey lock yet)
stays available to resume instead of being archived early.
"""

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


_TERMINAL_DECISION_LOCK_FILENAMES = (
    "qualification.lock.json",
    "qualification.no_winner.lock.json",
    "winner.lock.json",
    "no_winner.lock.json",
)
_DESIGNATED_TRACKS_LOCK_FILENAME = "qualification.designated-tracks.lock.json"
_DECIDED_EPOCH_ARCHIVE_MEMBERS = (
    "execution-state.json",
    "paid-authorization.json",
    "preflight-authorization.json",
    "preflight-result.json",
    "preflight",
    "locks",
)
_DECIDED_EPOCH_POINTER_FILENAME = "latest-decided-epoch-archival.json"
_DECIDED_EPOCH_COMPLETE_FILENAME = "archival-complete.json"
_DECIDED_EPOCH_MANIFEST_FILENAME = "archival-manifest.json"


def _require_decimal(value: object, *, label: str) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as error:
        raise ValueError(f"Benchmark v2 decided-epoch {label} is invalid") from error
    if not parsed.is_finite() or parsed < 0:
        raise ValueError(f"Benchmark v2 decided-epoch {label} is invalid")
    return parsed


def _require_regular_file(path: Path, *, label: str) -> bytes:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Benchmark v2 decided-epoch {label} is unavailable")
    try:
        return path.read_bytes()
    except OSError as error:
        raise ValueError(f"Benchmark v2 decided-epoch {label} is unreadable") from error


def _require_decided_epoch_details(journal_bytes: bytes) -> dict:
    journal = _parse_canonical_json_bytes(
        journal_bytes,
        label="decided-epoch execution journal",
    )
    _validate_execution_journal_state(journal)

    spent_usd = _require_decimal(journal.get("spent_usd"), label="spent_usd")
    completed_case_keys = journal.get("completed_case_keys")
    failed_case_keys = journal.get("failed_case_keys")
    attempts = journal.get("attempts")
    if (
        spent_usd <= 0
        or not isinstance(completed_case_keys, list)
        or not completed_case_keys
        or not isinstance(failed_case_keys, list)
        or not isinstance(attempts, list)
    ):
        raise ValueError(
            "Benchmark v2 decided-epoch archival requires a spent, resolved epoch"
        )

    resolved_case_key_set = set(completed_case_keys) | set(failed_case_keys)

    # A decided epoch is one where every attempt on record belongs to a case
    # that already reached one of the two durable terminal outcomes. Any
    # attempt outside that set means the epoch still has unresolved work (an
    # execution-integrity uncertainty, or a case mid-repair) and must go
    # through ``retire``/``reconcile`` or simply be resumed, never through
    # this archival path.
    for attempt in attempts:
        case_key = attempt.get("case_key") if isinstance(attempt, dict) else None
        if case_key not in resolved_case_key_set:
            raise ValueError(
                "Benchmark v2 decided-epoch archival requires every attempt's "
                "case to already be resolved"
            )

    return {
        "spent_usd": spent_usd,
        "completed_case_count": len(completed_case_keys),
        "failed_case_count": len(failed_case_keys),
    }


def _read_dual_track_decision_lock_filenames(
    lock_dir: Path,
    *,
    geoffrey_track_slug,
) -> list[str]:
    """Recognize the TASK-052 dual-track terminal decision shape.

    A dual-track epoch is decided when the designated-tracks lock exists and
    every *eligible* designated candidate already has its own terminal
    Geoffrey lock (``geoffrey/<slug>/{winner,no_winner}.lock.json`` under the
    same lock directory). An epoch where no designated candidate is eligible
    is decided vacuously -- no further Geoffrey attempt is possible.
    """

    designated_tracks_path = lock_dir / _DESIGNATED_TRACKS_LOCK_FILENAME
    designated_tracks_bytes = _require_regular_file(
        designated_tracks_path,
        label="designated-tracks decision lock",
    )
    designated_tracks_lock = _parse_canonical_json_bytes(
        designated_tracks_bytes,
        label="designated-tracks decision lock",
    )
    designated_tracks = (
        designated_tracks_lock.get("designated_tracks")
        if isinstance(designated_tracks_lock, dict)
        else None
    )
    if not isinstance(designated_tracks, dict) or not designated_tracks:
        raise ValueError(
            "Benchmark v2 decided-epoch designated-tracks lock is invalid"
        )

    present = [_DESIGNATED_TRACKS_LOCK_FILENAME]
    geoffrey_lock_dir = lock_dir / "geoffrey"
    for candidate_id, entry in designated_tracks.items():
        if not isinstance(entry, dict) or not isinstance(
            entry.get("eligible"), bool
        ):
            raise ValueError(
                "Benchmark v2 decided-epoch designated-tracks lock is invalid"
            )
        if entry["eligible"] is not True:
            continue

        slug = geoffrey_track_slug(candidate_id)
        track_lock_dir = geoffrey_lock_dir / slug
        winner_path = track_lock_dir / "winner.lock.json"
        no_winner_path = track_lock_dir / "no_winner.lock.json"
        winner_present = winner_path.exists() or winner_path.is_symlink()
        no_winner_present = no_winner_path.exists() or no_winner_path.is_symlink()

        if winner_present and no_winner_present:
            raise ValueError(
                "Benchmark v2 decided-epoch Geoffrey terminal lock is "
                f"ambiguous: {candidate_id}"
            )
        if not winner_present and not no_winner_present:
            raise ValueError(
                "Benchmark v2 decided-epoch archival requires every eligible "
                f"designated candidate to reach a terminal Geoffrey decision: "
                f"{candidate_id}"
            )

        terminal_path = winner_path if winner_present else no_winner_path
        if not terminal_path.is_file() or terminal_path.is_symlink():
            raise ValueError(
                "Benchmark v2 decided-epoch decision lock is invalid: "
                f"geoffrey/{slug}/{terminal_path.name}"
            )
        present.append(f"geoffrey/{slug}/{terminal_path.name}")

    return present


def _read_decision_lock_filenames(
    workspace: Path,
    *,
    geoffrey_track_slug,
) -> list[str]:
    lock_dir = workspace / "locks"
    if lock_dir.is_symlink():
        raise ValueError(
            "Benchmark v2 decided-epoch decision lock directory must not be a symlink"
        )
    if not lock_dir.exists():
        raise ValueError(
            "Benchmark v2 decided-epoch archival requires an existing decision lock"
        )
    if not lock_dir.is_dir():
        raise ValueError(
            "Benchmark v2 decided-epoch decision lock path must be a directory"
        )

    present = []
    for filename in _TERMINAL_DECISION_LOCK_FILENAMES:
        path = lock_dir / filename
        if path.exists() or path.is_symlink():
            if not path.is_file() or path.is_symlink():
                raise ValueError(
                    f"Benchmark v2 decided-epoch decision lock is invalid: {filename}"
                )
            present.append(filename)

    designated_tracks_path = lock_dir / _DESIGNATED_TRACKS_LOCK_FILENAME
    dual_track_present = (
        designated_tracks_path.exists() or designated_tracks_path.is_symlink()
    )

    if present and dual_track_present:
        raise ValueError(
            "Benchmark v2 decided-epoch decision lock shape is ambiguous"
        )

    if dual_track_present:
        return _read_dual_track_decision_lock_filenames(
            lock_dir,
            geoffrey_track_slug=geoffrey_track_slug,
        )

    if not present:
        raise ValueError(
            "Benchmark v2 decided-epoch archival requires an existing decision lock"
        )
    return present


def _member_shape_is_valid(path: Path, *, name: str) -> bool:
    if path.is_symlink():
        return False
    if name in ("preflight", "locks"):
        return path.is_dir()
    return path.is_file()


def _collect_archive_members(workspace: Path) -> list[str]:
    members = []
    for name in _DECIDED_EPOCH_ARCHIVE_MEMBERS:
        path = workspace / name
        if path.exists() or path.is_symlink():
            if not _member_shape_is_valid(path, name=name):
                raise ValueError(
                    f"Benchmark v2 decided-epoch archival source is invalid: {name}"
                )
            members.append(name)
    if (
        "execution-state.json" not in members
        or "paid-authorization.json" not in members
        or "locks" not in members
    ):
        raise ValueError(
            "Benchmark v2 decided-epoch archival requires the journal, paid "
            "authorization, and decision lock directory"
        )
    return members


def _build_decided_epoch_manifest(
    *,
    journal_bytes: bytes,
    archive_members: list[str],
    decision_lock_filenames: list[str],
) -> tuple[dict, bytes, dict]:
    if (
        not isinstance(archive_members, list)
        or len(archive_members) != len(set(archive_members))
        or any(name not in _DECIDED_EPOCH_ARCHIVE_MEMBERS for name in archive_members)
        or "execution-state.json" not in archive_members
        or "paid-authorization.json" not in archive_members
        or "locks" not in archive_members
    ):
        raise ValueError("Benchmark v2 decided-epoch archive member set is invalid")

    details = _require_decided_epoch_details(journal_bytes)
    manifest = {
        "archival_format": 1,
        "kind": "benchmark_v2_decided_epoch_archival",
        "hard_budget_usd": BENCHMARK_V2_BUDGET_USD,
        "next_epoch_budget_usd": BENCHMARK_V2_BUDGET_USD,
        "journal_sha256": _sha256_bytes(journal_bytes),
        "journal_spent_usd": format(details["spent_usd"], "f"),
        "completed_case_count": details["completed_case_count"],
        "failed_case_count": details["failed_case_count"],
        "archive_members": archive_members,
        "decision_lock_filenames": sorted(decision_lock_filenames),
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
        "decided-epoch-" + hashlib.sha256(archive_material).hexdigest()
    )


def _write_create_or_match(path: Path, data: bytes, *, label: str) -> None:
    if path.is_symlink():
        raise ValueError(f"Benchmark v2 decided-epoch {label} must not be a symlink")
    if path.exists():
        if not path.is_file():
            raise ValueError(f"Benchmark v2 decided-epoch {label} is invalid")
        try:
            existing = path.read_bytes()
        except OSError as error:
            raise ValueError(f"Benchmark v2 decided-epoch {label} is unreadable") from error
        if existing != data:
            raise ValueError(f"Benchmark v2 decided-epoch {label} conflicts")
        return
    try:
        with path.open("xb") as handle:
            handle.write(data)
    except FileExistsError:
        _write_create_or_match(path, data, label=label)
    except OSError as error:
        raise ValueError(f"Benchmark v2 decided-epoch {label} is unwritable") from error


def _write_latest_pointer(
    history_root: Path,
    *,
    archive_root: Path,
    manifest_bytes: bytes,
) -> None:
    try:
        history_root.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise ValueError("Benchmark v2 decided-epoch history is unavailable") from error
    if history_root.is_symlink() or not history_root.is_dir():
        raise ValueError("Benchmark v2 decided-epoch history path is invalid")

    pointer_path = history_root / _DECIDED_EPOCH_POINTER_FILENAME
    temp_path = history_root / f".{_DECIDED_EPOCH_POINTER_FILENAME}.tmp"
    if pointer_path.is_symlink() or temp_path.is_symlink():
        raise ValueError("Benchmark v2 decided-epoch archival pointer is invalid")
    if pointer_path.exists() and not pointer_path.is_file():
        raise ValueError("Benchmark v2 decided-epoch archival pointer is invalid")
    if temp_path.exists():
        if not temp_path.is_file():
            raise ValueError("Benchmark v2 decided-epoch archival pointer is invalid")
        try:
            temp_path.unlink()
        except OSError as error:
            raise ValueError(
                "Benchmark v2 decided-epoch archival pointer is unavailable"
            ) from error

    pointer = {
        "pointer_format": 1,
        "kind": "benchmark_v2_decided_epoch_archival_pointer",
        "archive_name": archive_root.name,
        "archival_manifest_sha256": _sha256_bytes(manifest_bytes),
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
            "Benchmark v2 decided-epoch archival pointer could not be recorded"
        ) from error


def _read_latest_pointer(history_root: Path) -> tuple[Path, dict]:
    pointer_path = history_root / _DECIDED_EPOCH_POINTER_FILENAME
    pointer_bytes = _require_regular_file(pointer_path, label="archival pointer")
    pointer = _parse_canonical_json_bytes(
        pointer_bytes,
        label="decided-epoch archival pointer",
    )
    archive_name = pointer.get("archive_name")
    manifest_sha256 = pointer.get("archival_manifest_sha256")
    if (
        set(pointer)
        != {
            "pointer_format",
            "kind",
            "archive_name",
            "archival_manifest_sha256",
        }
        or pointer.get("pointer_format") != 1
        or pointer.get("kind") != "benchmark_v2_decided_epoch_archival_pointer"
        or not isinstance(archive_name, str)
        or re.fullmatch(r"decided-epoch-[0-9a-f]{64}", archive_name) is None
        or not isinstance(manifest_sha256, str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", manifest_sha256) is None
    ):
        raise ValueError("Benchmark v2 decided-epoch archival pointer is invalid")
    archive_root = history_root / archive_name
    if (
        archive_root.resolve(strict=False).parent
        != history_root.resolve(strict=False)
    ):
        raise ValueError("Benchmark v2 decided-epoch archival pointer escapes history")
    return archive_root, pointer


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
        raise ValueError(f"Benchmark v2 decided-epoch {label} exists in two locations")
    if source_present:
        return _require_regular_file(source, label=label)
    if destination_present:
        return _require_regular_file(destination, label=label)
    raise ValueError(f"Benchmark v2 decided-epoch {label} is unavailable")


def _ensure_archive_root(archive_root: Path, manifest_bytes: bytes) -> None:
    if archive_root.is_symlink():
        raise ValueError("Benchmark v2 decided-epoch archive must not be a symlink")
    if archive_root.exists():
        if not archive_root.is_dir():
            raise ValueError("Benchmark v2 decided-epoch archive path is invalid")
    else:
        try:
            archive_root.mkdir(parents=True, exist_ok=False)
        except OSError as error:
            raise ValueError(
                "Benchmark v2 decided epoch could not be archived"
            ) from error
    _write_create_or_match(
        archive_root / _DECIDED_EPOCH_MANIFEST_FILENAME,
        manifest_bytes,
        label="archival manifest",
    )


def _move_archive_member(*, workspace: Path, archive_root: Path, name: str) -> None:
    source = workspace / name
    destination = archive_root / name
    source_present = source.exists() or source.is_symlink()
    destination_present = destination.exists() or destination.is_symlink()

    if source_present and destination_present:
        raise ValueError(f"Benchmark v2 decided-epoch archival member conflicts: {name}")
    if source_present:
        if not _member_shape_is_valid(source, name=name):
            raise ValueError(
                f"Benchmark v2 decided-epoch archival source is invalid: {name}"
            )
        if destination.is_symlink():
            raise ValueError(
                f"Benchmark v2 decided-epoch archival destination is invalid: {name}"
            )
        os.replace(source, destination)
        return
    if destination_present:
        if not _member_shape_is_valid(destination, name=name):
            raise ValueError(
                f"Benchmark v2 decided-epoch archival destination is invalid: {name}"
            )
        return
    raise ValueError(f"Benchmark v2 decided-epoch archival member is unavailable: {name}")


def _completion_bytes(manifest_bytes: bytes) -> bytes:
    return _canonical_json_bytes(
        {
            "completion_format": 1,
            "kind": "benchmark_v2_decided_epoch_archival_complete",
            "archival_manifest_sha256": _sha256_bytes(manifest_bytes),
        }
    )


def _result_from_manifest(
    *,
    status: str,
    archive_root: Path,
    manifest: dict,
    manifest_bytes: bytes,
) -> dict:
    return {
        "status": status,
        "archive_path": archive_root,
        "journal_spent_usd": manifest["journal_spent_usd"],
        "completed_case_count": manifest["completed_case_count"],
        "failed_case_count": manifest["failed_case_count"],
        "decision_lock_filenames": manifest["decision_lock_filenames"],
        "next_epoch_budget_usd": manifest["next_epoch_budget_usd"],
        "archival_manifest_sha256": _sha256_bytes(manifest_bytes),
    }


def install_decided_epoch_archival(namespace: dict) -> None:
    """Expose the canonical decided-epoch archival operation through the benchmark API."""

    geoffrey_track_slug = namespace["_geoffrey_track_slug"]

    def archive_benchmark_v2_decided_epoch(*, repository_root: Path) -> dict:
        """Archive one fully decided (WINNER/NO WINNER) epoch and reset the next epoch to $6.00.

        This is a deliberate operator act, never an automatic retry: it requires
        the live workspace to already hold a terminal decision lock and a fully
        resolved journal (every case ``completed`` or ``model_failed``, no
        dangling or in-progress attempts). A dangling/zero-spend epoch belongs to
        ``retire``/``reconcile`` instead; this function refuses those shapes.
        """

        repository_root = Path(repository_root)
        workspace = _workspace_root(repository_root)
        journal_path = workspace / "execution-state.json"
        history_root = workspace / "history"

        with _paid_runner_lock(repository_root):
            if workspace.is_symlink() or history_root.is_symlink():
                raise ValueError(
                    "Benchmark v2 decided-epoch archival workspace must not be a symlink"
                )

            active_journal = journal_path.exists() or journal_path.is_symlink()
            if active_journal:
                journal_bytes = _require_regular_file(
                    journal_path,
                    label="execution journal",
                )
                paid_authorization_bytes = _require_regular_file(
                    workspace / "paid-authorization.json",
                    label="paid authorization",
                )
                _parse_canonical_json_bytes(
                    paid_authorization_bytes,
                    label="decided-epoch paid authorization",
                )
                decision_lock_filenames = _read_decision_lock_filenames(
                    workspace,
                    geoffrey_track_slug=geoffrey_track_slug,
                )
                archive_members = _collect_archive_members(workspace)
                manifest, manifest_bytes, _details = _build_decided_epoch_manifest(
                    journal_bytes=journal_bytes,
                    archive_members=archive_members,
                    decision_lock_filenames=decision_lock_filenames,
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
                        "Benchmark v2 decided-epoch execution journal is unavailable"
                    )
                archive_root, pointer = _read_latest_pointer(history_root)
                if archive_root.is_symlink() or not archive_root.is_dir():
                    raise ValueError(
                        "Benchmark v2 decided-epoch archival archive is unavailable"
                    )
                manifest_bytes = _require_regular_file(
                    archive_root / _DECIDED_EPOCH_MANIFEST_FILENAME,
                    label="archival manifest",
                )
                if _sha256_bytes(manifest_bytes) != pointer["archival_manifest_sha256"]:
                    raise ValueError(
                        "Benchmark v2 decided-epoch archival manifest hash mismatch"
                    )
                manifest = _parse_canonical_json_bytes(
                    manifest_bytes,
                    label="decided-epoch archival manifest",
                )
                archive_members = manifest.get("archive_members")
                journal_bytes = _require_regular_file(
                    archive_root / "execution-state.json",
                    label="archived execution journal",
                )
                rebuilt_manifest, rebuilt_manifest_bytes, _details = (
                    _build_decided_epoch_manifest(
                        journal_bytes=journal_bytes,
                        archive_members=archive_members,
                        decision_lock_filenames=manifest.get("decision_lock_filenames") or [],
                    )
                )
                if rebuilt_manifest_bytes != manifest_bytes or rebuilt_manifest != manifest:
                    raise ValueError(
                        "Benchmark v2 decided-epoch archival manifest is invalid"
                    )
                paid_authorization_bytes = _read_member_file_from_active_or_archive(
                    workspace=workspace,
                    archive_root=archive_root,
                    name="paid-authorization.json",
                    label="paid authorization",
                )
                _parse_canonical_json_bytes(
                    paid_authorization_bytes,
                    label="decided-epoch paid authorization",
                )
                expected_archive_root = _archive_root_for_material(
                    history_root,
                    manifest_bytes=manifest_bytes,
                    journal_bytes=journal_bytes,
                    paid_authorization_bytes=paid_authorization_bytes,
                )
                if expected_archive_root != archive_root:
                    raise ValueError(
                        "Benchmark v2 decided-epoch archival archive identity mismatch"
                    )

                completion_path = archive_root / _DECIDED_EPOCH_COMPLETE_FILENAME
                if completion_path.exists() or completion_path.is_symlink():
                    completion = _require_regular_file(
                        completion_path,
                        label="archival completion marker",
                    )
                    if completion != _completion_bytes(manifest_bytes):
                        raise ValueError(
                            "Benchmark v2 decided-epoch archival completion marker conflicts"
                        )
                    for name in archive_members:
                        source = workspace / name
                        if source.exists() or source.is_symlink():
                            raise ValueError(
                                "Benchmark v2 completed decided-epoch archival conflicts "
                                "with active state"
                            )
                        if not _member_shape_is_valid(archive_root / name, name=name):
                            raise ValueError(
                                "Benchmark v2 completed decided-epoch archival is incomplete"
                            )
                    return _result_from_manifest(
                        status="already_archived",
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
                    "Benchmark v2 decided epoch could not be archived"
                ) from error

            _write_create_or_match(
                archive_root / _DECIDED_EPOCH_COMPLETE_FILENAME,
                _completion_bytes(manifest_bytes),
                label="archival completion marker",
            )
            return _result_from_manifest(
                status="archived",
                archive_root=archive_root,
                manifest=manifest,
                manifest_bytes=manifest_bytes,
            )

    namespace["archive_benchmark_v2_decided_epoch"] = archive_benchmark_v2_decided_epoch
