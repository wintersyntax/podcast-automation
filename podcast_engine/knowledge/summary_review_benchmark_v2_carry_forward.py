"""Zero-paid carry-forward of a retired epoch's terminal-case evidence."""

from __future__ import annotations

from pathlib import Path

from .summary_review_benchmark_v2_epoch_retirement import (
    _read_latest_pointer,
    _require_regular_file,
)
from .summary_review_benchmark_v2_journal import (
    seed_execution_journal_from_retired_epoch,
)
from .summary_review_benchmark_v2_safety import (
    _paid_runner_lock,
    _parse_canonical_json_bytes,
    _workspace_root,
)


def carry_forward_benchmark_v2_completed_cases(*, repository_root: Path) -> dict:
    """Seed a fresh epoch's journal with a retired epoch's durable terminal
    cases -- both ``completed`` and ``model_failed`` -- when one is
    available and no epoch is already active.

    This is a safe, explicit no-op (``seeded: False``) when there is
    nothing to carry forward yet, or when a journal is already active --
    neither is an anomaly, so operators can call this unconditionally
    between ``retire`` and ``preflight``/``authorize``. It fails closed
    (raises ``ValueError``) only on an actual inconsistency in the
    retirement archive itself.

    This does **not** independently recompute the current benchmark
    configuration's binding: the very next ``preflight``/``authorize``
    call already re-validates the seeded journal against a freshly
    computed binding -- the same check that already guards ordinary
    mid-epoch resumption -- so config drift since the archive was made
    is still caught, just one step later, at no extra paid cost.
    """
    repository_root = Path(repository_root)
    workspace = _workspace_root(repository_root)
    journal_path = workspace / "execution-state.json"
    history_root = workspace / "history"
    pointer_path = history_root / "latest-partial-spend-retirement.json"

    with _paid_runner_lock(repository_root):
        if journal_path.exists() or journal_path.is_symlink():
            return {"seeded": False, "reason": "epoch_already_active"}

        if not pointer_path.is_file() or pointer_path.is_symlink():
            return {"seeded": False, "reason": "no_retired_epoch"}

        archive_root, _pointer = _read_latest_pointer(history_root)
        retired_journal_bytes = _require_regular_file(
            archive_root / "execution-state.json",
            label="archived execution journal",
        )
        retired_journal = _parse_canonical_json_bytes(
            retired_journal_bytes,
            label="archived execution journal",
        )
        benchmark_binding = retired_journal.get("benchmark_binding")
        ordered_case_keys = retired_journal.get("ordered_case_keys")

        seeded_state = seed_execution_journal_from_retired_epoch(
            journal_path,
            retired_journal_bytes=retired_journal_bytes,
            benchmark_binding=benchmark_binding,
            ordered_case_keys=ordered_case_keys,
        )
        return {
            "seeded": True,
            "archive_root": archive_root.name,
            "carried_forward_case_keys": list(
                seeded_state["completed_case_keys"]
            ),
            "carried_forward_failed_case_keys": list(
                seeded_state["failed_case_keys"]
            ),
            "spent_usd": seeded_state["spent_usd"],
        }


def install_execution_journal_carry_forward(namespace: dict) -> None:
    """Expose the canonical carry-forward operation through the benchmark API."""

    namespace["carry_forward_benchmark_v2_completed_cases"] = (
        carry_forward_benchmark_v2_completed_cases
    )
