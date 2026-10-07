"""Read-only side-by-side comparison of the two dual Geoffrey tracks (TASK-052).

Two designated qualification candidates each independently attempt the real
Geoffrey gate (see ``select_qualification_designated_tracks`` /
``run_geoffrey_final_gate_with_designated_track_lock`` in the canonical
``.impl``). Neither track is auto-selected when both succeed -- a human
compares the two real outputs and picks one. This module never selects,
never mutates anything, and never triggers paid execution: it only reads
whatever designated-tracks lock, per-candidate Geoffrey final locks, and
execution-journal evidence already exist for the live epoch and renders
them side by side. Same safety class as ``agent-session-status``.
"""

from __future__ import annotations

import json
from pathlib import Path

from .summary_review_benchmark_v2_safety import (
    _parse_canonical_json_bytes,
    _workspace_root,
)


def _read_optional_canonical_json(path: Path, *, label: str) -> dict | None:
    if not path.exists() and not path.is_symlink():
        return None
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Benchmark v2 Geoffrey comparison {label} is invalid")
    try:
        data = path.read_bytes()
    except OSError as error:
        raise ValueError(
            f"Benchmark v2 Geoffrey comparison {label} is unreadable"
        ) from error
    return _parse_canonical_json_bytes(data, label=label)


def _read_optional_journal(path: Path) -> dict | None:
    if not path.exists() and not path.is_symlink():
        return None
    if not path.is_file() or path.is_symlink():
        raise ValueError(
            "Benchmark v2 Geoffrey comparison execution journal is invalid"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            "Benchmark v2 Geoffrey comparison execution journal is unreadable"
        ) from error
    if not isinstance(payload, dict):
        raise ValueError(
            "Benchmark v2 Geoffrey comparison execution journal is invalid"
        )
    return payload


def _track_runs_from_journal(journal: dict | None, *, slug: str) -> list[dict]:
    if journal is None:
        return []
    completed_cases = journal.get("completed_cases")
    if not isinstance(completed_cases, list):
        return []
    prefix = f"geoffrey::{slug}::run-"
    runs = []
    for item in completed_cases:
        if not isinstance(item, dict):
            continue
        case_key = item.get("case_key")
        if not isinstance(case_key, str) or not case_key.startswith(prefix):
            continue
        suffix = case_key[len(prefix):]
        if not suffix.isdigit():
            continue
        case_result = item.get("case_result")
        if not isinstance(case_result, dict):
            continue
        runs.append(
            {
                "run_number": int(suffix),
                "audit_result": case_result.get("audit_result"),
                "accepted_final_markdown": case_result.get(
                    "accepted_final_markdown"
                ),
                "served_provider": case_result.get("served_provider"),
            }
        )
    runs.sort(key=lambda run: run["run_number"])
    return runs


def install_geoffrey_candidate_comparison(namespace: dict) -> None:
    """Expose the read-only dual Geoffrey track comparison on the benchmark API."""

    def compare_benchmark_v2_geoffrey_candidates(*, repository_root: Path) -> dict:
        """Report both designated Geoffrey tracks' current status side by side.

        Reads whatever qualification designated-tracks lock, per-candidate
        Geoffrey final locks, and execution-journal evidence already exist
        for the live workspace. Makes no selection between the two tracks
        and writes nothing -- the choice of which real output to use, if
        both reach a terminal decision, remains a separate, explicitly
        authorized human action.
        """

        repository_root = Path(repository_root)
        workspace = _workspace_root(repository_root)
        lock_dir = workspace / "locks"

        config, _config_bytes = namespace["load_canonical_benchmark_v2_config"](
            root=repository_root
        )
        designated_candidate_ids = (
            config.get("geoffrey_designated_candidates")
            if isinstance(config, dict)
            else None
        )
        if (
            not isinstance(designated_candidate_ids, list)
            or not designated_candidate_ids
        ):
            raise ValueError(
                "Benchmark v2 Geoffrey comparison requires designated candidates"
            )

        designated_tracks_lock = _read_optional_canonical_json(
            lock_dir / "qualification.designated-tracks.lock.json",
            label="designated-tracks lock",
        )
        journal = _read_optional_journal(workspace / "execution-state.json")

        candidates = {}
        for candidate_id in designated_candidate_ids:
            slug = namespace["_geoffrey_track_slug"](candidate_id)
            track_lock_dir = lock_dir / "geoffrey" / slug

            eligible = None
            if designated_tracks_lock is not None:
                tracks = designated_tracks_lock.get("designated_tracks")
                entry = tracks.get(candidate_id) if isinstance(tracks, dict) else None
                if isinstance(entry, dict):
                    eligible = entry.get("eligible")

            winner_lock = _read_optional_canonical_json(
                track_lock_dir / "winner.lock.json",
                label="Geoffrey winner lock",
            )
            no_winner_lock = _read_optional_canonical_json(
                track_lock_dir / "no_winner.lock.json",
                label="Geoffrey no_winner lock",
            )
            if winner_lock is not None and no_winner_lock is not None:
                raise ValueError(
                    "Benchmark v2 Geoffrey comparison found conflicting terminal locks"
                )
            terminal_lock = winner_lock if winner_lock is not None else no_winner_lock

            candidates[candidate_id] = {
                "slug": slug,
                "eligible": eligible,
                "decision": (
                    terminal_lock.get("decision") if terminal_lock else None
                ),
                "sanity_review": (
                    terminal_lock.get("sanity_review") if terminal_lock else None
                ),
                "geoffrey_runs": _track_runs_from_journal(journal, slug=slug),
            }

        return {
            "designated_candidates": list(designated_candidate_ids),
            "candidates": candidates,
        }

    namespace["compare_benchmark_v2_geoffrey_candidates"] = (
        compare_benchmark_v2_geoffrey_candidates
    )
