"""Read-only live corpus checkpoint status and fail-closed CLI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import storage
from .corpus_checkpoints import (
    CheckpointDecision,
    CorpusEvidence,
    evaluate_corpus,
    evaluate_thresholds,
    load_checkpoint_policy,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY_PATH = ROOT / "config" / "corpus-checkpoints.json"

EXIT_CLEAR = 0
EXIT_BLOCKED = 2
EXIT_UNKNOWN = 3


def _load_report_json(gcs_path: str) -> dict:
    """Read and validate one canonical compiler report without mutating storage."""

    payload = storage.download_gcs_bytes(gcs_path)
    value = json.loads(payload.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Compiler report is not an object: {gcs_path}")
    return value


def collect_live_status(
    policy_path: str | Path = DEFAULT_POLICY_PATH,
) -> tuple[CorpusEvidence, CheckpointDecision, dict]:
    """Collect current canonical corpus evidence from GCS, read-only."""

    policy = load_checkpoint_policy(policy_path)
    episodes = storage.load_episodes()
    corpus = evaluate_corpus(episodes, _load_report_json, policy)
    decision = evaluate_thresholds(policy, corpus.eligible_count)
    return corpus, decision, policy


def _checkpoint_status_line(threshold: dict, eligible_count: int) -> str:
    count = threshold["count"]
    status = threshold["status"]
    if status == "accepted":
        label = "ACCEPTED"
    elif eligible_count >= count:
        label = "REACHED / NOT ACCEPTED"
    else:
        label = status
    return f"Checkpoint {count}: {label}"


def render_status(
    corpus: CorpusEvidence,
    decision: CheckpointDecision,
    policy: dict,
) -> str:
    """Render deterministic operator-facing checkpoint evidence."""

    lines = [
        "===== CORPUS CHECKPOINT STATUS =====",
        f"Eligibility policy: {policy['eligibility_policy']}",
        f"Eligible comparable episodes: {corpus.eligible_count}",
        f"Level 2 telemetry-bearing episodes: {len(corpus.level2_episode_keys)}",
        (
            "Q46 alignment-diagnostics-bearing episodes: "
            f"{len(corpus.alignment_episode_keys)}"
        ),
    ]
    lines.extend(
        _checkpoint_status_line(threshold, corpus.eligible_count)
        for threshold in policy["thresholds"]
    )
    lines.extend(
        [
            f"Blocking: {'YES' if decision.blocking_threshold is not None else 'NO'}",
            "Normal podcast processing: ALLOWED",
        ]
    )
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Report fail-closed Podcast Worker corpus checkpoint state."
    )
    parser.add_argument(
        "--gate",
        action="store_true",
        help="Use the status as a deployment gate (same evidence and exit contract).",
    )
    parser.add_argument(
        "--policy",
        default=str(DEFAULT_POLICY_PATH),
        help="Path to the versioned corpus checkpoint policy.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Render current status; return 0 clear, 2 blocked, or 3 unknown."""

    args = _build_parser().parse_args(argv)
    try:
        corpus, decision, policy = collect_live_status(args.policy)
    except Exception as exc:  # CLI boundary is intentionally fail-closed.
        print(f"CORPUS CHECKPOINT STATUS: UNKNOWN: {exc}", file=sys.stderr)
        return EXIT_UNKNOWN

    print(render_status(corpus, decision, policy))
    if decision.blocking_threshold is not None:
        return EXIT_BLOCKED
    return EXIT_CLEAR


if __name__ == "__main__":
    raise SystemExit(main())
