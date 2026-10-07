#!/usr/bin/env python3
"""Safely reconcile a stuck Human Review recompile with Cloud Run metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from podcast_engine.human_review import reconcile_review_recompile


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-key", required=True, help="Episode key to inspect.")
    parser.add_argument(
        "--review-generation",
        help="Optional exact review-generation fingerprint to inspect.",
    )
    parser.add_argument(
        "--expected-result-generation",
        help=(
            "Transitional operator-only result generation for a pre-linkage "
            "successful request; must equal the current resolver generation."
        ),
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--check",
        action="store_true",
        help="Read only (the default): inspect but never write GCS.",
    )
    action.add_argument(
        "--apply",
        action="store_true",
        help="Persist an exact Cloud Run reconciliation using a conditional GCS write.",
    )
    args = parser.parse_args(argv)
    try:
        result = reconcile_review_recompile(
            args.episode_key,
            apply=args.apply,
            review_generation=args.review_generation,
            expected_result_generation=args.expected_result_generation,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(json.dumps({"status": "error", "reason": str(error)[:240]}), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
