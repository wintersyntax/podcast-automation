#!/usr/bin/env python3
"""Inspect or reconcile legacy Third-ASR budget-ledger identities for one episode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from podcast_engine.ai_budget import (
    BudgetLedgerError,
    reconcile_third_asr_budget_identity,
)
from podcast_engine.human_review import load_review_record


def _classifications(args: argparse.Namespace) -> dict[str, str]:
    result: dict[str, str] = {}
    for fingerprint in args.legacy_current_fingerprint:
        if fingerprint in result:
            raise ValueError(f"Fingerprint classified more than once: {fingerprint}")
        result[fingerprint] = "legacy_current_generation"
    for fingerprint in args.previous_source_fingerprint:
        if fingerprint in result:
            raise ValueError(f"Fingerprint classified more than once: {fingerprint}")
        result[fingerprint] = "previous_source_generation"
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-key", required=True, help="Canonical 24-hex episode key.")
    parser.add_argument(
        "--legacy-current-fingerprint",
        action="append",
        default=[],
        help=(
            "Classify one inventoried non-canonical ledger as legacy spend from "
            "the CURRENT source generation. Repeat once per matching ledger."
        ),
    )
    parser.add_argument(
        "--previous-source-fingerprint",
        action="append",
        default=[],
        help=(
            "Classify one inventoried non-canonical ledger as a legitimate "
            "PREVIOUS source generation. Repeat once per matching ledger."
        ),
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--check",
        action="store_true",
        help="Read-only inventory/plan (default).",
    )
    action.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Apply the exact classifications with conditional writes. Run only "
            "after pre-fix Third-ASR writers are quiesced."
        ),
    )
    args = parser.parse_args(argv)

    try:
        record = load_review_record(args.episode_key)
        source_fingerprint = record.get("source_fingerprint")
        if not isinstance(source_fingerprint, str) or not source_fingerprint:
            raise ValueError("Current review record has no canonical source_fingerprint")
        result = reconcile_third_asr_budget_identity(
            args.episode_key,
            source_fingerprint,
            _classifications(args),
            apply=args.apply,
        )
    except (BudgetLedgerError, FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        print(
            json.dumps({"status": "error", "reason": str(error)[:400]}, sort_keys=True),
            file=sys.stderr,
        )
        return 1

    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
