"""List or reconcile Whisper or Third-ASR budget integrity failures (TASK-125, TASK-126).

Dry run (default) lists every budget ledger of the given episodes that holds
an unreconciled integrity failure. ``--execute`` settles one Whisper
transcription or Third-ASR attempt at its recorded actual cost; the operator restates the
exact attempt id and actual cost printed by the dry run, and any mismatch
fails closed. The failure stays in the ledger's append-only
``integrity_reconciliations`` list.
"""

from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from podcast_engine.ai_budget import (  # noqa: E402
    BudgetLedgerError,
    inventory_budget_integrity_failures,
    reconcile_whisper_integrity_failure,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("episode_keys", nargs="+", metavar="EPISODE_KEY")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--attempt")
    parser.add_argument("--actual-usd")
    parser.add_argument("--reason")
    args = parser.parse_args(argv)

    rows = []
    for key in args.episode_keys:
        rows.extend(inventory_budget_integrity_failures(key))
    for row in rows:
        print(
            f"{row['episode_key']}  {row['stage']}  {row['state']}  "
            f"attempt={row['attempt_id']}  reserved={row['reserved_usd']}  "
            f"actual={row['actual_usd']}  detected={row['detected_at']}"
        )
    if not args.execute:
        print(f"DRY RUN: {len(rows)} unreconciled integrity failure(s).")
        return 0

    if len(args.episode_keys) != 1 or not (args.attempt and args.actual_usd and args.reason):
        print("REFUSED: --execute needs exactly one EPISODE_KEY plus --attempt, --actual-usd and --reason")
        return 2
    matches = [row for row in rows if row["attempt_id"] == args.attempt]
    if len(matches) != 1:
        print(f"REFUSED: attempt {args.attempt} is not exactly one listed integrity failure")
        return 1
    try:
        actual = Decimal(args.actual_usd)
    except InvalidOperation:
        print("REFUSED: --actual-usd is not a decimal amount")
        return 2
    row = matches[0]
    try:
        attempt = reconcile_whisper_integrity_failure(
            row["episode_key"],
            row["source_fingerprint"],
            attempt_id=args.attempt,
            actual_usd=actual,
            reason=args.reason,
        )
    except (BudgetLedgerError, ValueError) as error:
        print(f"REFUSED: {error}")
        return 1
    print(
        f"RECONCILED {row['episode_key']} {args.attempt}: settled at "
        f"{attempt['settled_usd']} USD; the ledger admits reservations again"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
