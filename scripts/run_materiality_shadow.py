#!/usr/bin/env python3
"""TASK-133: run the shadow materiality filter for episodes already in Human Review.

The Worker runs the filter only when it sends a new episode to Human Review.
This operator command runs the same code for episodes that are already
waiting, so shadow evidence exists without waiting for new episodes.

Default (dry run): for every episode with pending review cards, print how
many cards there are, how many already carry current evidence, and what the
deterministic rule alone would do. No model call, nothing written.

``--execute``: run ``podcast_engine.materiality_shadow.run_materiality_shadow``
for those episodes (or only ``--episode KEY``) -- the Worker's exact path:
verified ``podcast-materiality-judge`` preset, per-call episode AI-budget
reservations, generation-CAS evidence writes onto the review record. It is
idempotent: cards that already carry current evidence are skipped and cost
nothing. Requires ``PODCAST_TRANSCRIPT_REVIEW_API_KEY`` and GCS write access
to the review record. The review page, tiers and decisions do not change.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from compiler.materiality import materiality_inputs, rule_verdict  # noqa: E402
from podcast_engine.human_review import load_review_record, pending_review_items  # noqa: E402
from podcast_engine.materiality_shadow import EVIDENCE_KEY, evidence_identity  # noqa: E402


def queued_episodes(episodes: list[dict], load_record: Callable[[str], dict]) -> list[dict]:
    """Episodes whose review record still has pending cards, with a dry-run plan."""

    rows = []
    for episode in episodes:
        key = episode.get("episode_key") if isinstance(episode, dict) else None
        if not isinstance(key, str):
            continue
        try:
            record = load_record(key)
        except FileNotFoundError:
            continue
        items = pending_review_items(record)
        if not items:
            continue
        plan = Counter()
        for item in items:
            inputs = materiality_inputs(item)
            if inputs is None:
                plan["no_text"] += 1
                continue
            existing = item.get(EVIDENCE_KEY)
            if isinstance(existing, dict) and existing.get("identity") == evidence_identity(inputs):
                plan["already_recorded"] += 1
                continue
            route, _ = rule_verdict(inputs)
            plan[{"advertisement": "rule_immaterial", "immaterial": "rule_immaterial", "reviewer": "reviewer_rule", "judge": "for_judge"}[route]] += 1
        rows.append({"episode": episode, "cards": len(items), "plan": dict(plan)})
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--execute", action="store_true", help="Run the filter (paid judge calls)")
    parser.add_argument("--episode", action="append", default=[], help="Only this episode_key (repeatable)")
    args = parser.parse_args(argv)

    from podcast_engine.storage import load_episodes

    rows = queued_episodes(load_episodes(), load_review_record)
    if args.episode:
        rows = [row for row in rows if row["episode"]["episode_key"] in set(args.episode)]
    if not rows:
        print("No episode with pending review cards.")
        return 0
    for row in rows:
        episode = row["episode"]
        print(f"{episode['episode_key']}  {episode.get('title')!s:.60}  cards={row['cards']}  {json.dumps(row['plan'])}")
    judge_cards = sum(row["plan"].get("for_judge", 0) for row in rows)
    print(f"\nCards for the judge: {judge_cards} (up to 3 calls each; about USD 0.002 per card)")
    if not args.execute:
        print("Dry run: nothing sent or written. Add --execute to run.")
        return 0

    from podcast_engine.materiality_shadow import run_materiality_shadow

    for row in rows:
        summary = run_materiality_shadow(row["episode"])
        print(f"{row['episode']['episode_key']}: {json.dumps(summary)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
