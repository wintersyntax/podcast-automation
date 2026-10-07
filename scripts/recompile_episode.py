"""List or request recompilation of episodes waiting in Human Review (TASK-131).

Dry run (default) lists every episode waiting in Human Review with its
pending cards and recorded human decisions. ``--execute KEY [KEY ...]``
marks those episodes' compiler state ready, so the next Worker run
compiles the existing sources again with the current compiler and replaces
the review queue. Nothing is re-transcribed. An episode with recorded human
decisions is refused unless ``--discard-decisions`` is given, because a
replaced queue may no longer contain the decided cards.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from podcast_engine.human_review import load_review_record, pending_review_items  # noqa: E402
from podcast_engine.storage import load_episodes, request_recompilation  # noqa: E402


def _review_counts(episode_key: str) -> tuple[int, int]:
    try:
        record = load_review_record(episode_key)
    except (FileNotFoundError, ValueError):
        return 0, 0
    decisions = [
        decision for decision in record.get("human_decisions", [])
        if isinstance(decision, dict)
    ]
    return len(pending_review_items(record)), len(decisions)


def candidates() -> list[dict]:
    rows = []
    for episode in load_episodes():
        if episode.get("status", {}).get("compiler", {}).get("state") != "review_required":
            continue
        key = episode.get("episode_key")
        pending, decisions = _review_counts(key)
        rows.append(
            {
                "episode_key": key,
                "podcast": episode.get("podcast"),
                "title": episode.get("title"),
                "pending_cards": pending,
                "human_decisions": decisions,
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--execute", nargs="+", metavar="EPISODE_KEY")
    parser.add_argument("--discard-decisions", action="store_true")
    args = parser.parse_args(argv)

    rows = candidates()
    for row in rows:
        print(
            f"{row['episode_key']}  cards={row['pending_cards']:<3} "
            f"decisions={row['human_decisions']:<3} {row['podcast']} -- {row['title']}"
        )
    if not args.execute:
        print(f"DRY RUN: {len(rows)} episode(s) wait in Human Review.")
        return 0

    by_key = {row["episode_key"]: row for row in rows}
    for key in args.execute:
        row = by_key.get(key)
        if row is None:
            print(f"REFUSED {key}: not an episode waiting in Human Review")
            return 1
        if row["human_decisions"] and not args.discard_decisions:
            print(
                f"REFUSED {key}: {row['human_decisions']} recorded human decision(s) "
                "may no longer apply; pass --discard-decisions to proceed"
            )
            return 1
    for key in args.execute:
        request_recompilation(key)
        print(f"REQUESTED {key}: the next Worker run recompiles it with the current compiler")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
