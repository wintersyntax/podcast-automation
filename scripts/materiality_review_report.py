#!/usr/bin/env python3
"""TASK-133: how the reviewer used the materiality groups -- the history for tuning the filter.

Read-only. For every episode with a Human Review record it reads the audited
decisions that carry a ``materiality`` snapshot and the record's
``materiality_review_log``, and reports per episode: decisions by group,
control-sample cards where the reviewer overruled the filter (the signal that
settled cards cannot be trusted), proposals he changed, "click one" picks,
time spent on quick cards, whether he opened the settled list, and his notes.
``--json`` prints the same as JSON. Nothing is written.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

GROUPS = ("sample", "click_one", "proposal", "settled")


def _reading(decision: dict, source: str | None) -> str:
    item = decision.get("review_item") if isinstance(decision.get("review_item"), dict) else {}
    focus = item.get("focus") if isinstance(item.get("focus"), dict) else {}
    if not source:
        return ""
    text = focus.get(f"{source}_text") if focus.get("scope") == "partial" else item.get(f"{source}_text")
    return str(text or "")[:80]


def summarize_record(record: dict) -> dict:
    """Per-episode summary of materiality decisions and the review log."""

    decisions = [d for d in record.get("human_decisions", []) if isinstance(d, dict)]
    grouped = [d for d in decisions if isinstance(d.get("materiality"), dict)]
    by_group = {group: 0 for group in GROUPS}
    overruled: list[dict] = []
    seconds = 0.0
    timed = 0
    for decision in grouped:
        snapshot = decision["materiality"]
        group = snapshot.get("group")
        by_group[group] = by_group.get(group, 0) + 1
        if isinstance(snapshot.get("seconds"), (int, float)):
            seconds += snapshot["seconds"]
            timed += 1
        if snapshot.get("agrees_with_filter") is False:
            overruled.append({
                "id": decision.get("id"),
                "group": group,
                "step": snapshot.get("step"),
                "filter": snapshot.get("filter_source"),
                "filter_text": _reading(decision, snapshot.get("filter_source")),
                "chosen": decision.get("chosen_source"),
                "chosen_text": _reading(decision, decision.get("chosen_source")),
            })
    log = [entry for entry in record.get("materiality_review_log", []) if isinstance(entry, dict)]
    return {
        "decisions": len(decisions),
        "outside_groups": len(decisions) - len(grouped),
        "by_group": by_group,
        "sample_overruled": [o for o in overruled if o["group"] == "sample"],
        "proposals_changed": [o for o in overruled if o["group"] == "proposal"],
        "quick_seconds": round(seconds),
        "quick_timed_cards": timed,
        "settled_list_opened": any(entry.get("settled_list_opened") for entry in log),
        "notes": [entry["note"] for entry in log if entry.get("note")],
        "saves": len(log),
    }


def _minutes(seconds: int) -> str:
    return f"{seconds // 60} min {seconds % 60} s"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    from podcast_engine.human_review import load_review_record
    from podcast_engine.storage import load_episodes

    rows = []
    for episode in load_episodes():
        key = episode.get("episode_key")
        try:
            record = load_review_record(key)
        except (FileNotFoundError, ValueError):
            continue
        summary = summarize_record(record)
        if not summary["saves"] and not any(summary["by_group"].values()):
            continue
        rows.append({"episode_key": key, "title": episode.get("title"), **summary})
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return 0
    for row in rows:
        groups = row["by_group"]
        print(f"\n{row['title']} ({row['episode_key']})")
        print(f"  decisions {row['decisions']}: settled {groups['settled']}, sample {groups['sample']}, "
              f"click one {groups['click_one']}, proposals {groups['proposal']}, other review {row['outside_groups']}")
        print(f"  control sample overruled: {len(row['sample_overruled'])} of {groups['sample']}")
        for o in row["sample_overruled"]:
            print(f"    #{o['id']} ({o['step']}): filter {o['filter']} {o['filter_text']!r} -> you {o['chosen']} {o['chosen_text']!r}")
        print(f"  proposals changed: {len(row['proposals_changed'])} of {groups['proposal']}")
        for o in row["proposals_changed"]:
            print(f"    #{o['id']} ({o['step']}): filter {o['filter']} {o['filter_text']!r} -> you {o['chosen']} {o['chosen_text']!r}")
        if row["quick_timed_cards"]:
            print(f"  quick cards: {_minutes(row['quick_seconds'])} over {row['quick_timed_cards']} cards "
                  f"({row['quick_seconds'] / row['quick_timed_cards']:.0f} s per card)")
        print(f"  settled list opened: {'yes' if row['settled_list_opened'] else 'no'}")
        for note in row["notes"]:
            print(f"  note: {note}")
    if not rows:
        print("No materiality review activity recorded yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
