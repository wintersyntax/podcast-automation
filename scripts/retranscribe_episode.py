"""List or request Whisper re-transcription for unfinished episodes (TASK-125).

Dry run (default) lists every unfinished episode whose canonical Whisper
source was produced by another producer, with its pending review cards and
recorded human decisions. ``--execute KEY [KEY ...]`` marks those episodes'
Whisper source pending, so the next Worker run re-transcribes them with the
current OpenRouter producer and recompiles them. An episode with recorded
human decisions is refused unless ``--discard-decisions`` is given, because a
new source generation invalidates them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from podcast_engine.episode_contract import paths_for  # noqa: E402
from podcast_engine.human_review import load_review_record, pending_review_items  # noqa: E402
from podcast_engine.storage import (  # noqa: E402
    RETRANSCRIBABLE_COMPILER_STATES,
    download_gcs_bytes,
    file_exists_in_gcs,
    load_episodes,
    request_whisper_retranscription,
)
from podcast_engine.whisper_provenance import (  # noqa: E402
    whisper_metadata_matches_current_producer,
)


def _whisper_is_current(episode_key: str) -> bool:
    metadata_path = paths_for(episode_key)["whisper_metadata"]
    if not file_exists_in_gcs(metadata_path):
        return False
    try:
        metadata = json.loads(download_gcs_bytes(metadata_path).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    return whisper_metadata_matches_current_producer(metadata)


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
        key = episode.get("episode_key")
        status = episode.get("status", {})
        compiler_state = status.get("compiler", {}).get("state")
        whisper_state = status.get("whisper", {}).get("state")
        if compiler_state not in RETRANSCRIBABLE_COMPILER_STATES or whisper_state != "ready":
            continue
        if _whisper_is_current(key):
            continue
        pending, decisions = _review_counts(key)
        rows.append(
            {
                "episode_key": key,
                "podcast": episode.get("podcast"),
                "title": episode.get("title"),
                "compiler_state": compiler_state,
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
            f"{row['episode_key']}  {row['compiler_state']:<16} "
            f"cards={row['pending_cards']:<3} decisions={row['human_decisions']:<3} "
            f"{row['podcast']} -- {row['title']}"
        )
    if not args.execute:
        print(f"DRY RUN: {len(rows)} unfinished episode(s) use another Whisper producer.")
        return 0

    by_key = {row["episode_key"]: row for row in rows}
    for key in args.execute:
        row = by_key.get(key)
        if row is None:
            print(f"REFUSED {key}: not an unfinished episode with a non-current Whisper source")
            return 1
        if row["human_decisions"] and not args.discard_decisions:
            print(
                f"REFUSED {key}: {row['human_decisions']} recorded human decision(s) "
                "would be invalidated; pass --discard-decisions to proceed"
            )
            return 1
    for key in args.execute:
        request_whisper_retranscription(key)
        print(f"REQUESTED {key}: Whisper source is pending; the next Worker run re-transcribes it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
