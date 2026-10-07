#!/usr/bin/env python3
"""One-time regeneration of existing knowledge notes with the TASK-118 writer.

The worker never revisits an episode whose summary is already ``ready``, so
notes written by the retired summary chain are replaced by this operator tool.
It runs on the operator's machine with operator GCS credentials and the
production ``PODCAST_KNOWLEDGE_API_KEY`` / ``PODCAST_KNOWLEDGE_WRITER_PRESET``
(from ``.env``), and goes through exactly the production path:
``orchestration.build_knowledge_note`` (writer, checks, render, tag registry,
manifest, per-episode AI budget ledger) followed by the same
``update_episode`` call the pipeline makes. Vault Sync then exports the changed
notes on its normal schedule.

Default is a dry run: it lists every ready episode, whether it already has a
writer note, its current AI-budget ledger commitment and headroom under the
hard cap, and the expected cost. ``--execute`` regenerates one episode at a
time, stops at the first failure, and never starts an episode once the actual
spend of this run plus one conservative writer reservation would exceed
``--budget-usd``.
"""

from __future__ import annotations

import argparse
from decimal import Decimal
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

WRITER_RESERVATION_USD = Decimal("0.75")
EXPECTED_WRITER_USD = Decimal("0.35")
READY_STATES = frozenset({"ready", "completed"})


def plan(
    episodes: list[dict],
    manifest_policy: dict,
    ledger_committed: dict,
    *,
    cap: Decimal,
    writer_policy: str,
    render_versions: dict | None = None,
    render_version: str | None = None,
) -> list[dict]:
    """Pure: one row per ready episode with its regeneration status.

    A writer note rendered by an older renderer only needs a free re-render
    from the stored checked note (``needs_rerender``), never a model call.
    """

    rows = []
    for episode in episodes:
        key = episode.get("episode_key")
        status = episode.get("status", {})
        if not key or status.get("summary", {}).get("state") not in READY_STATES:
            continue
        if status.get("compiler", {}).get("state") != "completed":
            continue
        committed = ledger_committed.get(key, Decimal("0"))
        headroom = cap - committed
        policy = manifest_policy.get(key)
        rows.append({
            "episode_key": key,
            "podcast": episode.get("podcast"),
            "title": episode.get("title"),
            "published": episode.get("published"),
            "current_policy": policy,
            "needs_regeneration": policy != writer_policy,
            "needs_rerender": (
                policy == writer_policy
                and render_version is not None
                and (render_versions or {}).get(key) != render_version
            ),
            "ledger_committed_usd": str(committed),
            "headroom_usd": str(headroom),
            "fits_one_attempt": headroom >= WRITER_RESERVATION_USD,
        })
    rows.sort(key=lambda row: str(row.get("published") or ""), reverse=True)
    return rows


def _load_state():
    from podcast_engine.ai_budget import EPISODE_AI_HARD_CAP_USD, episode_spend_summary
    from podcast_engine.episode_contract import paths_for
    from podcast_engine.episode_generation import source_generation_fingerprint_from_episode
    from podcast_engine.knowledge.models import NOTE_WRITER_POLICY_VERSION
    from podcast_engine.storage import get_bucket, load_episodes

    bucket = get_bucket()
    episodes = load_episodes()
    manifest_policy, ledger_committed, render_versions = {}, {}, {}
    for episode in episodes:
        key = episode.get("episode_key")
        if not key or episode.get("status", {}).get("summary", {}).get("state") not in READY_STATES:
            continue
        blob = bucket.blob(paths_for(key)["summary_metadata"])
        if blob.exists():
            try:
                manifest = json.loads(blob.download_as_text(encoding="utf-8"))
                manifest_policy[key] = (manifest.get("summary") or {}).get("policy_version")
                render_versions[key] = (manifest.get("summary") or {}).get("render_version") or "note-render-v1"
            except (ValueError, UnicodeDecodeError):
                manifest_policy[key] = "unreadable"
        try:
            fingerprint = source_generation_fingerprint_from_episode(episode)
            ledger_committed[key] = Decimal(episode_spend_summary(key, fingerprint)["committed_usd"])
        except Exception as error:  # noqa: BLE001 -- reported, not fatal, in a dry run
            print(f"warning: {key}: ledger unavailable ({type(error).__name__})", file=sys.stderr)
    return episodes, manifest_policy, ledger_committed, render_versions, EPISODE_AI_HARD_CAP_USD, NOTE_WRITER_POLICY_VERSION


def _regenerate(episode: dict) -> Decimal:
    from podcast_engine.episode_contract import paths_for
    from podcast_engine.knowledge import build_knowledge_note
    from podcast_engine.storage import get_bucket, update_episode

    key = episode["episode_key"]
    if build_knowledge_note(episode) is None:
        raise RuntimeError("writer path is not active (PODCAST_KNOWLEDGE_WRITER_PRESET unset)")
    paths = paths_for(key)
    if update_episode(
        key,
        summary_body_file=paths["summary_body"],
        summary_metadata_file=paths["summary_metadata"],
        markdown_file=paths["summary"],
    ) is None:
        raise RuntimeError("could not persist the episode record")
    manifest = json.loads(get_bucket().blob(paths["summary_metadata"]).download_as_text(encoding="utf-8"))
    cost = ((manifest.get("note_writer") or {}).get("usage") or {}).get("cost")
    return Decimal(str(cost)) if cost is not None else WRITER_RESERVATION_USD


def main(argv: list[str] | None = None) -> int:
    from dotenv import load_dotenv

    parser = argparse.ArgumentParser(description="Regenerate existing knowledge notes with the TASK-118 writer.")
    parser.add_argument("--execute", action="store_true", help="spend money; default is a dry run")
    parser.add_argument("--budget-usd", type=Decimal, default=Decimal("0"))
    parser.add_argument("--max-episodes", type=int, default=0, help="0 = all candidates")
    parser.add_argument("--only", nargs="*", help="episode keys or 8-character prefixes")
    args = parser.parse_args(argv)
    load_dotenv()

    from podcast_engine.knowledge.notes_v3.render import RENDER_VERSION

    episodes, manifest_policy, ledger_committed, render_versions, cap, writer_policy = _load_state()
    rows = plan(
        episodes, manifest_policy, ledger_committed, cap=cap, writer_policy=writer_policy,
        render_versions=render_versions, render_version=RENDER_VERSION,
    )
    if args.only:
        rows = [r for r in rows if r["episode_key"] in args.only or r["episode_key"][:8] in args.only]
    candidates = [r for r in rows if r["needs_regeneration"]]
    rerenders = [r for r in rows if r["needs_rerender"]]

    for row in rows:
        mark = "REGENERATE" if row["needs_regeneration"] else ("RE-RENDER " if row["needs_rerender"] else "done      ")
        fits = "" if row["fits_one_attempt"] else "  ! headroom below one writer reservation"
        print(
            f"{mark} {row['episode_key'][:8]} {str(row['published'] or '')[:10]:10} "
            f"committed ${row['ledger_committed_usd']:<10} headroom ${row['headroom_usd']:<10} "
            f"{(row['podcast'] or '')[:18]:18} {(row['title'] or '')[:60]}{fits}"
        )
    print(
        f"\n{len(candidates)} of {len(rows)} ready episodes need regeneration; "
        f"expected about USD {EXPECTED_WRITER_USD * len(candidates)} "
        f"(USD {EXPECTED_WRITER_USD} each, reservation {WRITER_RESERVATION_USD}). "
        f"{len(rerenders)} writer note(s) only need a free re-render."
    )
    waiting = [
        e for e in episodes
        if e.get("status", {}).get("compiler", {}).get("state") == "completed"
        and e.get("status", {}).get("summary", {}).get("state") not in READY_STATES
    ]
    if waiting:
        print(
            f"{len(waiting)} compiled episode(s) still wait for a note; the worker writes them "
            "on its next run once the writer preset is set: "
            + ", ".join(f"{e['episode_key'][:8]} ({(e.get('title') or '')[:40]})" for e in waiting)
        )
    if not args.execute:
        print("Dry run: nothing was spent. Re-run with --execute --budget-usd N to regenerate.")
        return 0

    for row in rerenders:
        episode = next(e for e in episodes if e.get("episode_key") == row["episode_key"])
        try:
            _regenerate(episode)
        except Exception as error:  # noqa: BLE001 -- stop at the first failure
            print(f"FAILED re-render {row['episode_key'][:8]}: {type(error).__name__}: {str(error)[:300]}")
            return 1
        print(f"re-rendered {row['episode_key'][:8]} (no model call)", flush=True)

    if args.max_episodes:
        candidates = candidates[: args.max_episodes]
    spent = Decimal("0")
    for row in candidates:
        if not row["fits_one_attempt"]:
            print(f"skip {row['episode_key'][:8]}: not enough ledger headroom", flush=True)
            continue
        if spent + WRITER_RESERVATION_USD > args.budget_usd:
            print(f"stop: USD {spent} spent; the next episode could exceed --budget-usd {args.budget_usd}")
            return 1
        episode = next(e for e in episodes if e.get("episode_key") == row["episode_key"])
        try:
            cost = _regenerate(episode)
        except Exception as error:  # noqa: BLE001 -- stop at the first failure
            print(f"FAILED {row['episode_key'][:8]}: {type(error).__name__}: {str(error)[:300]}")
            print(f"stopped after USD {spent}")
            return 1
        spent += cost
        print(f"regenerated {row['episode_key'][:8]} for USD {cost} (run total USD {spent})", flush=True)
    print(f"done: USD {spent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
