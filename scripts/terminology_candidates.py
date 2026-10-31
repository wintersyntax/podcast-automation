#!/usr/bin/env python3
"""Operate the non-authoritative terminology candidate registry.

No command in this tool can modify the active compiler terminology registry.
Promotion is intentionally a preview/code-review workflow only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from podcast_engine.terminology_candidates import (
    BACKFILL_STATUSES,
    CandidateObservation,
    CandidateRegistryStore,
    backfill_counts,
    candidate_registry_stats,
    compiler_candidate_observations,
    human_review_candidate_observations,
    initial_candidate_registry,
    validate_candidate_registry,
    validate_seed_payload,
)

DEFAULT_AI_SEED = REPOSITORY_ROOT / "config" / "terminology-candidates-ai-seed-v1.json"


def _load_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path} is not valid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def cmd_validate_seed(args: argparse.Namespace) -> int:
    observations = validate_seed_payload(_load_json(args.path))
    print(
        json.dumps(
            {
                "status": "valid",
                "path": str(args.path),
                "candidate_count": len(observations),
                "authority": "none",
                "compiler_visibility": "none",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def cmd_seed(args: argparse.Namespace) -> int:
    payload = _load_json(args.path)
    observations = validate_seed_payload(payload)
    store = CandidateRegistryStore()
    result = store.record_seed(observations, seed_id=payload["seed_id"])
    registry = store.load()
    print(
        json.dumps(
            {
                "seed": str(args.path),
                "input_candidates": len(observations),
                "write_result": result,
                "registry": candidate_registry_stats(registry),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def cmd_repair_seed(args: argparse.Namespace) -> int:
    payload = _load_json(args.path)
    observations = validate_seed_payload(payload)
    store = CandidateRegistryStore()
    result = store.repair_seed_duplicates(observations, seed_id=payload["seed_id"])
    registry = store.load()
    print(
        json.dumps(
            {
                "seed": str(args.path),
                "repair_result": result,
                "registry": candidate_registry_stats(registry),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def cmd_import(args: argparse.Namespace) -> int:
    payload = _load_json(args.path)
    values = payload.get("candidates")
    if not isinstance(values, list):
        raise ValueError("external import JSON must contain a candidates list")
    observations = []
    for index, item in enumerate(values):
        if not isinstance(item, dict):
            raise ValueError(f"candidate {index} must be an object")
        observations.append(
            CandidateObservation(
                text=item.get("canonical"),
                suggested_category=item.get("suggested_category"),
                provenance="external-curated",
                observed_form=item.get("canonical"),
                evidence={"source_note": args.source_note, "source_file": args.path.name},
            )
        )
    result = CandidateRegistryStore().record_many(observations)
    print(json.dumps({"input_candidates": len(observations), **result}, indent=2))
    return 0


def episode_backfill_observations(episode_key: str, *, read_report, read_review) -> list[CandidateObservation] | None:
    """The observations a compile of this episode would have recorded; None without a compiler report."""

    report = read_report(episode_key)
    if not isinstance(report, dict):
        return None
    observations = compiler_candidate_observations(episode_key=episode_key, report=report)
    review = read_review(episode_key)
    if isinstance(review, dict):
        observations.extend(human_review_candidate_observations(episode_key=episode_key, review_record=review))
    return observations


def _read_report(episode_key: str) -> dict | None:
    from google.api_core.exceptions import NotFound

    from podcast_engine.episode_contract import paths_for
    from podcast_engine.storage import download_gcs_bytes

    try:
        return json.loads(download_gcs_bytes(paths_for(episode_key)["compiler_report"]).decode("utf-8"))
    except NotFound:
        return None


def _read_review(episode_key: str) -> dict | None:
    from podcast_engine.human_review import load_review_record

    try:
        return load_review_record(episode_key)
    except FileNotFoundError:
        return None


def run_backfill(store, episode_keys, *, execute: bool, read_report, read_review) -> dict:
    """TASK-160: record past episodes' candidates at most once per candidate and episode.

    Dry run (default) merges into a copy of the current registry and writes
    nothing; ``execute`` writes once per episode through the same merge.
    """

    working = None
    if not execute:
        working = store.read_snapshot() or initial_candidate_registry()
    rows = []
    totals = {status: 0 for status in BACKFILL_STATUSES}
    for key in episode_keys:
        observations = episode_backfill_observations(key, read_report=read_report, read_review=read_review)
        if observations is None:
            rows.append({"episode_key": key, "status": "no_compiler_report"})
            continue
        counts = store.record_backfill(observations) if execute else backfill_counts(working, observations)
        for status, value in counts.items():
            totals[status] += value
        rows.append({"episode_key": key, "observations": len(observations), **counts})
    return {"mode": "execute" if execute else "dry_run", "episodes": rows, "totals": totals}


def cmd_backfill(args: argparse.Namespace) -> int:
    from podcast_engine.storage import load_episodes

    keys = args.episode or [
        episode["episode_key"] for episode in load_episodes()
        if isinstance(episode, dict) and isinstance(episode.get("episode_key"), str)
    ]
    store = CandidateRegistryStore()
    result = run_backfill(store, keys, execute=args.execute, read_report=_read_report, read_review=_read_review)
    if args.execute:
        result["registry"] = candidate_registry_stats(store.load())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def cmd_stats(_args: argparse.Namespace) -> int:
    registry = CandidateRegistryStore().load()
    print(json.dumps(candidate_registry_stats(registry), ensure_ascii=False, indent=2))
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    registry = CandidateRegistryStore().load()
    validate_candidate_registry(registry)
    content = json.dumps(registry, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(content, encoding="utf-8")
        print(str(args.output))
    else:
        print(content, end="")
    return 0


def cmd_promotion_preview(args: argparse.Namespace) -> int:
    registry = CandidateRegistryStore().load()
    normalized = args.term.strip().casefold()
    matches = [
        entry
        for entry in registry["candidates"].values()
        if entry.get("normalized") == normalized
    ]
    if len(matches) != 1:
        raise ValueError("candidate was not found exactly once")
    entry = matches[0]
    categories = entry["suggested_categories"]
    category = max(categories, key=lambda name: (categories[name], name))
    print(
        json.dumps(
            {
                "candidate": entry,
                "promotion": {
                    "automatic": False,
                    "requires_human_review": True,
                    "requires_active_registry_code_change": True,
                    "requires_semantic_policy_review": True,
                    "suggested_entry": (
                        "TerminologyEntry("
                        f"canonical={entry['canonical']!r}, "
                        f"category={category!r}, "
                        "provenance='human-reviewed-promotion')"
                    ),
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)

    validate_seed = sub.add_parser("validate-seed")
    validate_seed.add_argument("path", nargs="?", type=Path, default=DEFAULT_AI_SEED)
    validate_seed.set_defaults(func=cmd_validate_seed)

    seed = sub.add_parser("seed-ai")
    seed.add_argument("path", nargs="?", type=Path, default=DEFAULT_AI_SEED)
    seed.set_defaults(func=cmd_seed)

    repair_seed = sub.add_parser("repair-seed-ai")
    repair_seed.add_argument("path", nargs="?", type=Path, default=DEFAULT_AI_SEED)
    repair_seed.set_defaults(func=cmd_repair_seed)

    external = sub.add_parser("import-external")
    external.add_argument("path", type=Path)
    external.add_argument("--source-note", required=True)
    external.set_defaults(func=cmd_import)

    backfill = sub.add_parser(
        "backfill-episodes",
        help="TASK-160: record candidates past compiles could not; dry run unless --execute",
    )
    backfill.add_argument("--execute", action="store_true")
    backfill.add_argument("--episode", action="append", default=[], help="Only this episode_key (repeatable)")
    backfill.set_defaults(func=cmd_backfill)

    stats = sub.add_parser("stats")
    stats.set_defaults(func=cmd_stats)

    export = sub.add_parser("export")
    export.add_argument("--output", type=Path)
    export.set_defaults(func=cmd_export)

    preview = sub.add_parser("promotion-preview")
    preview.add_argument("term")
    preview.set_defaults(func=cmd_promotion_preview)

    return root


def main() -> int:
    args = parser().parse_args()
    try:
        return args.func(args)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
