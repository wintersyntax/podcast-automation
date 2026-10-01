#!/usr/bin/env python3
"""Per-episode AI spend report from the episode AI budget ledgers (TASK-118).

The worker already records every paid attempt, per episode and source
generation, in ``episodes/<episode_key>/ai/budgets/<digest>.json``. This
read-only operator tool lists those ledgers (it needs the operator's GCS
credentials: the worker runtime identity may not list objects), joins them
with the episode index for podcast/title/date, and writes:

* ``var/ai-spend/report.json`` -- one row per ledger, per-stage breakdown,
  plus totals and per-episode statistics;
* ``var/ai-spend/report.csv``  -- the same rows flattened for a spreadsheet.

It makes no model call and writes nothing to GCS.
"""

from __future__ import annotations

import argparse
import csv
from decimal import Decimal
import json
from pathlib import Path
import statistics
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "var" / "ai-spend"
LEDGER_GLOB = "episodes/*/ai/budgets/*.json"
CSV_STAGES = ("resolver", "triage", "third_asr", "summary", "summary_review", "metadata", "note_writer")


def build_report(breakdowns: list[dict], episodes: dict[str, dict]) -> dict:
    """Pure: join spend breakdowns with episode records and add statistics."""

    rows = []
    for item in breakdowns:
        episode = episodes.get(item["episode_key"], {})
        rows.append({
            "episode_key": item["episode_key"],
            "podcast": episode.get("podcast"),
            "title": episode.get("title"),
            "published": episode.get("published"),
            **item,
        })
    rows.sort(key=lambda row: (str(row.get("published") or ""), row["episode_key"]), reverse=True)

    committed = [Decimal(row["committed_usd"]) for row in rows]
    by_stage: dict[str, Decimal] = {}
    for row in rows:
        for stage, entry in row["stages"].items():
            by_stage[stage] = by_stage.get(stage, Decimal("0")) + Decimal(entry["settled_usd"]) + Decimal(
                entry["uncertain_usd"]
            )
    writer_spend = [
        Decimal(row["stages"]["note_writer"]["settled_usd"])
        for row in rows
        if "note_writer" in row["stages"]
    ]

    def usd(value) -> str:
        return str(Decimal(value).quantize(Decimal("0.000001")))

    return {
        "ledgers": len(rows),
        "episodes": len({row["episode_key"] for row in rows}),
        "total_committed_usd": usd(sum(committed, Decimal("0"))),
        "per_ledger": {
            "mean_usd": usd(statistics.fmean(committed)) if committed else None,
            "median_usd": usd(statistics.median(committed)) if committed else None,
            "max_usd": usd(max(committed)) if committed else None,
        },
        "note_writer_settled": {
            "count": len(writer_spend),
            "mean_usd": usd(statistics.fmean(writer_spend)) if writer_spend else None,
            "max_usd": usd(max(writer_spend)) if writer_spend else None,
        },
        "by_stage_usd": {stage: usd(value) for stage, value in sorted(by_stage.items())},
        "rows": rows,
    }


def write_outputs(report: dict, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "report.json"
    csv_path = output_dir / "report.csv"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "published", "podcast", "title", "episode_key", "source_fingerprint",
            "committed_usd", "settled_usd", "uncertain_usd", "reserved_usd", "hard_cap_usd",
            *(f"{stage}_usd" for stage in CSV_STAGES),
        ])
        for row in report["rows"]:
            stage_values = []
            for stage in CSV_STAGES:
                entry = row["stages"].get(stage)
                stage_values.append(
                    str(Decimal(entry["settled_usd"]) + Decimal(entry["uncertain_usd"])) if entry else ""
                )
            writer.writerow([
                row.get("published") or "", row.get("podcast") or "", row.get("title") or "",
                row["episode_key"], row["source_fingerprint"], row["committed_usd"],
                row["settled_usd"], row["uncertain_usd"], row["reserved_usd"], row["hard_cap_usd"],
                *stage_values,
            ])
    return json_path, csv_path


def collect() -> tuple[list[dict], dict[str, dict]]:
    if str(REPOSITORY_ROOT) not in sys.path:
        sys.path.insert(0, str(REPOSITORY_ROOT))
    from podcast_engine.ai_budget import spend_breakdown
    from podcast_engine.storage import get_bucket, load_episodes

    breakdowns = []
    for blob in get_bucket().list_blobs(match_glob=LEDGER_GLOB):
        ledger = json.loads(blob.download_as_text(encoding="utf-8"))
        breakdowns.append(spend_breakdown(ledger))
    episodes = {
        episode["episode_key"]: episode
        for episode in load_episodes()
        if isinstance(episode, dict) and episode.get("episode_key")
    }
    return breakdowns, episodes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    report = build_report(*collect())
    json_path, csv_path = write_outputs(report, args.output_dir)
    print(
        f"{report['episodes']} episodes / {report['ledgers']} ledgers, "
        f"total USD {report['total_committed_usd']}, "
        f"mean per ledger USD {report['per_ledger']['mean_usd']}, "
        f"max USD {report['per_ledger']['max_usd']}"
    )
    print(f"by stage: {report['by_stage_usd']}")
    print(f"wrote {json_path} and {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
