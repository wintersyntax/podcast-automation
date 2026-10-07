#!/usr/bin/env python3
"""TASK-133: evaluate shadow materiality evidence with the reviewer.

Read-only. Reads ``episodes.json`` and each episode's
``review/resolver.json`` from GCS (``exists`` and download only) and
collects every card that carries ``materiality_shadow`` evidence -- still
pending or already decided (the decision snapshot keeps the card).

Without ``--answers`` it reports, per episode, how many cards the filter
would have settled (rule / judge) and writes a blind check page: every
card the filter called immaterial plus an equal number of cards it kept
for the reviewer (fixed seed), shuffled, each with one question -- would it
bother you if the system chose either reading on its own? The key that
maps the page back to the evidence is written next to it.

With ``--answers "01 DA, 02 NE, ..."`` it scores the answers against the
key: an immaterial card answered DA is a leak. Nothing is written to GCS.
"""

from __future__ import annotations

import argparse
import html
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from compiler.materiality import materiality_inputs  # noqa: E402
from podcast_engine.materiality_shadow import EVIDENCE_KEY  # noqa: E402

DEFAULT_BUCKET = "YOUR_GCS_BUCKET"
DEFAULT_OUTPUT = REPO_ROOT / "var" / "eval" / "materiality-shadow"


def _read_json(bucket: Any, name: str) -> Any:
    blob = bucket.blob(name)
    if not blob.exists():
        return None
    return json.loads(blob.download_as_bytes().decode("utf-8"))


def collect_cards(bucket: Any) -> list[dict]:
    """Cards with shadow evidence, one per (episode, card id)."""

    cards: dict[tuple[str, int], dict] = {}
    for episode in _read_json(bucket, "episodes.json") or []:
        key = episode.get("episode_key") if isinstance(episode, dict) else None
        if not isinstance(key, str):
            continue
        record = _read_json(bucket, f"episodes/{key}/review/resolver.json")
        if not isinstance(record, dict):
            continue
        sources = [(item, None) for item in record.get("human_review") or []]
        sources += [
            (decision.get("review_item"), decision)
            for decision in record.get("human_decisions") or []
            if isinstance(decision, dict)
        ]
        for item, decision in sources:
            if not isinstance(item, dict) or not isinstance(item.get(EVIDENCE_KEY), dict):
                continue
            inputs = materiality_inputs(item)
            if inputs is None or not isinstance(item.get("id"), int):
                continue
            entry = {
                "episode": key,
                "title": episode.get("title"),
                "id": item["id"],
                "evidence": item[EVIDENCE_KEY],
                "inputs": inputs,
                "decision": None if decision is None else {
                    "chosen": decision.get("chosen_source"),
                    "text": decision.get("chosen_text"),
                },
            }
            current = cards.get((key, item["id"]))
            if current is None or (decision is not None and current["decision"] is None):
                cards[(key, item["id"])] = entry
    return list(cards.values())


def summarize(cards: list[dict]) -> dict:
    per_episode: dict[str, Counter] = {}
    for card in cards:
        counts = per_episode.setdefault(card["title"] or card["episode"], Counter())
        evidence = card["evidence"]
        counts["cards"] += 1
        counts[evidence.get("route", "unknown")] += 1
        counts["would_settle"] += bool(evidence.get("immaterial"))
    return {title: dict(counts) for title, counts in per_episode.items()}


def check_page(cards: list[dict], *, seed: int = 133) -> tuple[str, list[dict]]:
    immaterial = [card for card in cards if card["evidence"].get("immaterial")]
    kept = [card for card in cards if not card["evidence"].get("immaterial")]
    rng = random.Random(seed)
    sample = immaterial + rng.sample(kept, min(len(kept), len(immaterial)))
    rng.shuffle(sample)
    key = []
    sections = []
    esc = html.escape
    for number, card in enumerate(sample, start=1):
        inputs = card["inputs"]
        pair = [inputs["apple"], inputs["whisper"]]
        rng.shuffle(pair)
        key.append({"n": number, "episode": card["episode"], "id": card["id"],
                    "immaterial": bool(card["evidence"].get("immaterial")),
                    "route": card["evidence"].get("route")})
        sections.append(
            f'<section><h2>{number:02d}</h2><p class="ctx">…{esc(inputs["left"][-160:])} <b>[ ? ]</b> '
            f'{esc(inputs["right"][:160])}…</p><div><b>Verzija 1:</b> {esc(pair[0] or "(ništa)")}</div>'
            f'<div><b>Verzija 2:</b> {esc(pair[1] or "(ništa)")}</div>'
            f'<label><input type="radio" name="q{number}" value="DA"> Da, o ovome nešto ovisi</label>'
            f'<label><input type="radio" name="q{number}" value="NE"> Ne, svejedno mi je koja</label></section>'
        )
    page = f"""<!doctype html><html lang="hr"><meta charset="utf-8"><title>Shadow materijalnost</title>
<style>body{{font:16px system-ui;max-width:780px;margin:24px auto;padding:0 16px}}section{{border:1px solid #ccc;border-radius:8px;padding:12px;margin:12px 0}}
label{{display:block;margin:6px 0}}.ctx{{color:#555}}textarea{{width:100%;height:90px}}
@media (prefers-color-scheme: dark){{body{{background:#1b1b1b;color:#eee}}.ctx{{color:#aaa}}section{{border-color:#444}}}}</style>
<h1>Treba li ti ova kartica?</h1><p>Ako bi sustav sam izabrao jednu od dvije verzije, a ti je nikad ne vidiš, bi li ti to smetalo?</p>
{''.join(sections)}<button onclick="collect()">Prikaži rezultat</button><textarea id="out" readonly></textarea>
<script>function collect(){{const out=[];for(let n=1;n<={len(sample)};n++){{const c=document.querySelector(`input[name=q${{n}}]:checked`);out.push(String(n).padStart(2,"0")+" "+(c?c.value:"-"))}}document.getElementById("out").value=out.join(", ")}}</script></html>"""
    return page, key


def score(answers: str, key: list[dict]) -> dict:
    by_number = {entry["n"]: entry for entry in key}
    result = Counter()
    leaks = []
    for part in answers.split(","):
        fields = part.split()
        if len(fields) != 2 or not fields[0].isdigit() or fields[1] not in {"DA", "NE"}:
            continue
        entry = by_number.get(int(fields[0]))
        if entry is None:
            continue
        label = "immaterial" if entry["immaterial"] else "kept"
        result[f"{label}_{fields[1]}"] += 1
        if entry["immaterial"] and fields[1] == "DA":
            leaks.append(entry)
    return {"counts": dict(result), "leaks": leaks}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--answers", help='Reviewer answers, e.g. "01 DA, 02 NE"; scores the existing key')
    args = parser.parse_args(argv)
    output = Path(args.output_dir)

    if args.answers:
        key = json.loads((output / "key.json").read_text(encoding="utf-8"))
        result = score(args.answers, key)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    from google.cloud import storage

    bucket = storage.Client().bucket(args.bucket.removeprefix("gs://").strip("/"))
    cards = collect_cards(bucket)
    summary = summarize(cards)
    page, key = check_page(cards)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "check.html").write_text(page, encoding="utf-8")
    (output / "key.json").write_text(json.dumps(key, indent=1) + "\n", encoding="utf-8")
    for title, counts in summary.items():
        print(f"{title}: {counts}")
    print(f"Check page: {output / 'check.html'} ({len(key)} cards; key.json next to it)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
