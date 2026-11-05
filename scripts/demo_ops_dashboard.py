#!/usr/bin/env python3
"""Serve the real PodcastOps frontend over synthetic, read-only episode data.

Run: python scripts/demo_ops_dashboard.py
Open http://127.0.0.1:8767/ops

No cloud client, provider credentials, live feeds, local Vault access or actual
episode records are used. Binds to loopback only.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from podcast_engine.ops_dashboard_page import OPS_DASHBOARD_PAGE


# All show names, titles, episode keys and times below are fictional.
# Keep these data separate from production episode_status and GCS reads.
DEMO_DAY = datetime(2026, 10, 11, 8, 0, tzinfo=timezone.utc)
SHOWS = (
    "Example Strength & Nutrition",
    "Demo Coaching Lab",
    "The Fictional Training Show",
)
TITLES = (
    "Training Volume and Recovery",
    "Protein Distribution Across Meals",
    "Making Sense of Hypertrophy Studies",
    "Strength Blocks and Fatigue",
    "Exercise Technique: Hip Hinge",
    "A Practical Guide to Rest Intervals",
    "Creatine and Training Adaptations",
    "Autoregulation and RPE",
    "Energy Balance for Athletes",
    "Progressive Overload in Practice",
    "Planning Your Training Week",
    "Interpreting Scientific Claims",
)
# The newest two are incomplete so the dashboard demonstrates both a
# review queue and a pending Mac sync rather than an implausibly green table.
SCENARIOS = (
    "review", "sync", "ready", "ready", "ready", "ready",
    "ready", "audit", "ready", "unavailable", "ready", "ready",
)


def synthetic_episodes() -> list[dict]:
    """Return self-contained API rows in the real dashboard's projection shape."""
    rows = []
    for i, (title, scenario) in enumerate(zip(TITLES, SCENARIOS, strict=True)):
        day = DEMO_DAY - timedelta(days=i)
        stamp = day.strftime("%Y-%m-%d %H:%M UTC")
        confirmed = (day + timedelta(hours=2)).strftime("%Y-%m-%d %H:%M UTC")
        is_review = scenario == "review"
        is_sync = scenario == "sync"
        no_evidence = scenario == "unavailable"
        has_audit = scenario == "audit"
        is_ready = not is_review
        vault = (
            "not_ready" if is_review
            else "unconfirmed" if is_sync
            else "unavailable" if no_evidence
            else "confirmed"
        )
        delivery = (
            "processing" if is_review
            else "awaiting_vault_sync" if is_sync
            else "unverified" if no_evidence
            else "synced_to_mac"
        )
        review = (
            {"state": "pending", "decided": 9, "pending": 6,
             "groups": {"settled": 2, "sample": 1, "click_one": 1,
                        "proposal": 1, "full": 1}}
            if is_review else
            {"state": "unavailable", "decided": None, "pending": None,
             "groups": None} if no_evidence else
            {"state": "reviewed", "decided": 18 + i, "pending": 0,
             "groups": {"settled": 0, "sample": 0, "click_one": 0,
                        "proposal": 0, "full": 0}}
            if i % 3 == 0 else
            {"state": "no_cards", "decided": 0, "pending": 0,
             "groups": {"settled": 0, "sample": 0, "click_one": 0,
                        "proposal": 0, "full": 0}}
        )
        rows.append({
            "episode_key": f"demo-episode-{12-i:03d}",
            "podcast": SHOWS[i % len(SHOWS)],
            "title": title,
            "published": day.isoformat(),
            "apple": "completed",
            "whisper": "completed",
            "compiler": "review_required" if is_review else "completed",
            "review": review,
            "recompile": {
                "status": "unavailable" if no_evidence else
                          "completed" if has_audit else "not_recorded",
                "request_id": f"demo-recompile-{i:03d}" if has_audit else None,
                "updated_at": stamp if has_audit else None,
                "completed_at": stamp if has_audit else None,
            },
            "summary": "pending" if is_review else "ready",
            "summary_ready_at": stamp if is_ready else None,
            "vault_sync": {
                "status": vault,
                "confirmed_at": confirmed if vault == "confirmed" else None,
            },
            "delivery": delivery,
            "finished_at": confirmed if delivery == "synced_to_mac" else None,
        })
    return rows


def demo_html() -> str:
    """Keep upstream dashboard layout and client-side behavior unchanged."""
    return OPS_DASHBOARD_PAGE.replace(
        '<a href="/">← Human Review</a>',
        '<a href="/">← Synthetic demo index</a>',
        1,
    ).replace(
        "<h1>PodcastOps — episode status</h1>",
        '<h1>PodcastOps — episode status</h1>'
        '<p style="font-size:.87rem;color:#516178;margin:.1rem 0 0">'
        'SYNTHETIC DEMO · Fictional episodes · No cloud connection</p>',
        1,
    )


class DemoHandler(BaseHTTPRequestHandler):
    """The only API is a static GET projection; there are no write routes."""

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path in ("/ops", "/ops/"):
            self.send_payload(demo_html().encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/api/ops/episodes":
            payload = {"episodes": synthetic_episodes()}
            self.send_payload(json.dumps(payload).encode("utf-8"), "application/json")
        elif path == "/":
            body = (
                '<!doctype html><html lang="en"><meta charset="utf-8">'
                '<title>PodcastOps demo</title>'
                '<main style="font:16px/1.5 system-ui;padding:2rem">'
                '<h1>PodcastOps synthetic demo</h1><p>Fictional data only. '
                'No production access, review actions, or writes.</p>'
                '<a href="/ops">Open the read-only episode dashboard →</a>'
                '</main></html>'
            )
            self.send_payload(body.encode("utf-8"), "text/html; charset=utf-8")
        else:
            self.send_error(404)

    def send_payload(self, payload: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(payload)


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 8767), DemoHandler)
    print("Synthetic PodcastOps demo: http://127.0.0.1:8767/ops")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
