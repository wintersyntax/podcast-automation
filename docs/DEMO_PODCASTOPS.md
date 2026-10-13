# Synthetic PodcastOps dashboard demo

This is an **offline, read-only preview** of the private-production PodcastOps episode-status interface. The demo ships the real frontend template (`podcast_engine/ops_dashboard_page.py`) against an in-memory synthetic status API, **not** the private Google Cloud status backend or production episode data.

## Start the local server

From the repository root:

```sh
python scripts/demo_ops_dashboard.py
```

Open **http://127.0.0.1:8767/ops**. Stop with Ctrl-C. The script uses only Python's standard library and binds to `127.0.0.1`, never a public interface.

## What the demo shows

The 12 invented episodes demonstrate a complete processing path, a review-required episode with materiality groups, a finished note still awaiting Mac Vault confirmation, a successful local Mac copy, an unavailable-evidence state and a completed recompile audit. The find box, status filters and refresh control run against the synthetic JSON at `GET /api/ops/episodes`.

**Mac copy confirmed** means a copy of the exact summary was reported on the Mac at the recorded time. It is **not** proof of iCloud upload or continuing local file existence. A recompile `started` audit is **not** live Cloud Run status. The demo cannot mutate state, run AI, fetch feeds, dispatch jobs or reach private services.

## Screenshot in the public README

![Synthetic PodcastOps episode status dashboard](../assets/demo-podcastops-dashboard.png)

The screenshot is generated from the real read-only frontend and the same 12 wholly fictional episodes served by this local demo. It is a static, public-safe preview, **not** a production status report.

## Capture a public-safe screenshot

1. Start the demo and open the URL above in your browser.
2. Choose **All episodes** and leave the synthetic notice visible.
3. Capture the page or its visible dashboard area to a PNG.
4. Check that only fictitious show names, generated episode keys and demonstration timestamps appear.
5. When adding the PNG to the public README, label it **Synthetic PodcastOps demo** — it is not a live production screenshot.

The image may be captured again at any time without a cloud account. The canonical public source includes the launcher, synthetic data and frontend template; no binary screenshot generation is required at runtime.
