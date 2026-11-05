"""TASK-126: Worker-side Third-ASR prefetch before the review notification.

The Human Review page already prefetches the third voice for tier-B and
tier-C cards when an episode is opened.  That means the notification email
and the first view still show the unsorted queue.  This module runs the same
per-item, budget-safe ``ensure_third_asr`` path inside the Worker right after
compilation, so tiers are already settled when the reviewer is notified.

It is best-effort and never blocks the review notification:

* without ``PODCAST_REVIEW_ASR_API_KEY`` the step is skipped;
* any budget, pricing or reconciliation refusal stops the run;
* a single card failure is recorded and the run moves on, but repeated
  consecutive failures stop it;
* Third ASR stays evidence only -- nothing is decided here.
"""

from __future__ import annotations

import os
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from compiler.review_tiers import derive_review_tier
from compiler.third_asr_window import anchored_third_asr_window

from .ai_budget import BudgetConcurrencyError, BudgetLedgerError
from .ai_pricing import PricingResolutionError
from .observability import emit_event
from .human_review import (
    ThirdAsrClaimError,
    ensure_third_asr,
    load_review_record,
    pending_review_items,
)
from .review_audio import review_clip_window

PREFETCH_TIERS = frozenset({"B", "C"})
PREFETCH_WORKERS = 3
PREFETCH_MAX_ITEMS = 120
PREFETCH_MAX_CONSECUTIVE_FAILURES = 5
# Three cards reserve against the same episode budget ledger at once, so a
# reservation can lose its generation race (BudgetConcurrencyError). Nothing
# was reserved or claimed then, so the card is retried after a short jittered
# wait instead of stopping the whole prefetch (2026-10-12: stopped after 5 of
# 74 cards).
PREFETCH_CONCURRENCY_RETRIES = 4
PREFETCH_WALL_CLOCK_SECONDS = 900
# 2026-10-30: on the four episodes after the contention fix, 14 of 106 cards
# still failed in the parallel pass with no recorded reason, and every one of
# them transcribed fine when repeated later. Failed cards get one more try,
# one at a time, after the parallel pass; each failure is logged by type.
PREFETCH_RETRY_PAUSE_SECONDS = 2.0


class _Stop(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _log_card_failure(episode_key: str, difference_id: int, error: Exception, attempt: int) -> None:
    """One structured line per failed card: the error type and any HTTP status, never the message."""

    response = getattr(error, "response", None)
    status = getattr(response, "status_code", None)
    if not isinstance(status, int):
        status = getattr(error, "code", None) if isinstance(getattr(error, "code", None), int) else None
    emit_event(
        "third_asr_prefetch_card_failed",
        "Third-ASR prefetch failed for one card",
        severity="WARNING",
        episode_key=episode_key,
        difference_id=difference_id,
        attempt=attempt,
        error_type=type(error).__name__,
        http_status=status,
    )


def prefetch_enabled() -> bool:
    """Prefetch needs the dedicated Third-ASR key and is on by default."""

    if os.getenv("PODCAST_WORKER_THIRD_ASR_PREFETCH", "1").strip().lower() in {"0", "false", "off", "no"}:
        return False
    return bool(os.getenv("PODCAST_REVIEW_ASR_API_KEY", "").strip())


def _current_window(item: dict) -> dict | None:
    try:
        return review_clip_window(item)
    except (TypeError, ValueError):
        return None


def third_asr_refresh_needed(item: dict) -> bool:
    """Existing evidence that cannot be anchored and came from another clip.

    TASK-126: evidence recorded for a clip shorter than the card's current
    clip window is fetched once more for the longer clip. Evidence that
    already anchors, or that was recorded for the current window, is kept.
    """

    evidence = item.get("third_asr")
    if not isinstance(evidence, dict) or not evidence.get("text"):
        return False
    if not isinstance(evidence.get("window"), dict):
        return False
    if anchored_third_asr_window(item) is not None:
        return False
    window = _current_window(item)
    return window is not None and evidence.get("window") != window


def prefetch_candidates(record: dict) -> list[int]:
    """Pending tier-B/C cards with an audio window and no usable third evidence."""

    ids: list[int] = []
    for item in pending_review_items(record):
        if item.get("third_asr") and not third_asr_refresh_needed(item):
            continue
        if derive_review_tier(item)["tier"] not in PREFETCH_TIERS:
            continue
        if _current_window(item) is None:
            continue
        if isinstance(item.get("id"), int):
            ids.append(item["id"])
    return ids[:PREFETCH_MAX_ITEMS]


def _tier_counts(items: list) -> dict[str, int]:
    counts = Counter(derive_review_tier(item)["tier"] for item in items)
    return {tier: counts.get(tier, 0) for tier in ("A", "B", "C")}


def review_tier_counts(record: dict) -> dict[str, int]:
    return _tier_counts(pending_review_items(record))


def prefetch_third_asr(
    episode: dict,
    *,
    review_items: list | None = None,
    ensure: Callable[[dict, int], dict] = ensure_third_asr,
    load_record: Callable[[str], dict] = load_review_record,
    workers: int = PREFETCH_WORKERS,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    """Fetch Third-ASR evidence for tier-B/C cards; return a summary.

    ``review_items`` (the freshly compiled queue) only provides tier counts
    when the prefetch is disabled, so a disabled Worker never reads the
    review record. ``tiers`` is None when no count could be derived.
    """

    summary: dict = {
        "status": "skipped",
        "candidates": 0,
        "fetched": 0,
        "in_flight": 0,
        "failed": 0,
        "stopped_reason": None,
        "tiers": None,
    }
    episode_key = episode["episode_key"]

    if not prefetch_enabled():
        summary["status"] = "disabled"
        if isinstance(review_items, list):
            summary["tiers"] = _tier_counts(review_items)
        return summary

    try:
        record = load_record(episode_key)
    except Exception as error:  # noqa: BLE001 - notification must still go out
        summary["status"] = "record_unavailable"
        summary["stopped_reason"] = type(error).__name__
        return summary

    candidates = prefetch_candidates(record)
    summary["candidates"] = len(candidates)
    if not candidates:
        summary["status"] = "nothing_to_fetch"
        summary["tiers"] = review_tier_counts(record)
        return summary

    deadline = monotonic() + PREFETCH_WALL_CLOCK_SECONDS
    state = {"consecutive_failures": 0, "stop": None}
    attempt = {"pass": 1}

    def run_one(difference_id: int) -> str:
        if state["stop"] is not None:
            return "skipped"
        if monotonic() > deadline:
            state["stop"] = "wall_clock"
            return "skipped"
        try:
            for retry in range(PREFETCH_CONCURRENCY_RETRIES + 1):
                try:
                    ensure(episode, difference_id)
                    break
                except BudgetConcurrencyError:
                    if retry == PREFETCH_CONCURRENCY_RETRIES:
                        raise
                    sleep(random.uniform(0.5, 1.5) * (retry + 1))
        except ThirdAsrClaimError:
            # Another request owns the claim, or the generation moved on.
            return "in_flight"
        except (BudgetLedgerError, PricingResolutionError) as error:
            state["stop"] = type(error).__name__
            return "stopped"
        except Exception as error:  # noqa: BLE001 - one card must not stop the queue
            _log_card_failure(episode_key, difference_id, error, attempt["pass"])
            state["consecutive_failures"] += 1
            if state["consecutive_failures"] >= PREFETCH_MAX_CONSECUTIVE_FAILURES:
                state["stop"] = "consecutive_failures"
            return "failed"
        state["consecutive_failures"] = 0
        return "fetched"

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        outcomes = dict(zip(candidates, pool.map(run_one, candidates)))

    failed = [difference_id for difference_id, outcome in outcomes.items() if outcome == "failed"]
    summary["retried"] = 0
    if failed and state["stop"] is None:
        # One more try for each failed card, one at a time: no competing
        # writes on the episode's ledger and review record.
        attempt["pass"] = 2
        state["consecutive_failures"] = 0
        sleep(PREFETCH_RETRY_PAUSE_SECONDS)
        for difference_id in failed:
            outcomes[difference_id] = run_one(difference_id)
        summary["retried"] = len(failed)

    counts = Counter(outcomes.values())
    summary["fetched"] = counts.get("fetched", 0)
    summary["in_flight"] = counts.get("in_flight", 0)
    summary["failed"] = counts.get("failed", 0)
    summary["stopped_reason"] = state["stop"]
    summary["status"] = "stopped" if state["stop"] else "completed"

    try:
        summary["tiers"] = review_tier_counts(load_record(episode_key))
    except Exception:  # noqa: BLE001
        summary["tiers"] = None
    return summary
