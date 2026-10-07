"""TASK-126: Worker-side Third-ASR prefetch before the review notification."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from podcast_engine import pipeline
from podcast_engine import third_asr_prefetch as prefetch
from podcast_engine.ai_budget import BudgetAdmissionError
from podcast_engine.human_review import ThirdAsrInFlight
from podcast_engine.notifications import (
    build_email_review_payload,
    build_email_review_refreshed_payload,
    build_slack_review_payload,
    review_tier_summary,
)

EPISODE = {"episode_key": "a" * 24, "podcast": "Iron Culture", "title": "Ep 1"}
KEY_ENV = {"PODCAST_REVIEW_ASR_API_KEY": "third-key", "PODCAST_WORKER_THIRD_ASR_PREFETCH": "1"}


def _item(item_id: int, *, third: bool = False, start: float | None = 10.0) -> dict:
    item = {"id": item_id, "whisper_start_timestamp": start, "whisper_end_timestamp": start}
    if third:
        item["third_asr"] = {"text": "already here"}
    return item


def _tiers_by_id(mapping: dict[int, str]):
    return lambda item: {"tier": mapping.get(item.get("id"), "B"), "reason": "test"}


class PrefetchEnabledTests(unittest.TestCase):
    def test_disabled_without_key(self):
        with patch.dict(os.environ, {"PODCAST_REVIEW_ASR_API_KEY": ""}, clear=False):
            self.assertFalse(prefetch.prefetch_enabled())

    def test_explicit_off_switch_wins_over_key(self):
        env = dict(KEY_ENV, PODCAST_WORKER_THIRD_ASR_PREFETCH="off")
        with patch.dict(os.environ, env, clear=False):
            self.assertFalse(prefetch.prefetch_enabled())

    def test_enabled_with_key(self):
        with patch.dict(os.environ, KEY_ENV, clear=False):
            self.assertTrue(prefetch.prefetch_enabled())


class PrefetchRunTests(unittest.TestCase):
    def _run(self, record, *, ensure, tiers, workers=1):
        with (
            patch.dict(os.environ, KEY_ENV, clear=False),
            patch.object(prefetch, "derive_review_tier", side_effect=_tiers_by_id(tiers)),
        ):
            return prefetch.prefetch_third_asr(
                EPISODE, ensure=ensure, load_record=lambda key: record, workers=workers
            )

    def test_disabled_never_reads_record_and_counts_compiled_queue(self):
        def load(_key):
            raise AssertionError("disabled prefetch must not read the review record")

        with (
            patch.dict(os.environ, {"PODCAST_REVIEW_ASR_API_KEY": ""}, clear=False),
            patch.object(prefetch, "derive_review_tier", side_effect=_tiers_by_id({1: "A", 2: "C"})),
        ):
            summary = prefetch.prefetch_third_asr(
                EPISODE,
                review_items=[_item(1), _item(2), _item(3)],
                ensure=lambda *_: self.fail("no paid call"),
                load_record=load,
            )
        self.assertEqual(summary["status"], "disabled")
        self.assertEqual(summary["tiers"], {"A": 1, "B": 1, "C": 1})

    def test_fetches_only_tier_b_and_c_without_evidence(self):
        record = {"human_review": [_item(1), _item(2), _item(3, third=True), _item(4)]}
        calls = []
        summary = self._run(
            record,
            ensure=lambda episode, item_id: calls.append(item_id) or {"text": "x"},
            tiers={1: "A", 2: "B", 3: "B", 4: "C"},
        )
        self.assertEqual(sorted(calls), [2, 4])
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["fetched"], 2)
        self.assertEqual(summary["tiers"], {"A": 1, "B": 2, "C": 1})

    def test_budget_refusal_stops_the_run(self):
        record = {"human_review": [_item(1), _item(2), _item(3)]}
        calls = []

        def ensure(episode, item_id):
            calls.append(item_id)
            raise BudgetAdmissionError("cap")

        summary = self._run(record, ensure=ensure, tiers={})
        self.assertEqual(calls, [1])
        self.assertEqual(summary["status"], "stopped")
        self.assertEqual(summary["stopped_reason"], "BudgetAdmissionError")
        self.assertIsNotNone(summary["tiers"])

    def test_claim_in_flight_is_skipped_not_failed(self):
        record = {"human_review": [_item(1)]}

        def ensure(episode, item_id):
            raise ThirdAsrInFlight(retry_after_seconds=5, cache_key="k")

        summary = self._run(record, ensure=ensure, tiers={})
        self.assertEqual(summary["in_flight"], 1)
        self.assertEqual(summary["failed"], 0)
        self.assertEqual(summary["status"], "completed")

    def test_repeated_failures_stop_after_limit(self):
        items = [_item(i) for i in range(1, 10)]
        calls = []

        def ensure(episode, item_id):
            calls.append(item_id)
            raise RuntimeError("provider down")

        summary = self._run({"human_review": items}, ensure=ensure, tiers={})
        self.assertEqual(len(calls), prefetch.PREFETCH_MAX_CONSECUTIVE_FAILURES)
        self.assertEqual(summary["stopped_reason"], "consecutive_failures")

    def test_unreadable_record_returns_no_tiers(self):
        def load(_key):
            raise FileNotFoundError("missing")

        with patch.dict(os.environ, KEY_ENV, clear=False):
            summary = prefetch.prefetch_third_asr(EPISODE, ensure=lambda *_: None, load_record=load)
        self.assertEqual(summary["status"], "record_unavailable")
        self.assertIsNone(summary["tiers"])


class NotificationTierTests(unittest.TestCase):
    TIERS = {"A": 22, "B": 23, "C": 12}

    def test_summary_line(self):
        self.assertEqual(
            review_tier_summary(self.TIERS),
            "35 need a decision (23 listening, 12 protected); 22 can be confirmed together in tier A.",
        )
        self.assertIsNone(review_tier_summary(None))
        self.assertIsNone(review_tier_summary({"A": "x"}))

    def test_email_includes_tiers_only_when_known(self):
        with_tiers = build_email_review_payload(EPISODE, 57, "https://review.example.test", tiers=self.TIERS)
        without = build_email_review_payload(EPISODE, 57, "https://review.example.test")
        self.assertIn("35 need a decision", with_tiers["text"])
        self.assertIn("35 need a decision", with_tiers["html"])
        self.assertNotIn("need a decision", without["text"])
        refreshed = build_email_review_refreshed_payload(
            EPISODE, 50, 57, "https://review.example.test", tiers=self.TIERS
        )
        self.assertIn("22 can be confirmed together", refreshed["text"])

    def test_slack_includes_tiers(self):
        payload = build_slack_review_payload(EPISODE, 57, "https://review.example.test", tiers=self.TIERS)
        self.assertIn("35 need a decision", str(payload))


class PipelineWiringTests(unittest.TestCase):
    def test_prefetch_failure_degrades_to_no_tiers(self):
        with (
            patch.object(pipeline, "prefetch_third_asr", side_effect=RuntimeError("boom")),
            patch("builtins.print"),
        ):
            self.assertEqual(
                pipeline._prefetch_third_asr_before_notification(EPISODE, []),
                {"tiers": None},
            )

    def test_prefetch_summary_is_logged_and_returned(self):
        summary = {"status": "completed", "fetched": 3, "tiers": {"A": 1, "B": 2, "C": 0}}
        with (
            patch.object(pipeline, "prefetch_third_asr", return_value=summary) as run,
            patch.object(pipeline, "emit_event") as event,
        ):
            result = pipeline._prefetch_third_asr_before_notification(EPISODE, [_item(1)])
        self.assertEqual(result, summary)
        run.assert_called_once_with(EPISODE, review_items=[_item(1)])
        self.assertEqual(event.call_args.kwargs["tiers"], {"A": 1, "B": 2, "C": 0})



class ReviewClipWindowTests(unittest.TestCase):
    def test_short_card_keeps_the_default_clip(self):
        from podcast_engine.review_audio import clip_window, review_clip_window

        item = {"whisper_start_timestamp": 100.0, "whisper_end_timestamp": 101.0,
                "apple_text": "five words in this card", "whisper_text": "five words in the card"}
        self.assertEqual(review_clip_window(item), clip_window(100.0, 101.0))

    def test_long_card_gets_a_clip_that_covers_its_words_with_margins(self):
        from podcast_engine.review_audio import review_clip_window

        words = " ".join(["word"] * 40)  # 16 s at 2.5 words/s, plus 2 x 3 s margin
        item = {"whisper_start_timestamp": 100.0, "whisper_end_timestamp": 101.0,
                "apple_text": words, "whisper_text": ""}
        window = review_clip_window(item)
        self.assertEqual(window["duration"], 22.0)
        self.assertLessEqual(window["start"], 100.0)
        self.assertGreaterEqual(window["end"], 101.0)

    def test_long_clip_duration_is_whole_seconds(self):
        from podcast_engine.review_audio import review_clip_window

        words = " ".join(["word"] * 24)  # 9.6 s + 6 s margin = 15.6 s -> 16 s
        item = {"whisper_start_timestamp": 700.0, "whisper_end_timestamp": 701.0,
                "apple_text": words, "whisper_text": ""}
        self.assertEqual(review_clip_window(item)["duration"], 16.0)

    def test_clip_is_capped_and_reversed_timestamps_are_a_span(self):
        from podcast_engine.review_audio import REVIEW_CLIP_LONG_MAX_SECONDS, review_clip_window

        item = {"whisper_start_timestamp": 200.0, "whisper_end_timestamp": 190.0,
                "apple_text": " ".join(["word"] * 200), "whisper_text": ""}
        window = review_clip_window(item)
        self.assertEqual(window["duration"], REVIEW_CLIP_LONG_MAX_SECONDS)
        self.assertLessEqual(window["start"], 190.0)
        self.assertGreaterEqual(window["end"], 200.0)


class ThirdAsrRefreshTests(unittest.TestCase):
    def _item(self, evidence_window, text="the clip words"):
        return {
            "id": 1,
            "whisper_start_timestamp": 100.0,
            "whisper_end_timestamp": 101.0,
            "apple_text": " ".join(["word"] * 40),
            "whisper_text": "",
            "third_asr": {"text": text, "window": evidence_window},
        }

    def test_unanchored_evidence_from_a_shorter_clip_is_refreshed_once(self):
        from podcast_engine.review_audio import clip_window, review_clip_window

        old = self._item(clip_window(100.0, 101.0))
        self.assertTrue(prefetch.third_asr_refresh_needed(old))
        current = self._item(review_clip_window(old))
        self.assertFalse(prefetch.third_asr_refresh_needed(current))

    def test_anchored_or_missing_evidence_is_not_refreshed(self):
        from podcast_engine.review_audio import clip_window

        item = self._item(clip_window(100.0, 101.0))
        with patch.object(prefetch, "anchored_third_asr_window", return_value={"text": "x"}):
            self.assertFalse(prefetch.third_asr_refresh_needed(item))
        item["third_asr"] = None
        self.assertFalse(prefetch.third_asr_refresh_needed(item))

    def test_refresh_candidates_join_the_prefetch(self):
        from podcast_engine.review_audio import clip_window

        stale = self._item(clip_window(100.0, 101.0))
        record = {"human_review": [stale]}
        with patch.object(prefetch, "derive_review_tier", return_value={"tier": "B"}):
            self.assertEqual(prefetch.prefetch_candidates(record), [1])


if __name__ == "__main__":
    unittest.main()
