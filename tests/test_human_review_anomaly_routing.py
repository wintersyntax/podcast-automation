from __future__ import annotations

from types import SimpleNamespace
import unittest

from compiler import review_policy
from podcast_engine import compilation


def _difference(
    *,
    difference_id: int = 7,
    source_only: bool = True,
    source_only_source: str | None = "whisper",
    apple_words: list[str] | None = None,
    whisper_words: list[str] | None = None,
    apple_text: str = "",
    whisper_text: str = "",
    apple_start_timestamp: float | None = None,
    apple_end_timestamp: float | None = None,
    whisper_start_timestamp: float | None = 30.0,
    whisper_end_timestamp: float | None = 60.0,
):
    apple_words = list(apple_words or [])
    whisper_words = list(whisper_words or [])
    return SimpleNamespace(
        id=difference_id,
        kind="number_mismatch",
        severity="high",
        resolver_category="protocol_number",
        apple_text=apple_text,
        whisper_text=whisper_text,
        apple_context="episode speech before and after the disputed span",
        whisper_context="episode speech before and after the disputed span",
        apple_start_timestamp=apple_start_timestamp,
        apple_end_timestamp=apple_end_timestamp,
        whisper_start_timestamp=whisper_start_timestamp,
        whisper_end_timestamp=whisper_end_timestamp,
        apple_start_word=10,
        apple_end_word=10,
        whisper_start_word=10,
        whisper_end_word=10 + len(whisper_words),
        changed_apple_words=apple_words,
        changed_whisper_words=whisper_words,
        local_similarity=0.0,
        source_only=source_only,
        source_only_source=source_only_source,
        review_required=True,
        selected_source="apple",
        selected_text=None,
        merge_action="review_kept_primary",
        selection_reason="Meaning-sensitive conflict; automatic replacement is disabled",
    )


class ReviewAnomalyClassificationTests(unittest.TestCase):
    def test_large_one_sided_span_is_alignment_mismatch_candidate(self):
        words = [f"adword{index}" for index in range(45)]
        difference = _difference(
            whisper_words=words,
            whisper_text=" ".join(words),
        )

        anomaly = review_policy.classify_review_anomaly(difference)

        self.assertIsNotNone(anomaly)
        self.assertEqual(
            anomaly["kind"],
            "source_alignment_mismatch_candidate",
        )
        self.assertTrue(anomaly["reason"])

    def test_compact_one_sided_span_remains_ordinary_review(self):
        words = ["short", "source", "only", "phrase"]
        difference = _difference(
            whisper_words=words,
            whisper_text=" ".join(words),
        )

        self.assertIsNone(
            review_policy.classify_review_anomaly(difference)
        )

    def test_reversed_audio_interval_is_localization_unreliable(self):
        difference = _difference(
            source_only=False,
            source_only_source=None,
            apple_words=["seven"],
            whisper_words=["seven"],
            apple_text="seven",
            whisper_text="seven",
            apple_start_timestamp=42.0,
            apple_end_timestamp=40.0,
            whisper_start_timestamp=42.0,
            whisper_end_timestamp=43.0,
        )

        anomaly = review_policy.classify_review_anomaly(difference)

        self.assertIsNotNone(anomaly)
        self.assertEqual(
            anomaly["kind"],
            "audio_localization_unreliable",
        )

    def test_unknown_or_missing_geometry_fails_closed_to_no_anomaly_label(self):
        difference = _difference(
            source_only=False,
            source_only_source=None,
            apple_words=["alpha"],
            whisper_words=["beta"],
            apple_text="alpha",
            whisper_text="beta",
            apple_start_timestamp=None,
            apple_end_timestamp=None,
            whisper_start_timestamp=None,
            whisper_end_timestamp=None,
        )

        self.assertIsNone(
            review_policy.classify_review_anomaly(difference)
        )


class ReviewAnomalyRoutingTests(unittest.TestCase):
    def test_anomaly_is_removed_from_resolver_batch_and_kept_for_human_review(self):
        words = [f"adword{index}" for index in range(45)]
        difference = _difference(
            whisper_words=words,
            whisper_text=" ".join(words),
        )
        result = SimpleNamespace(
            differences=[difference],
            recommended_source="apple",
        )
        resolver_batch = {
            "batch": {
                "diff_items": [
                    {
                        "id": difference.id,
                        "kind": difference.kind,
                    }
                ],
            },
            "bypassed_items": [],
            "deferred_items": [],
        }

        routed = compilation._route_review_anomalies(
            result,
            resolver_batch,
        )

        self.assertEqual(routed["batch"]["diff_items"], [])
        self.assertEqual(len(routed["bypassed_items"]), 1)
        self.assertEqual(
            routed["bypassed_items"][0]["id"],
            difference.id,
        )
        self.assertIn(
            "source_alignment_mismatch_candidate",
            routed["bypassed_items"][0]["reason"],
        )

        review = compilation._human_review_items(
            result,
            routed,
            {
                "accepted": [],
                "review": [],
                "outcomes": [],
            },
        )

        self.assertEqual(len(review), 1)
        self.assertEqual(
            review[0]["anomaly"]["kind"],
            "source_alignment_mismatch_candidate",
        )

    def test_bypassed_pending_anomaly_is_classified_before_triage(self):
        difference = _difference(
            source_only=False,
            source_only_source=None,
            apple_words=["alpha", "beta", "gamma", "delta"],
            whisper_words=["alpha", "theta", "gamma", "delta"],
            apple_text="alpha beta gamma delta",
            whisper_text="alpha theta gamma delta",
            apple_start_timestamp=42.0,
            apple_end_timestamp=40.0,
            whisper_start_timestamp=42.0,
            whisper_end_timestamp=43.0,
        )
        difference.kind = "transcription_difference"
        difference.severity = "medium"
        difference.resolver_category = "other"

        result = SimpleNamespace(
            differences=[difference],
            recommended_source="apple",
        )
        resolver_batch = {
            "batch": {
                "diff_items": [],
            },
            "bypassed_items": [
                {
                    "id": difference.id,
                    "reason": "structurally_ineligible_no_acceptance_path",
                }
            ],
            "deferred_items": [],
        }

        routed = compilation._route_review_anomalies(
            result,
            resolver_batch,
        )

        self.assertEqual(routed["batch"]["diff_items"], [])
        self.assertEqual(len(routed["bypassed_items"]), 1)
        routed_item = routed["bypassed_items"][0]
        self.assertEqual(routed_item["id"], difference.id)
        self.assertIn("anomaly", routed_item)
        self.assertEqual(
            routed_item["anomaly"]["kind"],
            "audio_localization_unreliable",
        )

        anomaly_ids = {
            item["id"]
            for item in routed["bypassed_items"]
            if (
                isinstance(item, dict)
                and isinstance(item.get("id"), int)
                and isinstance(item.get("anomaly"), dict)
            )
        }
        self.assertEqual(anomaly_ids, {difference.id})

        review = compilation._human_review_items(
            result,
            routed,
            {
                "accepted": [],
                "review": [],
                "outcomes": [],
            },
        )
        self.assertEqual(len(review), 1)
        self.assertEqual(
            review[0]["anomaly"]["kind"],
            "audio_localization_unreliable",
        )

    def test_anomaly_projection_does_not_mutate_transcript_decision_fields(self):
        words = [f"adword{index}" for index in range(45)]
        difference = _difference(
            whisper_words=words,
            whisper_text=" ".join(words),
        )
        original = (
            difference.selected_source,
            difference.selected_text,
            difference.merge_action,
        )
        result = SimpleNamespace(
            differences=[difference],
            recommended_source="apple",
        )
        resolver_batch = {
            "batch": {
                "diff_items": [
                    {
                        "id": difference.id,
                        "kind": difference.kind,
                    }
                ],
            },
            "bypassed_items": [],
            "deferred_items": [],
        }

        routed = compilation._route_review_anomalies(
            result,
            resolver_batch,
        )
        review = compilation._human_review_items(
            result,
            routed,
            {
                "accepted": [],
                "review": [],
                "outcomes": [],
            },
        )

        self.assertEqual(
            (
                difference.selected_source,
                difference.selected_text,
                difference.merge_action,
            ),
            original,
        )
        self.assertEqual(
            review[0]["suggestion"]["source"],
            original[0],
        )
        self.assertFalse(
            review[0]["suggestion"]["automatic_resolution"],
        )


if __name__ == "__main__":
    unittest.main()
