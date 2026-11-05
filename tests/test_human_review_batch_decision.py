from __future__ import annotations

import copy
import unittest
from unittest.mock import Mock, patch

from google.api_core.exceptions import PreconditionFailed

from podcast_engine import human_review


FIXED_TIME = "2026-09-15T08:59:00+00:00"


def _card(
    difference_id: int,
    *,
    recommendation_source: str,
    apple_text: str,
    whisper_text: str,
) -> dict:
    recommended_text = (
        apple_text
        if recommendation_source == "apple"
        else whisper_text
    )
    return {
        "id": difference_id,
        "reason": "compiler_requires_human_review",
        "kind": "wording_difference",
        "severity": "low",
        "category": "other",
        "apple_text": apple_text,
        "whisper_text": whisper_text,
        "source_only": False,
        "risk_reasons": [],
        "domain_terms": [],
        "citation_signal": False,
        "preservation_class": "not_source_only",
        "merge_action": "review_kept_primary",
        "triage": {
            "status": "advisory",
            "recommendation": f"recommend_{recommendation_source}",
            "confidence": "high",
            "source": recommendation_source,
            "text": recommended_text,
        },
        "batch_recommendation": {
            "id": difference_id,
            "source": recommendation_source,
            "text": recommended_text,
            "reason": "high_confidence_exact_source_low_risk",
        },
    }


def _record() -> dict:
    cards = [
        _card(
            7,
            recommendation_source="apple",
            apple_text="creatine",
            whisper_text="creating",
        ),
        _card(
            8,
            recommendation_source="whisper",
            apple_text="hypertrophy",
            whisper_text="hypertrophic",
        ),
    ]
    input_fingerprint = "sha256:input"
    return {
        "episode_key": "episode-1",
        "input_fingerprint": input_fingerprint,
        "human_decisions": [],
        "human_review": cards,
        "human_review_queue_fingerprint": (
            human_review.review_queue_fingerprint(cards)
        ),
        "human_review_generation_fingerprint": (
            human_review.review_generation_fingerprint(
                input_fingerprint,
                cards,
            )
        ),
    }


def _requests() -> list[dict]:
    return [
        {"id": 7, "source": "apple"},
        {"id": 8, "source": "whisper"},
    ]


class HumanReviewBatchDecisionTests(unittest.TestCase):
    def test_batch_matches_two_individual_source_decisions(self):
        original = _record()
        generation = original["human_review_generation_fingerprint"]

        individual_store = copy.deepcopy(original)

        def load_individual(_episode_key):
            return copy.deepcopy(individual_store)

        def save_individual(_episode_key, value, **_kwargs):
            individual_store.clear()
            individual_store.update(copy.deepcopy(value))
            return value

        with (
            patch(
                "podcast_engine.human_review.load_review_record",
                side_effect=load_individual,
            ),
            patch(
                "podcast_engine.human_review.save_review_record",
                side_effect=save_individual,
            ),
            patch(
                "podcast_engine.human_review.now_iso",
                return_value=FIXED_TIME,
            ),
        ):
            human_review.record_human_decision(
                "episode-1",
                7,
                source="apple",
            )
            human_review.record_human_decision(
                "episode-1",
                8,
                source="whisper",
            )

        batch_saved = []

        def save_batch(
            _episode_key,
            value,
            *,
            if_generation_match=None,
        ):
            batch_saved.append(
                (
                    if_generation_match,
                    copy.deepcopy(value),
                )
            )
            return value

        with (
            patch(
                "podcast_engine.human_review.load_review_record_with_generation",
                return_value=(copy.deepcopy(original), 41),
            ),
            patch(
                "podcast_engine.human_review.save_review_record",
                side_effect=save_batch,
            ),
            patch(
                "podcast_engine.human_review.now_iso",
                return_value=FIXED_TIME,
            ),
        ):
            result = human_review.record_human_decision_batch(
                "episode-1",
                _requests(),
                expected_generation_fingerprint=generation,
            )

        self.assertEqual(len(batch_saved), 1)
        self.assertEqual(batch_saved[0][0], 41)
        self.assertEqual(
            result["human_decisions"],
            individual_store["human_decisions"],
        )
        self.assertEqual(
            result["human_review"],
            individual_store["human_review"],
        )
        self.assertEqual(
            result["human_review_queue_fingerprint"],
            individual_store["human_review_queue_fingerprint"],
        )
        self.assertEqual(result["human_review"], [])
        # TASK-076 Task 11: strict low-risk Batch decisions snapshot the
        # same routing provenance as an individual Detailed Review
        # decision -- audit parity across every decision channel.
        for decision in result["human_decisions"]:
            self.assertIn("routing_provenance", decision)
            self.assertEqual(decision["routing_provenance"]["reason_codes"], [])

    def test_invalid_member_makes_batch_atomic_with_no_write(self):
        original = _record()
        generation = original["human_review_generation_fingerprint"]
        invalid_batches = (
            [
                {"id": 7, "source": "apple"},
                {"id": 8, "source": "apple"},
            ],
            [
                {"id": 7, "source": "apple"},
                {"id": 999, "source": "whisper"},
            ],
            [
                {"id": 7, "source": "apple"},
                {"id": 8, "source": "third"},
            ],
        )

        for decisions in invalid_batches:
            with self.subTest(decisions=decisions):
                save = Mock()
                with (
                    patch(
                        "podcast_engine.human_review.load_review_record_with_generation",
                        return_value=(copy.deepcopy(original), 51),
                    ),
                    patch(
                        "podcast_engine.human_review.save_review_record",
                        save,
                    ),
                ):
                    with self.assertRaises(ValueError):
                        human_review.record_human_decision_batch(
                            "episode-1",
                            decisions,
                            expected_generation_fingerprint=generation,
                        )

                save.assert_not_called()
                self.assertEqual(original["human_decisions"], [])
                self.assertEqual(
                    [item["id"] for item in original["human_review"]],
                    [7, 8],
                )

    def test_batch_revalidates_live_eligibility_after_third_asr(self):
        original = _record()
        generation = original["human_review_generation_fingerprint"]

        item = original["human_review"][0]
        item.update(
            {
                "source_only": False,
                "risk_reasons": [],
                "domain_terms": [],
                "citation_signal": False,
                "preservation_class": "not_source_only",
                "merge_action": "review_kept_primary",
                "triage": {
                    "status": "advisory",
                    "recommendation": "recommend_apple",
                    "confidence": "high",
                    "source": "apple",
                    "text": "creatine",
                },
                "third_asr": {
                    "text": "creatine",
                },
            }
        )

        save = Mock()
        with (
            patch(
                "podcast_engine.human_review.load_review_record_with_generation",
                return_value=(copy.deepcopy(original), 59),
            ),
            patch(
                "podcast_engine.human_review.save_review_record",
                save,
            ),
        ):
            with self.assertRaisesRegex(
                ValueError,
                "eligible",
            ):
                human_review.record_human_decision_batch(
                    "episode-1",
                    [{"id": 7, "source": "apple"}],
                    expected_generation_fingerprint=generation,
                )

        save.assert_not_called()
        self.assertEqual(original["human_decisions"], [])

    def test_generation_mismatch_fails_before_write(self):
        original = _record()
        save = Mock()

        with (
            patch(
                "podcast_engine.human_review.load_review_record_with_generation",
                return_value=(copy.deepcopy(original), 61),
            ),
            patch(
                "podcast_engine.human_review.save_review_record",
                save,
            ),
        ):
            with self.assertRaisesRegex(
                ValueError,
                "generation",
            ):
                human_review.record_human_decision_batch(
                    "episode-1",
                    _requests(),
                    expected_generation_fingerprint="sha256:stale",
                )

        save.assert_not_called()

    def test_precondition_race_reloads_fresh_record_and_never_duplicates(self):
        original = _record()
        generation = original["human_review_generation_fingerprint"]
        load = Mock(
            side_effect=[
                (copy.deepcopy(original), 71),
                (copy.deepcopy(original), 72),
            ]
        )
        saved = []

        def save(
            _episode_key,
            value,
            *,
            if_generation_match=None,
        ):
            saved.append(
                (
                    if_generation_match,
                    copy.deepcopy(value),
                )
            )
            if len(saved) == 1:
                raise PreconditionFailed(
                    "concurrent review write"
                )
            return value

        with (
            patch(
                "podcast_engine.human_review.load_review_record_with_generation",
                load,
            ),
            patch(
                "podcast_engine.human_review.save_review_record",
                side_effect=save,
            ),
            patch(
                "podcast_engine.human_review.now_iso",
                return_value=FIXED_TIME,
            ),
        ):
            result = human_review.record_human_decision_batch(
                "episode-1",
                _requests(),
                expected_generation_fingerprint=generation,
            )

        self.assertEqual(load.call_count, 2)
        self.assertEqual(
            [entry[0] for entry in saved],
            [71, 72],
        )
        self.assertEqual(
            [item["id"] for item in result["human_decisions"]],
            [7, 8],
        )
        self.assertEqual(
            len(
                {
                    item["id"]
                    for item in result["human_decisions"]
                }
            ),
            2,
        )

    def test_precondition_retry_fails_closed_if_fresh_record_is_stale(self):
        original = _record()
        generation = original["human_review_generation_fingerprint"]

        changed = copy.deepcopy(original)
        changed["human_review_generation_fingerprint"] = "sha256:new-generation"

        load = Mock(
            side_effect=[
                (copy.deepcopy(original), 81),
                (changed, 82),
            ]
        )
        save = Mock(
            side_effect=[
                PreconditionFailed("concurrent review write"),
            ]
        )

        with (
            patch(
                "podcast_engine.human_review.load_review_record_with_generation",
                load,
            ),
            patch(
                "podcast_engine.human_review.save_review_record",
                save,
            ),
        ):
            with self.assertRaisesRegex(
                ValueError,
                "generation",
            ):
                human_review.record_human_decision_batch(
                    "episode-1",
                    _requests(),
                    expected_generation_fingerprint=generation,
                )

        self.assertEqual(load.call_count, 2)
        self.assertEqual(save.call_count, 1)

    def test_empty_duplicate_or_malformed_batch_is_rejected(self):
        original = _record()
        generation = original["human_review_generation_fingerprint"]

        invalid_batches = (
            [],
            [
                {"id": 7, "source": "apple"},
                {"id": 7, "source": "apple"},
            ],
            [{"id": "7", "source": "apple"}],
            [{"id": 7}],
        )

        for decisions in invalid_batches:
            with self.subTest(decisions=decisions):
                save = Mock()
                with (
                    patch(
                        "podcast_engine.human_review.load_review_record_with_generation",
                        return_value=(copy.deepcopy(original), 91),
                    ),
                    patch(
                        "podcast_engine.human_review.save_review_record",
                        save,
                    ),
                ):
                    with self.assertRaises(ValueError):
                        human_review.record_human_decision_batch(
                            "episode-1",
                            decisions,
                            expected_generation_fingerprint=generation,
                        )
                save.assert_not_called()


if __name__ == "__main__":
    unittest.main()
