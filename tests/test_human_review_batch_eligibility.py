from __future__ import annotations

from types import SimpleNamespace
import unittest

from compiler import review_policy
from podcast_engine.compilation import _human_review_items


def _card(**overrides):
    card = {
        "id": 7,
        "reason": "compiler_requires_human_review",
        "kind": "wording_difference",
        "severity": "low",
        "category": "other",
        "apple_text": "creatine",
        "whisper_text": "creating",
        "apple_context": "we discussed creatine today",
        "whisper_context": "we discussed creating today",
        "source_only": False,
        "source_only_source": None,
        "risk_reasons": [],
        "domain_terms": [],
        "citation_signal": False,
        "preservation_class": "not_source_only",
        "merge_action": "review_kept_primary",
        "representation_modified": False,
        "generation_stale": False,
        "custom_edit": None,
        "third_asr": None,
        "anomaly": None,
        "triage": {
            "status": "advisory",
            "recommendation": "recommend_apple",
            "source": "apple",
            "text": "creatine",
            "confidence": "high",
            "reason": "Apple matches the supplied source evidence",
        },
    }
    card.update(overrides)
    return card


class BatchRecommendationPolicyTests(unittest.TestCase):
    def test_high_confidence_exact_source_low_risk_card_is_eligible(self):
        for kind in (
            "wording_difference",
            "transcription_difference",
        ):
            with self.subTest(kind=kind):
                recommendation = review_policy.batch_recommendation(
                    _card(kind=kind)
                )

                self.assertEqual(
                    recommendation,
                    {
                        "id": 7,
                        "source": "apple",
                        "text": "creatine",
                        "reason": "high_confidence_exact_source_low_risk",
                    },
                )

    def test_recommended_source_text_must_match_current_card_exactly(self):
        card = _card()
        card["triage"] = {
            **card["triage"],
            "text": "Creatine",
        }

        self.assertIsNone(
            review_policy.batch_recommendation(card)
        )

    def test_only_high_confidence_source_recommendations_are_eligible(self):
        for recommendation, confidence in (
            ("recommend_apple", "medium"),
            ("recommend_whisper", "low"),
            ("likely_custom", "high"),
            ("needs_audio", "high"),
        ):
            with self.subTest(
                recommendation=recommendation,
                confidence=confidence,
            ):
                card = _card()
                card["triage"] = {
                    **card["triage"],
                    "recommendation": recommendation,
                    "confidence": confidence,
                }
                self.assertIsNone(
                    review_policy.batch_recommendation(card)
                )

    def test_hard_review_categories_are_ineligible(self):
        for category in (
            "negation",
            "protocol_number",
            "unit",
            "proper_name",
            "scientific_medical_term",
            "citation",
            "exercise_name",
            "supplement",
            "training_term",
        ):
            with self.subTest(category=category):
                self.assertIsNone(
                    review_policy.batch_recommendation(
                        _card(category=category)
                    )
                )

    def test_hard_review_kinds_are_ineligible(self):
        for kind in (
            "number_mismatch",
            "negation_mismatch",
            "unit_mismatch",
            "citation_mismatch",
            "domain_term_mismatch",
        ):
            with self.subTest(kind=kind):
                self.assertIsNone(
                    review_policy.batch_recommendation(
                        _card(kind=kind)
                    )
                )

    def test_non_low_severity_or_unknown_category_is_ineligible(self):
        for changes in (
            {"severity": "medium"},
            {"severity": "high"},
            {"severity": "unknown"},
            {"category": "unknown"},
            {"category": None},
        ):
            with self.subTest(changes=changes):
                self.assertIsNone(
                    review_policy.batch_recommendation(
                        _card(**changes)
                    )
                )

    def test_source_only_anomaly_and_nonempty_risk_are_ineligible(self):
        for changes in (
            {
                "source_only": True,
                "source_only_source": "apple",
            },
            {
                "anomaly": {
                    "kind": "source_alignment_mismatch_candidate",
                    "reason": "bad geometry",
                }
            },
            {
                "risk_reasons": ["suspected_repetition"],
            },
        ):
            with self.subTest(changes=changes):
                self.assertIsNone(
                    review_policy.batch_recommendation(
                        _card(**changes)
                    )
                )

    def test_custom_third_asr_representation_or_stale_state_is_ineligible(self):
        for changes in (
            {"custom_edit": {"text": "manual correction"}},
            {"third_asr": {"text": "third source"}},
            {"representation_modified": True},
            {"generation_stale": True},
        ):
            with self.subTest(changes=changes):
                self.assertIsNone(
                    review_policy.batch_recommendation(
                        _card(**changes)
                    )
                )

    def test_missing_or_malformed_policy_evidence_fails_closed(self):
        required_fields = (
            "id",
            "kind",
            "severity",
            "category",
            "apple_text",
            "whisper_text",
            "source_only",
            "risk_reasons",
            "domain_terms",
            "citation_signal",
            "preservation_class",
            "merge_action",
            "triage",
        )
        for field in required_fields:
            with self.subTest(field=field):
                card = _card()
                card.pop(field)
                self.assertIsNone(
                    review_policy.batch_recommendation(card)
                )


class BatchRecommendationProjectionTests(unittest.TestCase):
    def _difference(self):
        return SimpleNamespace(
            id=7,
            kind="wording_difference",
            severity="low",
            resolver_category="other",
            apple_text="creatine",
            whisper_text="creating",
            apple_context="we discussed creatine today",
            whisper_context="we discussed creating today",
            apple_start_timestamp=10.0,
            whisper_start_timestamp=10.0,
            whisper_end_timestamp=11.0,
            selected_source="apple",
            selection_reason="primary source retained for review",
            review_required=True,
            merge_action="review_kept_primary",
            source_only=False,
            source_only_source=None,
            risk_reasons=[],
            domain_terms=[],
            citation_signal=False,
            preservation_class="not_source_only",
        )

    def test_eligible_card_keeps_human_review_and_adds_batch_recommendation(self):
        difference = self._difference()
        result = SimpleNamespace(
            differences=[difference],
            recommended_source="apple",
        )
        resolver_batch = {
            "batch": {
                "diff_items": [],
            },
            "bypassed_items": [],
            "deferred_items": [],
        }
        triage = {
            "accepted": [],
            "recommendations": [
                {
                    "id": 7,
                    "recommendation": "recommend_apple",
                    "source": "apple",
                    "text": "creatine",
                    "confidence": "high",
                    "reason": "Apple matches the supplied source evidence",
                }
            ],
            "outcomes": [
                {
                    "id": 7,
                    "status": "advisory",
                    "recommendation": "recommend_apple",
                    "confidence": "high",
                    "reason": "Apple matches the supplied source evidence",
                }
            ],
        }

        review = _human_review_items(
            result,
            resolver_batch,
            {
                "accepted": [],
                "review": [],
                "outcomes": [],
            },
            triage=triage,
        )

        self.assertEqual(len(review), 1)
        self.assertEqual(review[0]["id"], 7)
        self.assertEqual(
            review[0]["batch_recommendation"],
            {
                "id": 7,
                "source": "apple",
                "text": "creatine",
                "reason": "high_confidence_exact_source_low_risk",
            },
        )


if __name__ == "__main__":
    unittest.main()
