import unittest

import compiler.review_policy as review_policy


def admissible_item(**overrides):
    item = {
        "kind": "wording_difference",
        "category": "other",
        "apple_text": "off stage and traveled back across Austria. You coach, you're a",
        "whisper_text": "offstage and traveled back from Austria. You coach your",
        "source_only": False,
        "risk_reasons": [],
        "domain_terms": [],
        "citation_signal": False,
        "preservation_class": "not_source_only",
        "merge_action": "review_kept_primary",
        "anomaly": None,
        "custom_edit": None,
        "third_asr": None,
        "representation_modified": False,
        "generation_stale": False,
        "whisper_start_timestamp": 10.0,
        "whisper_end_timestamp": 12.0,
    }
    item.update(overrides)
    return item


class AssistedRoutingPolicyVersionTests(unittest.TestCase):
    def test_policy_version_constant(self):
        self.assertEqual(
            review_policy.ASSISTED_REVIEW_POLICY_VERSION,
            "human-review-assisted-v1",
        )


class AssistedRoutingAdmissionTests(unittest.TestCase):
    def test_bounded_generic_conflict_is_eligible_with_no_reason_codes(self):
        item = admissible_item()
        result = review_policy.assisted_routing(item)

        self.assertEqual(result["policy_version"], "human-review-assisted-v1")
        self.assertTrue(result["eligible"])
        self.assertEqual(result["reason_codes"], [])
        self.assertEqual(result["decision_scope"], "full")
        self.assertEqual(
            result["token_counts"],
            {
                "apple": len(review_policy.comparison_tokens(item["apple_text"])),
                "whisper": len(review_policy.comparison_tokens(item["whisper_text"])),
            },
        )
        # ID-2-like boundary case: exactly 11 comparison tokens on the Apple side.
        self.assertEqual(result["token_counts"]["apple"], 11)

    def test_assisted_routing_never_inspects_suggestion_or_triage(self):
        item = admissible_item(
            suggestion={"source": "whisper", "reason": "x", "automatic_resolution": False},
            triage={"status": "advisory", "recommendation": "recommend_whisper", "confidence": "high"},
        )
        result = review_policy.assisted_routing(item)
        self.assertTrue(result["eligible"])
        self.assertNotIn("source", result)
        self.assertNotIn("text", result)

    def test_current_third_asr_evidence_does_not_exclude_a_candidate(self):
        item = admissible_item(third_asr={"text": "third source evidence", "cache_key": "x"})
        result = review_policy.assisted_routing(item)
        self.assertTrue(result["eligible"])
        self.assertEqual(result["reason_codes"], [])

    def test_assisted_candidate_wrapper_matches_routing_eligibility(self):
        eligible = admissible_item()
        ineligible = admissible_item(category="negation")
        self.assertTrue(review_policy.assisted_candidate(eligible))
        self.assertFalse(review_policy.assisted_candidate(ineligible))


class AssistedRoutingExclusionReasonTests(unittest.TestCase):
    def test_exclusion_reason_codes(self):
        cases = (
            ({"category": "negation"}, "protected_negation"),
            ({"kind": "negation_mismatch"}, "protected_negation"),
            ({"category": "protocol_number"}, "protected_protocol_number"),
            ({"kind": "number_mismatch"}, "protected_protocol_number"),
            ({"category": "unit"}, "protected_unit"),
            ({"category": "proper_name"}, "protected_proper_name"),
            ({"category": "citation"}, "protected_citation"),
            ({"citation_signal": True}, "protected_citation"),
            ({"category": "scientific_medical_term"}, "protected_domain_term"),
            ({"category": "supplement"}, "protected_domain_term"),
            ({"category": "training_term"}, "protected_domain_term"),
            ({"category": "exercise_name"}, "protected_domain_term"),
            ({"domain_terms": ["glycogen"]}, "protected_domain_term"),
            ({"risk_reasons": ["negation"]}, "protected_negation"),
            ({"risk_reasons": ["some_unrecognized_signal"]}, "unknown_category"),
            ({"source_only": True, "preservation_class": "source_only_semantic"}, "source_only_semantic"),
            ({"kind": "source_only"}, "source_only_semantic"),
            ({"preservation_class": "something_else"}, "source_only_semantic"),
            (
                {"anomaly": {"kind": "source_alignment_mismatch_candidate", "reason": "x"}},
                "anomaly",
            ),
            ({"merge_action": "review_suspected_whisper_repetition"}, "suspected_repetition"),
            ({"custom_edit": {"text": "human edit"}}, "custom_edit"),
            ({"representation_modified": True}, "representation_modified"),
            ({"generation_stale": True}, "stale_generation"),
            ({"whisper_start_timestamp": None, "whisper_end_timestamp": None}, "invalid_localization"),
            ({"whisper_end_timestamp": 5.0, "whisper_start_timestamp": 10.0}, "invalid_localization"),
            ({"whisper_start_timestamp": float("nan")}, "invalid_localization"),
            ({"whisper_end_timestamp": float("inf")}, "invalid_localization"),
            ({"kind": "totally_unrecognized_kind"}, "unknown_kind"),
            ({"category": "totally_unrecognized_category"}, "unknown_category"),
            (
                {"apple_text": "one two three four five six seven eight nine ten eleven twelve"},
                "span_over_limit",
            ),
            (
                {"whisper_text": "one two three four five six seven eight nine ten eleven twelve"},
                "span_over_limit",
            ),
        )

        for override, expected_reason in cases:
            with self.subTest(override=override, expected_reason=expected_reason):
                result = review_policy.assisted_routing(admissible_item(**override))
                self.assertFalse(result["eligible"])
                self.assertIn(expected_reason, result["reason_codes"])


class AssistedRoutingMalformedShapeTests(unittest.TestCase):
    def test_non_dict_item(self):
        result = review_policy.assisted_routing("not a dict")
        self.assertEqual(
            result,
            {
                "policy_version": "human-review-assisted-v1",
                "eligible": False,
                "reason_codes": ["unknown_kind", "unknown_category"],
                "decision_scope": None,
                "token_counts": None,
            },
        )

    def test_missing_required_fields(self):
        result = review_policy.assisted_routing({"id": 1})
        self.assertFalse(result["eligible"])
        self.assertEqual(result["reason_codes"], ["unknown_kind", "unknown_category"])
        self.assertIsNone(result["decision_scope"])
        self.assertIsNone(result["token_counts"])

    def test_wrong_field_types(self):
        item = admissible_item(risk_reasons="not a list")
        result = review_policy.assisted_routing(item)
        self.assertFalse(result["eligible"])
        self.assertEqual(result["reason_codes"], ["unknown_kind", "unknown_category"])


class AssistedRoutingFocusSemanticsTests(unittest.TestCase):
    def test_focus_text_is_the_comparison_scope(self):
        long_context = " ".join(f"context{i}" for i in range(30))
        item = admissible_item(
            apple_text=f"{long_context} the disputed word here",
            whisper_text=f"{long_context} the disputed phrase here",
            focus={"apple_text": "the disputed word here", "whisper_text": "the disputed phrase here"},
        )
        result = review_policy.assisted_routing(item)

        self.assertEqual(result["decision_scope"], "focus")
        self.assertTrue(result["eligible"], result["reason_codes"])
        self.assertEqual(result["token_counts"], {"apple": 4, "whisper": 4})

    def test_malformed_focus_falls_back_to_full_text(self):
        item = admissible_item(focus={"apple_text": "only apple present"})
        result = review_policy.assisted_routing(item)
        self.assertEqual(result["decision_scope"], "full")

    def test_focus_absent_uses_full_text(self):
        item = admissible_item()
        result = review_policy.assisted_routing(item)
        self.assertEqual(result["decision_scope"], "full")


if __name__ == "__main__":
    unittest.main()
