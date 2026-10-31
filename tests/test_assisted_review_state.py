from copy import deepcopy
import importlib.util
import unittest

from compiler.review_policy import assisted_routing

if importlib.util.find_spec("compiler.assisted_review"):
    from compiler import assisted_review
else:
    assisted_review = None


def card(apple="we traveled back", whisper="we traveled home", audio=None, **extra):
    result = dict(
        kind="wording_difference", category="other", apple_text=apple,
        whisper_text=whisper, source_only=False, risk_reasons=[], domain_terms=[],
        citation_signal=False, preservation_class="not_source_only",
        merge_action="review_kept_primary", whisper_start_timestamp=10.0,
        whisper_end_timestamp=12.0,
        third_asr=None if audio is None else dict(text=audio, cache_key="cache", model="test", window={"start": 8, "end": 14}),
    )
    result.update(extra)
    return result


def triage(recommendation, confidence="high"):
    return dict(status="advisory", recommendation=recommendation, confidence=confidence)


class AssistedStateTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(assisted_review, "Task 8 state engine is not implemented")

    def derive(self, item):
        before = deepcopy(item)
        result = assisted_review.derive_assisted_state(item)
        self.assertEqual(item, before, "canonical inputs were mutated")
        return result

    def assert_state(self, item, state, source=None):
        result = self.derive(item)
        self.assertEqual(result["state"], state)
        self.assertEqual(result["recommendation"], source)
        return result

    def test_nonadmitted_card_has_only_routing_and_no_assisted_state(self):
        result = self.derive(card(category="negation", audio="we traveled back"))
        self.assertFalse(result["routing"]["eligible"])
        self.assertNotIn("state", result)

    def test_pending_unavailable_and_budget_unavailable(self):
        self.assert_state(card(), "audio_pending")
        self.assert_state(card(third_asr={"status": "pending"}), "audio_pending")
        for evidence in [{}, {"text": "?!"}, {"text": 3}, "invalid", {"status": "unavailable"}, {"status": "unavailable", "reason_code": "budget_exhausted"}]:
            with self.subTest(evidence=evidence):
                result = self.assert_state(card(third_asr=evidence), "audio_unavailable")
                if isinstance(evidence, dict) and evidence.get("reason_code") == "budget_exhausted":
                    self.assertIn("budget_exhausted", result["reason_codes"])

    def test_identical_both_present_and_neither_supported(self):
        self.assert_state(card("we traveled back", "WE TRAVELED BACK!", "we traveled back"), "ambiguous_audio")
        self.assert_state(card(audio="we traveled back then we traveled home"), "ambiguous_audio")
        self.assert_state(card(audio="totally different sounds"), "neither_source_supported")

    def test_short_alternatives_require_exclusive_exact_occurrence(self):
        self.assert_state(card("back", "home", "well back then"), "machine_supported_apple", "apple")
        self.assert_state(card("go back", "go home", "well go home then"), "machine_supported_whisper", "whisper")
        self.assert_state(card("go back", "go home", "go away", triage=triage("recommend_apple")), "ambiguous_audio")
        self.assert_state(card("back", "home", "back home"), "ambiguous_audio")

    def test_calibrated_one_word_substitutions_use_exclusive_exact_support(self):
        pairs = [
            ("we traveled back", "we traveled home", 2 / 3),
            ("we traveled back from Austria", "we traveled home from Austria", .8),
            ("you coach your athletes through the process", "you guide your athletes through the process", 6 / 7),
            ("we traveled back from Austria after the show last weekend", "we traveled home from Austria after the show last weekend", .9),
        ]
        for apple, whisper, loser_score in pairs:
            with self.subTest(apple=apple):
                item = card(apple, whisper, "well " + apple + " anyway")
                self.assertTrue(assisted_routing(item)["eligible"])
                result = self.assert_state(item, "machine_supported_apple", "apple")
                self.assertIn("exclusive_exact_phrase", result["reason_codes"])
                self.assertAlmostEqual(result["matches"]["whisper"]["score"], loser_score)

    def test_nonexact_strong_audio(self):
        item = card("a b c d e f g", "h i j k l m n", "a b c d e f x")
        result = self.assert_state(item, "machine_supported_apple", "apple")
        self.assertIn("strong_audio", result["reason_codes"])

    def test_nonexact_audio_plus_high_triage_and_boundary(self):
        item = card("a b c d", "w x y z", "a b c q", triage=triage("recommend_apple"))
        result = self.assert_state(item, "machine_supported_apple", "apple")
        self.assertIn("audio_triage_agreement", result["reason_codes"])
        for advisory in [None, triage("recommend_apple", "medium"), triage("recommend_whisper")]:
            self.assert_state({**item, "triage": advisory}, "ambiguous_audio")
        self.assert_state(card("a b c", "x y z", "a b q", triage=triage("recommend_apple")), "ambiguous_audio")

    def test_exact_point_fifteen_margin_is_inclusive_without_float_drift(self):
        result = self.assert_state(card("a b c d", "a b c x y", "a b c q", triage=triage("recommend_apple")), "machine_supported_apple", "apple")
        self.assertEqual(result["matches"]["apple"]["score"], .75)
        self.assertEqual(result["matches"]["whisper"]["score"], .6)
        self.assertEqual(result["margin"], .15)
        self.assertIn("audio_triage_agreement", result["reason_codes"])

    def test_nonexact_margin_below_threshold_and_loser_above_ceiling_abstain(self):
        item = card("a b c d e f g", "a b c d e x y", "a b c d e f z")
        self.assert_state(item, "ambiguous_audio")
        self.assert_state({**item, "triage": triage("recommend_apple")}, "ambiguous_audio")

    def test_strong_audio_is_symmetric_and_high_opposite_is_conflict(self):
        item = card("h i j k l m n", "a b c d e f g", "a b c d e f x")
        self.assert_state(item, "machine_supported_whisper", "whisper")
        self.assert_state({**item, "triage": triage("recommend_apple")}, "evidence_conflict")

    def test_empty_alternative_cannot_receive_exclusive_exact_support(self):
        self.assert_state(card("?!", "go home", "go home"), "ambiguous_audio")

    def test_high_opposite_triage_veto_and_low_medium_disagreement(self):
        item = card(audio="we traveled back")
        result = self.assert_state({**item, "triage": triage("recommend_whisper")}, "evidence_conflict")
        self.assertIn("high_confidence_opposite_source", result["reason_codes"])
        for confidence in ["low", "medium"]:
            result = self.assert_state({**item, "triage": triage("recommend_whisper", confidence)}, "machine_supported_apple", "apple")
            self.assertIn("triage_disagreement", result["reason_codes"])

    def test_custom_unresolved_needs_audio_neutral_and_malformed_triage(self):
        item = card(audio="we traveled back")
        self.assert_state({**item, "triage": triage("likely_custom")}, "ambiguous_audio")
        self.assert_state({**item, "triage": triage("needs_audio")}, "machine_supported_apple", "apple")
        for advisory in [[], {"recommendation": "recommend_whisper"}, triage("invalid"), {**triage("recommend_whisper"), "source": "apple"}]:
            self.assert_state({**item, "triage": advisory}, "machine_supported_apple", "apple")

    def test_focus_and_audit_metadata_compiler_has_zero_vote(self):
        item = card("full original apple text", "full original whisper text", "well we traveled back", focus={"scope": "partial", "apple_text": "we traveled back", "whisper_text": "we traveled home"})
        item["third_asr"].update(prepare_session_id="session", budget_attempt_id="attempt")
        for source in ["apple", "whisper", "third", None]:
            result = self.assert_state({**item, "suggestion": {"source": source}}, "machine_supported_apple", "apple")
            self.assertEqual(result["routing"]["decision_scope"], "focus")
            self.assertEqual(result["third_asr"]["budget_attempt_id"], "attempt")
            self.assertEqual(result["compiler_suggestion"], {"source": source})
            self.assertNotIn("confidence", result)

    def test_returned_nested_metadata_cannot_mutate_input(self):
        item = card(audio="we traveled back")
        result = self.derive(item)
        result["third_asr"]["window"]["start"] = 999
        self.assertEqual(item["third_asr"]["window"]["start"], 8)
