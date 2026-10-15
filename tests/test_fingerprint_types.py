import unittest
from typing import get_type_hints

from podcast_engine import ai_budget, review_audio, review_web


class FingerprintTypeTests(unittest.TestCase):
    def test_budget_and_evidence_signatures_have_distinct_nominal_types(self):
        source = get_type_hints(ai_budget.budget_ledger_path)["source_fingerprint"]
        evidence = get_type_hints(review_audio.third_asr_cache_key)["input_fingerprint"]
        self.assertNotEqual(source, str, "budget_ledger_path must accept SourceFingerprint")
        self.assertNotEqual(evidence, str, "third_asr_cache_key must accept InputFingerprint")
        self.assertIsNot(source, evidence)
        self.assertIs(get_type_hints(ai_budget.budget_summary)["source_fingerprint"], source)
        self.assertEqual(source.__supertype__, str)
        self.assertEqual(evidence.__supertype__, str)
        fingerprint = "sha256:" + "c" * 64
        self.assertEqual(ai_budget.budget_ledger_path("a" * 24, fingerprint),
                         f"episodes/{'a' * 24}/ai/budgets/{'c' * 64}.json")
        self.assertEqual(review_audio.third_asr_cache_key(
            input_fingerprint=fingerprint, item_id=1, window={"start": 0.0}),
            review_audio.third_asr_cache_key(
                input_fingerprint=evidence(fingerprint), item_id=1, window={"start": 0.0}))

    def test_web_source_boundary_is_typed_and_selects_source_not_input(self):
        helper = getattr(review_web, "_source_fingerprint", None)
        self.assertIsNotNone(helper, "review_web source type boundary missing")
        self.assertIs(helper, review_web._source_fingerprint)
        source = get_type_hints(ai_budget.budget_ledger_path)["source_fingerprint"]
        fields = get_type_hints(review_web.ReviewFingerprintFields)
        self.assertIs(fields["source_fingerprint"], source)
        self.assertIsNot(fields["input_fingerprint"], source)
        self.assertIs(get_type_hints(helper)["return"], source)
        self.assertEqual(helper({"source_fingerprint": "source", "input_fingerprint": "input"}), "source")
