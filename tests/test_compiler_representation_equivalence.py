from __future__ import annotations

import unittest

from compiler import review_policy
from compiler.transcript import compile_transcripts


class RepresentationEquivalencePolicyTests(unittest.TestCase):
    def test_et_cetera_abbreviation_is_equivalent(self):
        result = review_policy.representation_equivalence(
            "et cetera",
            "etc.",
        )

        self.assertIsNotNone(result)
        self.assertEqual(result["equivalence_class"], "et_cetera")
        self.assertTrue(result["reason"])

    def test_repeated_et_cetera_abbreviation_is_equivalent(self):
        result = review_policy.representation_equivalence(
            "et cetera, et cetera",
            "Etc etc",
        )

        self.assertIsNotNone(result)
        self.assertEqual(result["equivalence_class"], "et_cetera")

    def test_clock_notation_requires_matching_value_and_clock_context(self):
        result = review_policy.representation_equivalence(
            "7.30",
            "7, 30",
            apple_context="the conference starts at 7.30",
            whisper_context="the conference starts at 7, 30",
        )

        self.assertIsNotNone(result)
        self.assertEqual(result["equivalence_class"], "clock_time")

    def test_second_clock_fixture_is_equivalent(self):
        result = review_policy.representation_equivalence(
            "10.30",
            "10, 30",
            apple_context="the conference starts at 10.30",
            whisper_context="the conference starts at 10, 30",
        )

        self.assertIsNotNone(result)
        self.assertEqual(result["equivalence_class"], "clock_time")

    def test_clock_shape_without_clock_cue_is_not_equivalent(self):
        self.assertIsNone(
            review_policy.representation_equivalence(
                "7.30",
                "7, 30",
                apple_context="the result was 7.30",
                whisper_context="the result was 7, 30",
            )
        )

    def test_clock_shape_with_duration_cue_is_not_equivalent(self):
        self.assertIsNone(
            review_policy.representation_equivalence(
                "7.30",
                "7, 30",
                apple_context="it lasted 7.30 minutes",
                whisper_context="it lasted 7, 30 minutes",
            )
        )

    def test_et_cetera_normalization_does_not_hide_lexical_difference(self):
        self.assertIsNone(
            review_policy.representation_equivalence(
                "tension, etc",
                "attention, et cetera, et cetera",
            )
        )

    def test_different_numeric_values_are_not_equivalent(self):
        cases = (
            ("six", "16"),
            ("7", "7,000"),
            ("332", "three, three, four, three, three, two"),
        )
        for apple_text, whisper_text in cases:
            with self.subTest(apple_text=apple_text, whisper_text=whisper_text):
                self.assertIsNone(
                    review_policy.representation_equivalence(
                        apple_text,
                        whisper_text,
                    )
                )

    def test_units_do_not_make_ambiguous_clock_shape_safe(self):
        self.assertIsNone(
            review_policy.representation_equivalence(
                "7.30 kg",
                "7, 30 kg",
                apple_context="the load was 7.30 kg",
                whisper_context="the load was 7, 30 kg",
            )
        )


class CompoundSpacingEquivalenceTests(unittest.TestCase):
    def test_spacing_and_hyphenation_variants_are_equivalent(self):
        for apple, whisper in (
            ("pull down", "pulldown"),
            ("eye line", "eyeline"),
            ("side delts", "sidedelts"),
            ("Meta-analysis", "metaanalysis"),
        ):
            with self.subTest(apple=apple, whisper=whisper):
                result = review_policy.representation_equivalence(apple, whisper)
                self.assertIsNotNone(result)
                self.assertEqual(result["equivalence_class"], "compound_spacing")
                self.assertEqual(result["selection_policy"], "preserve_primary_source")

    def test_different_letters_or_numbers_are_not_equivalent(self):
        for apple, whisper in (
            ("pull down", "pull town"),
            ("metanalysis", "meta-analysis"),
            ("1 80", "180"),
            ("set 3 down", "set3 down"),
            ("pull down", "pull down"),
            ("", "pulldown"),
            ("healthpromoting", "health promoting"),
        ):
            with self.subTest(apple=apple, whisper=whisper):
                result = review_policy.representation_equivalence(apple, whisper)
                self.assertTrue(result is None or result["equivalence_class"] != "compound_spacing")


class RepresentationEquivalenceCompilerIntegrationTests(unittest.TestCase):
    def test_et_cetera_equivalence_removes_review_without_rewriting_primary(self):
        apple = "We discussed tension, et cetera."
        whisper = "We discussed tension, etc."

        apple_result = compile_transcripts(apple, whisper, primary="apple")
        whisper_result = compile_transcripts(apple, whisper, primary="whisper")

        self.assertEqual(apple_result.review_required, 0)
        self.assertEqual(whisper_result.review_required, 0)
        self.assertEqual(apple_result.compiled_transcript, apple)
        self.assertEqual(whisper_result.compiled_transcript, whisper)
        self.assertTrue(
            any(
                item.kind == "representation_equivalent"
                and not item.review_required
                for item in apple_result.differences
            )
        )

    def test_context_proven_clock_notation_removes_review(self):
        apple = "The conference starts at 7.30."
        whisper = "The conference starts at 7, 30."

        result = compile_transcripts(apple, whisper, primary="apple")

        self.assertEqual(result.review_required, 0)
        self.assertEqual(result.compiled_transcript, apple)
        self.assertTrue(
            any(
                item.kind == "representation_equivalent"
                and not item.review_required
                for item in result.differences
            )
        )

    def test_spacing_variant_of_a_domain_term_removes_review(self):
        apple = "Then I do a lat pull down for three sets."
        whisper = "Then I do a lat pulldown for three sets."

        result = compile_transcripts(apple, whisper, primary="apple")

        self.assertEqual(result.review_required, 0)
        self.assertEqual(result.compiled_transcript, apple)
        self.assertTrue(
            any(
                item.kind == "representation_equivalent"
                and not item.review_required
                for item in result.differences
            )
        )

    def test_different_domain_term_is_not_a_spacing_equivalent(self):
        result = compile_transcripts(
            "Then I do a lat pull down for three sets.",
            "Then I do a lat pull over for three sets.",
        )

        self.assertFalse(
            any(item.kind == "representation_equivalent" for item in result.differences)
        )

    def test_ambiguous_scalar_shape_stays_in_review(self):
        result = compile_transcripts(
            "The result was 7.30.",
            "The result was 7, 30.",
        )

        self.assertGreater(result.review_required, 0)


if __name__ == "__main__":
    unittest.main()
