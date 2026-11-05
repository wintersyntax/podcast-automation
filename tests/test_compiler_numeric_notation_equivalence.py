import unittest

from compiler.transcript import compile_transcripts


class CompilerNumericNotationEquivalenceTests(unittest.TestCase):
    def test_scale_word_notation_matches_same_digit_value(self):
        apple = "We tested 1,500 people."
        whisper = "We tested a thousand five hundred people."

        result = compile_transcripts(apple, whisper)

        self.assertEqual(result.review_required, 0)
        self.assertEqual(len(result.differences), 1)
        self.assertEqual(result.differences[0].kind, "numeric_unit_equivalent")
        self.assertFalse(result.differences[0].review_required)

    def test_scale_word_equivalence_preserves_selected_source_text(self):
        apple = "We tested 1,500 people."
        whisper = "We tested a thousand five hundred people."

        self.assertEqual(
            compile_transcripts(apple, whisper, primary="apple").compiled_transcript,
            apple,
        )
        self.assertEqual(
            compile_transcripts(apple, whisper, primary="whisper").compiled_transcript,
            whisper,
        )

    def test_genuinely_different_values_still_require_review(self):
        result = compile_transcripts(
            "The weight moved between 103 and 104 kg.",
            "The weight moved to 304 kg.",
        )

        self.assertGreater(result.review_required, 0)
        self.assertTrue(
            any(item.kind == "number_mismatch" for item in result.differences)
        )


if __name__ == "__main__":
    unittest.main()
