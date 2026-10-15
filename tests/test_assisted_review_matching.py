import importlib.util
import unittest

from compiler.review_policy import comparison_tokens

if importlib.util.find_spec("compiler.assisted_review"):
    from compiler import assisted_review
else:
    assisted_review = None


class ComparisonTokenTests(unittest.TestCase):
    def test_unicode_case_punctuation_and_internal_apostrophes(self):
        self.assertEqual(
            comparison_tokens("ＦＯＯ Café! YOU'RE you’re 'quoted' １２"),
            ["foo", "café", "you're", "you’re", "quoted", "12"],
        )


class LocalWordMatchTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(assisted_review, "Task 8 matcher is not implemented")

    def match(self, source, audio):
        return assisted_review.best_local_word_edit_match(source, audio)

    def test_context_window_and_first_exact_occurrence(self):
        result = self.match("we traveled back", "well we traveled back then we traveled back")
        self.assertEqual(result["method"], "local-word-edit-v1")
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["edit_distance"], 0)
        self.assertEqual(result["token_range"], [1, 4])

    def test_unit_substitution_distance_and_normalized_score(self):
        result = self.match("we traveled home", "we traveled back")
        self.assertAlmostEqual(result["score"], 2 / 3)
        self.assertEqual(result["edit_distance"], 1)
        # Deleting "home" ties substituting "back": same score/distance/start.
        self.assertEqual(result["token_range"], [0, 2])

    def test_insertion_window_beats_shorter_inexact_window(self):
        result = self.match("a b", "a x b q a")
        self.assertAlmostEqual(result["score"], 2 / 3)
        self.assertEqual(result["token_range"], [0, 3])

    def test_equal_score_prefers_lower_edit_distance(self):
        result = self.match("a b", "a a a a x x b")
        # [3,7] has score 1/2 and distance 2; [0,1] has 1/2 and distance 1.
        self.assertEqual(result["score"], .5)
        self.assertEqual(result["edit_distance"], 1)
        self.assertEqual(result["token_range"], [0, 1])

    def test_zero_score_tie_prefers_shortest_earliest_window(self):
        result = self.match("a b", "x y z")
        self.assertEqual(result["score"], 0)
        self.assertEqual(result["edit_distance"], 2)
        self.assertEqual(result["token_range"], [0, 1])

    def test_window_lower_bound_excludes_too_short_occurrence(self):
        result = self.match("a b c d e f g h i j k", "a b c")
        self.assertEqual(result["score"], 0)
        self.assertIsNone(result["token_range"])

    def test_empty_and_punctuation_only_inputs_have_no_match(self):
        for source, audio in [("", "abc"), ("abc", ""), ("?!", "...")]:
            with self.subTest(source=source, audio=audio):
                result = self.match(source, audio)
                self.assertEqual(result["score"], 0)
                self.assertIsNone(result["edit_distance"])
                self.assertIsNone(result["token_range"])
