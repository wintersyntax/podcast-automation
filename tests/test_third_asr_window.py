"""TASK-123: "Use Third" may insert only the clip words that cover the card."""

from __future__ import annotations

import unittest

from compiler.third_asr_window import WINDOW_METHOD, anchored_third_asr_window


# Invented drawing-class example: a clip spans more words than one review
# card. Inserting it whole would duplicate the surrounding sentences.
SYNTHETIC_DRAWING_CARD = {
    "apple_text": "squares on a board. Next",
    "apple_context": (
        "In this invented drawing class we can draw squares on a board. Next "
        "we put the pencils away before taking a break in the imaginary garden."
    ),
    "whisper_text": "circles beside the door so",
    "whisper_context": (
        "In this invented drawing class we can draw circles beside the door so "
        "we put the pencils away before taking a break in the imaginary garden."
    ),
    "third_asr": {
        "text": (
            "Our fictional class begins and we can draw triangles on paper. Next "
            "we put the pencils away and take a break in the imaginary garden."
        )
    },
}


class AnchoredThirdAsrWindowTests(unittest.TestCase):
    def test_synthetic_clip_yields_only_the_card_words(self):
        window = anchored_third_asr_window(dict(SYNTHETIC_DRAWING_CARD))

        self.assertIsNotNone(window)
        self.assertEqual(window["text"], "triangles on paper. Next")
        self.assertEqual(window["method"], WINDOW_METHOD)
        self.assertEqual(window["anchor_source"], "apple")

    def test_missing_or_blank_evidence_has_no_window(self):
        for evidence in (None, {}, {"text": "  "}, "text"):
            item = dict(SYNTHETIC_DRAWING_CARD, third_asr=evidence)
            self.assertIsNone(anchored_third_asr_window(item))

    def test_ambiguous_anchor_fails_closed(self):
        item = dict(SYNTHETIC_DRAWING_CARD)
        item["third_asr"] = {
            "text": (
                "we can draw triangles on paper. Next we put the pencils away. "
                "we can draw stars on paper. Next we put the pencils away."
            )
        }
        item["whisper_context"] = "unrelated words only"
        self.assertIsNone(anchored_third_asr_window(item))

    def test_window_far_longer_than_the_span_fails_closed(self):
        item = dict(SYNTHETIC_DRAWING_CARD)
        filler = " ".join(["word"] * 40)
        item["third_asr"] = {"text": f"we can draw {filler}. Next we put the pencils away"}
        item["whisper_context"] = "unrelated words only"
        self.assertIsNone(anchored_third_asr_window(item))

    def test_falls_back_to_whisper_context_anchors(self):
        item = dict(SYNTHETIC_DRAWING_CARD)
        item["apple_context"] = "context that does not contain the span"
        window = anchored_third_asr_window(item)
        self.assertIsNotNone(window)
        self.assertEqual(window["anchor_source"], "whisper")

    def test_partial_focus_scope_uses_the_focus_span(self):
        item = {
            "apple_text": "gonna have messed up RPE",
            "apple_context": "when you are tired you are gonna have messed up RPE on every hard set",
            "whisper_text": "gonna have messed up RP",
            "whisper_context": "when you are tired you are gonna have messed up RP on every hard set",
            "focus": {"scope": "partial", "apple_text": "RPE", "whisper_text": "RP"},
            "third_asr": {"text": "you are gonna have messed up R.P.E. on every hard set"},
        }
        window = anchored_third_asr_window(item)
        self.assertIsNotNone(window)
        self.assertEqual(window["text"], "R.P.E")

    def test_clip_with_no_words_between_anchors_is_an_empty_window(self):
        item = {
            "apple_text": "you know",
            "apple_context": "we trained hard you know for three whole months",
            "whisper_text": "",
            "whisper_context": "we trained hard for three whole months",
            "third_asr": {"text": "we trained hard for three whole months"},
        }
        window = anchored_third_asr_window(item)
        self.assertIsNotNone(window)
        self.assertEqual(window["text"], "")


    def _repeated_anchor_item(self, third_text):
        return {
            "apple_text": "one two three four five six seven",
            "apple_context": "we said there's lots of one two three four five six seven thinking about that today",
            "whisper_text": "uno",
            "whisper_context": "unrelated words only",
            "third_asr": {"text": third_text},
        }

    def test_repeated_anchor_picks_the_pair_closest_to_the_card_length(self):
        # TASK-126: longer clips repeat short phrases. The left anchor occurs
        # twice; only the pair whose window matches the card's length is used.
        item = self._repeated_anchor_item(
            "there's lots of a b c d e f g h i j k l there's lots of uno dos tres "
            "cuatro cinco seis siete thinking about that"
        )
        window = anchored_third_asr_window(item)
        self.assertIsNotNone(window)
        self.assertEqual(window["text"], "uno dos tres cuatro cinco seis siete")
        self.assertTrue(window["disambiguated"])

    def test_repeated_anchor_with_equally_close_pairs_fails_closed(self):
        item = self._repeated_anchor_item(
            "there's lots of a b c there's lots of d e f g thinking about that"
        )
        # Windows of 10 and 4 words around a 7-word card: a tie, so no window.
        self.assertIsNone(anchored_third_asr_window(item))

    def test_unambiguous_window_is_not_marked_disambiguated(self):
        window = anchored_third_asr_window(dict(SYNTHETIC_DRAWING_CARD))
        self.assertFalse(window["disambiguated"])


if __name__ == "__main__":
    unittest.main()
