"""TASK-143: review notes survive distinct saves and uncertain-response retries."""

import copy
import unittest
from unittest.mock import patch

from podcast_engine.human_review import record_review_note

KEY = "a" * 24
GENERATION = "sha256:review-generation"


class ReviewNoteRetriesTests(unittest.TestCase):
    def test_multiple_distinct_notes_append_without_new_card_decisions(self):
        state = {
            "episode_key": KEY,
            "human_review_generation_fingerprint": GENERATION,
            "human_review": [{"id": 17, "apple_text": "A", "whisper_text": "B"}],
            "human_decisions": [],
            "materiality_review_log": [],
        }
        generation = 7

        def read(key):
            self.assertEqual(key, KEY)
            return copy.deepcopy(state), generation

        def save(key, snapshot, *, if_generation_match):
            nonlocal state, generation
            self.assertEqual(key, KEY)
            self.assertEqual(if_generation_match, generation)
            state = copy.deepcopy(snapshot)
            generation += 1

        with (
            patch("podcast_engine.human_review.load_review_record_with_generation",
                  side_effect=read),
            patch("podcast_engine.human_review.save_review_record", side_effect=save),
        ):
            first = record_review_note(
                KEY, expected_generation_fingerprint=GENERATION,
                note="Quick cards could have resolved more.",
            )
            self.assertEqual(first, {"saved": True, "idempotent": False})
            first_generation = generation
            uncertain_retry = record_review_note(
                KEY, expected_generation_fingerprint=GENERATION,
                note="Quick cards could have resolved more.",
            )
            self.assertEqual(uncertain_retry, {"saved": True, "idempotent": True})
            self.assertEqual(generation, first_generation)
            second = record_review_note(
                KEY, expected_generation_fingerprint=GENERATION,
                note="Full review needs a better category overview.",
            )
            self.assertEqual(second, {"saved": True, "idempotent": False})
            later = record_review_note(
                KEY, expected_generation_fingerprint=GENERATION,
                note="Quick cards could have resolved more.",
            )
            self.assertEqual(later, {"saved": True, "idempotent": False})

        notes = state["materiality_review_log"]
        self.assertEqual(
            [entry["note"] for entry in notes],
            [
                "Quick cards could have resolved more.",
                "Full review needs a better category overview.",
                "Quick cards could have resolved more.",
            ],
        )
        self.assertTrue(all(entry["note_only"] is True for entry in notes))
        self.assertTrue(all(entry["decisions"] == 0 for entry in notes))
        self.assertEqual(state["human_decisions"], [])
        self.assertEqual([x["id"] for x in state["human_review"]], [17])
        self.assertEqual(state["human_review_generation_fingerprint"], GENERATION)

    def test_stale_review_generation_rejects_a_note_without_writing(self):
        current = {
            "episode_key": KEY,
            "human_review_generation_fingerprint": GENERATION,
            "materiality_review_log": [],
        }
        with (
            patch("podcast_engine.human_review.load_review_record_with_generation",
                  return_value=(current, 1)),
            patch("podcast_engine.human_review.save_review_record") as save,
        ):
            with self.assertRaisesRegex(ValueError, "generation changed"):
                record_review_note(
                    KEY, expected_generation_fingerprint="sha256:obsolete",
                    note="Do not save on the wrong generation.",
                )
            save.assert_not_called()


if __name__ == "__main__":
    unittest.main()
