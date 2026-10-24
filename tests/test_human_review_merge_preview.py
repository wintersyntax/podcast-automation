import unittest
from unittest.mock import patch

import compiler.transcript as transcript
import podcast_engine.compilation as compilation
import podcast_engine.human_review as human_review
from podcast_engine.review_web import create_review_app


class HumanReviewMergePreviewTests(unittest.TestCase):
    def _full_item(self):
        context = "He said, and the body's not a car, so bad."
        span = "body's not a car"
        start = context.index(span)
        return {
            "id": 1,
            "apple_text": span,
            "whisper_text": "body is not a car",
            "focus": {"scope": "full", "apple_text": span, "whisper_text": "body is not a car"},
            "merge_preview": {
                "base_source": "apple",
                "context_text": context,
                "span_start": start,
                "span_end": start + len(span),
                "span_text": span,
            },
        }

    def _partial_item(self):
        context = "before gonna have messed up RP after"
        span = "gonna have messed up RP"
        start = context.index(span)
        return {
            "id": 2,
            "apple_text": "gonna have messed up RPE",
            "whisper_text": span,
            "focus": {"scope": "partial", "apple_text": "RPE", "whisper_text": "RP"},
            "merge_preview": {
                "base_source": "whisper",
                "context_text": context,
                "span_start": start,
                "span_end": start + len(span),
                "span_text": span,
            },
        }

    def _underscoped_item(self):
        context = "You can probably reflect upon to know even having vegetables with meals."
        span = "to"
        start = context.index(span, context.index("reflect upon"))
        return {
            "id": 1,
            "apple_text": "to",
            "whisper_text": "I don't",
            "focus": {"scope": "full", "apple_text": "to", "whisper_text": "I don't"},
            "merge_preview": {
                "base_source": "apple",
                "context_text": context,
                "span_start": start,
                "span_end": start + len(span),
                "span_text": span,
            },
        }

    def test_full_scope_custom_preview_exposes_boundary_duplication(self):
        merged = human_review.preview_custom_edit(
            self._full_item(),
            "and the bo the body's not a car",
        )
        self.assertEqual(
            merged,
            "He said, and the and the bo the body's not a car, so bad.",
        )

    def test_partial_scope_custom_preview_replaces_only_the_focus(self):
        merged = human_review.preview_custom_edit(
            self._partial_item(),
            "RPE",
        )
        self.assertEqual(merged, "before gonna have messed up RPE after")

    def test_custom_preview_can_expand_into_one_adjacent_agreed_word(self):
        item = self._underscoped_item()
        self.assertEqual(
            human_review.preview_custom_edit(item, "to not"),
            "You can probably reflect upon to not know even having vegetables with meals.",
        )
        self.assertEqual(
            human_review.preview_custom_edit(
                item,
                "to not",
                expand_right_words=1,
            ),
            "You can probably reflect upon to not even having vegetables with meals.",
        )

    def test_custom_preview_expansion_is_strictly_bounded(self):
        item = self._underscoped_item()
        with self.assertRaisesRegex(ValueError, "at most 3"):
            human_review.preview_custom_edit(
                item,
                "replacement",
                expand_right_words=4,
            )
        with self.assertRaisesRegex(ValueError, "whole words"):
            human_review.preview_custom_edit(
                item,
                "replacement",
                expand_left_words=-1,
            )

    def test_preview_uses_compiler_boundary_spacing_for_insertions(self):
        merged = transcript.render_review_merge_preview(
            "hello world",
            span_start=6,
            span_end=6,
            replacement="brave",
        )
        self.assertEqual(merged, "hello brave world")

    def test_preview_fails_closed_when_geometry_does_not_match_the_base_span(self):
        item = self._full_item()
        item["merge_preview"]["span_text"] = "different text"
        with self.assertRaisesRegex(ValueError, "preview"):
            human_review.preview_custom_edit(item, "replacement")

    def test_review_merge_preview_geometry_uses_the_primary_source_word_span(self):
        result = transcript.compile_transcripts(
            "before alpha after",
            "before beta after",
            primary="apple",
        )
        difference = result.differences[0]
        preview = compilation.build_review_merge_preview(
            difference,
            base_source="apple",
            base_text=transcript.clean_transcript("before alpha after"),
        )
        self.assertEqual(preview["base_source"], "apple")
        self.assertEqual(preview["span_text"], difference.apple_text)
        self.assertEqual(
            preview["context_text"][preview["span_start"] : preview["span_end"]],
            difference.apple_text,
        )

    def test_preview_endpoint_is_read_only_and_returns_final_context(self):
        record = {
            "episode_key": "episode-1",
            "human_review": [self._full_item()],
            "human_decisions": [],
        }
        with (
            patch("podcast_engine.review_web.load_review_record", return_value=record),
            patch("podcast_engine.review_web.record_human_decision") as decide,
        ):
            client = create_review_app().test_client()
            response = client.post(
                "/api/review/episodes/episode-1/items/1/preview",
                json={"source": "custom", "text": "bo the body's not a car"},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.get_json()["merged_context"],
            "He said, and the bo the body's not a car, so bad.",
        )
        decide.assert_not_called()

    def test_preview_endpoint_returns_the_exact_expanded_replacement_range(self):
        record = {
            "episode_key": "episode-1",
            "human_review": [self._underscoped_item()],
            "human_decisions": [],
        }
        with patch("podcast_engine.review_web.load_review_record", return_value=record):
            response = create_review_app().test_client().post(
                "/api/review/episodes/episode-1/items/1/preview",
                json={
                    "source": "custom",
                    "text": "to not",
                    "expand_right_words": 1,
                },
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["replaced_text"], "to know")
        self.assertEqual(
            response.get_json()["merged_context"],
            "You can probably reflect upon to not even having vegetables with meals.",
        )

    def test_expanded_custom_decision_replays_the_same_bounded_edit(self):
        item = self._underscoped_item()
        record = {
            "episode_key": "episode-1",
            "human_review": [item],
            "human_decisions": [],
        }
        with (
            patch("podcast_engine.human_review.load_review_record", return_value=record),
            patch("podcast_engine.human_review.save_review_record"),
        ):
            decision = human_review.record_human_decision(
                "episode-1",
                1,
                source="custom",
                text="to not",
                expand_right_words=1,
            )

        self.assertEqual(decision["scope"], "expanded")
        self.assertEqual(decision["edit_base_source"], "apple")
        self.assertEqual(decision["expand_left_words"], 0)
        self.assertEqual(decision["expand_right_words"], 1)
        self.assertEqual(decision["expected_base_text"], "to know")

        resolutions = human_review.validated_human_resolutions(record, [item])
        self.assertEqual(
            resolutions,
            [{
                "id": 1,
                "source": "human",
                "text": "to not",
                "scope": "expanded",
                "reviewed_by": "human",
                "edit_base_source": "apple",
                "expand_left_words": 0,
                "expand_right_words": 1,
                "expected_base_text": "to know",
            }],
        )
        result = transcript.compile_transcripts(
            "reflect upon to know even having vegetables with meals",
            "reflect upon I don't know even having vegetables with meals",
            primary="apple",
            resolver_resolutions=resolutions,
        )
        self.assertEqual(
            result.compiled_transcript,
            "reflect upon to not even having vegetables with meals",
        )

    def test_review_ui_requires_exact_final_merge_preview_for_custom_edit(self):
        client = create_review_app().test_client()
        page = client.get("/")
        self.assertIn(b"Final merged context", page.data)
        self.assertIn(b"Expand edit range", page.data)
        self.assertIn(b"Replacing exactly", page.data)
        self.assertIn(b"expand_right_words", page.data)
        self.assertIn(b"/preview", page.data)
        self.assertIn(b"customPreviewReady", page.data)


if __name__ == "__main__":
    unittest.main()
