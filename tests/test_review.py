import unittest
from unittest.mock import patch

from compiler.transcript import build_resolver_batch, compile_transcripts
from podcast_engine.review import _completion_metadata, resolve_compiler_batch


class TranscriptReviewTests(unittest.TestCase):
    def test_empty_review_batch_skips_openrouter(self):
        result = compile_transcripts("The result is clear.", "The result is clear.")
        batch = build_resolver_batch(result, "episode-42")

        with patch("podcast_engine.review.requests.post") as post:
            self.assertEqual(resolve_compiler_batch(batch), {"accepted": [], "review": []})

        post.assert_not_called()

    def test_completion_metadata_keeps_only_documented_returned_values(self):
        metadata = _completion_metadata({
            "id": "chatcmpl-1",
            "model": "google/gemini-test",
            "openrouter_metadata": {"endpoints": {"available": [
                {"provider": "Unselected", "selected": False},
                {"provider": "Selected", "selected": True},
            ]}},
        })

        self.assertEqual(metadata, {
            "completion_id": "chatcmpl-1",
            "served_model": "google/gemini-test",
            "served_provider": "Selected",
        })

    def test_completion_metadata_does_not_invent_absent_provider_or_completion(self):
        self.assertEqual(_completion_metadata({"model": "google/gemini-test"}), {"served_model": "google/gemini-test"})

    def test_completion_metadata_does_not_guess_provider_from_unselected_routing(self):
        metadata = _completion_metadata({
            "id": "chatcmpl-1",
            "model": "google/gemini-test",
            "openrouter_metadata": {"endpoints": {"available": [{"provider": "Configured Provider", "selected": False}]}},
        })

        self.assertEqual(metadata, {"completion_id": "chatcmpl-1", "served_model": "google/gemini-test"})
