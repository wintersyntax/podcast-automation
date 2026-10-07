import os

# Legacy-path tests: a developer .env (loaded by knowledge.client via
# python-dotenv) may activate the TASK-118 note writer. Keep these tests
# hermetic -- the writer path has its own tests.
os.environ["PODCAST_KNOWLEDGE_WRITER_PRESET"] = ""
import unittest
from unittest.mock import patch

import podcast_engine.knowledge.orchestration as knowledge
import podcast_engine.knowledge.summary_review as summary_review
from podcast_engine.episode_contract import new_episode_record
from podcast_engine.pipeline import resume_tracked_episode


class SummaryReviewActivationTests(unittest.TestCase):
    def test_no_explicit_preset_is_inactive(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(summary_review.active_summary_review_preset())

    def test_blank_explicit_preset_is_inactive(self):
        with patch.dict(
            os.environ,
            {
                "PODCAST_SUMMARY_REVIEW_PRESET": "   ",
                "PODCAST_SUMMARY_REVIEW_API_KEY": "review-key",
            },
            clear=True,
        ):
            self.assertIsNone(summary_review.active_summary_review_preset())

    def test_explicit_preset_without_dedicated_key_fails_closed(self):
        with patch.dict(
            os.environ,
            {"PODCAST_SUMMARY_REVIEW_PRESET": "winner-reviewer"},
            clear=True,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "PODCAST_SUMMARY_REVIEW_API_KEY",
            ):
                summary_review.active_summary_review_preset()

    def test_explicit_preset_and_dedicated_key_activate_reviewer(self):
        with patch.dict(
            os.environ,
            {
                "PODCAST_SUMMARY_REVIEW_PRESET": "winner-reviewer",
                "PODCAST_SUMMARY_REVIEW_API_KEY": "review-key",
            },
            clear=True,
        ):
            self.assertEqual(
                summary_review.active_summary_review_preset(),
                "winner-reviewer",
            )

    def test_explicit_preset_normalization_matches_request_identity(self):
        with patch.dict(
            os.environ,
            {
                "PODCAST_SUMMARY_REVIEW_PRESET": "  winner-reviewer  ",
                "PODCAST_SUMMARY_REVIEW_API_KEY": "review-key",
            },
            clear=True,
        ):
            self.assertEqual(
                summary_review.active_summary_review_preset(),
                "winner-reviewer",
            )
            self.assertEqual(
                summary_review.summary_review_preset(),
                "winner-reviewer",
            )

    def test_inactive_reviewer_stops_before_transcript_download_or_draft(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(
                knowledge,
                "download_file_from_gcs",
                side_effect=AssertionError("inactive reviewer must short-circuit"),
            ) as download,
            patch.object(knowledge.summary, "generate") as summary_generate,
        ):
            result = knowledge.build_knowledge_note({"episode_key": "episode-1"})

        self.assertIsNone(result)
        download.assert_not_called()
        summary_generate.assert_not_called()

    def test_pipeline_returns_resumable_waiting_state_when_summary_is_inactive(self):
        episode = new_episode_record(
            episode_key="episode-1",
            podcast="Test Podcast",
            podcast_id="test-podcast",
            feed_url="https://example.test/feed.xml",
            rss_guid="episode-1-guid",
            title="Activation safety",
            published="2026-09-12T00:00:00Z",
            link="https://example.test/episode",
            audio_url="https://example.test/audio.mp3",
        )
        episode["status"]["download"]["state"] = "completed"
        episode["status"]["whisper"]["state"] = "ready"
        episode["status"]["apple_transcript"]["state"] = "ready"
        episode["status"]["compiler"]["state"] = "completed"
        episode["files"]["sources"]["whisper"]["text"] = "whisper.txt"
        episode["files"]["sources"]["apple"]["text"] = "apple.txt"

        with (
            patch(
                "podcast_engine.pipeline._transcribe_if_needed",
                return_value=episode,
            ),
            patch(
                "podcast_engine.pipeline.build_knowledge_note",
                return_value=None,
            ) as build,
            patch("podcast_engine.pipeline.update_episode") as update,
        ):
            result = resume_tracked_episode(
                {
                    "id": "test-podcast",
                    "name": "Test Podcast",
                    "rss": "https://example.test/feed.xml",
                },
                episode,
            )

        self.assertEqual(result["status"], "waiting_for_summary_review_activation")
        self.assertEqual(result["reason"], "reviewer_not_activated")
        self.assertIs(result["episode"], episode)
        build.assert_called_once_with(episode)
        update.assert_not_called()


if __name__ == "__main__":
    unittest.main()
