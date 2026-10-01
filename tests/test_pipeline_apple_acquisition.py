import unittest
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from podcast_engine.episode_identity import episode_key_for
from podcast_engine.pipeline import (
    APPLE_TRANSCRIPT_DELAY_DAYS,
    APPLE_TRANSCRIPT_LATE_DAYS,
    _apple_acquisition_due,
    _emit_apple_transcript_late_if_needed,
    process_episode,
)


class PipelineAppleAcquisitionTests(
    unittest.TestCase
):
    feed_url = (
        "https://example.test/feed.xml"
    )

    guid = "episode-guid-123"

    podcast_config = {
        "id": "test-podcast",
        "name": "Test Podcast",
        "rss": feed_url,
        "apple_podcasts": {
            "show_id": "123456789",
            "show_title": "Test Podcast",
            "storefront": "us",
        },
    }

    def _episode(
        self,
        published=(
            "Sun, 23 Aug 2026 "
            "12:00:00 +0000"
        ),
    ):
        key = episode_key_for(
            self.feed_url,
            self.guid,
        )

        return {
            "schema_version": 3,
            "id": key,
            "episode_key": key,
            "podcast": "Test Podcast",
            "podcast_id": "test-podcast",
            "feed_url": self.feed_url,
            "guid": self.guid,
            "title": "Test Episode",
            "published": published,
            "audio_url": (
                "https://example.test/audio.mp3"
            ),
            "status": {
                "download": {"state": "completed"},
                "apple_transcript": {"state": "pending"},
                "whisper": {"state": "ready"},
                "compiler": {
                    "state": "blocked"
                },
                "summary": {"state": "pending"},
            },
            "files": {
                "sources": {
                    "apple": {
                        "text": None,
                        "metadata": None,
                    },
                    "whisper": {
                        "text": (
                            "episodes/test/"
                            "sources/whisper/"
                            "transcript.txt"
                        ),
                        "metadata": (
                            "episodes/test/"
                            "sources/whisper/"
                            "transcript.json"
                        ),
                    },
                },
                "compiled": {"transcript": None, "report": None},
                "summary": {"body": None, "metadata": None, "markdown": None},
            },
        }

    def test_apple_acquisition_becomes_due_at_two_days(
        self,
    ):
        now = datetime(
            2026,
            8,
            25,
            11,
            59,
            59,
            tzinfo=timezone.utc,
        )

        not_due = self._episode(
            "Sun, 23 Aug 2026 "
            "12:00:00 +0000"
        )

        self.assertFalse(
            _apple_acquisition_due(
                not_due,
                now=now,
            )
        )

        due_at_two_days = now.replace(
            hour=12,
            minute=0,
            second=0,
        )

        self.assertTrue(
            _apple_acquisition_due(
                not_due,
                now=due_at_two_days,
            )
        )

    def test_apple_transcript_delay_remains_two_days(
        self,
    ):
        self.assertEqual(
            APPLE_TRANSCRIPT_DELAY_DAYS,
            2,
        )

    def test_apple_transcript_late_threshold_remains_ten_days(
        self,
    ):
        self.assertEqual(
            APPLE_TRANSCRIPT_LATE_DAYS,
            10,
        )

    def test_no_late_event_before_ten_days(
        self,
    ):
        episode = self._episode(
            "Sun, 16 Aug 2026 "
            "12:00:00 +0000"
        )
        now = datetime(
            2026,
            8,
            26,
            11,
            59,
            59,
            tzinfo=timezone.utc,
        )

        with (
            patch(
                "podcast_engine.pipeline.update_episode"
            ) as update,
            patch(
                "podcast_engine.pipeline.emit_event"
            ) as emit,
        ):
            result = _emit_apple_transcript_late_if_needed(
                episode,
                now=now,
            )

        self.assertIs(result, episode)
        update.assert_not_called()
        emit.assert_not_called()

    def test_late_event_is_persisted_once_with_safe_metadata(
        self,
    ):
        episode = self._episode(
            "Sun, 16 Aug 2026 "
            "12:00:00 +0000"
        )
        now = datetime(
            2026,
            8,
            26,
            12,
            0,
            0,
            tzinfo=timezone.utc,
        )
        marked = deepcopy(episode)
        marked["apple_transcript_late_notified_at"] = (
            now.isoformat()
        )

        with (
            patch(
                "podcast_engine.pipeline.update_episode",
                return_value=marked,
            ) as update,
            patch(
                "podcast_engine.pipeline.emit_event"
            ) as emit,
        ):
            result = _emit_apple_transcript_late_if_needed(
                episode,
                now=now,
            )
            repeated = _emit_apple_transcript_late_if_needed(
                result,
                now=now,
            )

        self.assertIs(result, marked)
        self.assertIs(repeated, marked)
        update.assert_called_once_with(
            episode["id"],
            apple_transcript_late_notified_at=now.isoformat(),
        )
        emit.assert_called_once_with(
            "apple_transcript_late",
            "Apple transcript is still unavailable after ten days.",
            severity="NOTICE",
            podcast="Test Podcast",
            title="Test Episode",
            episode_key=episode["episode_key"],
            published=episode["published"],
            age_days=10,
            apple_eligible_at=(
                "2026-08-18T12:00:00+00:00"
            ),
        )

    def test_late_event_does_not_stop_apple_acquisition(
        self,
    ):
        episode = self._episode(
            "Mon, 10 Aug 2026 "
            "12:00:00 +0000"
        )
        marked = deepcopy(episode)
        marked["apple_transcript_late_notified_at"] = (
            "2026-08-27T12:00:00+00:00"
        )
        rss_episode = {
            "podcast": "Test Podcast",
            "feed_url": self.feed_url,
            "guid": self.guid,
            "title": "Test Episode",
        }
        unavailable = {
            "status": "TRANSCRIPT_NOT_READY",
            "episode_key": episode["episode_key"],
        }

        with (
            patch(
                "podcast_engine.pipeline._rss_episode",
                return_value=rss_episode,
            ),
            patch(
                "podcast_engine.pipeline.get_episode_by_key",
                return_value=None,
            ),
            patch(
                "podcast_engine.pipeline.add_episode",
                return_value=episode,
            ),
            patch(
                "podcast_engine.pipeline._transcribe_if_needed",
                return_value=episode,
            ),
            patch(
                "podcast_engine.pipeline._apple_acquisition_due",
                return_value=True,
            ),
            patch(
                "podcast_engine.pipeline.update_episode",
                return_value=marked,
            ) as update,
            patch(
                "podcast_engine.pipeline.emit_event"
            ) as emit,
            patch(
                "podcast_engine.pipeline.acquire_apple_transcript",
                return_value=unavailable,
            ) as acquire,
        ):
            outcome = process_episode(
                self.podcast_config
            )

        update.assert_called_once()
        emit.assert_called_once()
        acquire.assert_called_once_with(
            self.podcast_config,
            marked,
        )
        self.assertEqual(
            outcome["status"],
            "waiting_for_apple_transcript",
        )
        self.assertEqual(
            outcome["reason"], "not_ready")

    def test_pipeline_does_not_call_apple_before_delay(
        self,
    ):
        episode = self._episode()

        rss_episode = {
            "podcast": "Test Podcast",
            "feed_url": self.feed_url,
            "guid": self.guid,
            "title": "Test Episode",
        }

        with (
            patch(
                "podcast_engine.pipeline._rss_episode",
                return_value=rss_episode,
            ),
            patch(
                "podcast_engine.pipeline.get_episode_by_key",
                return_value=None,
            ),
            patch(
                "podcast_engine.pipeline.add_episode",
                return_value=episode,
            ),
            patch(
                "podcast_engine.pipeline._transcribe_if_needed",
                return_value=episode,
            ),
            patch(
                "podcast_engine.pipeline._apple_acquisition_due",
                return_value=False,
            ),
            patch(
                "podcast_engine.pipeline.acquire_apple_transcript"
            ) as acquire,
        ):
            outcome = process_episode(
                self.podcast_config
            )

        self.assertEqual(
            outcome["status"],
            "waiting_for_apple_transcript",
        )

        self.assertEqual(
            outcome["reason"],
            "delay",
        )

        acquire.assert_not_called()

    def test_eventual_apple_upload_uses_normal_ingest_flow(
        self,
    ):
        published = (
            datetime.now(timezone.utc)
            - timedelta(days=3)
        ).strftime(
            "%a, %d %b %Y %H:%M:%S +0000"
        )
        episode = self._episode(published)

        rss_episode = {
            "podcast": "Test Podcast",
            "feed_url": self.feed_url,
            "guid": self.guid,
            "title": "Test Episode",
        }

        uploaded = {
            "status": "uploaded",
            "episode_key": (
                episode["episode_key"]
            ),
        }

        with (
            patch(
                "podcast_engine.pipeline._rss_episode",
                return_value=rss_episode,
            ),
            patch(
                "podcast_engine.pipeline.get_episode_by_key",
                return_value=None,
            ),
            patch(
                "podcast_engine.pipeline.add_episode",
                return_value=episode,
            ),
            patch(
                "podcast_engine.pipeline._transcribe_if_needed",
                return_value=episode,
            ),
            patch(
                "podcast_engine.pipeline._apple_acquisition_due",
                return_value=True,
            ),
            patch(
                "podcast_engine.pipeline.acquire_apple_transcript",
                return_value=uploaded,
            ) as acquire,
        ):
            outcome = process_episode(
                self.podcast_config
            )

        acquire.assert_called_once_with(
            self.podcast_config,
            episode,
        )

        self.assertEqual(
            outcome["status"],
            "waiting_for_apple_transcript",
        )

        self.assertEqual(
            outcome["reason"],
            "ingest_pending",
        )

        self.assertEqual(
            outcome["apple"],
            uploaded,
        )

    def test_pipeline_keeps_knowledge_behind_compiler_review_gate(
        self,
    ):
        episode = self._episode()
        episode["files"]["sources"]["apple"]["text"] = (
            "episodes/test/sources/apple/transcript.txt"
        )
        episode["status"]["compiler"] = {"state": "ready"}
        review_episode = deepcopy(episode)
        review_episode["status"]["compiler"] = {
            "state": "review_required"
        }
        compiled = {
            "transcript": "episodes/test/compiled/transcript.txt",
            "report": "episodes/test/compiled/report.json",
            "review_required": True,
            "review": [{"id": 1}],
        }

        with (
            patch("podcast_engine.pipeline._rss_episode", return_value={"podcast": "Test Podcast", "feed_url": self.feed_url, "guid": self.guid, "title": "Test Episode"}),
            patch("podcast_engine.pipeline.get_episode_by_key", return_value=None),
            patch("podcast_engine.pipeline.add_episode", return_value=episode),
            patch("podcast_engine.pipeline._transcribe_if_needed", return_value=episode),
            patch("podcast_engine.pipeline.compile_episode_sources", return_value=compiled),
            patch("podcast_engine.pipeline.update_episode", return_value=review_episode),
            patch("podcast_engine.pipeline.build_knowledge_note") as knowledge,
        ):
            outcome = process_episode(self.podcast_config)

        self.assertEqual(outcome["status"], "review_required")
        knowledge.assert_not_called()

    def test_pipeline_runs_knowledge_note_without_compatibility_export(
        self,
    ):
        episode = self._episode()
        episode["files"]["sources"]["apple"]["text"] = "apple.txt"
        episode["files"]["compiled"] = {"transcript": "compiled.txt"}
        episode["status"]["compiler"] = {"state": "completed"}
        summarized = deepcopy(episode)
        summarized["status"]["summary"] = {"state": "ready"}

        with (
            patch("podcast_engine.pipeline._rss_episode", return_value={"podcast": "Test Podcast", "feed_url": self.feed_url, "guid": self.guid, "title": "Test Episode"}),
            patch("podcast_engine.pipeline.get_episode_by_key", return_value=None),
            patch("podcast_engine.pipeline.add_episode", return_value=episode),
            patch("podcast_engine.pipeline._transcribe_if_needed", return_value=episode),
            patch("podcast_engine.pipeline.build_knowledge_note", return_value="/tmp/summary.md") as knowledge,
            patch("podcast_engine.pipeline.update_episode", return_value=summarized) as update,
            patch(
                "podcast_engine.pipeline.canonical_summary_generated_at",
                return_value="2026-08-30T12:34:56+00:00",
            ),
            patch("podcast_engine.pipeline.send_slack_summary_ready_notification"),
        ):
            outcome = process_episode(self.podcast_config)

        knowledge.assert_called_once_with(episode)
        self.assertEqual(
            update.call_args_list[0].kwargs["markdown_file"],
            f"episodes/{episode['episode_key']}/summary/summary.md",
        )
        self.assertEqual(
            update.call_args_list[0].kwargs["summary_body_file"],
            f"episodes/{episode['episode_key']}/summary/body.md",
        )
        update.assert_called_once()
        self.assertEqual(outcome["status"], "completed")

    def test_completed_episode_is_not_reprocessed(self):
        episode = self._episode()
        episode["files"]["sources"]["apple"]["text"] = "apple.txt"
        episode["files"]["compiled"] = {"transcript": "compiled.txt", "report": "report.json"}
        episode["files"]["summary"] = {
            "body": f"episodes/{episode['episode_key']}/summary/body.md",
            "metadata": f"episodes/{episode['episode_key']}/summary/metadata.json",
            "markdown": f"episodes/{episode['episode_key']}/summary/summary.md",
        }
        episode["status"]["compiler"] = {"state": "completed"}
        episode["status"]["summary"] = {"state": "ready"}

        with (
            patch("podcast_engine.pipeline._rss_episode", return_value={"podcast": "Test Podcast", "feed_url": self.feed_url, "guid": self.guid, "title": "Test Episode"}),
            patch("podcast_engine.pipeline.get_episode_by_key", return_value=None),
            patch("podcast_engine.pipeline.add_episode", return_value=episode),
            patch("podcast_engine.pipeline._transcribe_if_needed", return_value=episode),
            patch("podcast_engine.pipeline.build_knowledge_note", return_value="/tmp/summary.md") as knowledge,
            patch("podcast_engine.pipeline.update_episode", return_value=episode),
        ):
            outcome = process_episode(self.podcast_config)

        knowledge.assert_not_called()
        self.assertEqual(outcome["status"], "skipped")
        self.assertEqual(outcome["reason"], "completed")


if __name__ == "__main__":
    unittest.main()
