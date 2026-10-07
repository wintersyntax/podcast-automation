import json
import unittest
from copy import deepcopy
from unittest.mock import patch

from podcast_engine import pipeline
from podcast_engine.episode_contract import new_episode_record


class PipelineNotificationTests(unittest.TestCase):
    podcast = {
        "id": "test-podcast",
        "name": "Test Podcast",
        "rss": "https://example.test/feed.xml",
    }

    def _ready_for_summary(self):
        episode = new_episode_record(
            episode_key="episode-123",
            podcast="Test Podcast",
            podcast_id="test-podcast",
            feed_url=self.podcast["rss"],
            rss_guid="guid-123",
            title="Example Episode",
            published=None,
            link=None,
            audio_url=None,
        )
        episode["status"]["whisper"]["state"] = "ready"
        episode["files"]["sources"]["whisper"]["text"] = "whisper.txt"
        episode["status"]["apple_transcript"]["state"] = "ready"
        episode["files"]["sources"]["apple"]["text"] = "apple.txt"
        episode["status"]["compiler"]["state"] = "completed"
        return episode

    def test_summary_ready_notification_follows_durable_ready_transition(self):
        episode = self._ready_for_summary()
        persisted = deepcopy(episode)
        persisted["status"]["summary"]["state"] = "ready"
        calls: list[str] = []

        def durable_update(*_args, **_kwargs):
            calls.append("update")
            return persisted

        def send_after_update(*_args, **_kwargs):
            self.assertEqual(calls, ["update"])

        with (
            patch.object(pipeline, "build_knowledge_note"),
            patch.object(pipeline, "update_episode", side_effect=durable_update),
            patch.object(
                pipeline,
                "canonical_summary_generated_at",
                return_value="2026-08-30T12:34:56+00:00",
            ) as generated_at,
            patch.object(
                pipeline,
                "send_slack_summary_ready_notification",
                side_effect=send_after_update,
            ) as slack,
        ):
            outcome = pipeline.resume_tracked_episode(self.podcast, episode)

        self.assertEqual(outcome["status"], "completed")
        generated_at.assert_called_once_with("episode-123")
        slack.assert_called_once_with(persisted, "2026-08-30T12:34:56+00:00")

    def test_summary_ready_notification_is_silent_without_ready_transition(self):
        episode = self._ready_for_summary()
        episode["status"]["summary"]["state"] = "ready"

        with patch.object(
            pipeline,
            "send_slack_summary_ready_notification",
        ) as slack:
            outcome = pipeline.resume_tracked_episode(self.podcast, episode)

        self.assertEqual(outcome["status"], "skipped")
        self.assertEqual(outcome["reason"], "completed")
        slack.assert_not_called()

    def test_summary_ready_notification_requires_successful_durable_update(self):
        episode = self._ready_for_summary()

        with (
            patch.object(pipeline, "build_knowledge_note"),
            patch.object(pipeline, "update_episode", return_value=None),
            patch.object(
                pipeline,
                "send_slack_summary_ready_notification",
            ) as slack,
        ):
            with self.assertRaisesRegex(RuntimeError, "persist canonical summary state"):
                pipeline.resume_tracked_episode(self.podcast, episode)

        slack.assert_not_called()

    def test_summary_ready_notification_requires_persisted_vault_sync_readiness(self):
        episode = self._ready_for_summary()
        persisted = deepcopy(episode)

        with (
            patch.object(pipeline, "build_knowledge_note"),
            patch.object(pipeline, "update_episode", return_value=persisted),
            patch.object(
                pipeline,
                "send_slack_summary_ready_notification",
            ) as slack,
        ):
            outcome = pipeline.resume_tracked_episode(self.podcast, episode)

        self.assertEqual(outcome["status"], "completed")
        slack.assert_not_called()

    def test_summary_ready_notification_is_not_sent_after_knowledge_failure(self):
        episode = self._ready_for_summary()

        with (
            patch.object(
                pipeline,
                "build_knowledge_note",
                side_effect=RuntimeError("knowledge failed"),
            ),
            patch.object(pipeline, "update_episode") as update,
            patch.object(
                pipeline,
                "send_slack_summary_ready_notification",
            ) as slack,
        ):
            with self.assertRaisesRegex(RuntimeError, "knowledge failed"):
                pipeline.resume_tracked_episode(self.podcast, episode)

        update.assert_not_called()
        slack.assert_not_called()

    def test_summary_ready_timestamp_lookup_failure_is_non_fatal_and_secret_safe(self):
        episode = self._ready_for_summary()
        persisted = deepcopy(episode)
        persisted["status"]["summary"]["state"] = "ready"

        with (
            patch.object(pipeline, "build_knowledge_note"),
            patch.object(pipeline, "update_episode", return_value=persisted),
            patch.object(
                pipeline,
                "canonical_summary_generated_at",
                side_effect=RuntimeError("private storage error"),
            ),
            patch.object(pipeline, "emit_event") as event,
            patch.object(
                pipeline,
                "send_slack_summary_ready_notification",
            ) as slack,
        ):
            outcome = pipeline.resume_tracked_episode(self.podcast, episode)

        self.assertEqual(outcome["status"], "completed")
        self.assertEqual(event.call_args.args[0], "summary_ready_slack_failed")
        self.assertEqual(event.call_args.kwargs["episode_key"], "episode-123")
        self.assertEqual(event.call_args.kwargs["error_type"], "RuntimeError")
        self.assertNotIn("private storage error", str(event.call_args))
        slack.assert_not_called()

    def test_human_review_notification_is_structured_json(self):
        episode = {
            "podcast": "Example Nutrition Podcast",
            "title": "Example Episode",
            "episode_key": "episode-123",
        }

        with (
            patch.object(
                pipeline,
                "PODCAST_REVIEW_URL",
                "https://review.example.test",
            ),
            patch("builtins.print") as output,
        ):
            pipeline._log_human_review_required(
                episode,
                3,
            )

        output.assert_called_once()

        payload = json.loads(
            output.call_args.args[0]
        )

        self.assertEqual(
            payload,
            {
                "severity": "NOTICE",
                "message": "Podcast human review required",
                "event": "human_review_required",
                "podcast": "Example Nutrition Podcast",
                "title": "Example Episode",
                "episode_key": "episode-123",
                "pending_count": 3,
                "review_url": "https://review.example.test",
            },
        )

    def test_refreshed_notification_is_structured_and_excludes_transcript_text(self):
        episode = {
            "podcast": "Example Nutrition Podcast",
            "title": "Example Episode",
            "episode_key": "episode-123",
        }
        previous_review = [
            {
                "id": 1,
                "apple_text": "private old Apple transcript",
                "whisper_text": "private old Whisper transcript",
            }
        ]
        review = [
            {
                "id": 1,
                "apple_text": "private new Apple transcript",
                "whisper_text": "private new Whisper transcript",
            }
        ]

        with (
            patch.object(
                pipeline,
                "PODCAST_REVIEW_URL",
                "https://review.example.test",
            ),
            patch("builtins.print") as output,
        ):
            pipeline._log_human_review_refreshed(
                episode,
                previous_review,
                review,
            )

        payload = json.loads(output.call_args.args[0])

        self.assertEqual(payload["event"], "human_review_refreshed")
        self.assertEqual(payload["severity"], "NOTICE")
        self.assertEqual(payload["previous_pending_count"], 1)
        self.assertEqual(payload["pending_count"], 1)
        self.assertTrue(payload["previous_queue_fingerprint"].startswith("sha256:"))
        self.assertTrue(payload["queue_fingerprint"].startswith("sha256:"))
        self.assertNotIn("private old Apple transcript", output.call_args.args[0])
        self.assertNotIn("private new Whisper transcript", output.call_args.args[0])

    def test_notification_emits_only_on_first_review_required_transition(self):
        rss_episode = {
            "podcast": "Example Nutrition Podcast",
            "feed_url": "https://example.test/feed.xml",
            "guid": "guid-123",
            "title": "Example Episode",
        }

        compiled = {
            "transcript": "compiled/transcript.txt",
            "report": "compiled/report.json",
            "review_required": 2,
            "review": [
                {"id": 1},
                {"id": 2},
            ],
        }

        for previous_state, expected_calls in (
            ("ready", 1),
            ("review_required", 0),
        ):
            with self.subTest(
                previous_state=previous_state
            ):
                episode = {
                    "id": "episode-123",
                    "episode_key": "episode-123",
                    "podcast": "Example Nutrition Podcast",
                    "title": "Example Episode",
                    "status": {
                        "compiler": {
                            "state": previous_state,
                        },
                    },
                    "files": {
                        "sources": {
                            "apple": {
                                "text": "apple/transcript.txt",
                            },
                            "whisper": {
                                "text": "whisper/transcript.txt",
                            },
                        },
                    },
                }

                updated_episode = {
                    **episode,
                    "status": {
                        **episode["status"],
                        "compiler": {
                            "state": "review_required",
                            "review_required": 2,
                        },
                    },
                }

                with (
                    patch.object(
                        pipeline,
                        "_rss_episode",
                        return_value=rss_episode,
                    ),
                    patch.object(
                        pipeline,
                        "get_episode_by_key",
                        return_value=episode,
                    ),
                    patch.object(
                        pipeline,
                        "merge_episode_metadata",
                        return_value=episode,
                    ),
                    patch.object(
                        pipeline,
                        "ensure_v3_record",
                        return_value=episode,
                    ),
                    patch.object(
                        pipeline,
                        "_transcribe_if_needed",
                        return_value=episode,
                    ),
                    patch.object(
                        pipeline,
                        "load_review_record",
                        return_value={
                            "episode_key": "episode-123",
                            "human_review": [
                                {"id": 1},
                                {"id": 2},
                            ],
                            "human_decisions": [],
                        },
                    ),
                    patch.object(
                        pipeline,
                        "review_record_is_current",
                        return_value=True,
                    ),
                    patch.object(
                        pipeline,
                        "compile_episode_sources",
                        return_value=compiled,
                    ) as compile_sources,
                    patch.object(
                        pipeline,
                        "update_episode",
                        return_value=updated_episode,
                    ),
                    patch.object(
                        pipeline,
                        "_log_human_review_required",
                    ) as notification,
                    patch.object(
                        pipeline,
                        "send_slack_review_notification",
                    ) as slack_notification,
                ):
                    result = pipeline.process_episode(
                        {
                            "name": "Example Nutrition Podcast",
                        }
                    )

                self.assertEqual(
                    result["status"],
                    "review_required",
                )

                self.assertEqual(
                    notification.call_count,
                    expected_calls,
                )

                self.assertEqual(
                    compile_sources.call_count,
                    1 if previous_state == "ready" else 0,
                )

                self.assertEqual(
                    slack_notification.call_count,
                    expected_calls,
                )


if __name__ == "__main__":
    unittest.main()
