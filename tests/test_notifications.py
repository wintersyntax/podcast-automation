import json
import unittest
from unittest.mock import Mock, patch

import requests

from podcast_engine.notifications import (
    build_slack_review_payload,
    build_slack_review_refreshed_payload,
    build_slack_summary_ready_payload,
    send_slack_review_notification,
    send_slack_review_refreshed_notification,
    send_slack_summary_ready_notification,
)


class NotificationTests(unittest.TestCase):
    def setUp(self):
        self.episode = {
            "episode_key": "episode-123",
            "podcast": "Example Nutrition Podcast",
            "title": "Example Episode",
        }

        self.review_url = (
            "https://review.example.test"
        )

    def test_slack_review_payload_contains_expected_content(self):
        payload = build_slack_review_payload(
            self.episode,
            3,
            self.review_url,
        )

        self.assertEqual(
            payload["text"],
            (
                "Podcast ready for review: "
                "Example Nutrition Podcast — Example Episode"
            ),
        )

        button = (
            payload["blocks"][-1]
            ["elements"][0]
        )

        self.assertEqual(
            button["action_id"],
            "open_human_review",
        )

        self.assertEqual(
            button["url"],
            self.review_url,
        )

        serialized = json.dumps(payload)

        self.assertIn(
            "3 transcript differences",
            serialized,
        )

    def test_missing_slack_webhook_is_a_no_op(self):
        with (
            patch.dict(
                "os.environ",
                {
                    "PODCAST_SLACK_WEBHOOK_URL": "",
                },
                clear=False,
            ),
            patch(
                "podcast_engine.notifications.requests.post"
            ) as post,
        ):
            result = send_slack_review_notification(
                self.episode,
                3,
                self.review_url,
            )

        self.assertFalse(result)
        post.assert_not_called()

    def test_refreshed_slack_payload_distinguishes_a_queue_update(self):
        payload = build_slack_review_refreshed_payload(
            self.episode,
            36,
            21,
            self.review_url,
        )

        serialized = json.dumps(payload)
        self.assertIn("Podcast review updated", serialized)
        self.assertIn("Review queue refreshed", serialized)
        self.assertIn("36 \\u2192 21 transcript differences", serialized)
        self.assertEqual(
            payload["blocks"][-1]["elements"][0]["url"],
            self.review_url,
        )

    def test_slack_notification_posts_block_kit_payload(self):
        response = Mock()
        response.raise_for_status.return_value = None

        with patch(
            "podcast_engine.notifications.requests.post",
            return_value=response,
        ) as post:
            result = send_slack_review_notification(
                self.episode,
                3,
                self.review_url,
                webhook_url=(
                    "https://hooks.slack.test/services/test"
                ),
            )

        self.assertTrue(result)

        post.assert_called_once()

        call = post.call_args

        self.assertEqual(
            call.kwargs["timeout"],
            10,
        )

        self.assertEqual(
            call.kwargs["json"]["blocks"][-1]
            ["elements"][0]["action_id"],
            "open_human_review",
        )

    def test_slack_delivery_failure_does_not_break_worker(self):
        error = requests.ConnectionError(
            "secret provider details"
        )

        with (
            patch(
                "podcast_engine.notifications.requests.post",
                side_effect=error,
            ),
            patch("builtins.print") as output,
        ):
            result = send_slack_review_notification(
                self.episode,
                3,
                self.review_url,
                webhook_url=(
                    "https://hooks.slack.test/services/test"
                ),
            )

        self.assertFalse(result)

        warning = json.loads(
            output.call_args.args[0]
        )

        self.assertEqual(
            warning["event"],
            "human_review_slack_failed",
        )

        self.assertEqual(
            warning["episode_key"],
            "episode-123",
        )

        self.assertNotIn(
            "secret provider details",
            output.call_args.args[0],
        )

    def test_refreshed_slack_delivery_failure_does_not_break_worker(self):
        with patch(
            "podcast_engine.notifications.requests.post",
            side_effect=requests.ConnectionError("secret provider details"),
        ):
            result = send_slack_review_refreshed_notification(
                self.episode,
                36,
                21,
                self.review_url,
                webhook_url="https://hooks.slack.test/services/test",
            )

        self.assertFalse(result)

    def test_summary_ready_payload_is_content_minimal_and_vault_sync_specific(self):
        episode = {
            **self.episode,
            "summary_body": "private summary body",
            "transcript": "private transcript body",
        }
        payload = build_slack_summary_ready_payload(
            episode,
            "2026-08-30T21:41:19.510315+00:00",
        )
        serialized = json.dumps(payload)

        self.assertEqual(
            payload["blocks"][0]["text"]["text"],
            "✅ Summary ready",
        )
        self.assertIn("Example Nutrition Podcast", serialized)
        self.assertIn("Example Episode", serialized)
        self.assertIn("30 Aug 2026, 23:41", serialized)
        self.assertNotIn("2026-08-30T21:41:19.510315+00:00", serialized)
        self.assertIn(
            "Ready \\u2014 will sync to Obsidian on the next Vault Sync.",
            serialized,
        )
        self.assertNotIn("private summary body", serialized)
        self.assertNotIn("private transcript body", serialized)

    def test_summary_ready_timestamp_uses_oslo_timezone_and_dst(self):
        cases = (
            ("2026-01-15T12:30:00+00:00", "15 Jan 2026, 13:30"),
            ("2026-07-15T12:30:00+00:00", "15 Jul 2026, 14:30"),
        )

        for generated_at, expected in cases:
            with self.subTest(generated_at=generated_at):
                payload = build_slack_summary_ready_payload(
                    self.episode,
                    generated_at,
                )

                self.assertEqual(
                    payload["blocks"][2]["fields"][0]["text"],
                    f"*Generated*\n{expected}",
                )

    def test_invalid_summary_ready_timestamp_stays_non_fatal_and_safe(self):
        payload = build_slack_summary_ready_payload(
            self.episode,
            "not-a-canonical-timestamp",
        )

        self.assertEqual(
            payload["blocks"][2]["fields"][0]["text"],
            "*Generated*\nUnavailable",
        )

    def test_missing_slack_webhook_is_a_no_op_for_summary_ready(self):
        with (
            patch.dict("os.environ", {"PODCAST_SLACK_WEBHOOK_URL": ""}, clear=False),
            patch("podcast_engine.notifications.requests.post") as post,
        ):
            result = send_slack_summary_ready_notification(
                self.episode,
                "2026-08-30T12:34:56+00:00",
            )

        self.assertFalse(result)
        post.assert_not_called()

    def test_summary_ready_slack_failure_is_non_fatal_and_secret_safe(self):
        with (
            patch(
                "podcast_engine.notifications.requests.post",
                side_effect=requests.ConnectionError("secret provider details"),
            ),
            patch("builtins.print") as output,
        ):
            result = send_slack_summary_ready_notification(
                self.episode,
                "2026-08-30T12:34:56+00:00",
                webhook_url="https://hooks.slack.test/services/test",
            )

        self.assertFalse(result)
        warning = json.loads(output.call_args.args[0])
        self.assertEqual(warning["event"], "summary_ready_slack_failed")
        self.assertEqual(warning["episode_key"], "episode-123")
        self.assertNotIn("secret provider details", output.call_args.args[0])

    def test_summary_ready_unexpected_delivery_failure_is_non_fatal(self):
        with patch(
            "podcast_engine.notifications.requests.post",
            side_effect=ValueError("private delivery detail"),
        ):
            result = send_slack_summary_ready_notification(
                self.episode,
                "2026-08-30T12:34:56+00:00",
                webhook_url="https://hooks.slack.test/services/test",
            )

        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
