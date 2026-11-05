import hashlib
import hmac
import os
import unittest
from unittest.mock import patch

from flask import Flask, request

from slack_ack.main import _valid_slack_signature, slack_interactions


SIGNING_SECRET = "test-slack-signing-secret"


def slack_signature(body: bytes, timestamp: str) -> str:
    """Build a Slack v0 signature for a deterministic test request."""

    base = b"v0:" + timestamp.encode("utf-8") + b":" + body
    digest = hmac.new(
        SIGNING_SECRET.encode("utf-8"),
        base,
        hashlib.sha256,
    ).hexdigest()
    return f"v0={digest}"


class SlackAckFunctionTests(unittest.TestCase):
    def setUp(self):
        self.app = Flask(__name__)

    def test_valid_signature_is_accepted(self):
        body = b"payload=%7B%22type%22%3A%22block_actions%22%7D"
        timestamp = "1000"

        self.assertTrue(
            _valid_slack_signature(
                body,
                timestamp,
                slack_signature(body, timestamp),
                SIGNING_SECRET,
                now=1000,
            )
        )

    def test_stale_signature_is_rejected(self):
        body = b"payload=test"
        timestamp = "1000"

        self.assertFalse(
            _valid_slack_signature(
                body,
                timestamp,
                slack_signature(body, timestamp),
                SIGNING_SECRET,
                now=1301,
            )
        )

    def test_invalid_signature_is_rejected(self):
        self.assertFalse(
            _valid_slack_signature(
                b"payload=test",
                "1000",
                "v0=invalid",
                SIGNING_SECRET,
                now=1000,
            )
        )

    def test_non_integer_timestamp_is_rejected(self):
        self.assertFalse(
            _valid_slack_signature(
                b"payload=test",
                "not-a-number",
                "v0=irrelevant",
                SIGNING_SECRET,
                now=1000,
            )
        )

    def test_valid_request_returns_empty_ack(self):
        body = b"payload=%7B%22type%22%3A%22block_actions%22%7D"
        timestamp = "1000"

        with patch.dict(
            os.environ,
            {"PODCAST_SLACK_SIGNING_SECRET": SIGNING_SECRET},
            clear=False,
        ):
            with self.app.test_request_context(
                "/",
                method="POST",
                data=body,
                headers={
                    "X-Slack-Request-Timestamp": timestamp,
                    "X-Slack-Signature": slack_signature(body, timestamp),
                },
            ):
                with patch("slack_ack.main.time.time", return_value=1000):
                    response_body, status_code = slack_interactions(request)

        self.assertEqual((response_body, status_code), ("", 200))

    def test_missing_secret_returns_service_unavailable(self):
        with patch.dict(
            os.environ,
            {"PODCAST_SLACK_SIGNING_SECRET": ""},
            clear=False,
        ):
            with self.app.test_request_context("/", method="POST"):
                response_body, status_code = slack_interactions(request)

        self.assertEqual((response_body, status_code), ("", 503))

    def test_invalid_request_returns_unauthorized(self):
        with patch.dict(
            os.environ,
            {"PODCAST_SLACK_SIGNING_SECRET": SIGNING_SECRET},
            clear=False,
        ):
            with self.app.test_request_context(
                "/",
                method="POST",
                data=b"payload=test",
                headers={
                    "X-Slack-Request-Timestamp": "1000",
                    "X-Slack-Signature": "v0=invalid",
                },
            ):
                response_body, status_code = slack_interactions(request)

        self.assertEqual((response_body, status_code), ("", 401))
