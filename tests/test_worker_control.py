import unittest
from unittest.mock import patch

import requests

from podcast_engine.worker_control import (
    WorkerRunRequestError,
    request_worker_run,
)


class FakeResponse:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}

    def json(self):
        return self._payload


class WorkerControlTests(unittest.TestCase):
    @patch("podcast_engine.worker_control.AuthorizedSession")
    @patch("podcast_engine.worker_control.google_auth_default")
    def test_success_requests_worker_once(self, credentials, session_class):
        credentials.return_value = (object(), None)
        session_class.return_value.post.return_value = FakeResponse(
            200, {"name": "operations/worker-1"}
        )

        result = request_worker_run(
            requested_by="apple_ingest",
            episode_key="episode-1",
        )

        self.assertEqual(result["operation"], "operations/worker-1")
        self.assertEqual(result["attempts"], 1)
        self.assertEqual(result["requested_by"], "apple_ingest")
        self.assertEqual(result["episode_key"], "episode-1")
        session_class.return_value.post.assert_called_once()
        self.assertEqual(session_class.return_value.post.call_args.kwargs["json"], {})

    @patch("podcast_engine.worker_control.AuthorizedSession")
    @patch("podcast_engine.worker_control.google_auth_default")
    def test_recompile_correlation_uses_only_exact_non_secret_overrides(
        self, credentials, session_class
    ):
        credentials.return_value = (object(), None)
        session_class.return_value.post.return_value = FakeResponse(200, {"name": "operations/worker-1"})

        request_worker_run(
            requested_by="reviewer@example.com",
            episode_key="episode-1",
            correlation={
                "request_id": "request-123",
                "review_generation": "sha256:review",
                "episode_key": "episode-1",
            },
        )

        body = session_class.return_value.post.call_args.kwargs["json"]
        env = body["overrides"]["containerOverrides"][0]["env"]
        self.assertEqual(
            env,
            [
                {"name": "PODCAST_RECOMPILE_REQUEST_ID", "value": "request-123"},
                {"name": "PODCAST_REVIEW_GENERATION", "value": "sha256:review"},
                {"name": "PODCAST_RECOMPILE_EPISODE_KEY", "value": "episode-1"},
            ],
        )
        self.assertNotIn("reviewer@example.com", str(body))

    @patch("podcast_engine.worker_control.AuthorizedSession")
    @patch("podcast_engine.worker_control.google_auth_default")
    def test_transient_503_is_ambiguous_and_is_not_retried(
        self, credentials, session_class
    ):
        credentials.return_value = (object(), None)
        session_class.return_value.post.return_value = FakeResponse(503)

        with self.assertRaisesRegex(WorkerRunRequestError, "outcome is unknown") as raised:
            request_worker_run(requested_by="apple_ingest", episode_key="episode-1")

        self.assertFalse(raised.exception.definitely_not_accepted)
        session_class.return_value.post.assert_called_once()

    @patch("podcast_engine.worker_control.time.sleep")
    @patch("podcast_engine.worker_control.AuthorizedSession")
    @patch("podcast_engine.worker_control.google_auth_default")
    def test_429_is_retried(self, credentials, session_class, sleep):
        credentials.return_value = (object(), None)
        session_class.return_value.post.side_effect = [
            FakeResponse(429),
            FakeResponse(200, {"name": "operations/worker-3"}),
        ]

        result = request_worker_run(requested_by="test")

        self.assertEqual(result["attempts"], 2)
        sleep.assert_called_once()

    @patch("podcast_engine.worker_control.time.sleep")
    @patch("podcast_engine.worker_control.AuthorizedSession")
    @patch("podcast_engine.worker_control.google_auth_default")
    def test_transport_failure_is_ambiguous_and_is_not_retried(
        self, credentials, session_class, sleep
    ):
        credentials.return_value = (object(), None)
        session_class.return_value.post.side_effect = requests.ConnectionError("temporary")

        with self.assertRaisesRegex(WorkerRunRequestError, "outcome is unknown") as raised:
            request_worker_run(requested_by="test")

        self.assertFalse(raised.exception.definitely_not_accepted)
        session_class.return_value.post.assert_called_once()
        sleep.assert_not_called()

    @patch("podcast_engine.worker_control.time.sleep")
    @patch("podcast_engine.worker_control.AuthorizedSession")
    @patch("podcast_engine.worker_control.google_auth_default")
    def test_ordinary_4xx_is_not_retried(
        self, credentials, session_class, sleep
    ):
        credentials.return_value = (object(), None)
        session_class.return_value.post.return_value = FakeResponse(403)

        with self.assertRaisesRegex(WorkerRunRequestError, "HTTP 403"):
            request_worker_run(requested_by="test")

        session_class.return_value.post.assert_called_once()
        sleep.assert_not_called()

    @patch("podcast_engine.worker_control.time.sleep")
    @patch("podcast_engine.worker_control.AuthorizedSession")
    @patch("podcast_engine.worker_control.google_auth_default")
    def test_5xx_is_ambiguous_and_is_not_retried(
        self, credentials, session_class, sleep
    ):
        credentials.return_value = (object(), None)
        session_class.return_value.post.return_value = FakeResponse(503)

        with self.assertRaisesRegex(WorkerRunRequestError, "outcome is unknown") as raised:
            request_worker_run(requested_by="test")

        self.assertFalse(raised.exception.definitely_not_accepted)
        session_class.return_value.post.assert_called_once()
        sleep.assert_not_called()
