import unittest
from unittest.mock import patch

from podcast_engine.apple_ingest import gcs_apple_transcript_finalize


class AppleIngestResumeTests(unittest.TestCase):
    @patch("podcast_engine.apple_ingest.emit_event")
    @patch("podcast_engine.apple_ingest.request_worker_run")
    @patch("podcast_engine.apple_ingest.ingest_apple_transcript")
    def test_new_ingest_requests_exactly_one_worker_run(
        self, ingest, request_worker, emit
    ):
        ingest.return_value = {
            "status": "ingested",
            "episode_key": "abc123",
        }
        request_worker.return_value = {
            "operation": "operations/worker-1",
            "attempts": 1,
        }

        result = gcs_apple_transcript_finalize(
            {"bucket": "bucket", "name": "incoming/apple/abc123/apple-transcript.txt"}
        )

        request_worker.assert_called_once_with(
            requested_by="apple_ingest",
            episode_key="abc123",
        )
        self.assertEqual(result["status"], "ingested")
        self.assertEqual(result["worker_resume"]["status"], "requested")
        self.assertEqual(
            result["worker_resume"]["operation"],
            "operations/worker-1",
        )
        self.assertEqual(
            emit.call_args.args[0],
            "apple_ingest_worker_resume_requested",
        )

    @patch("podcast_engine.apple_ingest.request_worker_run")
    @patch("podcast_engine.apple_ingest.ingest_apple_transcript")
    def test_already_ingested_duplicate_does_not_request_worker(
        self, ingest, request_worker
    ):
        ingest.return_value = {
            "status": "already_ingested",
            "episode_key": "abc123",
        }

        result = gcs_apple_transcript_finalize(
            {"bucket": "bucket", "name": "incoming/apple/abc123/apple-transcript.txt"}
        )

        request_worker.assert_not_called()
        self.assertEqual(result["status"], "already_ingested")
        self.assertNotIn("worker_resume", result)

    @patch("podcast_engine.apple_ingest.request_worker_run")
    @patch("podcast_engine.apple_ingest.ingest_apple_transcript")
    def test_ignored_event_does_not_request_worker(
        self, ingest, request_worker
    ):
        ingest.return_value = {
            "status": "ignored",
            "reason": "not_an_apple_completion_object",
        }

        result = gcs_apple_transcript_finalize(
            {"bucket": "bucket", "name": "unrelated/object.txt"}
        )

        request_worker.assert_not_called()
        self.assertEqual(result["status"], "ignored")

    @patch("podcast_engine.apple_ingest.emit_event")
    @patch("podcast_engine.apple_ingest.request_worker_run")
    @patch("podcast_engine.apple_ingest.ingest_apple_transcript")
    def test_resume_failure_does_not_turn_successful_ingest_into_failure(
        self, ingest, request_worker, emit
    ):
        ingest.return_value = {
            "status": "ingested",
            "episode_key": "abc123",
        }
        request_worker.side_effect = RuntimeError("simulated failure")

        result = gcs_apple_transcript_finalize(
            {"bucket": "bucket", "name": "incoming/apple/abc123/apple-transcript.txt"}
        )

        self.assertEqual(result["status"], "ingested")
        self.assertEqual(result["worker_resume"]["status"], "failed")
        self.assertEqual(
            emit.call_args.args[0],
            "apple_ingest_worker_resume_failed",
        )
