import os
import unittest
from unittest.mock import patch

import worker
from podcast_engine.human_review import RecompileAuditPersistenceError


class WorkerRecompileLifecycleTests(unittest.TestCase):
    correlation = {
        "PODCAST_RECOMPILE_REQUEST_ID": "request-123",
        "PODCAST_REVIEW_GENERATION": "sha256:original-review",
        "PODCAST_RECOMPILE_EPISODE_KEY": "episode-1",
    }

    def test_successful_target_pipeline_persists_original_request_completion(self):
        outcomes = [{"status": "completed", "episode": {"episode_key": "episode-1"}}]
        with patch.dict(os.environ, self.correlation, clear=False), patch(
            "worker.run_pipeline", return_value=outcomes
        ), patch("worker.complete_recompile_request") as complete:
            result = worker.main()

        self.assertEqual(result, outcomes)
        complete.assert_called_once_with(
            "episode-1",
            request_id="request-123",
            review_generation="sha256:original-review",
        )

    def test_summary_review_activation_wait_persists_recompile_completion(self):
        outcomes = [
            {
                "status": "waiting_for_summary_review_activation",
                "reason": "reviewer_not_activated",
                "episode": {"episode_key": "episode-1"},
            }
        ]
        with patch.dict(os.environ, self.correlation, clear=False), patch(
            "worker.run_pipeline", return_value=outcomes
        ), patch("worker.complete_recompile_request") as complete:
            result = worker.main()

        self.assertEqual(result, outcomes)
        complete.assert_called_once_with(
            "episode-1",
            request_id="request-123",
            review_generation="sha256:original-review",
        )

    def test_correlated_review_required_target_fails_without_terminal_write(self):
        outcomes = [{"status": "review_required", "episode": {"episode_key": "episode-1"}}]
        with patch.dict(os.environ, self.correlation, clear=False), patch(
            "worker.run_pipeline", return_value=outcomes
        ), patch("worker.complete_recompile_request") as complete:
            with self.assertRaisesRegex(RuntimeError, "did not complete the full pipeline"):
                worker.main()

        complete.assert_not_called()

    def test_correlated_target_missing_from_outcomes_fails_without_terminal_write(self):
        outcomes = [{"status": "completed", "episode": {"episode_key": "other-episode"}}]
        with patch.dict(os.environ, self.correlation, clear=False), patch(
            "worker.run_pipeline", return_value=outcomes
        ), patch("worker.complete_recompile_request") as complete:
            with self.assertRaisesRegex(RuntimeError, "not found exactly once"):
                worker.main()

        complete.assert_not_called()

    def test_correlated_non_completed_target_fails_without_terminal_write(self):
        outcomes = [{"status": "blocked", "episode": {"episode_key": "episode-1"}}]
        with patch.dict(os.environ, self.correlation, clear=False), patch(
            "worker.run_pipeline", return_value=outcomes
        ), patch("worker.complete_recompile_request") as complete:
            with self.assertRaisesRegex(RuntimeError, "did not complete the full pipeline"):
                worker.main()

        complete.assert_not_called()

    def test_uncorrelated_review_required_outcome_remains_successful(self):
        outcomes = [{"status": "review_required", "episode": {"episode_key": "episode-1"}}]
        with patch.dict(os.environ, {}, clear=True), patch(
            "worker.run_pipeline", return_value=outcomes
        ), patch("worker.complete_recompile_request") as complete:
            result = worker.main()

        self.assertEqual(result, outcomes)
        complete.assert_not_called()

    def test_terminal_audit_persistence_failure_keeps_successful_pipeline_successful(self):
        outcomes = [{"status": "completed", "episode": {"episode_key": "episode-1"}}]
        with patch.dict(os.environ, self.correlation, clear=False), patch(
            "worker.run_pipeline", return_value=outcomes
        ), patch(
            "worker.complete_recompile_request",
            side_effect=RecompileAuditPersistenceError("conditional write exhausted"),
        ) as complete, self.assertLogs("worker", level="WARNING") as logs:
            result = worker.main()

        self.assertEqual(result, outcomes)
        complete.assert_called_once()
        self.assertIn("terminal audit persistence deferred", "\n".join(logs.output))

    def test_pipeline_exception_still_propagates(self):
        with patch.dict(os.environ, self.correlation, clear=False), patch(
            "worker.run_pipeline", side_effect=RuntimeError("pipeline failed")
        ), patch("worker.complete_recompile_request") as complete:
            with self.assertRaisesRegex(RuntimeError, "pipeline failed"):
                worker.main()

        complete.assert_not_called()

    def test_incomplete_correlation_fails_before_running_pipeline(self):
        environment = {**self.correlation, "PODCAST_REVIEW_GENERATION": ""}
        with patch.dict(os.environ, environment, clear=False), patch("worker.run_pipeline") as pipeline:
            with self.assertRaisesRegex(RuntimeError, "correlation environment is incomplete"):
                worker.main()

        pipeline.assert_not_called()
