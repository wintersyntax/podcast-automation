import copy
import json
import unittest
from unittest.mock import patch

from google.api_core.exceptions import PreconditionFailed

from podcast_engine.human_review import (
    RecompileAuditPersistenceError,
    complete_recompile_request,
    load_review_record_with_generation,
    request_worker_recompile,
    save_review_record,
)
from podcast_engine.worker_control import WorkerRunRequestError


class HumanReviewWorkerControlTests(unittest.TestCase):
    def _store(self, record):
        state = {"record": copy.deepcopy(record), "generation": 1}

        def load(_episode_key):
            return copy.deepcopy(state["record"]), state["generation"]

        def save(_episode_key, record, *, if_generation_match=None):
            self.assertEqual(if_generation_match, state["generation"])
            state["record"] = copy.deepcopy(record)
            state["generation"] += 1
            return record

        return state, load, save

    def test_snapshot_read_retries_instead_of_pairing_stale_content_with_new_generation(self):
        state = {"generation": 1, "content": '{"revision": "old"}', "reads": 0}
        case = self

        class Blob:
            generation = 1

            def exists(self):
                return True

            def reload(self):
                self.generation = state["generation"]

            def download_as_text(self, *, encoding, if_generation_match):
                state["reads"] += 1
                if state["reads"] == 1:
                    state.update(generation=2, content='{"revision": "new"}')
                    raise PreconditionFailed("changed during read")
                case.assertEqual(if_generation_match, state["generation"])
                return state["content"]

            def upload_from_string(self, value, *, content_type, if_generation_match):
                case.assertEqual(if_generation_match, state["generation"])
                state["content"] = value
                state["generation"] += 1

        blob = Blob()

        class Bucket:
            def blob(self, _name):
                return blob

        with patch("podcast_engine.human_review.get_bucket", return_value=Bucket()):
            record, generation = load_review_record_with_generation("episode-1")
            record["saved"] = True
            save_review_record("episode-1", record, if_generation_match=generation)

        self.assertEqual(record["revision"], "new")
        self.assertEqual(generation, 2)
        self.assertEqual(state["reads"], 2)
        self.assertEqual(json.loads(state["content"]), {"revision": "new", "saved": True})

    @patch("podcast_engine.human_review.request_worker_run")
    def test_recompile_uses_shared_worker_control_once_and_persists_it(
        self, request_worker
    ):
        state, load, save = self._store({
            "episode_key": "episode-1",
            "input_fingerprint": "sha256:input-a",
            "human_review": [],
        })
        request_worker.return_value = {
            "operation": "operations/worker-1",
            "attempts": 1,
        }

        with patch("podcast_engine.human_review.load_review_record_with_generation", side_effect=load), patch("podcast_engine.human_review.save_review_record", side_effect=save):
            result = request_worker_recompile("episode-1", requested_by="test-review")
            repeated = request_worker_recompile("episode-1", requested_by="second-tab")

        request_worker.assert_called_once()
        worker_kwargs = request_worker.call_args.kwargs
        self.assertEqual(worker_kwargs["requested_by"], "test-review")
        self.assertEqual(worker_kwargs["episode_key"], "episode-1")
        self.assertEqual(worker_kwargs["correlation"]["episode_key"], "episode-1")
        self.assertEqual(
            worker_kwargs["correlation"]["review_generation"],
            saved_generation := result["review_generation_fingerprint"],
        )
        self.assertEqual(worker_kwargs["correlation"]["request_id"], result["request_id"])
        self.assertEqual(result["operation"], "operations/worker-1")
        self.assertEqual(result["attempts"], 1)
        self.assertEqual(result["requested_by"], "test-review")
        self.assertTrue(repeated["idempotent"])
        self.assertEqual(repeated["status"], "started")
        saved = state["record"]
        self.assertEqual(len(saved["recompile_requests"]), 1)
        self.assertEqual(
            saved["recompile_requests"][0]["operation"],
            "operations/worker-1",
        )

    @patch("podcast_engine.human_review.request_worker_run")
    def test_starting_reservation_blocks_near_simultaneous_second_launch(self, request_worker):
        state, load, save = self._store({"episode_key": "episode-1", "human_review": []})

        def start_worker(**_kwargs):
            second = request_worker_recompile("episode-1", requested_by="second-tab")
            self.assertTrue(second["idempotent"])
            self.assertEqual(second["status"], "starting")
            return {"operation": "operations/worker-1", "attempts": 1}

        request_worker.side_effect = start_worker
        with patch("podcast_engine.human_review.load_review_record_with_generation", side_effect=load), patch("podcast_engine.human_review.save_review_record", side_effect=save):
            request_worker_recompile("episode-1", requested_by="first-tab")
        self.assertEqual(request_worker.call_count, 1)
        self.assertEqual(len(state["record"]["recompile_requests"]), 1)

    @patch("podcast_engine.human_review.request_worker_run")
    def test_recompile_rejects_pending_and_failed_launch_is_retryable(self, request_worker):
        pending, load, save = self._store({"episode_key": "episode-1", "human_review": [{"id": 1}]})
        with patch("podcast_engine.human_review.load_review_record_with_generation", side_effect=load), patch("podcast_engine.human_review.save_review_record", side_effect=save):
            with self.assertRaisesRegex(ValueError, "All human-review"):
                request_worker_recompile("episode-1", requested_by="test")
        self.assertEqual(pending["record"]["human_review"], [{"id": 1}])

        state, load, save = self._store({"episode_key": "episode-1", "human_review": []})
        request_worker.side_effect = [
            WorkerRunRequestError("request rejected", definitely_not_accepted=True),
            {"operation": "operations/retry", "attempts": 1},
        ]
        with patch("podcast_engine.human_review.load_review_record_with_generation", side_effect=load), patch("podcast_engine.human_review.save_review_record", side_effect=save):
            with self.assertRaisesRegex(WorkerRunRequestError, "request rejected"):
                request_worker_recompile("episode-1", requested_by="test")
            retried = request_worker_recompile("episode-1", requested_by="test")
        self.assertEqual(request_worker.call_count, 2)
        self.assertEqual(retried["status"], "started")
        self.assertEqual(state["record"]["recompile_requests"][0]["status"], "failed")

    @patch("podcast_engine.human_review.request_worker_run")
    def test_unknown_or_crash_window_reservation_is_not_retryable(self, request_worker):
        state, load, save = self._store({"episode_key": "episode-1", "human_review": []})
        request_worker.side_effect = WorkerRunRequestError(
            "transport outcome unknown", definitely_not_accepted=False
        )
        with patch("podcast_engine.human_review.load_review_record_with_generation", side_effect=load), patch("podcast_engine.human_review.save_review_record", side_effect=save):
            with self.assertRaisesRegex(WorkerRunRequestError, "outcome unknown"):
                request_worker_recompile("episode-1", requested_by="first-tab")
            repeated = request_worker_recompile("episode-1", requested_by="second-tab")
        request_worker.assert_called_once()
        self.assertTrue(repeated["idempotent"])
        self.assertEqual(repeated["status"], "unknown")

        state, load, save = self._store({
            "episode_key": "episode-1",
            "human_review": [],
            "human_review_generation_fingerprint": "sha256:complete",
            "recompile_requests": [{
                "request_id": "crash-window",
                "review_generation_fingerprint": "sha256:complete",
                "status": "starting",
            }],
        })
        with patch("podcast_engine.human_review.load_review_record_with_generation", side_effect=load), patch("podcast_engine.human_review.save_review_record", side_effect=save):
            crashed = request_worker_recompile("episode-1", requested_by="reloaded-tab")
        self.assertTrue(crashed["idempotent"])
        self.assertEqual(crashed["status"], "starting")

    @patch("podcast_engine.human_review.request_worker_run")
    def test_new_review_generation_can_request_recompile_after_previous_one(self, request_worker):
        state, load, save = self._store({"episode_key": "episode-1", "human_review": [], "human_review_generation_fingerprint": "sha256:first"})
        request_worker.side_effect = [
            {"operation": "operations/first", "attempts": 1},
            {"operation": "operations/second", "attempts": 1},
        ]
        with patch("podcast_engine.human_review.load_review_record_with_generation", side_effect=load), patch("podcast_engine.human_review.save_review_record", side_effect=save):
            request_worker_recompile("episode-1", requested_by="test")
            state["record"]["human_review_generation_fingerprint"] = "sha256:second"
            request_worker_recompile("episode-1", requested_by="test")
        self.assertEqual(request_worker.call_count, 2)

    def test_worker_completion_after_later_generation_preserves_rebuild_result_link(self):
        state, load, save = self._store({
            "episode_key": "episode-1",
            "human_review": [],
            "human_review_generation_fingerprint": "sha256:later-c",
            "recompile_requests": [{
                "request_id": "request-123",
                "review_generation_fingerprint": "sha256:original",
                "result_review_generation_fingerprint": "sha256:rebuilt",
                "status": "started",
            }],
        })

        with patch("podcast_engine.human_review.load_review_record_with_generation", side_effect=load), patch(
            "podcast_engine.human_review.save_review_record", side_effect=save
        ):
            completed = complete_recompile_request(
                "episode-1",
                request_id="request-123",
                review_generation="sha256:original",
            )

        self.assertEqual(completed["status"], "completed")
        request = state["record"]["recompile_requests"][0]
        self.assertEqual(request["status"], "completed")
        self.assertIn("completed_at", request)
        self.assertEqual(
            request["result_review_generation_fingerprint"], "sha256:rebuilt"
        )

    def test_terminal_completion_reports_conditional_write_exhaustion(self):
        state, load, _save = self._store({
            "episode_key": "episode-1",
            "human_review": [],
            "human_review_generation_fingerprint": "sha256:rebuilt",
            "recompile_requests": [{
                "request_id": "request-123",
                "review_generation_fingerprint": "sha256:original",
                "result_review_generation_fingerprint": "sha256:rebuilt",
                "status": "started",
            }],
        })

        with patch("podcast_engine.human_review.load_review_record_with_generation", side_effect=load), patch(
            "podcast_engine.human_review.save_review_record",
            side_effect=PreconditionFailed("concurrent review update"),
        ):
            with self.assertRaises(RecompileAuditPersistenceError):
                complete_recompile_request(
                    "episode-1",
                    request_id="request-123",
                    review_generation="sha256:original",
                )

        self.assertEqual(state["record"]["recompile_requests"][0]["status"], "started")
