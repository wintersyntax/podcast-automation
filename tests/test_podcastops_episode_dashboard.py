"""TASK-142: PodcastOps status reporting must be read-only and evidence-bounded."""

import copy
from podcast_engine.vault_sync_receipts import summary_digest
import unittest
from unittest.mock import patch

from google.api_core.exceptions import Forbidden

from podcast_engine.ops_dashboard import episode_status, sorted_episode_statuses
from podcast_engine.review_web import PAGE, create_review_app


def episode(key, compiler="review_required", summary="pending", published="2026-10-08"):
    return {
        "episode_key": key, "podcast": "Iron Culture", "title": "Ep 391",
        "published": published, "status": {
            "apple_transcript": {"state": "ready"},
            "whisper": {"state": "ready"},
            "compiler": {"state": compiler},
            "summary": {"state": summary},
        },
    }


def record(key, pending=1, decided=2, status="started"):
    return {
        "episode_key": key,
        "human_review": [
            {"id": i, "apple_text": "A", "whisper_text": "B",
             "focus": {"scope": "full"}}
            for i in range(1, pending + 1)
        ],
        "human_decisions": [
            {"id": i + 100, "chosen_source": "apple"}
            for i in range(1, decided + 1)
        ],
        "recompile_requests": [{
            "request_id": "one-request",
            "status": status,
            "updated_at": "2026-10-08T23:38:00+00:00",
            "review_generation_fingerprint": "sha256:old",
        }],
    }


class EpisodeStatusProjectionTests(unittest.TestCase):
    def test_compiler_completed_does_not_imply_recompile_completed(self):
        item = episode("ep-391", compiler="completed")
        source = record("ep-391", pending=0, decided=90, status="started")
        original = copy.deepcopy(source)
        output = episode_status(item, source)
        self.assertEqual(output["compiler"], "completed")
        self.assertEqual(output["summary"], "pending")
        self.assertEqual(output["review"], {
            "state": "reviewed", "pending": 0, "decided": 90,
            "groups": {
                "settled": 0, "sample": 0, "click_one": 0,
                "proposal": 0, "full": 0,
            },
        })
        self.assertEqual(output["recompile"]["status"], "started")
        self.assertEqual(source, original, "Status view cannot alter any audit data")

        source["recompile_requests"][-1]["status"] = "completed"
        source["recompile_requests"][-1]["completed_at"] = "2026-10-08T23:37:58Z"
        finished = episode_status(item, source)
        self.assertEqual(finished["recompile"]["status"], "completed")
        self.assertEqual(finished["recompile"]["completed_at"], "2026-10-08T23:37:58Z")

    def test_materiality_group_counts_keep_real_pending_ids(self):
        item = episode("ep-grouped")
        source = record("ep-grouped", pending=3, decided=1)
        groups = {1: {"group": "settled"}, 2: {"group": "proposal"}, 3: {"group": "full"}}
        with patch("podcast_engine.ops_dashboard.materiality_groups", return_value=groups):
            data = episode_status(item, source)
        self.assertEqual(data["review"]["pending"], 3)
        self.assertEqual(data["review"]["decided"], 1)
        self.assertEqual(data["review"]["groups"]["settled"], 1)
        self.assertEqual(data["review"]["groups"]["proposal"], 1)
        self.assertEqual(data["review"]["groups"]["full"], 1)

    def test_missing_and_unreadable_audits_never_claim_zero_cards_or_success(self):
        active = episode("ep-active")
        completed = episode("ep-complete", compiler="completed")
        self.assertEqual(episode_status(active, None)["review"]["state"], "unavailable")
        self.assertIsNone(episode_status(active, None)["review"]["pending"])
        self.assertEqual(episode_status(completed, None)["review"]["state"], "not_recorded")
        broken = episode_status(completed, None, audit_error="unavailable")
        self.assertEqual(broken["review"]["state"], "unavailable")
        self.assertEqual(broken["recompile"]["status"], "unavailable")

    def test_malformed_decisions_and_duplicate_ids_fail_closed(self):
        source = record("ep-1")
        for mutation in ("duplicate", "overlap", "wrong_key", "wrong_status", "wrong_list"):
            with self.subTest(mutation=mutation):
                bad = copy.deepcopy(source)
                if mutation == "duplicate":
                    bad["human_decisions"].append(bad["human_decisions"][0])
                elif mutation == "overlap":
                    bad["human_decisions"][0]["id"] = 1
                elif mutation == "wrong_key":
                    bad["episode_key"] = "another-episode"
                elif mutation == "wrong_status":
                    bad["recompile_requests"][-1]["status"] = "impossible"
                else:
                    bad["human_review"] = {}
                data = episode_status(episode("ep-1"), bad)
                if mutation == "wrong_status":
                    self.assertEqual(data["recompile"]["status"], "unavailable")
                else:
                    self.assertEqual(data["review"]["state"], "unavailable")
                    self.assertIsNone(data["review"]["pending"])

    def test_delivery_is_byte_bound_and_exposes_distinct_completion_times(self):
        e = episode("a" * 24, compiler="completed", summary="ready")
        e["status"]["compiler"]["updated_at"] = "2026-10-09T00:30:00Z"
        e["status"]["summary"]["updated_at"] = "2026-10-09T01:30:00Z"
        note = b"---\\nkind: summary\\n---\\n"
        receipt = {"digest": summary_digest(note), "confirmed_at": "2026-10-09T03:00:00Z"}
        status = episode_status(e, None, summary_bytes=note, sync_receipt=receipt)
        self.assertEqual(status["delivery"], "synced_to_mac")
        self.assertEqual(status["vault_sync"]["status"], "confirmed")
        self.assertEqual(status["finished_at"], receipt["confirmed_at"])
        self.assertEqual(status["compiler_completed_at"], "2026-10-09T00:30:00Z")
        self.assertEqual(status["summary_ready_at"], "2026-10-09T01:30:00Z")
        status = episode_status(e, None, summary_bytes=note+b"revision", sync_receipt=receipt)
        self.assertEqual(status["delivery"], "awaiting_vault_sync")
        self.assertEqual(status["vault_sync"]["status"], "outdated")
        self.assertIsNone(status["finished_at"])
        self.assertEqual(episode_status(e, None, summary_bytes=note)["vault_sync"]["status"], "unconfirmed")
        self.assertEqual(episode_status(e, None, sync_error=True)["vault_sync"]["status"], "unavailable")
        e["status"]["summary"]["state"] = "pending"
        self.assertEqual(episode_status(e, None)["delivery"], "awaiting_summary")

    def test_sorting_is_stable_and_latest_first(self):
        first = episode_status(episode("older", published="2026-09-01"), None)
        newer = episode_status(episode("newer", published="2026-10-08"), None)
        twin = episode_status(episode("same-day", published="2026-10-08"), None)
        ordered = sorted_episode_statuses([first, newer, twin])
        self.assertEqual([x["episode_key"] for x in ordered], [
            "same-day", "newer", "older",
        ])


class EpisodeDashboardHttpTests(unittest.TestCase):
    def test_all_episodes_visible_while_review_queue_remains_review_only(self):
        active = episode("review-1")
        finished = episode("done-1", compiler="completed", summary="ready")
        fresh = episode("new-1", compiler="pending", published="2026-09-01")

        def load(key):
            if key == "new-1":
                raise FileNotFoundError("No resolver record")
            return record(key, pending=(1 if key == "review-1" else 0))

        with (
            patch("podcast_engine.review_web.load_episodes",
                  return_value=[fresh, active, finished]) as episodes,
            patch("podcast_engine.review_web.load_review_record",
                  side_effect=load) as reviews,
            patch("podcast_engine.review_web.load_vault_sync_receipts", return_value=None),
            patch("podcast_engine.review_web.download_gcs_bytes", return_value=b"summary"),
            patch("podcast_engine.review_web.request_worker_recompile") as submit_worker,
            patch("podcast_engine.review_web.record_human_decision") as write_card,
        ):
            client = create_review_app().test_client()
            response = client.get("/api/ops/episodes")
            review_queue = client.get("/api/review/episodes")
        self.assertEqual(response.status_code, 200)
        rows = response.get_json()["episodes"]
        self.assertEqual({row["episode_key"] for row in rows},
                         {"new-1", "review-1", "done-1"})
        self.assertEqual(rows[0]["episode_key"], "review-1")
        self.assertEqual(rows[1]["episode_key"], "done-1")
        finished_row = next(row for row in rows if row["episode_key"] == "done-1")
        self.assertEqual(finished_row["compiler"], "completed")
        self.assertEqual(finished_row["review"]["pending"], 0)
        self.assertEqual(finished_row["recompile"]["status"], "started")
        self.assertEqual(
            [row["episode_key"] for row in review_queue.get_json()["episodes"]],
            ["review-1"],
        )
        self.assertGreaterEqual(episodes.call_count, 2)
        self.assertGreaterEqual(reviews.call_count, 2)
        submit_worker.assert_not_called()
        write_card.assert_not_called()

    def test_unavailable_episode_audit_is_not_suppressed(self):
        item = episode("ep-err")
        with (
            patch("podcast_engine.review_web.load_episodes", return_value=[item]),
            patch("podcast_engine.review_web.load_review_record",
                  side_effect=ValueError("Invalid audit bytes")),
            patch("podcast_engine.review_web.load_vault_sync_receipts", return_value=None),
        ):
            response = create_review_app().test_client().get("/api/ops/episodes")
        self.assertEqual(response.status_code, 200)
        body = response.get_json()["episodes"][0]
        self.assertEqual(body["review"]["state"], "unavailable")
        self.assertIsNone(body["review"]["pending"])

    def test_api_confirms_only_mac_receipt_matching_current_summary_bytes(self):
        key = "a" * 24
        e = episode(key, compiler="completed", summary="ready")
        summary = b"final canonical summary"
        manifest = {"schema_version": 1, "receipts": {
            key: {"digest": summary_digest(summary), "confirmed_at": "2026-10-09T02:00:00Z"},
        }}
        with (
            patch("podcast_engine.review_web.load_episodes", return_value=[e]),
            patch("podcast_engine.review_web.load_review_record", side_effect=FileNotFoundError()),
            patch("podcast_engine.review_web.load_vault_sync_receipts", return_value=manifest),
            patch("podcast_engine.review_web.download_gcs_bytes", return_value=summary) as summary_read,
        ):
            result = create_review_app().test_client().get("/api/ops/episodes").get_json()
        row = result["episodes"][0]
        self.assertEqual(row["vault_sync"]["status"], "confirmed")
        self.assertEqual(row["delivery"], "synced_to_mac")
        self.assertEqual(row["finished_at"], "2026-10-09T02:00:00Z")
        summary_read.assert_called_once_with(f"episodes/{key}/summary/summary.md")

    def test_api_cannot_fake_confirmation_on_missing_summary_or_corrupt_receipts(self):
        key = "a" * 24
        e = episode(key, compiler="completed", summary="ready")
        with (
            patch("podcast_engine.review_web.load_episodes", return_value=[e]),
            patch("podcast_engine.review_web.load_review_record", side_effect=FileNotFoundError()),
            patch("podcast_engine.review_web.load_vault_sync_receipts",
                  side_effect=ValueError("unsupported receipt schema")),
            patch("podcast_engine.review_web.download_gcs_bytes", side_effect=FileNotFoundError()),
        ):
            row = create_review_app().test_client().get("/api/ops/episodes").get_json()["episodes"][0]
        self.assertEqual(row["vault_sync"]["status"], "unavailable")
        self.assertEqual(row["delivery"], "unverified")

    def test_receipt_forbidden_yields_http_200_with_unavailable_mac_proof(self):
        """Production incident TASK-143: get on management receipt denied."""
        key = "a" * 24
        e = episode(key, compiler="completed", summary="ready")
        with (
            patch("podcast_engine.review_web.load_episodes", return_value=[e]),
            patch("podcast_engine.review_web.load_review_record", return_value=record(key)),
            patch(
                "podcast_engine.review_web.load_vault_sync_receipts",
                side_effect=Forbidden("storage.objects.get denied"),
            ),
            patch("podcast_engine.review_web.download_gcs_bytes",
                  return_value=b"valid current summary"),
        ):
            response = create_review_app().test_client().get("/api/ops/episodes")
        self.assertEqual(response.status_code, 200)
        row = response.get_json()["episodes"][0]
        self.assertEqual(row["compiler"], "completed")
        self.assertEqual(row["review"]["pending"], 1)
        self.assertEqual(row["recompile"]["status"], "started")
        self.assertEqual(row["vault_sync"]["status"], "unavailable")
        self.assertEqual(row["delivery"], "unverified")
        self.assertIsNone(row["finished_at"])
        self.assertNotIn("storage.objects.get", response.get_data(as_text=True))

    def test_per_episode_gcs_permission_errors_isolate_review_and_summary(self):
        key = "a" * 24
        item = episode(key, compiler="completed", summary="ready")
        with (
            patch("podcast_engine.review_web.load_episodes", return_value=[item]),
            patch("podcast_engine.review_web.load_review_record",
                  side_effect=Forbidden("resolver access forbidden")),
            patch("podcast_engine.review_web.load_vault_sync_receipts", return_value=None),
            patch("podcast_engine.review_web.download_gcs_bytes",
                  side_effect=Forbidden("summary access forbidden")),
        ):
            response = create_review_app().test_client().get("/api/ops/episodes")
        self.assertEqual(response.status_code, 200)
        row = response.get_json()["episodes"][0]
        self.assertEqual(row["review"]["state"], "unavailable")
        self.assertEqual(row["vault_sync"]["status"], "unavailable")
        self.assertEqual(row["delivery"], "unverified")
        self.assertNotIn("forbidden", response.get_data(as_text=True))

    def test_note_save_is_timeout_bounded_and_does_not_block_card_buttons(self):
        page = PAGE
        note = page.split("async function saveReviewNote()", 1)[1].split(
            "q('#save-review-note').onclick", 1
        )[0]
        self.assertIn("new AbortController()", note)
        self.assertIn("NOTE_SAVE_TIMEOUT_MS", note)
        self.assertIn("controller.abort()", note)
        self.assertIn("clearTimeout(timer)", note)
        self.assertIn("noteSaving=false", note)
        self.assertIn("if(button)button.disabled=busy", note)
        self.assertIn("if(fb&&fb.value.trim()===note)fb.value=''", note)
        self.assertIn("It may have saved; retrying the same note is safe.", note)
        self.assertNotIn("setBusy(true)", note)
        self.assertNotIn("setBusy(false)", note)
        self.assertIn("if(switchingEpisode&&noteSaving)", page)
        self.assertIn("if(error?.name==='AbortError')throw error", page)

    def test_dashboard_is_static_and_dynamic_titles_are_not_injected_in_html(self):
        self.assertIn('href="/ops"', PAGE)
        response = create_review_app().test_client().get("/ops")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn('textContent=row.title', html)
        self.assertIn('fetch("/api/ops/episodes"', html)
        self.assertNotIn("Recompile &amp; continue", html)

    def test_dashboard_and_api_use_existing_google_login_boundary(self):
        with patch.dict("os.environ", {
            "REVIEW_REQUIRE_AUTH": "true",
            "GOOGLE_OAUTH_CLIENT_ID": "client-id",
            "REVIEW_SESSION_SECRET": "test-cookie-secret",
        }):
            client = create_review_app().test_client()
            html = client.get("/ops")
            api = client.get("/api/ops/episodes")
        self.assertEqual(html.status_code, 302)
        self.assertIn("/login", html.headers["Location"])
        self.assertEqual(api.status_code, 401)

    def test_episode_titles_are_api_data_not_server_side_html(self):
        dangerous = episode("ep-malicious")
        dangerous["title"] = '<img src=x onerror=alert(1)>'
        with (
            patch("podcast_engine.review_web.load_episodes", return_value=[dangerous]),
            patch("podcast_engine.review_web.load_review_record",
                  side_effect=FileNotFoundError()),
            patch("podcast_engine.review_web.load_vault_sync_receipts", return_value=None),
        ):
            client = create_review_app().test_client()
            html = client.get("/ops").get_data(as_text=True)
            data = client.get("/api/ops/episodes").get_json()
        self.assertNotIn(dangerous["title"], html)
        self.assertEqual(data["episodes"][0]["title"], dangerous["title"])


if __name__ == "__main__":
    unittest.main()
