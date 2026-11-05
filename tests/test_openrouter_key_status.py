"""warn before an OpenRouter key the Worker uses expires."""

from __future__ import annotations

from datetime import datetime, timezone
import unittest
from unittest.mock import patch

from podcast_engine import openrouter_key_status as keys
from podcast_engine.pipeline import run_pipeline

NOW = datetime(2026, 10, 31, 17, 0, tzinfo=timezone.utc)


class _Response:
    def __init__(self, status_code=200, data=None):
        self.status_code = status_code
        self._data = data

    def json(self):
        return {"data": self._data}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def _get_for(responses, calls=None):
    def get(url, *, headers, timeout):
        assert url == keys.OPENROUTER_KEY_URL
        key = headers["Authorization"].removeprefix("Bearer ")
        if calls is not None:
            calls.append(key)
        response = responses[key]
        if isinstance(response, Exception):
            raise response
        return response
    return get


class CheckOpenRouterKeysTests(unittest.TestCase):
    def _check(self, environ, responses):
        with patch.object(keys, "emit_event") as events:
            results = keys.check_openrouter_keys(environ=environ, now=NOW, get=_get_for(responses))
        return results, events

    def test_classifies_each_configured_key_and_skips_missing_ones(self):
        environ = {
            "PODCAST_WHISPER_API_KEY": "k-whisper",
            "PODCAST_KNOWLEDGE_API_KEY": "k-knowledge",
            "PODCAST_TRANSCRIPT_REVIEW_API_KEY": "k-review",
            "PODCAST_REVIEW_ASR_API_KEY": "",
        }
        responses = {
            "k-whisper": _Response(data={"label": "podcast-whisper", "expires_at": "2026-11-05T17:00:00Z"}),
            "k-knowledge": _Response(data={"label": "knowledge", "expires_at": "2027-10-20T21:08:00Z"}),
            "k-review": _Response(data={"label": "review", "expires_at": None}),
        }
        results, events = self._check(environ, responses)
        by_env = {item["env"]: item for item in results}
        self.assertEqual(set(by_env), {"PODCAST_WHISPER_API_KEY", "PODCAST_KNOWLEDGE_API_KEY",
                                       "PODCAST_TRANSCRIPT_REVIEW_API_KEY"})
        self.assertEqual((by_env["PODCAST_WHISPER_API_KEY"]["status"], by_env["PODCAST_WHISPER_API_KEY"]["remaining_days"]),
                         ("expiring", 5.0))
        self.assertEqual(by_env["PODCAST_KNOWLEDGE_API_KEY"]["status"], "ok")
        self.assertEqual(by_env["PODCAST_TRANSCRIPT_REVIEW_API_KEY"]["status"], "no_expiry")
        self.assertEqual([call.args[0] for call in events.call_args_list], ["openrouter_key_expiring"])
        self.assertEqual(events.call_args.kwargs["severity"], "WARNING")
        self.assertEqual(keys.keys_needing_alert(results), [by_env["PODCAST_WHISPER_API_KEY"]])

    def test_rejected_or_expired_key_is_an_error_and_alerts(self):
        environ = {"PODCAST_WHISPER_API_KEY": "k-old", "PODCAST_KNOWLEDGE_API_KEY": "k-past"}
        responses = {
            "k-old": _Response(status_code=401),
            "k-past": _Response(data={"label": "knowledge", "expires_at": "2026-10-30T00:00:00Z"}),
        }
        results, events = self._check(environ, responses)
        self.assertEqual([item["status"] for item in results], ["invalid", "expired"])
        self.assertEqual([call.args[0] for call in events.call_args_list], ["openrouter_key_invalid"] * 2)
        self.assertEqual(len(keys.keys_needing_alert(results)), 2)

    def test_a_key_due_in_eight_to_fourteen_days_warns_without_failing_the_run(self):
        results, events = self._check(
            {"PODCAST_WHISPER_API_KEY": "k"},
            {"k": _Response(data={"label": "w", "expires_at": "2026-11-10T17:00:00Z"})},
        )
        self.assertEqual(results[0]["status"], "expiring")
        self.assertEqual(keys.keys_needing_alert(results), [])

    def test_unreachable_check_is_quiet_and_never_raises(self):
        results, events = self._check({"PODCAST_WHISPER_API_KEY": "k"}, {"k": TimeoutError("slow")})
        self.assertEqual(results, [{"env": "PODCAST_WHISPER_API_KEY", "status": "unavailable"}])
        self.assertEqual(events.call_args.args[0], "openrouter_key_check_unavailable")
        self.assertEqual(events.call_args.kwargs["severity"], "INFO")
        self.assertEqual(keys.keys_needing_alert(results), [])

    def test_events_never_carry_the_key(self):
        _, events = self._check(
            {"PODCAST_WHISPER_API_KEY": "synthetic-secret"},
            {"synthetic-secret": _Response(status_code=401)},
        )
        self.assertNotIn("synthetic-secret", repr(events.call_args_list))


class PipelineKeyAlertTests(unittest.TestCase):
    def _run(self, key_status):
        with (
            patch("podcast_engine.pipeline.ENABLED_PODCASTS", []),
            patch("podcast_engine.pipeline._retain_review_audio"),
            patch("podcast_engine.pipeline.check_openrouter_keys", return_value=key_status),
        ):
            return run_pipeline()

    def test_due_key_ends_the_scheduled_run_failed_after_the_work(self):
        due = [{"env": "PODCAST_WHISPER_API_KEY", "status": "expiring", "remaining_days": 3.0}]
        with self.assertRaisesRegex(keys.OpenRouterKeysExpiring, "PODCAST_WHISPER_API_KEY"):
            self._run(due)

    def test_healthy_or_unreadable_keys_leave_the_run_green(self):
        self.assertEqual(self._run([{"env": "PODCAST_WHISPER_API_KEY", "status": "ok", "remaining_days": 300.0},
                                    {"env": "PODCAST_KNOWLEDGE_API_KEY", "status": "unavailable"}]), [])

    def test_targeted_recompile_does_not_check_keys(self):
        with (
            patch("podcast_engine.pipeline.get_episode_by_key", return_value=None),
            patch("podcast_engine.pipeline.check_openrouter_keys") as check,
        ):
            with self.assertRaises(RuntimeError):
                run_pipeline(target_episode_key="missing")
        check.assert_not_called()


if __name__ == "__main__":
    unittest.main()
