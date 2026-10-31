import base64
import json
import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from macos_agent import apple_token_maintenance as maintenance


def _jwt(expires_at):
    payload = base64.urlsafe_b64encode(
        json.dumps({"exp": expires_at}).encode("utf-8")
    ).decode("ascii").rstrip("=")
    return f"header.{payload}.signature"


class AppleTokenMaintenanceTests(unittest.TestCase):
    now = 1_800_000_000

    def _write_token(self, directory, token):
        path = Path(directory) / "bearer_token.txt"
        path.write_text(token, encoding="utf-8")
        return path

    def test_expiry_threshold_is_strictly_more_than_five_days(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_token(
                directory,
                _jwt(self.now + maintenance.REFRESH_THRESHOLD_SECONDS),
            )
            with patch.object(maintenance, "TOKEN_PATH", path):
                _, status = maintenance._read_local_token(now=self.now)

        self.assertTrue(maintenance.token_needs_refresh(status))

        status["remaining_seconds"] += 1
        self.assertFalse(maintenance.token_needs_refresh(status))

    def test_missing_or_expired_token_runs_helper(self):
        for token in (None, _jwt(self.now - 1)):
            with self.subTest(token=token):
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "bearer_token.txt"
                    if token is not None:
                        path.write_text(token, encoding="utf-8")

                    fresh = _jwt(
                        self.now + maintenance.REFRESH_THRESHOLD_SECONDS + 1
                    )
                    with patch.object(maintenance, "TOKEN_PATH", path), patch.object(
                        maintenance,
                        "_refresh_with_helper",
                        return_value=(True, None),
                    ) as refresh, patch.object(
                        maintenance,
                        "_read_local_token",
                        side_effect=[
                            (None if token is None else token, {"status": "missing"} if token is None else {"status": "expired", "remaining_seconds": -1}),
                            (fresh, {"status": "valid", "remaining_seconds": maintenance.REFRESH_THRESHOLD_SECONDS + 1}),
                        ],
                    ), patch.object(
                        maintenance,
                        "probe_apple_api",
                        return_value=(True, "Apple API probe succeeded."),
                    ), patch.object(
                        maintenance,
                        "upload_secret_version",
                        return_value=(True, None),
                    ):
                        self.assertEqual(maintenance.maintain_token(now=self.now), 0)

                refresh.assert_called_once_with()

    def test_helper_failure_does_not_probe_or_update_secret(self):
        with patch.object(
            maintenance,
            "_read_local_token",
            return_value=(None, {"status": "missing"}),
        ), patch.object(
            maintenance,
            "_refresh_with_helper",
            return_value=(False, "FetchTranscript helper failed with exit 1."),
        ), patch.object(maintenance, "probe_apple_api") as probe, patch.object(
            maintenance,
            "upload_secret_version",
        ) as upload:
            self.assertEqual(maintenance.maintain_token(now=self.now), 1)

        probe.assert_not_called()
        upload.assert_not_called()

    def test_probe_failure_does_not_update_secret(self):
        fresh = _jwt(self.now + maintenance.REFRESH_THRESHOLD_SECONDS + 1)
        with patch.object(
            maintenance,
            "_read_local_token",
            side_effect=[
                (None, {"status": "missing"}),
                (fresh, {"status": "valid", "remaining_seconds": maintenance.REFRESH_THRESHOLD_SECONDS + 1}),
            ],
        ), patch.object(
            maintenance,
            "_refresh_with_helper",
            return_value=(True, None),
        ), patch.object(
            maintenance,
            "probe_apple_api",
            return_value=(False, "Apple API probe failed."),
        ), patch.object(maintenance, "upload_secret_version") as upload:
            self.assertEqual(maintenance.maintain_token(now=self.now), 1)

        upload.assert_not_called()

    def test_successful_refresh_probes_then_adds_secret_version(self):
        fresh = _jwt(self.now + maintenance.REFRESH_THRESHOLD_SECONDS + 1)
        with patch.object(
            maintenance,
            "_read_local_token",
            side_effect=[
                (None, {"status": "missing"}),
                (fresh, {"status": "valid", "remaining_seconds": maintenance.REFRESH_THRESHOLD_SECONDS + 1}),
            ],
        ), patch.object(
            maintenance,
            "_refresh_with_helper",
            return_value=(True, None),
        ), patch.object(
            maintenance,
            "probe_apple_api",
            return_value=(True, "Apple API probe succeeded."),
        ) as probe, patch.object(
            maintenance,
            "upload_secret_version",
            return_value=(True, None),
        ) as upload:
            self.assertEqual(maintenance.maintain_token(now=self.now), 0)

        probe.assert_called_once_with(fresh)
        upload.assert_called_once_with(fresh)

    def test_successful_noop_sends_the_optional_heartbeat(self):
        status = {
            "status": "valid",
            "remaining_seconds": maintenance.REFRESH_THRESHOLD_SECONDS + 1,
        }
        with patch.object(
            maintenance,
            "_read_local_token",
            return_value=("not-returned", status),
        ), patch.object(
            maintenance,
            "send_success_heartbeat",
            return_value=True,
        ) as heartbeat, patch.object(
            maintenance,
            "_refresh_with_helper",
        ) as refresh:
            self.assertEqual(maintenance.maintain_token(now=self.now), 0)

        heartbeat.assert_called_once_with()
        refresh.assert_not_called()

    def test_failure_never_sends_success_heartbeat(self):
        with patch.object(
            maintenance,
            "_read_local_token",
            return_value=(None, {"status": "missing"}),
        ), patch.object(
            maintenance,
            "_refresh_with_helper",
            return_value=(False, "helper failed"),
        ), patch.object(
            maintenance,
            "send_success_heartbeat",
        ) as heartbeat:
            self.assertEqual(maintenance.maintain_token(now=self.now), 1)

        heartbeat.assert_not_called()

    def test_secret_update_uses_stdin_and_keeps_old_versions(self):
        result = subprocess.CompletedProcess([], 0, "", "")
        with patch.object(
            maintenance.subprocess,
            "run",
            return_value=result,
        ) as run:
            success, error = maintenance.upload_secret_version("not-logged")

        self.assertTrue(success)
        self.assertIsNone(error)
        command = run.call_args.args[0]
        self.assertEqual(command[-1], "--data-file=-")
        self.assertNotIn("disable", command)
        self.assertNotIn("destroy", command)
        self.assertEqual(run.call_args.kwargs["input"], "not-logged")


_FAKE_HELPER = """#!/usr/bin/env python3
# Emulates the upstream FetchTranscript --cache-bearer-token semantics
# (dado3212/apple-podcast-transcript-downloader @535e0ce): a bearer_token.txt in
# the working directory whose mtime is younger than 30 days is reused verbatim,
# regardless of the JWT's own expiry; otherwise a fresh token is fetched and
# written there. The transcript download that follows may still fail.
import os, sys, time
path = "bearer_token.txt"
if "--cache-bearer-token" in sys.argv and os.path.exists(path) and time.time() - os.path.getmtime(path) < 30 * 86400:
    sys.exit(int(os.environ.get("FAKE_HELPER_EXIT", "0")))
fresh = os.environ.get("FAKE_HELPER_FRESH_TOKEN", "")
if fresh:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(fresh + "\\n")
sys.exit(int(os.environ.get("FAKE_HELPER_EXIT", "0")))
"""


class HelperCacheRefreshTests(unittest.TestCase):
    """TASK-116: the refresh must never be satisfied by the helper's own
    30-day file cache, which reuses a near-expiry or expired token."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.runtime = root / "apple-api"
        self.runtime.mkdir()
        self.token_path = self.runtime / "bearer_token.txt"
        self.helper = root / "FetchTranscript"
        self.helper.write_text(_FAKE_HELPER, encoding="utf-8")
        self.helper.chmod(0o700)
        self.now = time.time()
        self._patches = [
            patch.object(maintenance, "RUNTIME_DIR", self.runtime),
            patch.object(maintenance, "TOKEN_PATH", self.token_path),
            patch.object(maintenance, "FETCHER_PATH", self.helper),
        ]
        for item in self._patches:
            item.start()

    def tearDown(self):
        for item in self._patches:
            item.stop()
        self._tmp.cleanup()

    def _env(self, fresh, exit_code=0):
        return patch.dict(
            os.environ,
            {"FAKE_HELPER_FRESH_TOKEN": fresh, "FAKE_HELPER_EXIT": str(exit_code)},
        )

    def test_recently_written_near_expiry_cache_is_not_reused(self):
        stale = _jwt(self.now + 2 * 86_400)
        self.token_path.write_text(stale, encoding="utf-8")  # fresh mtime
        fresh = _jwt(self.now + 20 * 86_400)
        with self._env(fresh):
            ok, error = maintenance._refresh_with_helper()
        self.assertTrue(ok, error)
        self.assertEqual(self.token_path.read_text(encoding="utf-8").strip(), fresh)

    def test_expired_cache_is_not_reused(self):
        self.token_path.write_text(_jwt(self.now - 86_400), encoding="utf-8")
        fresh = _jwt(self.now + 20 * 86_400)
        with self._env(fresh):
            ok, error = maintenance._refresh_with_helper()
        self.assertTrue(ok, error)
        self.assertEqual(self.token_path.read_text(encoding="utf-8").strip(), fresh)

    def test_fresh_token_is_kept_when_helper_fails_after_writing_it(self):
        self.token_path.write_text(_jwt(self.now - 86_400), encoding="utf-8")
        fresh = _jwt(self.now + 20 * 86_400)
        with self._env(fresh, exit_code=1):
            ok, error = maintenance._refresh_with_helper()
        self.assertTrue(ok, error)
        self.assertEqual(self.token_path.read_text(encoding="utf-8").strip(), fresh)

    def test_helper_without_new_token_leaves_existing_token_untouched(self):
        existing = _jwt(self.now + 2 * 86_400)
        self.token_path.write_text(existing, encoding="utf-8")
        with self._env("", exit_code=1):
            ok, error = maintenance._refresh_with_helper()
        self.assertFalse(ok)
        self.assertIn("exit 1", error)
        self.assertEqual(self.token_path.read_text(encoding="utf-8"), existing)
        self.assertEqual(sorted(p.name for p in self.runtime.iterdir()), ["bearer_token.txt"])

    def test_never_downgrades_to_a_token_expiring_sooner(self):
        existing = _jwt(self.now + 4 * 86_400)
        self.token_path.write_text(existing, encoding="utf-8")
        with self._env(_jwt(self.now + 3 * 86_400)):
            ok, error = maintenance._refresh_with_helper()
        self.assertFalse(ok)
        self.assertEqual(self.token_path.read_text(encoding="utf-8"), existing)

    def test_end_to_end_refresh_uploads_the_fresh_token(self):
        self.token_path.write_text(_jwt(self.now + 2 * 86_400), encoding="utf-8")
        fresh = _jwt(self.now + 20 * 86_400)
        with self._env(fresh), patch.object(
            maintenance, "probe_apple_api", return_value=(True, "ok")
        ), patch.object(
            maintenance, "upload_secret_version", return_value=(True, None)
        ) as upload, patch.object(maintenance, "send_success_heartbeat", return_value=True):
            self.assertEqual(maintenance.maintain_token(now=self.now), 0)
        upload.assert_called_once_with(fresh)


if __name__ == "__main__":
    unittest.main()
