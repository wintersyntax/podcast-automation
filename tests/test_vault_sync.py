import io
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from macos_agent.vault_sync import (
    destination_for_episode,
    export_vault_summary,
    main,
    ready_summary_episodes,
    sanitize_filename,
    send_vault_sync_heartbeat,
    vault_root_from_env,
)


SUMMARY = b"---\ntype: podcast-summary\nepisode_key: episode-1\n---\n\n# Summary\n"


class VaultSyncTests(unittest.TestCase):
    def _episode(self):
        return {
            "id": "episode-1",
            "episode_key": "episode-1",
            "podcast": "Example / Strength: Podcast",
            "title": "Ep 386: Caffeine / Creatine?",
            "published": "Thu, 20 Aug 2026 14:58:00 +0000",
            "status": {"summary": {"state": "ready"}},
        }

    def test_sanitize_filename_keeps_safe_unicode_and_removes_problem_characters(self):
        self.assertEqual(
            sanitize_filename("Čučanj: 5/3?*", "fallback"),
            "Čučanj 5 3",
        )

    def test_destination_uses_configured_subfolder_podcast_date_and_title(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = destination_for_episode(
                self._episode(),
                Path(directory),
                Path("Podcasts/Strength"),
            )

            self.assertEqual(
                destination.relative_to(Path(directory)).as_posix(),
                "Podcasts/Strength/Example Strength Podcast/"
                "2026-08-20 - Ep 386 Caffeine Creatine.md",
            )

    def test_existing_identical_summary_is_a_no_op(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = destination_for_episode(
                self._episode(),
                root,
                Path("Podcasts"),
            )
            destination.parent.mkdir(parents=True)
            destination.write_bytes(SUMMARY)
            before_mtime = destination.stat().st_mtime_ns

            with patch(
                "macos_agent.vault_sync.download_gcs_bytes",
                return_value=SUMMARY,
            ), patch.object(
                Path,
                "write_bytes",
                wraps=Path.write_bytes,
            ) as write_bytes:
                actual, wrote = export_vault_summary(
                    self._episode(),
                    vault_root=root,
                    vault_subdir=Path("Podcasts"),
                )

            self.assertEqual(actual, destination)
            self.assertFalse(wrote)
            self.assertEqual(
                destination.stat().st_mtime_ns,
                before_mtime,
            )
            write_bytes.assert_not_called()

    def test_changed_summary_overwrites_the_same_canonical_note(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = destination_for_episode(
                self._episode(),
                root,
                Path("Podcasts"),
            )
            destination.parent.mkdir(parents=True)
            destination.write_bytes(b"old")
            changed = SUMMARY + b"Updated.\n"

            with patch(
                "macos_agent.vault_sync.download_gcs_bytes",
                return_value=changed,
            ):
                actual, wrote = export_vault_summary(
                    self._episode(),
                    vault_root=root,
                    vault_subdir=Path("Podcasts"),
                )

            self.assertEqual(actual, destination)
            self.assertTrue(wrote)
            self.assertEqual(destination.read_bytes(), changed)

    def test_missing_vault_environment_is_a_clear_local_error(self):
        with self.assertRaisesRegex(
            ValueError,
            "PODCAST_VAULT_DIR is not set",
        ):
            vault_root_from_env({"UNRELATED": "value"})

    def test_all_ready_selection_excludes_pending_and_accepts_completed(self):
        episodes = [
            self._episode(),
            {
                "id": "pending",
                "status": {"summary": {"state": "pending"}},
            },
            {
                "id": "complete",
                "status": {"summary": {"state": "completed"}},
            },
        ]

        self.assertEqual(
            [
                episode["id"]
                for episode in ready_summary_episodes(episodes)
            ],
            ["episode-1", "complete"],
        )

    def test_heartbeat_without_url_is_a_no_op(self):
        with patch(
            "macos_agent.vault_sync.subprocess.run"
        ) as run:
            sent = send_vault_sync_heartbeat({})

        self.assertFalse(sent)
        run.assert_not_called()

    def test_successful_heartbeat_uses_curl_once(self):
        heartbeat_url = "https://hc-ping.com/test-vault-heartbeat"

        with patch(
            "macos_agent.vault_sync.subprocess.run",
            return_value=subprocess.CompletedProcess(
                args=[],
                returncode=0,
            ),
        ) as run:
            sent = send_vault_sync_heartbeat(
                {
                    "PODCAST_VAULT_SYNC_HEARTBEAT_URL":
                        heartbeat_url
                }
            )

        self.assertTrue(sent)
        run.assert_called_once()

        command = run.call_args.args[0]
        self.assertEqual(command[0], "/usr/bin/curl")
        self.assertIn("--fail", command)
        self.assertIn("--silent", command)
        self.assertIn("--show-error", command)
        self.assertIn(heartbeat_url, command)

    def test_heartbeat_failure_is_secret_safe(self):
        heartbeat_url = (
            "https://hc-ping.com/"
            "this-secret-heartbeat-url-must-not-be-logged"
        )
        output = io.StringIO()

        with patch(
            "macos_agent.vault_sync.subprocess.run",
            return_value=subprocess.CompletedProcess(
                args=[],
                returncode=22,
            ),
        ), redirect_stdout(output):
            sent = send_vault_sync_heartbeat(
                {
                    "PODCAST_VAULT_SYNC_HEARTBEAT_URL":
                        heartbeat_url
                }
            )

        rendered = output.getvalue()

        self.assertFalse(sent)
        self.assertIn(
            "vault_sync_heartbeat_failed",
            rendered,
        )
        self.assertNotIn(
            heartbeat_url,
            rendered,
        )
        self.assertNotIn(
            "this-secret-heartbeat-url-must-not-be-logged",
            rendered,
        )

    def test_successful_all_ready_run_sends_one_heartbeat(self):
        with patch(
            "macos_agent.vault_sync.load_repo_env"
        ), patch(
            "macos_agent.vault_sync.vault_root_from_env",
            return_value=Path("/tmp/vault"),
        ), patch(
            "macos_agent.vault_sync.vault_subdir_from_env",
            return_value=Path("Podcasts"),
        ), patch(
            "macos_agent.vault_sync.sync_all_ready",
            return_value=(1, 2),
        ), patch(
            "macos_agent.vault_sync.send_vault_sync_heartbeat",
            return_value=True,
        ) as heartbeat:
            result = main(["--all-ready"])

        self.assertEqual(result, 0)
        heartbeat.assert_called_once_with()

    def test_failed_all_ready_run_sends_no_success_heartbeat(self):
        with patch(
            "macos_agent.vault_sync.load_repo_env"
        ), patch(
            "macos_agent.vault_sync.vault_root_from_env",
            return_value=Path("/tmp/vault"),
        ), patch(
            "macos_agent.vault_sync.vault_subdir_from_env",
            return_value=Path("Podcasts"),
        ), patch(
            "macos_agent.vault_sync.sync_all_ready",
            side_effect=ValueError("simulated sync failure"),
        ), patch(
            "macos_agent.vault_sync.send_vault_sync_heartbeat"
        ) as heartbeat:
            result = main(["--all-ready"])

        self.assertEqual(result, 2)
        heartbeat.assert_not_called()

    def test_heartbeat_failure_does_not_fail_successful_vault_sync(self):
        with patch(
            "macos_agent.vault_sync.load_repo_env"
        ), patch(
            "macos_agent.vault_sync.vault_root_from_env",
            return_value=Path("/tmp/vault"),
        ), patch(
            "macos_agent.vault_sync.vault_subdir_from_env",
            return_value=Path("Podcasts"),
        ), patch(
            "macos_agent.vault_sync.sync_all_ready",
            return_value=(0, 3),
        ), patch(
            "macos_agent.vault_sync.send_vault_sync_heartbeat",
            return_value=False,
        ) as heartbeat:
            result = main(["--all-ready"])

        self.assertEqual(result, 0)
        heartbeat.assert_called_once_with()

    def test_episode_key_sync_does_not_send_scheduled_heartbeat(self):
        with patch(
            "macos_agent.vault_sync.load_repo_env"
        ), patch(
            "macos_agent.vault_sync.vault_root_from_env",
            return_value=Path("/tmp/vault"),
        ), patch(
            "macos_agent.vault_sync.vault_subdir_from_env",
            return_value=Path("Podcasts"),
        ), patch(
            "macos_agent.vault_sync.sync_episode"
        ) as sync_episode, patch(
            "macos_agent.vault_sync.send_vault_sync_heartbeat"
        ) as heartbeat:
            result = main(
                [
                    "--episode-key",
                    "episode-1",
                ]
            )

        self.assertEqual(result, 0)
        sync_episode.assert_called_once_with("episode-1")
        heartbeat.assert_not_called()


if __name__ == "__main__":
    unittest.main()
