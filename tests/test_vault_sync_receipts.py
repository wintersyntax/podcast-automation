"""Deterministic Mac Vault Sync receipt tests (TASK-142)."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from google.api_core.exceptions import PreconditionFailed

from macos_agent.vault_sync import export_vault_summary, sync_all_ready, sync_episode
from podcast_engine.vault_sync_receipts import (
    VAULT_SYNC_RECEIPT_PATH,
    load_vault_sync_receipts,
    receipt_status,
    record_vault_sync_receipts,
    summary_digest,
)


KEY = "a" * 24
OTHER = "b" * 24
CONTENT = b"---\nepisode_key: test\n---\n\nA canonical summary\n"


class FakeBlob:
    def __init__(self):
        self.value = None
        self.generation = 0
        self.writes = 0
        self.fail_once = False

    def exists(self):
        return self.value is not None

    def reload(self):
        if not self.exists():
            raise FileNotFoundError()

    def download_as_text(self, encoding="utf-8", if_generation_match=None):
        if if_generation_match is not None and if_generation_match != self.generation:
            raise PreconditionFailed("stale")
        return self.value

    def upload_from_string(self, data, *, content_type=None, if_generation_match=None):
        if self.fail_once:
            self.fail_once = False
            self.generation += 1
            self.value = json.dumps({
                "schema_version": 1,
                "receipts": {
                    OTHER: {
                        "digest": summary_digest(b"other"),
                        "confirmed_at": "2026-10-09T00:01:00Z",
                    },
                },
            })
            raise PreconditionFailed("simulated concurrent writer")
        if if_generation_match != self.generation:
            raise PreconditionFailed("stale")
        self.value = data
        self.generation += 1
        self.writes += 1


class FakeBucket:
    def __init__(self):
        self.receipt_blob = FakeBlob()
        self.paths = []

    def blob(self, path):
        self.paths.append(path)
        if path != VAULT_SYNC_RECEIPT_PATH:
            raise AssertionError("Unexpected receipt path")
        return self.receipt_blob


def _episode():
    return {
        "episode_key": KEY,
        "id": KEY,
        "podcast": "Iron Culture",
        "title": "Ep 391",
        "published": "Thu, 08 Oct 2026 14:58:00 +0000",
        "status": {"summary": {"state": "ready"}},
    }


class VaultSyncReceiptsTests(unittest.TestCase):
    def test_initial_write_repeat_and_changed_content_are_generation_safe(self):
        bucket = FakeBucket()
        first = summary_digest(CONTENT)
        result = record_vault_sync_receipts(
            {KEY: first}, bucket=bucket, clock=lambda: "2026-10-09T01:00:00Z"
        )
        self.assertEqual(result, "recorded")
        self.assertEqual(bucket.receipt_blob.writes, 1)
        snapshot = copy.deepcopy(load_vault_sync_receipts(bucket=bucket))
        self.assertEqual(snapshot["receipts"][KEY]["digest"], first)
        again = record_vault_sync_receipts(
            {KEY: first}, bucket=bucket, clock=lambda: "2026-10-09T02:00:00Z"
        )
        self.assertEqual(again, "unchanged")
        self.assertEqual(bucket.receipt_blob.writes, 1)
        self.assertEqual(load_vault_sync_receipts(bucket=bucket), snapshot)
        updated = record_vault_sync_receipts(
            {KEY: summary_digest(CONTENT+b"new")},
            bucket=bucket, clock=lambda: "2026-10-09T03:00:00Z",
        )
        self.assertEqual(updated, "recorded")
        self.assertEqual(bucket.receipt_blob.writes, 2)
        self.assertEqual(
            load_vault_sync_receipts(bucket=bucket)["receipts"][KEY]["confirmed_at"],
            "2026-10-09T03:00:00Z",
        )

    def test_cas_race_retries_without_discarding_concurrent_receipt(self):
        bucket = FakeBucket()
        bucket.receipt_blob.fail_once = True
        result = record_vault_sync_receipts(
            {KEY: summary_digest(CONTENT)}, bucket=bucket,
            clock=lambda: "2026-10-09T01:00:00Z",
        )
        self.assertEqual(result, "recorded")
        receipts = load_vault_sync_receipts(bucket=bucket)["receipts"]
        self.assertIn(KEY, receipts)
        self.assertIn(OTHER, receipts)

    def test_malformed_object_cannot_be_overwritten_and_missing_is_not_success(self):
        bucket = FakeBucket()
        self.assertIsNone(load_vault_sync_receipts(bucket=bucket))
        bucket.receipt_blob.value = '{"schema_version": 99, "receipts": {}}'
        bucket.receipt_blob.generation = 5
        with self.assertRaisesRegex(ValueError, "schema"):
            record_vault_sync_receipts({KEY: summary_digest(CONTENT)}, bucket=bucket)
        self.assertEqual(bucket.receipt_blob.writes, 0)
        with self.assertRaises(ValueError):
            load_vault_sync_receipts(bucket=bucket)

    def test_unknown_or_changed_source_is_never_mac_confirmed(self):
        receipt = {
            "digest": summary_digest(CONTENT),
            "confirmed_at": "2026-10-09T01:00:00Z",
        }
        self.assertEqual(
            receipt_status(summary_state="ready", summary_bytes=CONTENT, receipt=receipt)["status"],
            "confirmed",
        )
        self.assertEqual(
            receipt_status(summary_state="ready", summary_bytes=CONTENT+b"x", receipt=receipt)["status"],
            "outdated",
        )
        self.assertEqual(
            receipt_status(summary_state="ready", summary_bytes=CONTENT, receipt=None)["status"],
            "unconfirmed",
        )
        self.assertEqual(
            receipt_status(summary_state="pending", summary_bytes=None, receipt=receipt)["status"],
            "not_ready",
        )
        self.assertEqual(
            receipt_status(summary_state="ready", summary_bytes=None, receipt=receipt)["status"],
            "unavailable",
        )
        self.assertEqual(
            receipt_status(summary_state="ready", summary_bytes=CONTENT, receipt={"digest": "wrong"})["status"],
            "unavailable",
        )


class MacReceiptsProductionBoundaryTests(unittest.TestCase):
    def test_export_callback_only_after_successful_byte_identical_local_copy(self):
        callbacks = []
        with tempfile.TemporaryDirectory() as directory:
            with patch("macos_agent.vault_sync.download_gcs_bytes", return_value=CONTENT):
                destination, wrote = export_vault_summary(
                    _episode(), vault_root=Path(directory), vault_subdir=Path("Podcasts"),
                    on_synced=lambda key, content: callbacks.append((key, content)),
                )
                again, changed = export_vault_summary(
                    _episode(), vault_root=Path(directory), vault_subdir=Path("Podcasts"),
                    on_synced=lambda key, content: callbacks.append((key, content)),
                )
        self.assertTrue(wrote)
        self.assertFalse(changed)
        self.assertEqual(destination, again)
        self.assertEqual(callbacks, [(KEY, CONTENT), (KEY, CONTENT)])

    def test_failed_local_write_never_emits_receipt_callback(self):
        callbacks = []
        with tempfile.TemporaryDirectory() as directory, patch(
            "macos_agent.vault_sync.download_gcs_bytes", return_value=CONTENT,
        ), patch(
            "macos_agent.vault_sync.os.replace", side_effect=OSError("disk failure"),
        ):
            with self.assertRaises(OSError):
                export_vault_summary(
                    _episode(), vault_root=Path(directory), vault_subdir=Path("Podcasts"),
                    on_synced=lambda key, content: callbacks.append((key, content)),
                )
        self.assertEqual(callbacks, [])

    def test_full_sweep_writes_exactly_one_batch_only_after_all_local_success(self):
        e = _episode()
        def export(episode, *, on_synced):
            on_synced(episode["episode_key"], CONTENT)
            return Path("/tmp/summary.md"), True
        with (
            patch("macos_agent.vault_sync.load_episodes", return_value=[e]),
            patch("macos_agent.vault_sync.export_vault_summary", side_effect=export),
            patch("macos_agent.vault_sync.record_vault_sync_receipts") as publish,
        ):
            self.assertEqual(sync_all_ready(), (1, 0))
        publish.assert_called_once_with({KEY: summary_digest(CONTENT)})

    def test_mid_sweep_failure_does_not_publish_any_claim(self):
        e = _episode()
        other = {**e, "episode_key": OTHER, "id": OTHER}
        def export(episode, *, on_synced):
            if episode["episode_key"] == OTHER:
                raise OSError("second note not copied")
            on_synced(episode["episode_key"], CONTENT)
            return Path("/tmp/summary.md"), True
        with (
            patch("macos_agent.vault_sync.load_episodes", return_value=[e, other]),
            patch("macos_agent.vault_sync.export_vault_summary", side_effect=export),
            patch("macos_agent.vault_sync.record_vault_sync_receipts") as publish,
        ):
            with self.assertRaises(OSError):
                sync_all_ready()
        publish.assert_not_called()

    def test_manual_sync_publishes_but_does_not_require_healthchecks(self):
        e = _episode()
        def export(episode, *, on_synced):
            on_synced(episode["episode_key"], CONTENT)
            return Path("/tmp/summary.md"), False
        with (
            patch("macos_agent.vault_sync.load_episodes", return_value=[e]),
            patch("macos_agent.vault_sync.export_vault_summary", side_effect=export),
            patch("macos_agent.vault_sync.record_vault_sync_receipts") as publish,
        ):
            path, written = sync_episode(KEY)
        self.assertFalse(written)
        self.assertEqual(path.name, "summary.md")
        publish.assert_called_once_with({KEY: summary_digest(CONTENT)})


if __name__ == "__main__":
    unittest.main()
