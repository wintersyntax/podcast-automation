import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from macos_agent.knowledge_sync import (
    KnowledgeRoot,
    KnowledgeSyncAgent,
    SingleRunLock,
    atomic_write,
    knowledge_roots_from_env,
)
from podcast_engine.knowledge.tags import (
    AGENT_STATUS_PATH,
    ARTIFACT_INDEX_PATH,
    KnowledgeAgentStatus,
    TagRegistry,
)
from podcast_engine.review_web import TAG_PAGE, create_review_app
from tests.test_tag_vocabulary import _Bucket


NOTE_A = """---
type: article
tags:
  - creatine
  - sleep
  - recovery
---

Test body A.
"""
NOTE_B = """---
type: article
title: Lengthened partials
topics:
  - User owned topic
tags:
  - hypertrophy
  - lengthened-partials
---

Test body B.
"""
NOTE_C = """---
type: book
tags:
  - protein
---

Test body C.
"""


class KnowledgeSyncTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name) / "knowledge"
        self.root.mkdir()
        self.bucket = _Bucket()
        self.registry = TagRegistry(bucket=self.bucket, episodes_loader=lambda: [])
        self.status = KnowledgeAgentStatus(bucket=self.bucket)
        self.agent = KnowledgeSyncAgent(
            roots=[KnowledgeRoot("test", self.root)],
            registry=self.registry,
            status=self.status,
            lock_path=Path(self.directory.name) / "knowledge-sync.lock",
        )

    def note(self, name, content):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def candidate(self, relative_path):
        return next(
            item for item in self.registry.artifact_index.local_candidates()
            if item["relative_path"] == relative_path
        )

    def adopt(self, relative_path):
        adopted = self.registry.adopt_local_artifact(self.candidate(relative_path))
        self.assertEqual(adopted["local_work_item"]["status"], "waiting_for_agent")
        return adopted

    def test_scan_only_configured_roots_and_ignores_unrelated_markdown(self):
        self.note("A.md", NOTE_A)
        self.note("README.md", "# Normal project Markdown\n")
        outside = Path(self.directory.name) / "outside.md"
        outside.write_text(NOTE_C, encoding="utf-8")

        result = self.agent.run()

        self.assertEqual(result["status"], "Healthy")
        self.assertEqual(result["last_run"]["scanned"], 1)
        self.assertEqual(len(self.registry.artifact_index.local_candidates()), 1)
        self.assertEqual(self.candidate("A.md")["source_type"], "article")

    def test_candidate_adoption_is_agent_only_and_preserves_body(self):
        path = self.note("Books/C.md", NOTE_C)
        original = path.read_text(encoding="utf-8")
        self.agent.run()

        adopted = self.adopt("Books/C.md")
        self.assertEqual(path.read_text(encoding="utf-8"), original)
        self.agent.run()

        saved = path.read_text(encoding="utf-8")
        artifact_id = adopted["artifact"]["artifact_id"]
        self.assertIn(f"knowledge_id: {artifact_id}\n", saved)
        self.assertTrue(saved.endswith("\nTest body C.\n"))
        self.assertEqual(self.registry.artifact_index.get(artifact_id)["relative_path"], "Books/C.md")
        self.assertEqual(self.registry.pending_adoption_work(), [])
        self.agent.run()
        self.assertEqual(path.read_text(encoding="utf-8").count("knowledge_id:"), 1)

    def test_unknown_tag_is_global_candidate_but_alias_never_changes_note_automatically(self):
        path = self.note("B.md", NOTE_B)
        self.agent.run()
        adopted = self.adopt("B.md")
        self.agent.run()
        before_review = path.read_text(encoding="utf-8")

        entry = self.registry.list_candidates("pending")[0]
        self.assertEqual(entry["slug"], "lengthened-partials")
        self.assertEqual(entry["source_distribution"], {"article": 1})
        self.registry.decide("lengthened-partials", "map", mapped_to="hypertrophy")
        self.assertEqual(path.read_text(encoding="utf-8"), before_review)

        queued = self.registry.run_backfill("lengthened-partials")["backfill"]
        self.assertEqual(queued["status"], "waiting_for_agent")
        self.assertEqual(path.read_text(encoding="utf-8"), before_review)
        completed = self.agent.run()
        saved = path.read_text(encoding="utf-8")

        self.assertEqual(completed["last_run"]["backfilled"], 1)
        self.assertIn("- hypertrophy\n", saved)
        self.assertNotIn("- lengthened-partials\n", saved)
        self.assertEqual(saved.count("- hypertrophy\n"), 1)
        self.assertIn("topics:\n  - User owned topic\n", saved)
        self.assertTrue(saved.endswith("\nTest body B.\n"))
        self.assertEqual(self.registry.artifact_index.get(adopted["artifact"]["artifact_id"])["tags"], ["hypertrophy"])

    def test_existing_identity_survives_a_path_change_and_body_is_not_indexed(self):
        path = self.note("Inbox/A.md", "---\nknowledge_id: note:stable\ntype: article\ntags:\n  - creatine\n---\n\nBody one.\n")
        self.agent.run()
        moved = self.root / "Research/A.md"
        moved.parent.mkdir()
        path.rename(moved)

        self.agent.run()

        artifact = self.registry.artifact_index.get("note:stable")
        self.assertEqual(artifact["relative_path"], "Research/A.md")
        stored = json.loads(self.bucket.objects[ARTIFACT_INDEX_PATH])["artifacts"]["note:stable"]
        self.assertNotIn("body", stored)
        self.assertNotIn("markdown", stored)

    def test_body_does_not_affect_taxonomy_or_trigger_a_patch(self):
        path = self.note("known.md", "---\nknowledge_id: note:known\ntype: article\ntags:\n  - creatine\n---\n\nFirst body.\n")
        self.agent.run()
        before = self.registry.artifact_index.get("note:known")["taxonomy_fingerprint"]
        path.write_text("---\nknowledge_id: note:known\ntype: article\ntags:\n  - creatine\n---\n\nChanged body only.\n", encoding="utf-8")

        result = self.agent.run()

        self.assertEqual(result["last_run"]["changed"], 0)
        self.assertEqual(self.registry.artifact_index.get("note:known")["taxonomy_fingerprint"], before)
        self.assertTrue(path.read_text(encoding="utf-8").endswith("Changed body only.\n"))

    def test_malformed_note_does_not_stop_healthy_notes_and_reports_error(self):
        self.note("bad.md", "---\ntags: [\n---\n\nBroken body\n")
        self.note("good.md", "---\nknowledge_id: note:good\ntype: article\ntags:\n  - creatine\n---\n\nGood.\n")

        result = self.agent.run()

        self.assertEqual(result["status"], "Error")
        self.assertEqual(result["last_run"]["errors"], 1)
        self.assertIsNotNone(self.registry.artifact_index.get("note:good"))

    def test_failed_backfill_does_not_stop_the_next_waiting_item(self):
        first = self.note("one.md", "---\nknowledge_id: note:one\ntype: article\ntags:\n  - lengthened-partials\n---\n\nOne.\n")
        second = self.note("two.md", "---\nknowledge_id: note:two\ntype: article\ntags:\n  - lengthened-partials\n---\n\nTwo.\n")
        self.agent.run()
        self.registry.decide("lengthened-partials", "promote")
        self.registry.run_backfill("lengthened-partials")
        first.unlink()

        result = self.agent.run()

        self.assertEqual(result["status"], "Error")
        self.assertGreaterEqual(result["last_run"]["errors"], 1)
        self.assertIn("- lengthened-partials\n", second.read_text(encoding="utf-8"))
        work = self.registry.preview_backfill("lengthened-partials")["backfill"]["local_work_items"]
        self.assertIn("failed", [item["status"] for item in work])
        self.assertIn("completed", [item["status"] for item in work])

    def test_atomic_write_and_single_run_lock(self):
        path = self.note("atomic.md", "old\n")
        with patch("macos_agent.knowledge_sync.os.replace", wraps=os.replace) as replace:
            atomic_write(path, "new\n")
        self.assertEqual(path.read_text(encoding="utf-8"), "new\n")
        replace.assert_called_once()

        lock = SingleRunLock(self.agent.lock_path)
        self.assertTrue(lock.acquire())
        try:
            self.assertEqual(self.agent.run(), {"status": "already_running"})
        finally:
            lock.release()

    def test_heartbeat_and_stale_presentation(self):
        self.note("known.md", "---\nknowledge_id: note:heart\ntype: article\ntags:\n  - sleep\n---\n\nBody.\n")
        result = self.agent.run()
        persisted = json.loads(self.bucket.objects[AGENT_STATUS_PATH])

        self.assertEqual(persisted["status"], "Healthy")
        self.assertEqual(persisted["last_run"], result["last_run"])
        fresh = datetime.fromisoformat(persisted["last_finished_at"].replace("Z", "+00:00"))
        stale = self.status.presented(now=fresh + timedelta(hours=19))
        self.assertEqual(stale["status"], "Stale")

    def test_root_configuration_supports_multiple_explicit_roots(self):
        other = Path(self.directory.name) / "other"
        roots = knowledge_roots_from_env({
            "PODCAST_KNOWLEDGE_ROOTS": f"personal={self.root}{os.pathsep}work={other}",
        })
        self.assertEqual([(root.vault, root.path) for root in roots], [("personal", self.root.resolve()), ("work", other.resolve())])
        with self.assertRaisesRegex(ValueError, "absolute"):
            knowledge_roots_from_env({"PODCAST_KNOWLEDGE_ROOTS": "relative"})

    def test_web_reads_saved_agent_status_without_contacting_the_mac(self):
        with patch("podcast_engine.review_web.KnowledgeAgentStatus") as status:
            status.return_value.presented.return_value = {"status": "Healthy", "last_run": {}}
            client = create_review_app().test_client()
            response = client.get("/api/tag-vocabulary/agent-status")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["agent"]["status"], "Healthy")
        self.assertIn("Mac Knowledge Agent", TAG_PAGE)


if __name__ == "__main__":
    unittest.main()
