import json
import unittest
from unittest.mock import patch

from google.api_core.exceptions import PreconditionFailed

from podcast_engine.knowledge import tags
from podcast_engine.knowledge.tags import REGISTRY_PATH, TagRegistry


class _Blob:
    def __init__(self, bucket, name):
        self.bucket, self.name = bucket, name

    @property
    def generation(self):
        return self.bucket.generations.get(self.name, 0)

    def exists(self):
        return self.name in self.bucket.objects

    def reload(self):
        return None

    def download_as_text(self, encoding="utf-8"):
        return self.bucket.objects[self.name]

    def upload_from_string(self, value, content_type=None, if_generation_match=None):
        if self.bucket.fail_next_registry_write and self.name == REGISTRY_PATH:
            self.bucket.fail_next_registry_write = False
            raise PreconditionFailed("simulated concurrent update")
        if if_generation_match is not None and if_generation_match != self.generation:
            raise PreconditionFailed("generation mismatch")
        self.bucket.objects[self.name] = value
        self.bucket.generations[self.name] = self.generation + 1
        self.bucket.writes.append(self.name)


class _Bucket:
    def __init__(self):
        self.objects, self.generations, self.writes = {}, {}, []
        self.fail_next_registry_write = False

    def blob(self, name):
        return _Blob(self, name)


def _episode(key):
    return {
        "episode_key": key,
        "podcast": "Example Strength Podcast",
        "podcast_id": "example-strength",
        "title": f"Episode {key}",
        "status": {"compiler": {"state": "completed"}},
    }


def _artifacts(bucket, episode, tags=None):
    root = f"episodes/{episode['episode_key']}/summary"
    body = "# Existing summary\n\nThis exact body must remain.\n"
    manifest = {
        "schema_version": 1,
        "episode_key": episode["episode_key"],
        "summary": {"generated_at": "2026-08-28T00:00:00+00:00"},
        "metadata": {
            "topics": ["Free-form topic"], "people": ["Alice"],
            "tags": tags if tags is not None else ["training-volume"], "tag_candidates": [],
        },
    }
    bucket.objects[f"{root}/body.md"] = body
    bucket.objects[f"{root}/metadata.json"] = json.dumps(manifest) + "\n"
    bucket.objects[f"{root}/summary.md"] = "unchanged-before-explicit-backfill"
    for name in list(bucket.objects):
        bucket.generations.setdefault(name, 1)


class TagVocabularyTests(unittest.TestCase):
    def setUp(self):
        self.bucket = _Bucket()
        self.episodes = [_episode("episode-a"), _episode("episode-b")]
        for episode in self.episodes:
            _artifacts(self.bucket, episode)
        self.service = TagRegistry(bucket=self.bucket, episodes_loader=lambda: self.episodes)

    def test_registry_current_schema_constant_drives_storage_contract(self):
        future = tags.REGISTRY_SCHEMA_VERSION + 7
        with patch.object(tags, "REGISTRY_SCHEMA_VERSION", future):
            registry = tags.initial_registry()
            self.assertEqual(registry["schema_version"], future)
            self.assertTrue(tags._is_current_registry(registry))

    def test_artifact_index_current_schema_constant_drives_storage_contract(self):
        future = tags.ARTIFACT_INDEX_SCHEMA_VERSION + 7
        with patch.object(tags, "ARTIFACT_INDEX_SCHEMA_VERSION", future):
            bucket = _Bucket()
            service = tags.ArtifactIndex(bucket=bucket)
            first = service.load()
            self.assertEqual(first["schema_version"], future)
            self.assertEqual(service.load()["schema_version"], future)

    def test_agent_status_current_schema_constant_drives_storage_contract(self):
        future = 9
        with patch.object(
            tags,
            "AGENT_STATUS_SCHEMA_VERSION",
            future,
            create=True,
        ):
            bucket = _Bucket()
            service = tags.KnowledgeAgentStatus(bucket=bucket)
            first = service.load()
            self.assertEqual(first["schema_version"], future)
            self.assertEqual(service.load()["schema_version"], future)

    def _candidate(self, slug="lengthened-partials"):
        return self.service.resolve_and_record(
            "episode-a",
            {
                "topics": ["Specific free-form topic"], "people": ["Alice"],
                "existing_tags": ["weekly volume"],
                "new_tag_candidates": [{"tag": slug, "category": "training"}],
            },
        )

    def _summary_snapshot(self):
        return {
            name: value for name, value in self.bucket.objects.items()
            if name.startswith("episodes/")
        }

    def test_registry_initialization_is_conditional_and_contains_small_initial_vocabulary(self):
        registry = self.service.load()
        self.assertIn(REGISTRY_PATH, self.bucket.objects)
        self.assertEqual(
            registry["schema_version"],
            tags.REGISTRY_SCHEMA_VERSION,
        )
        self.assertEqual(
            registry["canonical_tags"]["training-volume"]["aliases"],
            ["volume", "weekly volume"],
        )

    def test_resolves_canonical_and_alias_and_never_promotes_unknown_existing_tag(self):
        result = self._candidate()
        self.assertEqual(result["tags"], ["training-volume"])
        self.assertEqual(result["tag_candidates"], ["lengthened-partials"])
        unknown = self.service.resolve_and_record(
            "episode-b",
            {"topics": [], "people": [], "existing_tags": ["invented tag"], "new_tag_candidates": []},
        )
        self.assertEqual(unknown["tags"], [])
        self.assertEqual(unknown["unknown_existing_tags"], ["invented tag"])

    def test_candidate_occurrence_is_unique_per_episode_and_rejected_candidate_is_retained(self):
        self._candidate()
        self._candidate()
        self.service.resolve_and_record(
            "episode-b",
            {"topics": [], "people": [], "existing_tags": [], "new_tag_candidates": [{"tag": "lengthened partials", "category": "training"}]},
        )
        entry = self.service.list_candidates("pending")[0]
        self.assertEqual(entry["occurrences"], 2)
        self.assertEqual(entry["artifact_refs"], [
            {"artifact_id": "podcast:episode-a", "source_type": "podcast"},
            {"artifact_id": "podcast:episode-b", "source_type": "podcast"},
        ])
        self.assertNotIn("episode_keys", entry)
        self.service.decide("lengthened-partials", "reject")
        self.assertEqual(self.service.list_candidates("rejected")[0]["slug"], "lengthened-partials")

    def test_review_decisions_never_write_episode_artifacts(self):
        actions = [
            ("promote", {}),
            ("map", {"mapped_to": "hypertrophy"}),
            ("reject", {}),
            ("reopen", {}),
        ]
        for action, kwargs in actions:
            with self.subTest(action=action):
                self.setUp()
                self._candidate()
                if action == "reopen":
                    self.service.decide("lengthened-partials", "promote")
                before = self._summary_snapshot()
                result = self.service.decide("lengthened-partials", action, **kwargs)
                self.assertEqual(self._summary_snapshot(), before)
                self.assertEqual(result["entry"]["backfill"]["status"], "pending" if action != "reject" else "not_needed")

    def test_change_from_completed_promote_to_map_creates_corrective_pending_without_writing(self):
        self._candidate()
        self.service.decide("lengthened-partials", "promote")
        self.service.run_backfill("lengthened-partials")
        before_change = self._summary_snapshot()
        changed = self.service.decide("lengthened-partials", "map", mapped_to="hypertrophy")
        self.assertEqual(self._summary_snapshot(), before_change)
        self.assertEqual(changed["entry"]["backfill"]["status"], "pending")
        self.assertEqual(changed["entry"]["backfill"]["obsolete_tags"], ["lengthened-partials"])
        self.service.run_backfill("lengthened-partials")
        manifest = json.loads(self.bucket.objects["episodes/episode-a/summary/metadata.json"])
        self.assertEqual(manifest["metadata"]["tags"], ["training-volume", "hypertrophy"])
        self.assertEqual(self.bucket.objects["episodes/episode-a/summary/body.md"], "# Existing summary\n\nThis exact body must remain.\n")

    def test_reject_after_completed_promotion_schedules_corrective_removal(self):
        self._candidate()
        self.service.decide("lengthened-partials", "promote")
        self.service.run_backfill("lengthened-partials")
        before_reject = self._summary_snapshot()
        rejected = self.service.decide("lengthened-partials", "reject")
        self.assertEqual(self._summary_snapshot(), before_reject)
        self.assertEqual(rejected["entry"]["backfill"]["status"], "pending")
        self.assertEqual(rejected["entry"]["backfill"]["target_tag"], None)
        self.assertEqual(rejected["entry"]["backfill"]["obsolete_tags"], ["lengthened-partials"])
        self.service.run_backfill("lengthened-partials")
        manifest = json.loads(self.bucket.objects["episodes/episode-a/summary/metadata.json"])
        self.assertEqual(manifest["metadata"]["tags"], ["training-volume"])

    def test_explicit_backfill_is_idempotent_and_preserves_unrelated_metadata(self):
        self._candidate()
        decision = self.service.decide("lengthened-partials", "promote")
        self.assertEqual(decision["entry"]["backfill"]["status"], "pending")
        preview = self.service.preview_backfill("lengthened-partials")
        self.assertEqual([row["episode_key"] for row in preview["episodes"]], ["episode-a"])
        self.service.run_backfill("lengthened-partials")
        first = self.bucket.objects["episodes/episode-a/summary/summary.md"]
        self.service.run_backfill("lengthened-partials")
        manifest = json.loads(self.bucket.objects["episodes/episode-a/summary/metadata.json"])
        self.assertEqual(manifest["metadata"]["tags"], ["training-volume", "lengthened-partials"])
        self.assertEqual(manifest["metadata"]["topics"], ["Free-form topic"])
        self.assertEqual(manifest["metadata"]["people"], ["Alice"])
        self.assertEqual(self.bucket.objects["episodes/episode-a/summary/summary.md"], first)

    def test_partial_failure_is_recorded_without_sensitive_details_and_rerun_completes(self):
        self._candidate()
        self.service.resolve_and_record(
            "episode-b",
            {"topics": [], "people": [], "existing_tags": [], "new_tag_candidates": [{"tag": "lengthened-partials", "category": "training"}]},
        )
        self.service.decide("lengthened-partials", "promote")
        del self.bucket.objects["episodes/episode-b/summary/metadata.json"]
        failed = self.service.run_backfill("lengthened-partials")["backfill"]
        self.assertEqual(failed["status"], "failed")
        self.assertNotIn("Bearer", failed["last_error"])
        _artifacts(self.bucket, self.episodes[1])
        completed = self.service.run_backfill("lengthened-partials")["backfill"]
        self.assertEqual(completed["status"], "completed")
        self.assertEqual(completed["completed_episodes"], 2)

    def test_registry_retries_a_conditional_write_conflict(self):
        self.service.load()
        self.bucket.fail_next_registry_write = True
        result = self._candidate("new-concept")
        self.assertEqual(result["tag_candidates"], ["new-concept"])
        self.assertIn("new-concept", self.service.load()["candidates"])

    def test_invalid_map_target_does_not_corrupt_the_registry(self):
        self._candidate()
        before = self.service.load()
        with self.assertRaisesRegex(ValueError, "existing canonical"):
            self.service.decide("lengthened-partials", "map", mapped_to="does-not-exist")
        self.assertEqual(self.service.load(), before)


if __name__ == "__main__":
    unittest.main()
