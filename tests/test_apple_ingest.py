import json
import unittest
from unittest.mock import patch

from google.api_core.exceptions import NotFound, PreconditionFailed

from podcast_engine import apple_ingest
from podcast_engine.apple_ingest import _record_from_apple_source
from podcast_engine.episode_contract import incoming_apple_paths, paths_for
from podcast_engine.episode_identity import episode_key_for


class AppleIngestTests(unittest.TestCase):
    feed_url = "https://feeds.example.com/example-strength.xml"
    guid = "9a8bcf36-a027-48bc-976e-d441e201ee45"

    def test_gcs_completion_ingests_both_apple_files_and_updates_index(self):
        key = episode_key_for(self.feed_url, self.guid)
        incoming = incoming_apple_paths(key)
        source = {
            "source": "apple_podcasts_api",
            "episode": self._apple_episode(key),
        }
        bucket = _FakeBucket(
            {
                incoming["metadata"]: json.dumps(source),
                incoming["text"]: "Apple transcript text\n",
            }
        )
        original_client = apple_ingest.storage.Client
        apple_ingest.storage.Client = lambda: _FakeClient(bucket)
        try:
            outcome = apple_ingest.ingest_apple_transcript("test-bucket", incoming["text"])
        finally:
            apple_ingest.storage.Client = original_client
        destination = paths_for(key)
        self.assertEqual(outcome, {"status": "ingested", "episode_key": key})
        self.assertEqual(bucket.objects[destination["apple_text"]], "Apple transcript text\n")
        self.assertEqual(
            json.loads(bucket.objects[destination["apple_metadata"]]), source,
        )
        self.assertEqual(
            json.loads(bucket.objects["episodes.json"])[0]["status"]["apple_transcript"]["state"],
            "ready",
        )
        self.assertNotIn(incoming["metadata"], bucket.objects)
        self.assertNotIn(incoming["text"], bucket.objects)
        self.assertNotIn(f"episodes/{key}/metadata.json", bucket.objects)
        self.assertNotIn(f"episodes/{key}/status.json", bucket.objects)

    def test_duplicate_completion_after_cleanup_is_already_ingested(self):
        key = episode_key_for(self.feed_url, self.guid)
        incoming = incoming_apple_paths(key)
        source = {"episode": self._apple_episode(key)}
        bucket = _FakeBucket(
            {
                incoming["metadata"]: json.dumps(source),
                incoming["text"]: "Apple transcript text\n",
            }
        )
        original_client = apple_ingest.storage.Client
        apple_ingest.storage.Client = lambda: _FakeClient(bucket)
        try:
            first_outcome = apple_ingest.ingest_apple_transcript(
                "test-bucket", incoming["text"]
            )
            index_after_first_ingest = bucket.objects["episodes.json"]
            outcome = apple_ingest.ingest_apple_transcript("test-bucket", incoming["text"])
        finally:
            apple_ingest.storage.Client = original_client

        self.assertEqual(first_outcome, {"status": "ingested", "episode_key": key})
        self.assertEqual(outcome, {"status": "already_ingested", "episode_key": key})
        self.assertEqual(bucket.objects["episodes.json"], index_after_first_ingest)

    def test_missing_metadata_before_canonical_ingest_still_fails(self):
        key = episode_key_for(self.feed_url, self.guid)
        incoming = incoming_apple_paths(key)
        bucket = _FakeBucket({incoming["text"]: "Apple transcript text\n"})
        original_client = apple_ingest.storage.Client
        apple_ingest.storage.Client = lambda: _FakeClient(bucket)
        try:
            with self.assertRaisesRegex(
                ValueError, "Apple completion arrived without apple-transcript.json"
            ):
                apple_ingest.ingest_apple_transcript("test-bucket", incoming["text"])
        finally:
            apple_ingest.storage.Client = original_client

    def test_failed_index_update_keeps_incoming_staging_pair(self):
        key = episode_key_for(self.feed_url, self.guid)
        incoming = incoming_apple_paths(key)
        source = {"episode": self._apple_episode(key)}
        bucket = _FakeBucket(
            {
                incoming["metadata"]: json.dumps(source),
                incoming["text"]: "Apple transcript text\n",
            }
        )
        original_client = apple_ingest.storage.Client
        apple_ingest.storage.Client = lambda: _FakeClient(bucket)
        try:
            with patch.object(apple_ingest, "_save_index", side_effect=RuntimeError("index failed")):
                with self.assertRaisesRegex(RuntimeError, "index failed"):
                    apple_ingest.ingest_apple_transcript("test-bucket", incoming["text"])
        finally:
            apple_ingest.storage.Client = original_client

        self.assertIn(incoming["metadata"], bucket.objects)
        self.assertIn(incoming["text"], bucket.objects)

    def test_ready_index_without_canonical_blob_is_not_already_ingested(self):
        key = episode_key_for(self.feed_url, self.guid)
        incoming = incoming_apple_paths(key)
        destination = paths_for(key)
        record = _record_from_apple_source(
            {"episode": self._apple_episode(key)}, key
        )
        bucket = _FakeBucket(
            {
                "episodes.json": json.dumps([record]),
                destination["apple_metadata"]: json.dumps({"episode": self._apple_episode(key)}),
            }
        )
        original_client = apple_ingest.storage.Client
        apple_ingest.storage.Client = lambda: _FakeClient(bucket)
        try:
            with self.assertRaisesRegex(
                ValueError, "Apple completion arrived without apple-transcript.json"
            ):
                apple_ingest.ingest_apple_transcript("test-bucket", incoming["text"])
        finally:
            apple_ingest.storage.Client = original_client

    def test_cleanup_does_not_delete_a_newer_staging_object(self):
        key = episode_key_for(self.feed_url, self.guid)
        incoming = incoming_apple_paths(key)
        source = {"episode": self._apple_episode(key)}
        bucket = _FakeBucket(
            {
                incoming["metadata"]: json.dumps(source),
                incoming["text"]: "Original transcript text\n",
            }
        )
        real_merge_index = apple_ingest._merge_index

        def merge_then_replace_text(*args, **kwargs):
            result = real_merge_index(*args, **kwargs)
            bucket.objects[incoming["text"]] = "New transcript text\n"
            bucket.generations[incoming["text"]] += 1
            return result

        original_client = apple_ingest.storage.Client
        apple_ingest.storage.Client = lambda: _FakeClient(bucket)
        try:
            with patch.object(
                apple_ingest, "_merge_index", side_effect=merge_then_replace_text
            ):
                outcome = apple_ingest.ingest_apple_transcript(
                    "test-bucket", incoming["text"]
                )
        finally:
            apple_ingest.storage.Client = original_client

        self.assertEqual(outcome, {"status": "ingested", "episode_key": key})
        self.assertEqual(bucket.objects[incoming["text"]], "New transcript text\n")
        self.assertNotIn(incoming["metadata"], bucket.objects)

    def test_apple_ingest_keeps_same_guid_from_different_feeds_separate(self):
        feed_b = "https://example.test/other-feed.xml"
        key_a = episode_key_for(self.feed_url, self.guid)
        key_b = episode_key_for(feed_b, self.guid)
        incoming_a = incoming_apple_paths(key_a)
        incoming_b = incoming_apple_paths(key_b)
        source_a = {
            "episode": {
                "episode_key": key_a,
                "podcast": "Feed A",
                "podcast_id": "feed-a",
                "feed_url": self.feed_url,
                "rss_guid": self.guid,
                "title": "Episode A",
            }
        }
        source_b = {
            "episode": {
                "episode_key": key_b,
                "podcast": "Feed B",
                "podcast_id": "feed-b",
                "feed_url": feed_b,
                "rss_guid": self.guid,
                "title": "Episode B",
            }
        }
        bucket = _FakeBucket(
            {
                incoming_a["metadata"]: json.dumps(source_a),
                incoming_a["text"]: "Apple transcript A\n",
                incoming_b["metadata"]: json.dumps(source_b),
                incoming_b["text"]: "Apple transcript B\n",
            }
        )
        original_client = apple_ingest.storage.Client
        apple_ingest.storage.Client = lambda: _FakeClient(bucket)
        try:
            ingest_a = apple_ingest.ingest_apple_transcript(
                "test-bucket", incoming_a["text"]
            )
            ingest_b = apple_ingest.ingest_apple_transcript(
                "test-bucket", incoming_b["text"]
            )
        finally:
            apple_ingest.storage.Client = original_client

        records = json.loads(bucket.objects["episodes.json"])
        self.assertNotEqual(key_a, key_b)
        self.assertEqual(ingest_a["episode_key"], key_a)
        self.assertEqual(ingest_b["episode_key"], key_b)
        self.assertEqual(
            {record["episode_key"] for record in records}, {key_a, key_b}
        )

    def _apple_episode(self, key):
        return {
            "episode_key": key,
            "podcast": "Example Strength Podcast",
            "podcast_id": "example-strength",
            "feed_url": self.feed_url,
            "rss_guid": self.guid,
            "title": "Ep 384 - Is Ultrasound Lying About Your Gains?",
        }


class _FakeBlob:
    def __init__(self, bucket, name):
        self.bucket = bucket
        self.name = name
        self.generation = bucket.generations.get(name, 0)

    def exists(self):
        return self.name in self.bucket.objects

    def download_as_text(self, encoding="utf-8"):
        return self.bucket.objects[self.name]

    def reload(self):
        self.generation = self.bucket.generations.get(self.name, 0)

    def upload_from_string(self, value, content_type=None, if_generation_match=None):
        current_generation = self.bucket.generations.get(self.name, 0)
        if if_generation_match is not None:
            assert if_generation_match == current_generation
        self.bucket.objects[self.name] = value
        self.bucket.generations[self.name] = current_generation + 1
        self.generation = self.bucket.generations[self.name]

    def delete(self, if_generation_match=None):
        current_generation = self.bucket.generations.get(self.name, 0)
        if self.name not in self.bucket.objects:
            raise NotFound("object is missing")
        if if_generation_match is not None and if_generation_match != current_generation:
            raise PreconditionFailed("generation mismatch")
        del self.bucket.objects[self.name]
        self.bucket.generations[self.name] = current_generation + 1


class _FakeBucket:
    def __init__(self, objects):
        self.objects = dict(objects)
        self.generations = {name: 1 for name in self.objects}

    def blob(self, name):
        return _FakeBlob(self, name)

    def copy_blob(self, source_blob, destination_bucket, new_name, source_generation=None):
        if source_generation is not None and source_generation != source_blob.generation:
            raise PreconditionFailed("source generation mismatch")
        destination_bucket.objects[new_name] = source_blob.download_as_text()
        destination_bucket.generations[new_name] = destination_bucket.generations.get(new_name, 0) + 1


class _FakeClient:
    def __init__(self, bucket):
        self._bucket = bucket

    def bucket(self, name):
        return self._bucket


if __name__ == "__main__":
    unittest.main()
