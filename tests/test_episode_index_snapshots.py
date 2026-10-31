"""Episode-index races, including the SDK's download-generation side effect."""

import base64
import hashlib
from io import BytesIO
import json
import unittest
from unittest.mock import Mock, patch

from google.api_core.exceptions import Forbidden, NotFound, PreconditionFailed
from google.auth.credentials import AnonymousCredentials
from google.cloud import storage as gcs
from requests import Response
from urllib3.response import HTTPResponse

from podcast_engine import apple_ingest, storage
from podcast_engine.episode_contract import new_episode_record, paths_for


def record(key):
    return new_episode_record(
        episode_key=key, podcast="Podcast", podcast_id="podcast",
        feed_url="https://example.test/feed.xml", rss_guid=key,
        title=key, published=None, link=None, audio_url=None,
    )


class IndexBucket:
    """Non-versioned GCS index; replacement removes the old generation."""

    def __init__(self, episodes=None):
        self.content = None if episodes is None else json.dumps(episodes)
        self.generation = 0 if episodes is None else 10
        self.after_download = None
        self.download_error = None
        self.conflicts = 0

    def blob(self, name):
        if name != "episodes.json":
            raise AssertionError(f"Unexpected object: {name}")
        bucket = self

        class Blob:
            generation = None

            def exists(self):
                return bucket.content is not None

            def download_as_text(self, encoding):
                if bucket.download_error:
                    raise bucket.download_error
                # The SDK fills this from the media response, not a later GET.
                self.generation = bucket.generation
                content = bucket.content
                callback, bucket.after_download = bucket.after_download, None
                if callback:
                    callback()
                return content

            def reload(self):
                if self.generation != bucket.generation:
                    raise NotFound("Downloaded generation has been replaced")

            def upload_from_string(self, content, *, content_type, if_generation_match):
                if if_generation_match != bucket.generation:
                    bucket.conflicts += 1
                    raise PreconditionFailed("Index changed")
                bucket.content = content
                bucket.generation += 1

        return Blob()

    def replace(self, episodes):
        self.content = json.dumps(episodes)
        self.generation += 1


class EpisodeIndexSnapshotTests(unittest.TestCase):
    def read_worker(self, bucket):
        with patch.object(storage, "get_bucket", return_value=bucket):
            return storage.load_episodes_with_generation()

    def test_both_readers_keep_bytes_and_generation_when_index_is_replaced(self):
        for reader in (self.read_worker, apple_ingest._load_index):
            with self.subTest(reader=reader.__name__):
                bucket = IndexBucket([record("original")])
                bucket.after_download = lambda: bucket.replace([record("concurrent")])
                episodes, generation = reader(bucket)
                self.assertEqual([e["episode_key"] for e in episodes], ["original"])
                self.assertEqual(generation, 10)
                self.assertEqual(bucket.generation, 11)

    def test_worker_stale_write_retries_and_preserves_concurrent_changes(self):
        bucket = IndexBucket([record("target")])
        latest = [record("target"), record("concurrent")]
        latest[0]["title"] = "Concurrent title"
        bucket.after_download = lambda: bucket.replace(latest)
        with patch.object(storage, "get_bucket", return_value=bucket):
            storage.update_episode("target", download_completed=True)
            storage.update_episode("target", download_completed=True)
        episodes = json.loads(bucket.content)
        self.assertEqual([e["episode_key"] for e in episodes], ["target", "concurrent"])
        self.assertEqual(episodes[0]["title"], "Concurrent title")
        self.assertEqual(episodes[0]["status"]["download"]["state"], "completed")
        self.assertEqual(bucket.conflicts, 1)

    def test_ingest_stale_write_retries_and_duplicate_replay_preserves_other_episode(self):
        key = "a" * 24
        original = record(key)
        bucket = IndexBucket([original])
        bucket.after_download = lambda: bucket.replace([original, record("concurrent")])
        source = {"episode": {"feed_url": original["feed_url"], "podcast_id": "podcast"}}
        for _ in range(2):
            apple_ingest._merge_index(bucket, key, source, paths_for(key))
        episodes = json.loads(bucket.content)
        self.assertEqual([e["episode_key"] for e in episodes], [key, "concurrent"])
        self.assertEqual(episodes[0]["status"]["apple_transcript"]["state"], "ready")
        self.assertEqual(bucket.conflicts, 1)

    def test_missing_index_remains_create_only(self):
        for reader in (self.read_worker, apple_ingest._load_index):
            with self.subTest(reader=reader.__name__):
                bucket = IndexBucket()
                episodes, generation = reader(bucket)
                self.assertEqual((episodes, generation), ([], 0))
                bucket.replace([record("concurrent")])
                with patch.object(storage, "get_bucket", return_value=bucket):
                    with self.assertRaises(PreconditionFailed):
                        storage.save_episodes(episodes, if_generation_match=generation)
                self.assertEqual(json.loads(bucket.content)[0]["episode_key"], "concurrent")

    def test_permission_and_malformed_content_fail_closed(self):
        for reader in (self.read_worker, apple_ingest._load_index):
            for error in (Forbidden("denied"), NotFound("deleted during read")):
                with self.subTest(reader=reader.__name__, error=type(error).__name__):
                    bucket = IndexBucket([])
                    bucket.download_error = error
                    with self.assertRaises(type(error)):
                        reader(bucket)
            with self.subTest(reader=reader.__name__, malformed=True):
                bucket = IndexBucket([])
                bucket.content = "invalid JSON"
                with self.assertRaises(json.JSONDecodeError):
                    reader(bucket)

    def test_real_sdk_download_sets_generation_from_the_same_media_response(self):
        content = b"[]"
        response = Response()
        response.status_code = 200
        response.headers.update({
            "Content-Length": "2", "X-goog-generation": "10",
            "X-Goog-Hash": "md5=" + base64.b64encode(hashlib.md5(content).digest()).decode(),
        })
        response.raw = HTTPResponse(body=BytesIO(content), preload_content=False)
        transport = Mock()
        transport.request.return_value = response
        client = gcs.Client(project="test", credentials=AnonymousCredentials(), _http=transport)
        blob = client.bucket("test-bucket").blob("episodes.json")
        self.assertIsNone(blob.generation)
        self.assertEqual(blob.download_as_text(encoding="utf-8"), "[]")
        self.assertEqual(blob.generation, 10)


if __name__ == "__main__":
    unittest.main()
