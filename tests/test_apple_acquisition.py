import json
import unittest
from unittest.mock import patch

from podcast_engine.apple_acquisition import (
    _source_metadata,
    _validate_episode_identity,
    upload_incoming_apple_source,
)
from podcast_engine.apple_transcript import AppleTranscriptResult
from podcast_engine.episode_identity import episode_key_for


class AppleAcquisitionTests(unittest.TestCase):
    feed_url = "https://example.test/feed.xml"
    rss_guid = "episode-guid-123"

    def _episode(self):
        key = episode_key_for(
            self.feed_url,
            self.rss_guid,
        )

        return {
            "id": key,
            "episode_key": key,
            "podcast": "Test Podcast",
            "podcast_id": "test-podcast",
            "feed_url": self.feed_url,
            "guid": self.rss_guid,
            "title": "Test Episode",
            "published": "Tue, 25 Aug 2026 12:00:00 +0000",
            "link": "https://example.test/episode",
        }

    def _podcast_config(self):
        return {
            "id": "test-podcast",
            "name": "Test Podcast",
            "apple_podcasts": {
                "show_id": "123456789",
                "show_title": "Test Podcast",
                "storefront": "us",
            },
        }

    def _result(self):
        return AppleTranscriptResult(
            status="READY",
            message="Apple Podcasts API transcript downloaded.",
            transcript="First paragraph.\n\nSecond paragraph.",
            segments=[
                {
                    "text": "First paragraph.",
                    "start": 1.0,
                    "end": 2.0,
                },
                {
                    "text": "Second paragraph.",
                    "start": 2.0,
                    "end": 3.0,
                },
            ],
            metadata={
                "source": "apple_podcasts_api",
                "apple_episode_id": "1000000000000",
                "segment_count": 2,
            },
        )

    def test_episode_identity_must_match_feed_and_guid(self):
        episode = self._episode()

        self.assertEqual(
            _validate_episode_identity(episode),
            episode["episode_key"],
        )

        episode["episode_key"] = "0" * 24

        with self.assertRaises(ValueError):
            _validate_episode_identity(episode)

    def test_source_metadata_matches_existing_ingest_contract(self):
        source = _source_metadata(
            podcast_config=self._podcast_config(),
            episode=self._episode(),
            result=self._result(),
        )

        self.assertEqual(
            source["source"],
            "apple_podcasts_api",
        )

        self.assertEqual(
            source["episode"]["rss_guid"],
            self.rss_guid,
        )

        self.assertEqual(
            source["episode"]["apple_episode_id"],
            "1000000000000",
        )

        self.assertEqual(
            len(source["segments"]),
            2,
        )

    def test_upload_writes_json_before_txt_completion_signal(self):
        bucket = _FakeBucket()

        with patch(
            "podcast_engine.apple_acquisition.get_bucket",
            return_value=bucket,
        ):
            outcome = upload_incoming_apple_source(
                podcast_config=self._podcast_config(),
                episode=self._episode(),
                result=self._result(),
            )

        key = self._episode()["episode_key"]

        metadata_path = (
            f"incoming/apple/{key}/apple-transcript.json"
        )

        text_path = (
            f"incoming/apple/{key}/apple-transcript.txt"
        )

        self.assertEqual(
            bucket.upload_order,
            [
                metadata_path,
                text_path,
            ],
        )

        metadata = json.loads(
            bucket.objects[metadata_path]
        )

        self.assertEqual(
            metadata["episode"]["episode_key"],
            key,
        )

        self.assertEqual(
            bucket.objects[text_path],
            "First paragraph.\n\nSecond paragraph.\n",
        )

        self.assertEqual(
            outcome["status"],
            "uploaded",
        )



class BearerTokenEarlyWarningTests(unittest.TestCase):
    """TASK-116: the cloud worker warns while the mounted token still works,
    so a stalled macOS refresh is visible days before acquisition breaks."""

    def _read(self, remaining_seconds):
        import base64, os, tempfile, time
        from podcast_engine import apple_acquisition

        payload = base64.urlsafe_b64encode(
            json.dumps({"exp": time.time() + remaining_seconds}).encode()
        ).decode().rstrip("=")
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
            handle.write(f"h.{payload}.s")
        try:
            with patch.dict(os.environ, {"APPLE_PODCASTS_TOKEN_FILE": handle.name}), patch.object(
                apple_acquisition, "emit_event"
            ) as emit:
                token = apple_acquisition.read_bearer_token()
        finally:
            os.unlink(handle.name)
        return token, emit

    def test_token_inside_warning_window_is_returned_with_expiring_event(self):
        from podcast_engine import apple_acquisition

        token, emit = self._read(apple_acquisition.TOKEN_EARLY_WARNING_SECONDS - 3600)
        self.assertTrue(token.startswith("h."))
        emit.assert_called_once()
        self.assertEqual(emit.call_args.args[0], "apple_token_expiring")
        self.assertEqual(emit.call_args.kwargs["severity"], "WARNING")
        self.assertNotIn(token, json.dumps(emit.call_args.kwargs))

    def test_token_outside_warning_window_emits_nothing(self):
        from podcast_engine import apple_acquisition

        _, emit = self._read(apple_acquisition.TOKEN_EARLY_WARNING_SECONDS + 3600)
        emit.assert_not_called()


class _FakeBlob:
    def __init__(self, bucket, name):
        self.bucket = bucket
        self.name = name

    def upload_from_string(
        self,
        value,
        content_type=None,
    ):
        self.bucket.objects[self.name] = value
        self.bucket.content_types[self.name] = content_type
        self.bucket.upload_order.append(self.name)


class _FakeBucket:
    def __init__(self):
        self.objects = {}
        self.content_types = {}
        self.upload_order = []

    def blob(self, name):
        return _FakeBlob(
            self,
            name,
        )


if __name__ == "__main__":
    unittest.main()
