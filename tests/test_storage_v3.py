import unittest
from unittest.mock import patch

from podcast_engine.episode_contract import new_episode_record
from podcast_engine.episode_identity import episode_key_for
from podcast_engine.storage import add_episode, update_episode


class StorageV3Tests(unittest.TestCase):
    def test_add_episode_scopes_duplicate_detection_to_the_canonical_key(self):
        feed_a = "https://example.test/feed-a.xml"
        feed_b = "https://example.test/feed-b.xml"
        guid = "shared-guid"
        key_a = episode_key_for(feed_a, guid)
        key_b = episode_key_for(feed_b, guid)
        existing = new_episode_record(
            episode_key=key_a,
            podcast="Feed A",
            podcast_id="feed-a",
            feed_url=feed_a,
            rss_guid=guid,
            title="Feed A episode",
            published=None,
            link=None,
            audio_url=None,
        )
        saved = []
        source = {
            "podcast": "Feed B",
            "podcast_id": "feed-b",
            "feed_url": feed_b,
            "guid": guid,
            "title": "Feed B episode",
        }

        with (
            patch(
                "podcast_engine.storage.load_episodes_with_generation",
                return_value=([existing], 1),
            ),
            patch(
                "podcast_engine.storage.save_episodes",
                side_effect=lambda episodes, **_: saved.extend(episodes),
            ),
        ):
            added = add_episode(source)

        self.assertNotEqual(key_a, key_b)
        self.assertEqual(added["episode_key"], key_b)
        self.assertEqual([item["episode_key"] for item in saved], [key_a, key_b])

    def test_update_episode_writes_only_canonical_v3_fields(self):
        record = new_episode_record(
            episode_key="episode-1",
            podcast="Podcast",
            podcast_id="podcast",
            feed_url="https://example.test/feed.xml",
            rss_guid="guid-1",
            title="Episode",
            published=None,
            link=None,
            audio_url=None,
        )
        saved = []

        with (
            patch("podcast_engine.storage.load_episodes_with_generation", return_value=([record], 1)),
            patch("podcast_engine.storage.save_episodes", side_effect=lambda episodes, **_: saved.extend(episodes)),
        ):
            updated = update_episode(
                "episode-1",
                transcript_file="canonical-whisper.txt",
                whisper_metadata_file="canonical-whisper.json",
                compiled_transcript_file="compiled.txt",
                compiler_report_file="report.json",
                compiler_review_required=False,
                summary_body_file="body.md",
                summary_metadata_file="metadata.json",
                markdown_file="summary.md",
                apple_transcript_late_notified_at=(
                    "2026-08-30T12:00:00+00:00"
                ),
            )

        self.assertEqual(updated["schema_version"], 3)
        self.assertEqual(updated["files"]["sources"]["whisper"]["text"], "canonical-whisper.txt")
        self.assertEqual(updated["files"]["compiled"]["transcript"], "compiled.txt")
        self.assertEqual(updated["files"]["summary"]["markdown"], "summary.md")
        self.assertEqual(updated["status"]["compiler"]["state"], "completed")
        self.assertEqual(updated["status"]["summary"]["state"], "ready")
        self.assertEqual(
            updated["apple_transcript_late_notified_at"],
            "2026-08-30T12:00:00+00:00",
        )
        self.assertEqual(
            set(updated["status"]),
            {"download", "apple_transcript", "whisper", "compiler", "summary"},
        )
        self.assertEqual(set(updated["files"]), {"sources", "compiled", "summary"})
        self.assertEqual(saved[0], updated)

    def test_update_episode_uses_only_the_canonical_episode_key(self):
        record = new_episode_record(
            episode_key="episode-1",
            podcast="Podcast",
            podcast_id="podcast",
            feed_url="https://example.test/feed.xml",
            rss_guid="guid-1",
            title="Shared episode title",
            published=None,
            link=None,
            audio_url=None,
        )
        record["id"] = "legacy-id"

        with (
            patch(
                "podcast_engine.storage.load_episodes_with_generation",
                return_value=([record], 1),
            ),
            patch("podcast_engine.storage.save_episodes") as save_episodes,
        ):
            self.assertIsNone(update_episode("Shared episode title", download_completed=True))
            self.assertIsNone(update_episode("legacy-id", download_completed=True))
            updated = update_episode("episode-1", download_completed=True)

        self.assertEqual(updated["status"]["download"]["state"], "completed")
        save_episodes.assert_called_once()


if __name__ == "__main__":
    unittest.main()
