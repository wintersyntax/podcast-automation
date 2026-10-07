import unittest

from podcast_engine.apple_ingest import _incoming_key, _record_from_apple_source
from podcast_engine.episode_contract import (
    SCHEMA_VERSION,
    canonical_episode_url,
    ensure_v3_record,
    incoming_apple_paths,
    new_episode_record,
    paths_for,
    summary_is_vault_sync_ready,
)
from podcast_engine.episode_identity import episode_key_for


class EpisodeContractTests(unittest.TestCase):
    feed_url = "https://feeds.example.com/example-strength.xml"
    guid = "9a8bcf36-a027-48bc-976e-d441e201ee45"

    def test_canonical_episode_url_normalizes_only_the_broken_libsyn_pattern(self):
        broken = "https://example-strength.libsyn.com/ep-386-sleep-hypertrophy-and-creatine"
        correct = "https://example-strength.libsyn.com/website/ep-386-sleep-hypertrophy-and-creatine"

        self.assertEqual(canonical_episode_url(broken), correct)
        self.assertEqual(canonical_episode_url(correct), correct)
        self.assertEqual(
            canonical_episode_url("https://example.test/ep-386-sleep-hypertrophy-and-creatine"),
            "https://example.test/ep-386-sleep-hypertrophy-and-creatine",
        )
        self.assertIsNone(canonical_episode_url(None))

    def test_existing_record_is_normalized_in_memory(self):
        record = ensure_v3_record(
            {
                "schema_version": SCHEMA_VERSION,
                "id": "episode-1",
                "link": "https://example-strength.libsyn.com/ep-386-sleep-hypertrophy-and-creatine",
            }
        )

        self.assertEqual(
            record["link"],
            "https://example-strength.libsyn.com/website/ep-386-sleep-hypertrophy-and-creatine",
        )


    def test_podcast_url_is_preserved(self):
        record = new_episode_record(
            episode_key="episode-1",
            podcast="Example Nutrition Podcast",
            podcast_id="example-nutrition",
            feed_url="https://feeds.example.com/example-nutrition.xml",
            podcast_url="https://example.com/nutrition-podcast/",
            rss_guid="guid-1",
            title="Episode",
            published=None,
            link=None,
            audio_url=None,
        )

        self.assertEqual(
            record["podcast_url"],
            "https://example.com/nutrition-podcast/",
        )

        partial_v3 = ensure_v3_record(
            {
                "schema_version": SCHEMA_VERSION,
                "id": "episode-1",
            }
        )
        self.assertIsNone(partial_v3["podcast_url"])

    def test_episode_key_is_stable_and_scoped_to_feed(self):
        first = episode_key_for(self.feed_url, self.guid)
        self.assertEqual(first, episode_key_for(self.feed_url, self.guid))
        self.assertEqual(len(first), 24)
        self.assertNotEqual(first, episode_key_for("https://example.test/feed", self.guid))

    def test_incoming_and_canonical_paths_share_the_same_key(self):
        key = episode_key_for(self.feed_url, self.guid)
        paths = paths_for(key)
        self.assertEqual(
            incoming_apple_paths(key)["text"],
            f"incoming/apple/{key}/apple-transcript.txt",
        )
        self.assertEqual(paths, {
            "apple_text": f"episodes/{key}/sources/apple/transcript.txt",
            "apple_metadata": f"episodes/{key}/sources/apple/transcript.json",
            "whisper_text": f"episodes/{key}/sources/whisper/transcript.txt",
            "whisper_metadata": f"episodes/{key}/sources/whisper/transcript.json",
            "compiled_text": f"episodes/{key}/compiled/transcript.txt",
            "compiler_report": f"episodes/{key}/compiled/report.json",
            "resolver_record": f"episodes/{key}/review/resolver.json",
            "summary_body": f"episodes/{key}/summary/body.md",
            "summary_metadata": f"episodes/{key}/summary/metadata.json",
            "summary": f"episodes/{key}/summary/summary.md",
        })
        self.assertEqual(_incoming_key(incoming_apple_paths(key)["text"]), key)
        self.assertIsNone(_incoming_key(f"incoming/apple/{key}/apple-transcript.json"))

    def test_apple_source_creates_a_cloud_owned_ready_record(self):
        key = episode_key_for(self.feed_url, self.guid)
        source = {
            "episode": {
                "episode_key": key,
                "podcast": "Example Strength Podcast",
                "podcast_id": "example-strength",
                "feed_url": self.feed_url,
                "rss_guid": self.guid,
                "title": "Ep 384 - Is Ultrasound Lying About Your Gains?",
                "published": "Tue, 12 Aug 2026 14:00:00 +0000",
            }
        }
        record = _record_from_apple_source(source, key)
        self.assertEqual(record["episode_key"], key)
        self.assertEqual(record["status"]["apple_transcript"]["state"], "ready")
        self.assertEqual(record["status"]["compiler"]["state"], "blocked")
        self.assertEqual(record["files"]["sources"]["apple"]["text"], paths_for(key)["apple_text"])

    def test_new_record_has_only_v3_canonical_state_and_files(self):
        record = new_episode_record(
            episode_key="episode-1",
            podcast="Podcast",
            podcast_id="podcast",
            feed_url=self.feed_url,
            rss_guid=self.guid,
            title="Episode",
            published=None,
            link=None,
            audio_url=None,
        )

        self.assertEqual(record["schema_version"], SCHEMA_VERSION)
        self.assertIsNone(record["podcast_url"])
        self.assertIsNone(record["apple_transcript_late_notified_at"])
        self.assertEqual(
            set(record["status"]),
            {"download", "apple_transcript", "whisper", "compiler", "summary"},
        )
        self.assertNotIn("export", record["status"])
        self.assertNotIn("transcript", record["files"])
        self.assertNotIn("markdown", record["files"])
        self.assertNotIn("audio", record["files"])
        self.assertNotIn("export", record["files"])
        self.assertEqual(
            set(record["files"]["summary"]), {"body", "metadata", "markdown"}
        )

    def test_vault_sync_readiness_accepts_ready_and_compatible_completed_only(self):
        episode = {"status": {"summary": {"state": "pending"}}}

        self.assertFalse(summary_is_vault_sync_ready(episode))

        for state in ("ready", "completed"):
            with self.subTest(state=state):
                episode["status"]["summary"]["state"] = state
                self.assertTrue(summary_is_vault_sync_ready(episode))

    def test_historical_schema_is_rejected(self):
        legacy = {
            "id": "episode-1",
            "status": {
                "detected": True,
                "downloaded": True,
                "transcribed": True,
                "summarized": True,
            },
            "files": {
                "transcript": "sources/whisper/transcript.txt",
                "markdown": "summary/summary.md",
            },
        }

        for schema_version in (1, 2):
            with self.subTest(schema_version=schema_version):
                legacy["schema_version"] = schema_version
                with self.assertRaisesRegex(
                    ValueError,
                    "Unsupported Episode Contract schema version .*; expected 3",
                ):
                    ensure_v3_record(legacy)

    def test_v3_record_with_obsolete_alias_is_rejected(self):
        record = new_episode_record(
            episode_key="episode-1",
            podcast="Podcast",
            podcast_id="podcast",
            feed_url=self.feed_url,
            rss_guid=self.guid,
            title="Episode",
            published=None,
            link=None,
            audio_url=None,
        )
        record["status"]["downloaded"] = True

        with self.assertRaisesRegex(ValueError, "obsolete status field 'downloaded'"):
            ensure_v3_record(record)

if __name__ == "__main__":
    unittest.main()
