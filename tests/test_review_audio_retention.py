"""TASK-144: the stored review audio is deleted once an episode is finished."""

import unittest

from google.api_core.exceptions import NotFound

from podcast_engine.episode_contract import new_episode_record
from podcast_engine.review_audio_retention import (
    RETAINED_EPISODES,
    apply_review_audio_retention,
    retention_candidates,
)


def episode(key, *, compiler="completed", summary="ready", published="Mon, 05 Oct 2026 12:00:00 +0000", **extra):
    record = new_episode_record(
        episode_key=key, podcast="P", podcast_id="p", feed_url="https://example.test/feed.xml",
        rss_guid=key, title=key, published=published, link=None, audio_url=None,
    )
    record["status"]["compiler"]["state"] = compiler
    record["status"]["summary"]["state"] = summary
    record.update(extra)
    return record


class FakeBucket:
    def __init__(self, missing=()):
        self.deleted, self.missing = [], set(missing)

    def blob(self, name):
        bucket = self

        class Blob:
            def delete(self):
                if name in bucket.missing:
                    raise NotFound(name)
                bucket.deleted.append(name)

        return Blob()


class ReviewAudioRetentionTests(unittest.TestCase):
    def test_user_decision_keeps_no_finished_copy(self):
        self.assertEqual(RETAINED_EPISODES, 0)

    def test_only_finished_episodes_lose_their_audio(self):
        episodes = [
            episode("done"),
            episode("review", compiler="review_required", summary="pending"),
            episode("blocked", compiler="blocked", summary="pending"),
            episode("copy", compiler="blocked", summary="pending", duplicate_check={"result": "duplicate"}),
            episode("done-pending-summary", summary="pending"),
            episode("already", review_audio_deleted_at="2026-10-12T00:00:00+00:00"),
        ]
        self.assertEqual([e["episode_key"] for e in retention_candidates(episodes)], ["done", "copy"])

    def test_retained_count_keeps_the_most_recent_finished_episodes(self):
        episodes = [
            episode("old", published="Mon, 01 Jun 2026 12:00:00 +0000"),
            episode("new", published="Mon, 05 Oct 2026 12:00:00 +0000"),
        ]
        self.assertEqual([e["episode_key"] for e in retention_candidates(episodes, keep=1)], ["old"])

    def test_deletes_once_and_records_it_even_when_the_object_is_missing(self):
        bucket = FakeBucket(missing={"episodes/old/sources/whisper/audio.mp3"})
        updates = []
        deleted = apply_review_audio_retention(
            [episode("done"), episode("old"), episode("review", compiler="review_required", summary="pending")],
            bucket=bucket,
            update_episode=lambda key, **fields: updates.append((key, fields)),
        )
        self.assertEqual(sorted(deleted), ["done", "old"])
        self.assertEqual(bucket.deleted, ["episodes/done/sources/whisper/audio.mp3"])
        self.assertEqual(sorted(key for key, _ in updates), ["done", "old"])
        self.assertTrue(all("review_audio_deleted_at" in fields for _, fields in updates))


if __name__ == "__main__":
    unittest.main()
