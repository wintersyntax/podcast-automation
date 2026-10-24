"""TASK-135: republished episodes are recognized by transcript content."""

import random
import unittest
from copy import deepcopy
from unittest.mock import patch

from podcast_engine import episode_duplicates as dup
from podcast_engine.episode_contract import new_episode_record
from podcast_engine.pipeline import _tracked_episodes_to_resume, resume_tracked_episode

VOCABULARY = (
    "protein muscle training volume sets reps load hypertrophy strength diet calories "
    "carbohydrate fat sleep recovery study evidence research data effect athletes "
    "lifters week day meal intake gram per kilogram body weight energy deficit "
    "surplus maintenance coach client program progression failure proximity rest "
    "interval session frequency split upper lower legs back chest arms the a of to "
    "and in that is it you we they i so but like know think really just about "
    "what when how why because if then there this those these some more most less"
).split()

PODCAST = {
    "id": "example-strength-podcast",
    "name": "Example Strength Podcast",
    "rss": "https://example.test/iron.xml",
}


def conversation(seed: int, words: int = 3000) -> list[str]:
    rng = random.Random(seed)
    return [rng.choice(VOCABULARY) for _ in range(words)]


def asr_variant(words: list[str], seed: int, *, error_rate: float = 0.05) -> str:
    """The same speech as another ASR model would write it."""

    rng = random.Random(seed)
    out = []
    for word in words:
        roll = rng.random()
        if roll < error_rate / 2:
            continue
        if roll < error_rate:
            out.append(rng.choice(VOCABULARY))
            continue
        out.append(word.capitalize() if rng.random() < 0.1 else word)
        if rng.random() < 0.05:
            out[-1] += ","
    return " ".join(out)


def episode(key, *, podcast="Example Strength Podcast", feed="https://example.test/iron.xml",
            published, whisper=True, apple=False, compiler="blocked"):
    record = new_episode_record(
        episode_key=key,
        podcast=podcast,
        podcast_id=podcast.lower().replace(" ", "-"),
        feed_url=feed,
        rss_guid=f"guid-{key}",
        title=f"Title {key}",
        published=published,
        link=None,
        audio_url="https://example.test/audio.mp3",
    )
    if whisper:
        record["status"]["whisper"]["state"] = "ready"
        record["files"]["sources"]["whisper"]["text"] = f"episodes/{key}/sources/whisper/transcript.txt"
    if apple:
        record["status"]["apple_transcript"]["state"] = "ready"
        record["files"]["sources"]["apple"]["text"] = f"episodes/{key}/sources/apple/transcript.txt"
    record["status"]["compiler"]["state"] = compiler
    return record


class ContainmentTests(unittest.TestCase):
    def test_trimmed_copy_from_another_asr_model_reaches_the_threshold(self):
        original = conversation(1)
        trimmed = original[300:2700]
        score, words = dup.containment(asr_variant(trimmed, 2), asr_variant(original, 3))
        self.assertGreaterEqual(words, dup.MIN_WORDS)
        self.assertGreaterEqual(score, dup.MIN_CONTAINMENT + 0.1)

    def test_different_conversation_on_the_same_subject_stays_far_below(self):
        score, _ = dup.containment(
            asr_variant(conversation(4), 5), asr_variant(conversation(6), 7)
        )
        self.assertLess(score, 0.05)

    def test_normalization_ignores_case_and_punctuation(self):
        self.assertEqual(dup.normalized_words("Don't  STOP, now."), ["dont", "stop", "now"])


class FindOriginalTests(unittest.TestCase):
    def setUp(self):
        speech = conversation(10)
        self.original = episode(
            "orig", podcast="Example Fitness Podcast", feed="https://example.test/rs.xml",
            published="Sat, 03 Oct 2026 13:00:00 -0000", apple=True, compiler="review_required",
        )
        self.bonus = episode("bonus", published="Tue, 06 Oct 2026 15:00:00 +0000")
        self.texts = {
            "episodes/orig/sources/apple/transcript.txt": asr_variant(speech, 11),
            "episodes/orig/sources/whisper/transcript.txt": asr_variant(speech, 12),
            "episodes/bonus/sources/whisper/transcript.txt": asr_variant(speech[200:2800], 13),
        }

    def read(self, path):
        return self.texts[path]

    def test_later_copy_names_the_earlier_episode(self):
        finding = dup.find_original(self.bonus, [self.original, self.bonus], self.read)
        self.assertEqual(finding["original_episode_key"], "orig")
        self.assertEqual(finding["original_podcast"], "Example Fitness Podcast")
        self.assertGreaterEqual(finding["containment"], dup.MIN_CONTAINMENT)

    def test_the_earlier_episode_is_never_the_duplicate(self):
        self.original["status"]["compiler"]["state"] = "blocked"
        self.assertIsNone(dup.find_original(self.original, [self.original, self.bonus], self.read))

    def test_episodes_outside_the_window_are_not_compared(self):
        self.original["published"] = "Fri, 21 Aug 2026 13:00:00 -0000"
        self.assertIsNone(dup.find_original(self.bonus, [self.original, self.bonus], self.read))

    def test_unrelated_earlier_episode_is_not_a_match(self):
        self.texts["episodes/orig/sources/apple/transcript.txt"] = asr_variant(conversation(20), 21)
        self.texts["episodes/orig/sources/whisper/transcript.txt"] = asr_variant(conversation(20), 22)
        finding = dup.best_match(self.bonus, [self.original, self.bonus], self.read)
        self.assertLess(finding["containment"], dup.MIN_CONTAINMENT)
        self.assertIsNone(dup.find_original(self.bonus, [self.original, self.bonus], self.read))

    def test_short_episode_is_never_called_a_duplicate(self):
        speech = conversation(10)
        self.texts["episodes/bonus/sources/whisper/transcript.txt"] = asr_variant(speech[:500], 13)
        self.assertIsNone(dup.find_original(self.bonus, [self.original, self.bonus], self.read))

    def test_an_episode_already_recorded_as_duplicate_is_not_an_original(self):
        self.original["duplicate_check"] = {"result": "duplicate"}
        self.assertIsNone(dup.find_original(self.bonus, [self.original, self.bonus], self.read))

    def test_only_uncompiled_episodes_without_a_recorded_result_are_checked(self):
        self.assertTrue(dup.checkable(self.bonus))
        for state in ("review_required", "completed"):
            with self.subTest(state=state):
                other = deepcopy(self.bonus)
                other["status"]["compiler"]["state"] = state
                self.assertFalse(dup.checkable(other))
        overruled = deepcopy(self.bonus)
        overruled["duplicate_check"] = dup.not_duplicate_record(checked_at="now")
        self.assertFalse(dup.checkable(overruled))
        self.assertFalse(dup.is_duplicate(overruled))


class PipelineTests(FindOriginalTests):
    def test_worker_records_the_duplicate_and_stops_before_apple(self):
        stored = {"orig": self.original, "bonus": self.bonus}

        def update(key, *, duplicate_check):
            stored[key] = {**stored[key], "duplicate_check": duplicate_check}
            return stored[key]

        with (
            patch("podcast_engine.pipeline.load_episodes", return_value=list(stored.values())),
            patch("podcast_engine.pipeline.download_gcs_bytes",
                  side_effect=lambda path: self.texts[path].encode("utf-8")),
            patch("podcast_engine.pipeline.update_episode", side_effect=update),
            patch("podcast_engine.pipeline.emit_event") as event,
            patch("podcast_engine.pipeline.acquire_apple_transcript") as apple,
        ):
            outcome = resume_tracked_episode(PODCAST, self.bonus)

        self.assertEqual(outcome["status"], "skipped")
        self.assertEqual(outcome["reason"], "duplicate")
        self.assertEqual(outcome["duplicate_of"], "orig")
        apple.assert_not_called()
        record = stored["bonus"]["duplicate_check"]
        self.assertEqual(record["result"], "duplicate")
        self.assertEqual(record["policy_version"], dup.DUPLICATE_POLICY_VERSION)
        self.assertEqual(record["decided_by"], "worker")
        self.assertEqual(event.call_args.args[0], "episode_duplicate")

    def test_recorded_duplicate_is_skipped_and_not_resumed(self):
        self.bonus["duplicate_check"] = dup.duplicate_record(
            {"original_episode_key": "orig"}, checked_at="now"
        )
        with patch("podcast_engine.pipeline._transcribe_if_needed") as transcribe:
            outcome = resume_tracked_episode(PODCAST, self.bonus)
        transcribe.assert_not_called()
        self.assertEqual(outcome["reason"], "duplicate")
        with patch("podcast_engine.pipeline.load_episodes", return_value=[self.bonus]):
            self.assertEqual(_tracked_episodes_to_resume(PODCAST, processed_episode_keys=set()), [])

    def test_no_match_continues_to_apple_acquisition(self):
        self.texts["episodes/orig/sources/apple/transcript.txt"] = asr_variant(conversation(30), 31)
        self.texts["episodes/orig/sources/whisper/transcript.txt"] = asr_variant(conversation(30), 32)
        # Late-transcript notifications have separate coverage. Keep this
        # duplicate-detection test independent of the current calendar date.
        with (
            patch("podcast_engine.pipeline._apple_transcript_late_notification_due", return_value=None),
            patch("podcast_engine.pipeline.load_episodes", return_value=[self.original, self.bonus]),
            patch("podcast_engine.pipeline.download_gcs_bytes",
                  side_effect=lambda path: self.texts[path].encode("utf-8")),
            patch("podcast_engine.pipeline.update_episode") as update,
            patch(
                "podcast_engine.pipeline.acquire_apple_transcript",
                return_value={"status": "uploaded"},
            ) as apple,
        ):
            outcome = resume_tracked_episode(PODCAST, self.bonus)
        update.assert_not_called()
        apple.assert_called_once()
        self.assertEqual(outcome["status"], "waiting_for_apple_transcript")


if __name__ == "__main__":
    unittest.main()
