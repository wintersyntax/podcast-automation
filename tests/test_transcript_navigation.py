import json
import os
import unittest
from unittest.mock import patch

from podcast_engine.knowledge import summary
from podcast_engine.preset_provenance import PresetProvenance
from podcast_engine.knowledge.transcript_navigation import (
    MAX_SEGMENTS,
    NAVIGATION_CONTRACT,
    NAVIGATION_METHOD,
    TARGET_WORDS_PER_SEGMENT,
    render_navigated_transcript,
    segment_transcript,
)


class TranscriptNavigationTests(unittest.TestCase):
    def _episode(self):
        return {
            "episode_key": "episode-1",
            "podcast": "Example Strength Podcast",
            "podcast_id": "example-strength",
            "category": "exercise_strength",
            "prompt": "strength",
            "title": "Navigation test",
            "published": "2026-09-04T00:00:00Z",
            "link": "https://example.test/episode",
        }

    def _transcript(self):
        sections = [
            "Dieting rate of weight loss changed across the cut. Sleep quality and recovery stayed stable.",
            "Supplements were kept deliberately minimal. Creatine, electrolytes, and greens powder were discussed skeptically.",
            "Post-diet recovery required letting body fat rise while normal eating and training returned.",
            "Athletic side quests included sprinting, rowing, mobility work, and jumping practice.",
            "Training moved away from strict logbook tracking for some smaller muscle groups.",
            "Heavy body weight and excessive bulking were discussed as long-term health trade-offs.",
        ]
        return " ".join(sentence for sentence in sections for _ in range(120))

    @staticmethod
    def _provenance():
        return PresetProvenance(
            status="verified",
            slug="custom-summary",
            preset_id="preset-id",
            version_id="version-id-1",
            version=1,
            config={"model": "test/summary-model", "max_tokens": 900},
            system_prompt="Summary system prompt.",
            config_digest="sha256:config",
            system_prompt_digest="sha256:prompt",
            verified_at="2026-09-16T00:00:00Z",
        )

    def test_payload_uses_navigation_metadata_without_second_summary_policy(self):
        transcript = self._transcript()
        expected_segments = segment_transcript(transcript)

        payload = summary.openrouter_payload(self._episode(), transcript, self._provenance())

        self.assertEqual(payload["model"], "test/summary-model")
        self.assertNotIn("temperature", payload)
        self.assertNotIn("provider", payload)
        self.assertEqual(payload["messages"][0], {"role": "system", "content": "Summary system prompt."})
        content = json.loads(payload["messages"][1]["content"])

        self.assertEqual(content["transcript_navigation_method"], NAVIGATION_METHOD)
        self.assertEqual(content["transcript_navigation_segment_count"], len(expected_segments))
        self.assertEqual(content["transcript_navigation_contract"], NAVIGATION_CONTRACT)
        self.assertNotIn("coverage_method", content)
        self.assertNotIn("coverage_segment_count", content)
        self.assertNotIn("coverage_contract", content)
        self.assertEqual(
            content["compiled_transcript"],
            render_navigated_transcript(expected_segments),
        )
        self.assertEqual("".join(segment["text"] for segment in expected_segments), transcript)

        contract = NAVIGATION_CONTRACT.lower()
        self.assertIn("every numbered segment", contract)
        self.assertIn("segment boundaries are not topic boundaries", contract)
        self.assertIn("follow the summary preset exactly", contract)
        self.assertIn("only content authority", contract)
        self.assertNotIn("one primary home", contract)
        self.assertNotIn("research & evidence", contract)
        self.assertNotIn("highest-value conclusions", contract)
        self.assertNotIn("practical takeaways", contract)

    def test_navigation_scales_across_realistic_episode_length_range_losslessly(self):
        # Approximate transcript sizes for ~10, 30, 60, 95, 150, and 190 minute episodes.
        cases = [
            (1_700, 3),
            (5_000, 9),
            (10_000, 17),
            (16_000, 27),
            (25_000, 40),
            (32_000, 40),
        ]
        self.assertEqual(TARGET_WORDS_PER_SEGMENT, 600)
        self.assertEqual(MAX_SEGMENTS, 40)

        for word_count, expected_count in cases:
            with self.subTest(word_count=word_count):
                transcript = "word " * word_count
                segments = segment_transcript(transcript)
                self.assertEqual(len(segments), expected_count)
                self.assertEqual("".join(segment["text"] for segment in segments), transcript)

    def test_navigation_is_deterministic_lossless_and_preserves_whitespace(self):
        transcript = (
            "  Opening words with leading spaces.\n\n"
            "Second paragraph has\ttabs and punctuation!  "
            + "Training sleep recovery supplements and dieting. " * 1_800
            + "Trailing words.   \n"
        )

        first = segment_transcript(transcript)
        second = segment_transcript(transcript)

        self.assertEqual(first, second)
        self.assertEqual("".join(segment["text"] for segment in first), transcript)
        self.assertLessEqual(len(first), MAX_SEGMENTS)
        self.assertEqual([segment["index"] for segment in first], list(range(1, len(first) + 1)))
        self.assertTrue(all(segment["total"] == len(first) for segment in first))
        self.assertEqual(
            [segment["start_pct"] for segment in first],
            sorted(segment["start_pct"] for segment in first),
        )

    def test_single_segment_renderer_preserves_exact_transcript_payload(self):
        transcript = "Alpha beta gamma delta epsilon zeta eta theta."
        segments = segment_transcript(transcript)
        rendered = render_navigated_transcript(segments)

        self.assertEqual(len(segments), 1)
        self.assertEqual(rendered, transcript)
        self.assertNotIn("<<< TRANSCRIPT SEGMENT", rendered)

    def test_multi_segment_renderer_adds_markers_without_duplicating_source_text(self):
        transcript = "word " * 1_800
        segments = segment_transcript(transcript)
        rendered = render_navigated_transcript(segments)

        self.assertGreater(len(segments), 1)
        self.assertEqual("".join(segment["text"] for segment in segments), transcript)
        self.assertEqual(rendered.count("<<< TRANSCRIPT SEGMENT"), len(segments))
        for segment in segments:
            self.assertIn(segment["text"], rendered)

    def test_empty_and_whitespace_only_transcripts_are_preserved_without_invention(self):
        self.assertEqual(segment_transcript(""), [])
        self.assertEqual(render_navigated_transcript([]), "")
        whitespace = "   \n\t"
        segments = segment_transcript(whitespace)
        self.assertEqual("".join(segment["text"] for segment in segments), whitespace)


if __name__ == "__main__":
    unittest.main()
