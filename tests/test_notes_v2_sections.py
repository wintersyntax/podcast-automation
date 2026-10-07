"""RED/GREEN tests for transcript-sections-v1 (TASK-106 Phase A plan, Task 1).

Spec: docs/superpowers/specs/2026-09-24-grounded-knowledge-note-pipeline-design.md sec 5.0
Plan: docs/superpowers/plans/2026-09-24-grounded-knowledge-note-pipeline-phase-a.md Task 1
"""

import unittest

from podcast_engine.knowledge.notes_v2.sections import (
    OVERLAP_WORDS,
    SECTIONS_SCHEMA_VERSION,
    TARGET_WORDS,
    split_sections,
)


def _plain_words(n, start=0):
    """``n`` space-separated tokens with no punctuation, so no paragraph or
    sentence boundary can accidentally appear in the lookahead window."""

    return " ".join(f"w{start + i}" for i in range(n))


class SectionsSchemaVersionTests(unittest.TestCase):
    def test_schema_version_constant_is_versioned(self):
        self.assertEqual(SECTIONS_SCHEMA_VERSION, "transcript-sections-v1")


class EmptyAndShortTranscriptTests(unittest.TestCase):
    def test_empty_transcript_returns_no_sections(self):
        self.assertEqual(split_sections(""), [])

    def test_short_transcript_returns_one_section(self):
        transcript = _plain_words(50)
        sections = split_sections(transcript)
        self.assertEqual(len(sections), 1)
        self.assertEqual(sections[0]["section_id"], "sec-01")
        self.assertEqual(sections[0]["start"], 0)
        self.assertEqual(sections[0]["end"], len(transcript))
        self.assertEqual(sections[0]["text"], transcript)


class DeterminismTests(unittest.TestCase):
    def test_identical_input_gives_identical_sections(self):
        transcript = _plain_words(6500)
        self.assertEqual(split_sections(transcript), split_sections(transcript))


class OffsetExactnessTests(unittest.TestCase):
    def test_offsets_are_exact_substrings_with_no_gap(self):
        transcript = _plain_words(6500)
        sections = split_sections(transcript)
        self.assertGreater(len(sections), 1)
        self.assertEqual(sections[0]["start"], 0)
        self.assertEqual(sections[-1]["end"], len(transcript))
        for section in sections:
            self.assertEqual(
                transcript[section["start"]:section["end"]], section["text"]
            )
            self.assertLess(section["start"], section["end"])
        for prev, nxt in zip(sections, sections[1:]):
            # Overlapping (or at least contiguous): no uncovered gap between
            # consecutive sections.
            self.assertLessEqual(nxt["start"], prev["end"])

    def test_section_ids_are_sequential_two_digit(self):
        transcript = _plain_words(6500)
        sections = split_sections(transcript)
        expected_ids = [f"sec-{i:02d}" for i in range(1, len(sections) + 1)]
        self.assertEqual([s["section_id"] for s in sections], expected_ids)


class TargetSizeAndOverlapTests(unittest.TestCase):
    def _word_count(self, text):
        return len(text.split())

    def test_target_section_size_is_approximately_2000_words(self):
        transcript = _plain_words(8200)
        sections = split_sections(transcript)
        # every section except possibly the last should be close to the
        # target; the fallback (no paragraph/sentence nearby) cuts exactly
        # at the target word count.
        for section in sections[:-1]:
            self.assertEqual(self._word_count(section["text"]), TARGET_WORDS)

    def test_overlap_with_previous_section_is_exactly_the_configured_amount(self):
        transcript = _plain_words(8200)
        sections = split_sections(transcript)
        for prev, nxt in zip(sections, sections[1:]):
            overlap_text = transcript[nxt["start"]:prev["end"]]
            self.assertEqual(self._word_count(overlap_text), OVERLAP_WORDS)


class BoundarySnapTests(unittest.TestCase):
    def test_boundary_snaps_to_a_paragraph_break_within_the_lookahead_window(self):
        # Ideal cut lands at word index TARGET_WORDS (2000). Put a paragraph
        # break 30 words later, well inside the lookahead window, and
        # nothing else nearby that could match first.
        before = _plain_words(TARGET_WORDS + 30)
        after = _plain_words(400, start=TARGET_WORDS + 30)
        transcript = before + "\n\n" + after
        sections = split_sections(transcript)
        first = sections[0]
        # The paragraph break sits between word (TARGET_WORDS + 29) and
        # (TARGET_WORDS + 30); the first section should end exactly there.
        self.assertTrue(first["text"].endswith("\n\n"))
        self.assertEqual(
            self._nth_word(transcript, TARGET_WORDS + 30), first["end"]
        )

    def test_boundary_snaps_to_a_sentence_end_when_no_paragraph_is_near(self):
        words = [f"w{i}" for i in range(TARGET_WORDS + 400)]
        # End word (TARGET_WORDS + 9) with a period, 10 words into the
        # lookahead window, with no paragraph break anywhere nearby.
        words[TARGET_WORDS + 9] = words[TARGET_WORDS + 9] + "."
        transcript = " ".join(words)
        sections = split_sections(transcript)
        first = sections[0]
        self.assertTrue(first["text"].rstrip().endswith("."))
        self.assertEqual(
            self._nth_word(transcript, TARGET_WORDS + 10), first["end"]
        )

    def test_paragraph_break_is_preferred_over_a_closer_sentence_end(self):
        words = [f"w{i}" for i in range(TARGET_WORDS + 400)]
        # A sentence end only 10 words in (closer to the ideal cut) ...
        words[TARGET_WORDS + 9] = words[TARGET_WORDS + 9] + "."
        transcript = " ".join(words)
        # ... but a paragraph break 50 words in (farther away) must still win.
        before_words = words[: TARGET_WORDS + 50]
        after_words = words[TARGET_WORDS + 50 :]
        transcript = " ".join(before_words) + "\n\n" + " ".join(after_words)
        sections = split_sections(transcript)
        first = sections[0]
        self.assertTrue(first["text"].endswith("\n\n"))
        self.assertEqual(
            self._nth_word(transcript, TARGET_WORDS + 50), first["end"]
        )

    def test_fallback_cuts_exactly_at_target_word_count_with_no_nearby_boundary(self):
        transcript = _plain_words(TARGET_WORDS + 400)
        sections = split_sections(transcript)
        first = sections[0]
        self.assertEqual(self._nth_word(transcript, TARGET_WORDS), first["end"])

    @staticmethod
    def _nth_word(transcript, word_index):
        """Return the character offset where the word at ``word_index`` starts."""

        import re

        spans = list(re.finditer(r"\S+", transcript))
        return spans[word_index].start()


if __name__ == "__main__":
    unittest.main()
