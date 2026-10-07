"""Deterministic overlapping transcript sectioning (TASK-106).

``split_sections`` makes no semantic decisions: it is a pure function of the
transcript text, used unchanged by the Phase A evaluation harness and later
Phase B production integration. See
docs/superpowers/specs/2026-09-24-grounded-knowledge-note-pipeline-design.md
sec 5.0.

Sections target ``TARGET_WORDS`` words, with boundaries snapped forward (never
backward) to the nearest paragraph break within a bounded lookahead window,
then to the nearest sentence end in that same window, and otherwise cut
exactly at the target word count. Each section overlaps the previous one by
``OVERLAP_WORDS`` words so a claim split across a boundary still appears
whole in at least one section.
"""

from __future__ import annotations

import re

SECTIONS_SCHEMA_VERSION = "transcript-sections-v1"

TARGET_WORDS = 2000
OVERLAP_WORDS = 150

# How far past the ideal cut point to look for a paragraph/sentence boundary
# before giving up and cutting exactly at the target word count.
_LOOKAHEAD_WORDS = 100

_WORD_RE = re.compile(r"\S+")
_PARAGRAPH_BREAK_RE = re.compile(r"\n\s*\n")
_SENTENCE_END_RE = re.compile(r"[.!?][\"')\]]*$")


def _snap_boundary_word_index(transcript: str, word_spans: list, ideal_word: int, word_count: int) -> int:
    """Return the word index to cut at, at or after ``ideal_word``.

    Prefers a paragraph break, then a sentence end, both only within
    ``_LOOKAHEAD_WORDS`` of the ideal cut; otherwise falls back to
    ``ideal_word`` itself (no snap).
    """

    search_end = min(ideal_word + _LOOKAHEAD_WORDS, word_count)

    for word_index in range(ideal_word, search_end):
        gap = transcript[word_spans[word_index - 1].end():word_spans[word_index].start()]
        if _PARAGRAPH_BREAK_RE.search(gap):
            return word_index

    for word_index in range(ideal_word, search_end):
        previous_word_text = transcript[word_spans[word_index - 1].start():word_spans[word_index - 1].end()]
        if _SENTENCE_END_RE.search(previous_word_text):
            return word_index

    return ideal_word


def split_sections(transcript: str) -> list[dict]:
    """Split ``transcript`` into overlapping, offset-exact sections.

    Returns a list of ``{"section_id", "start", "end", "text"}`` dicts where
    ``text == transcript[start:end]`` exactly, sections are contiguous or
    overlapping (never gapped), the first section starts at 0, and the last
    section ends at ``len(transcript)``.
    """

    word_spans = list(_WORD_RE.finditer(transcript))
    word_count = len(word_spans)
    if word_count == 0:
        return []

    sections: list[dict] = []
    cursor = 0
    section_number = 1

    while cursor < word_count:
        remaining = word_count - cursor
        if remaining <= TARGET_WORDS:
            end_word = word_count
        else:
            ideal_word = cursor + TARGET_WORDS
            end_word = _snap_boundary_word_index(transcript, word_spans, ideal_word, word_count)
            end_word = max(end_word, cursor + 1)
            end_word = min(end_word, word_count)

        start_char = 0 if section_number == 1 else word_spans[cursor].start()
        end_char = len(transcript) if end_word >= word_count else word_spans[end_word].start()

        sections.append(
            {
                "section_id": f"sec-{section_number:02d}",
                "start": start_char,
                "end": end_char,
                "text": transcript[start_char:end_char],
            }
        )

        if end_word >= word_count:
            break

        cursor = max(cursor + 1, end_word - OVERLAP_WORDS)
        section_number += 1

    return sections
