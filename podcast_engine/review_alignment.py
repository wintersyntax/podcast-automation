"""Monotonic, evidence-only alignment helpers for human transcript review.

The compiler remains the authority for differences and their classification.
This module only maps an Apple position onto the Whisper timeline so a reviewer
can open the relevant part of the audio even when Apple timestamps have drifted.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import re
from typing import Sequence


WORD_PATTERN = re.compile(r"[\w]+(?:['’][\w]+)?", re.UNICODE)
ANCHOR_WORDS = 5


def tokens(text: str) -> list[str]:
    """Return comparison tokens without making any transcription decision."""

    return [match.group(0).casefold() for match in WORD_PATTERN.finditer(text)]


@dataclass(frozen=True)
class Alignment:
    """A monotonic Apple-to-Whisper token mapping and its retained anchors."""

    apple_to_whisper: dict[int, int]
    anchors: tuple[tuple[int, int, int], ...]
    apple_words: int
    whisper_words: int

    def locate_apple_span(self, start: int, end: int) -> dict:
        """Estimate the Whisper range corresponding to an Apple token range.

        ``start`` is zero-based and ``end`` is exclusive. Exact matched tokens
        are preferred. For a changed span (which naturally has no exact map),
        surrounding monotonic anchors supply an interpolation only.
        """

        if start < 0 or end < start:
            raise ValueError("Invalid Apple token range")

        mapped = [
            self.apple_to_whisper[index]
            for index in range(start, end)
            if index in self.apple_to_whisper
        ]

        if mapped:
            return {
                "method": "matched_tokens",
                "apple_start_word": start + 1,
                "apple_end_word": end,
                "whisper_start_word": min(mapped) + 1,
                "whisper_end_word": max(mapped) + 1,
            }

        before = [
            (apple, whisper)
            for apple, whisper in self.apple_to_whisper.items()
            if apple < start
        ]
        after = [
            (apple, whisper)
            for apple, whisper in self.apple_to_whisper.items()
            if apple >= end
        ]

        left = max(before, default=None)
        right = min(after, default=None)

        if left and right and right[0] > left[0]:
            ratio = (right[1] - left[1]) / (right[0] - left[0])
            whisper_start = round(left[1] + (start - left[0]) * ratio)
            whisper_end = round(left[1] + (end - left[0]) * ratio)
            method = "interpolated_anchors"
        elif left:
            whisper_start = left[1] + max(1, start - left[0])
            whisper_end = whisper_start + max(1, end - start)
            method = "left_anchor"
        elif right:
            whisper_end = right[1] - max(1, right[0] - end)
            whisper_start = whisper_end - max(1, end - start)
            method = "right_anchor"
        else:
            whisper_start = 0
            whisper_end = max(1, end - start)
            method = "unanchored"

        whisper_start = max(0, min(whisper_start, self.whisper_words))
        whisper_end = max(whisper_start, min(whisper_end, self.whisper_words))
        return {
            "method": method,
            "apple_start_word": start + 1,
            "apple_end_word": end,
            "whisper_start_word": whisper_start + 1,
            "whisper_end_word": whisper_end,
        }


def _unique_ngrams(values: Sequence[str], width: int) -> dict[tuple[str, ...], int]:
    positions: dict[tuple[str, ...], list[int]] = {}
    for index in range(max(0, len(values) - width + 1)):
        positions.setdefault(tuple(values[index : index + width]), []).append(index)
    return {ngram: indexes[0] for ngram, indexes in positions.items() if len(indexes) == 1}


def _anchors(apple: Sequence[str], whisper: Sequence[str]) -> list[tuple[int, int, int]]:
    """Find unique n-gram anchors and retain a monotonic subsequence."""

    apple_ngrams = _unique_ngrams(apple, ANCHOR_WORDS)
    whisper_ngrams = _unique_ngrams(whisper, ANCHOR_WORDS)
    candidates = sorted(
        (apple_start, whisper_ngrams[ngram], ANCHOR_WORDS)
        for ngram, apple_start in apple_ngrams.items()
        if ngram in whisper_ngrams
    )

    kept: list[tuple[int, int, int]] = []
    last_apple = last_whisper = -1
    for apple_start, whisper_start, width in candidates:
        if apple_start > last_apple and whisper_start > last_whisper:
            kept.append((apple_start, whisper_start, width))
            last_apple = apple_start + width - 1
            last_whisper = whisper_start + width - 1
    return kept


def _map_window(
    mapping: dict[int, int],
    apple: Sequence[str],
    whisper: Sequence[str],
    apple_start: int,
    apple_end: int,
    whisper_start: int,
    whisper_end: int,
) -> None:
    """Use local sequence alignment between neighboring anchors."""

    matcher = SequenceMatcher(
        None,
        apple[apple_start:apple_end],
        whisper[whisper_start:whisper_end],
        autojunk=False,
    )
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            mapping[apple_start + block.a + offset] = whisper_start + block.b + offset


def align_transcripts(apple_text: str, whisper_text: str) -> Alignment:
    """Build a monotonic anchor/sliding alignment without choosing text.

    Unique five-token anchors keep the long transcript mapping stable. Each gap
    is then locally aligned, which avoids a distant repeated phrase pulling a
    review card to the wrong audio position.
    """

    apple = tokens(apple_text)
    whisper = tokens(whisper_text)
    anchors = _anchors(apple, whisper)
    mapping: dict[int, int] = {}
    apple_cursor = whisper_cursor = 0

    for apple_start, whisper_start, width in anchors:
        _map_window(
            mapping,
            apple,
            whisper,
            apple_cursor,
            apple_start,
            whisper_cursor,
            whisper_start,
        )
        for offset in range(width):
            mapping[apple_start + offset] = whisper_start + offset
        apple_cursor = apple_start + width
        whisper_cursor = whisper_start + width

    _map_window(
        mapping,
        apple,
        whisper,
        apple_cursor,
        len(apple),
        whisper_cursor,
        len(whisper),
    )

    return Alignment(
        apple_to_whisper=mapping,
        anchors=tuple(anchors),
        apple_words=len(apple),
        whisper_words=len(whisper),
    )
