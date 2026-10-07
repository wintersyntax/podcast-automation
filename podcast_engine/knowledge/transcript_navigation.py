"""Lossless transcript navigation for single-pass podcast summaries."""

from __future__ import annotations

import math
import re


MAX_SEGMENTS = 40
TARGET_WORDS_PER_SEGMENT = 600

_NONSPACE_RE = re.compile(r"\S+")


def segment_transcript(transcript: str) -> list[dict]:
    """Split a transcript into ordered mechanical navigation segments.

    This function makes no semantic decisions. Concatenating every returned
    ``text`` value reproduces the input transcript exactly at the Python string
    level, including leading, trailing, and inter-segment whitespace.
    """

    if transcript == "":
        return []

    word_spans = list(_NONSPACE_RE.finditer(transcript))
    if not word_spans:
        return [
            {
                "index": 1,
                "total": 1,
                "start_pct": 0,
                "end_pct": 100,
                "text": transcript,
            }
        ]

    word_count = len(word_spans)
    segment_count = min(
        MAX_SEGMENTS,
        max(1, math.ceil(word_count / TARGET_WORDS_PER_SEGMENT)),
    )

    word_boundaries = [
        round(index * word_count / segment_count)
        for index in range(segment_count + 1)
    ]
    char_boundaries = [0]
    for word_index in word_boundaries[1:-1]:
        char_boundaries.append(word_spans[word_index].start())
    char_boundaries.append(len(transcript))

    segments = []
    for index in range(segment_count):
        start_char = char_boundaries[index]
        end_char = char_boundaries[index + 1]
        segments.append(
            {
                "index": index + 1,
                "total": segment_count,
                "start_pct": round(index * 100 / segment_count),
                "end_pct": round((index + 1) * 100 / segment_count),
                "text": transcript[start_char:end_char],
            }
        )

    return segments


def render_navigated_transcript(segments: list[dict]) -> str:
    """Render exact-source segments with clearly non-source navigation markers."""

    if not segments:
        return ""

    if len(segments) == 1:
        return segments[0]["text"]

    rendered: list[str] = []
    for segment in segments:
        rendered.append(
            "<<< TRANSCRIPT SEGMENT "
            f"{segment['index']:02d}/{segment['total']:02d} | "
            f"approx {segment['start_pct']}-{segment['end_pct']}% >>>\n"
        )
        rendered.append(segment["text"])
        if not segment["text"].endswith("\n"):
            rendered.append("\n")
    return "".join(rendered)


NAVIGATION_METHOD = "lossless_numbered_transcript_navigation_v1"

NAVIGATION_CONTRACT = (
    "The compiled_transcript field contains the complete canonical transcript. For multi-segment transcripts, "
    "it is divided only by mechanical numbered navigation markers; a single short transcript may be left unmarked. "
    "The markers are not source content, and segment boundaries are not topic boundaries. Before drafting a "
    "multi-segment transcript, inspect every numbered segment and form an internal inventory of the material developed "
    "across the complete transcript; for a single segment, review the complete transcript normally. A topic may span "
    "multiple segments. The complete canonical transcript remains the only content authority. After completing that "
    "navigation sweep, follow the summary preset exactly. These navigation instructions do not modify the preset's "
    "topic prioritization, podcast profile, evidence discipline, section requirements, redundancy rules, length guidance, "
    "or output format. Do not promote or omit material merely because of segment boundaries."
)
