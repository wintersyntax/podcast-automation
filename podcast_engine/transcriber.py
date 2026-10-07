"""OpenRouter Whisper transcription with bounded-size midpoint-stitched chunks.

TASK-125: each overlapping audio chunk is encoded as a small MP3 and sent to
OpenRouter's transcription endpoint (``openai/whisper-large-v3``, one provider
pinned by the API key's OpenRouter guardrail, ``verbose_json`` with word and
segment timestamps). The response is
converted into the same segment/word metadata the local Faster-Whisper
producer used to write, so chunk stitching, timestamp validation and the
compiler contract are unchanged. Every physical request is admitted through
the episode AI budget before it is sent.
"""

from __future__ import annotations

import base64
from decimal import Decimal
import gc
import hashlib
import json
import math
import os
from difflib import SequenceMatcher
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import requests

from .ai_budget import (
    STAGE_WHISPER_TRANSCRIPTION,
    mark_budget_attempt_uncertain,
    reserve_budget_batch,
    settle_budget_attempt,
)
from .ai_pricing import derive_audio_reservation_usd, resolve_audio_endpoint_pricing
from .episode_contract import paths_for
from .storage import upload_path_to_gcs
from .whisper_provenance import (
    AUDIO_BITRATE_KBPS,
    AUDIO_CHANNELS,
    AUDIO_FORMAT,
    AUDIO_SAMPLE_RATE,
    CHUNK_OVERLAP_SECONDS,
    CHUNK_SECONDS,
    MAX_ATTEMPTS,
    MODEL_ID,
    MODEL_PROVIDER,
    RATE_LIMIT_ATTEMPTS,
    RATE_LIMIT_BACKOFF_SECONDS,
    REQUEST_TIMEOUT_SECONDS,
    RESPONSE_FORMAT,
    RETRY_AFTER_CAP_SECONDS,
    STITCH_METHOD,
    STT_ENDPOINT,
    TIMESTAMP_GRANULARITIES,
    WHISPER_METADATA_SCHEMA_VERSION,
    WHISPER_METADATA_SOURCE,
    current_whisper_producer,
)


WHISPER_API_KEY_ENV = "PODCAST_WHISPER_API_KEY"
RETRY_DELAY_SECONDS = 5

MODEL = MODEL_ID

WHISPER_CHUNK_SECONDS = CHUNK_SECONDS
WHISPER_CHUNK_OVERLAP_SECONDS = CHUNK_OVERLAP_SECONDS

TIMESTAMP_EPSILON = 1e-6
# A chunk this long represents a meaningful blind spot. The threshold follows
# the configured chunk size when deployments use shorter chunks, while still
# allowing ordinary short trailing silence to remain empty.
MIN_SUBSTANTIAL_EMPTY_CHUNK_SECONDS = min(60.0, WHISPER_CHUNK_SECONDS)

@dataclass(frozen=True)
class AudioChunk:
    """One decoded chunk and its absolute start time in the episode."""

    path: Path
    start: float
    duration: float


def _chunk_has_usable_transcript(segments: list[dict]) -> bool:
    """Return whether Whisper supplied any lexical evidence for a chunk."""

    return any(
        str(segment.get("text") or "").strip()
        or any(str(word.get("word") or "").strip() for word in segment.get("words") or [])
        for segment in segments
    )


def _suspicious_empty_chunk(
    chunk: AudioChunk,
    *,
    chunk_number: int,
    chunk_count: int,
    segments: list[dict],
) -> bool:
    """Identify an empty substantial chunk that must not publish silently.

    Position is not evidence of safety: a substantial first, middle, final, or
    sole chunk with no lexical output leaves a blind spot the compiler cannot
    recover. A short edge chunk remains allowed because it may be trailing
    silence or a short silent lead-in.
    """

    return (
        chunk.duration >= MIN_SUBSTANTIAL_EMPTY_CHUNK_SECONDS
        and not _chunk_has_usable_transcript(segments)
    )


def _offset_timestamp(
    value,
    offset_seconds: float,
):
    """Shift a chunk-local timestamp into episode-global time."""

    if value is None:
        return None

    return float(value) + offset_seconds


def _word_midpoint(
    word: dict,
) -> float | None:
    """Return a stable timestamp used to assign a word to one side of a cut."""

    start = word.get("start")
    end = word.get("end")

    if start is not None and end is not None:
        return (
            float(start)
            + float(end)
        ) / 2.0

    if start is not None:
        return float(start)

    if end is not None:
        return float(end)

    return None


def _segment_midpoint(
    segment: dict,
) -> float | None:
    start = segment.get("start")
    end = segment.get("end")

    if start is not None and end is not None:
        return (
            float(start)
            + float(end)
        ) / 2.0

    if start is not None:
        return float(start)

    if end is not None:
        return float(end)

    return None


def _normalized_word_value(word: dict) -> str:
    """Return one normalized lexical value for overlap comparison."""

    return str(word.get("word") or "").strip().casefold()


def _segment_overlaps_window(
    segment: dict,
    *,
    start: float,
    end: float,
) -> bool:
    """Return whether a segment intersects the real chunk-overlap window."""

    segment_start = segment.get("start")
    segment_end = segment.get("end")
    if segment_start is None and segment_end is None:
        return False

    left = float(segment_start if segment_start is not None else segment_end)
    right = float(segment_end if segment_end is not None else segment_start)
    return right >= start - TIMESTAMP_EPSILON and left <= end + TIMESTAMP_EPSILON


def _overlap_word_entries(
    segments: list[dict],
    *,
    overlap_start: float,
    overlap_end: float,
) -> list[dict]:
    """Return lexical word occurrences that plausibly belong to the overlap.

    Untimestamped words are included only when their containing segment overlaps
    the overlap window. Occurrence identity is retained so duplicate handling
    never collapses all equal words into one set-membership decision.
    """

    entries: list[dict] = []
    for segment_index, segment in enumerate(segments):
        segment_words = segment.get("words") or []
        segment_values = [
            value
            for word in segment_words
            if (value := _normalized_word_value(word))
        ]
        segment_is_local = _segment_overlaps_window(
            segment,
            start=overlap_start,
            end=overlap_end,
        )

        for word_index, word in enumerate(segment_words):
            value = _normalized_word_value(word)
            if not value:
                continue

            midpoint = _word_midpoint(word)
            if midpoint is not None:
                if not (
                    overlap_start - TIMESTAMP_EPSILON
                    <= midpoint
                    <= overlap_end + TIMESTAMP_EPSILON
                ):
                    continue
            elif not segment_is_local:
                continue

            entries.append(
                {
                    "segment_index": segment_index,
                    "word_index": word_index,
                    "value": value,
                    "timestamped": midpoint is not None,
                    "segment_values": tuple(segment_values),
                    "segment_start": segment.get("start"),
                    "segment_end": segment.get("end"),
                }
            )

    return entries


def _entry_segments_overlap(previous: dict, current: dict) -> bool:
    """Require temporal support before accepting a one-token duplicate."""

    p_start = previous.get("segment_start")
    p_end = previous.get("segment_end")
    c_start = current.get("segment_start")
    c_end = current.get("segment_end")
    if None in {p_start, p_end, c_start, c_end}:
        return False
    return (
        float(p_end) >= float(c_start) - TIMESTAMP_EPSILON
        and float(c_end) >= float(p_start) - TIMESTAMP_EPSILON
    )


def _singleton_overlap_match_supported(previous: dict, current: dict) -> bool:
    """Return whether one equal token has enough local evidence to deduplicate.

    A bare lexical match such as ``the`` is insufficient. We require temporal
    segment overlap plus either identical segment lexical content or a
    single-token segment visibly contained at an edge of the other segment.
    Ambiguous cases are preserved on both sides rather than risking silent
    content loss.
    """

    if not _entry_segments_overlap(previous, current):
        return False

    previous_values = previous["segment_values"]
    current_values = current["segment_values"]
    if previous_values == current_values:
        return True

    if len(previous_values) == 1:
        return bool(
            current_values
            and (
                current_values[0] == previous_values[0]
                or current_values[-1] == previous_values[0]
            )
        )

    if len(current_values) == 1:
        return bool(
            previous_values
            and (
                previous_values[0] == current_values[0]
                or previous_values[-1] == current_values[0]
            )
        )

    return False


def _untimestamped_overlap_duplicate_locations(
    previous_segments: list[dict],
    current_segments: list[dict],
    *,
    chunk_start: float,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    """Identify only strongly supported duplicate untimestamped occurrences.

    Sequence alignment is restricted to the actual overlap window and retains
    occurrence multiplicity. Equal words in unrelated local contexts are not
    enough to delete either occurrence.
    """

    overlap_start = chunk_start
    overlap_end = chunk_start + WHISPER_CHUNK_OVERLAP_SECONDS
    previous_entries = _overlap_word_entries(
        previous_segments,
        overlap_start=overlap_start,
        overlap_end=overlap_end,
    )
    current_entries = _overlap_word_entries(
        current_segments,
        overlap_start=overlap_start,
        overlap_end=overlap_end,
    )

    previous_values = [entry["value"] for entry in previous_entries]
    current_values = [entry["value"] for entry in current_entries]
    matcher = SequenceMatcher(
        None,
        previous_values,
        current_values,
        autojunk=False,
    )

    drop_previous: set[tuple[int, int]] = set()
    drop_current: set[tuple[int, int]] = set()

    for block in matcher.get_matching_blocks():
        if block.size <= 0:
            continue

        for offset in range(block.size):
            previous = previous_entries[block.a + offset]
            current = current_entries[block.b + offset]

            if previous["timestamped"] and current["timestamped"]:
                continue

            if (
                block.size == 1
                and not _singleton_overlap_match_supported(previous, current)
            ):
                continue

            previous_location = (
                previous["segment_index"],
                previous["word_index"],
            )
            current_location = (
                current["segment_index"],
                current["word_index"],
            )

            # Prefer the occurrence with real word timing. If neither has word
            # timing, keep the earlier chunk's occurrence deterministically.
            if previous["timestamped"] and not current["timestamped"]:
                drop_current.add(current_location)
            elif current["timestamped"] and not previous["timestamped"]:
                drop_previous.add(previous_location)
            else:
                drop_current.add(current_location)

    return drop_previous, drop_current


def _optional_float(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _openrouter_segment_metadata(
    payload: dict,
    *,
    offset_seconds: float = 0.0,
) -> list[dict]:
    """Convert one OpenRouter ``verbose_json`` response into chunk segments.

    OpenRouter returns segments and a flat word list separately. Each word is
    attached to the segment containing its midpoint (or the nearest earlier
    segment), preserving the segment's acoustic fields for the compiler.
    Words carry no per-word probability, so the compiler falls back to the
    segment ``avg_logprob``.
    """

    raw_segments = payload.get("segments")
    raw_words = payload.get("words")
    if not isinstance(raw_segments, list) or not isinstance(raw_words, list):
        raise ValueError("OpenRouter transcription response has no segments/words")

    segments: list[dict] = []
    for raw in raw_segments:
        if not isinstance(raw, dict):
            raise ValueError("OpenRouter transcription segment is malformed")
        start = _optional_float(raw.get("start"))
        end = _optional_float(raw.get("end"))
        if start is None or end is None or end < start:
            raise ValueError("OpenRouter transcription segment has invalid timestamps")
        segments.append(
            {
                "id": len(segments),
                "start": _offset_timestamp(start, offset_seconds),
                "end": _offset_timestamp(end, offset_seconds),
                "text": str(raw.get("text") or "").strip(),
                "avg_logprob": _optional_float(raw.get("avg_logprob")),
                "no_speech_prob": _optional_float(raw.get("no_speech_prob")),
                "words": [],
                "_local_start": start,
                "_local_end": end,
            }
        )

    for raw in raw_words:
        if not isinstance(raw, dict) or not isinstance(raw.get("word"), str):
            raise ValueError("OpenRouter transcription word is malformed")
        start = _optional_float(raw.get("start"))
        end = _optional_float(raw.get("end"))
        if start is None or end is None or end < start:
            raise ValueError("OpenRouter transcription word has invalid timestamps")
        if not segments:
            raise ValueError("OpenRouter transcription returned words without segments")
        midpoint = (start + end) / 2
        owner = segments[0]
        for segment in segments:
            if segment["_local_start"] - TIMESTAMP_EPSILON <= midpoint:
                owner = segment
            else:
                break
        owner["words"].append(
            {
                "word": raw["word"],
                "start": _offset_timestamp(start, offset_seconds),
                "end": _offset_timestamp(end, offset_seconds),
            }
        )

    for segment in segments:
        segment.pop("_local_start")
        segment.pop("_local_end")
    return segments


def _probe_duration(
    audio_file: str | Path,
) -> float:
    """Read source duration with ffprobe."""

    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(audio_file),
    ]

    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )

        duration = float(
            result.stdout.strip()
        )

    except (
        subprocess.CalledProcessError,
        ValueError,
    ) as error:
        raise RuntimeError(
            f"Could not determine audio duration: {error}"
        ) from error

    if duration <= 0:
        raise RuntimeError(
            "Audio duration must be greater than zero."
        )

    return duration


def _chunk_ranges(
    total_duration: float,
) -> list[tuple[float, float]]:
    """Return chunk start/duration pairs with trailing overlap."""

    if WHISPER_CHUNK_SECONDS <= 0:
        raise ValueError(
            "WHISPER_CHUNK_SECONDS must be greater than zero."
        )

    if WHISPER_CHUNK_OVERLAP_SECONDS < 0:
        raise ValueError(
            "WHISPER_CHUNK_OVERLAP_SECONDS cannot be negative."
        )

    if (
        WHISPER_CHUNK_OVERLAP_SECONDS
        >= WHISPER_CHUNK_SECONDS
    ):
        raise ValueError(
            "Whisper chunk overlap must be shorter than the chunk."
        )

    ranges: list[
        tuple[float, float]
    ] = []

    start = 0.0

    while start < total_duration:
        nominal_end = min(
            start + WHISPER_CHUNK_SECONDS,
            total_duration,
        )

        chunk_end = min(
            nominal_end
            + (
                WHISPER_CHUNK_OVERLAP_SECONDS
                if nominal_end < total_duration
                else 0
            ),
            total_duration,
        )

        ranges.append(
            (
                start,
                chunk_end - start,
            )
        )

        start += WHISPER_CHUNK_SECONDS

    return ranges


def _prepare_audio_chunks(
    audio_file: str | Path,
    work_dir: Path,
) -> tuple[
    Path,
    list[AudioChunk],
    float,
]:
    """Encode audio into overlapping mono chunks in the manifest's request format."""

    total_duration = _probe_duration(
        audio_file
    )

    chunk_dir = work_dir / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    chunks: list[
        AudioChunk
    ] = []

    ranges = _chunk_ranges(
        total_duration
    )

    print(
        "Preparing Whisper audio chunks "
        f"({WHISPER_CHUNK_SECONDS}s + "
        f"{WHISPER_CHUNK_OVERLAP_SECONDS}s overlap)..."
    )

    for index, (
        start,
        duration,
    ) in enumerate(ranges):
        chunk_path = (
            chunk_dir
            / f"chunk_{index:03d}.{AUDIO_FORMAT}"
        )

        command = [
            "ffmpeg",
            "-y",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start:.3f}",
            "-i",
            str(audio_file),
            "-t",
            f"{duration:.3f}",
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            str(AUDIO_CHANNELS),
            "-ar",
            str(AUDIO_SAMPLE_RATE),
            "-b:a",
            f"{AUDIO_BITRATE_KBPS}k",
            str(chunk_path),
        ]

        try:
            subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            )

        except subprocess.CalledProcessError as error:
            detail = (
                error.stderr.strip()
                or error.stdout.strip()
                or str(error)
            )

            raise RuntimeError(
                f"FFmpeg chunking failed: {detail}"
            ) from error

        chunks.append(
            AudioChunk(
                path=chunk_path,
                start=start,
                duration=duration,
            )
        )

    if not chunks:
        raise RuntimeError(
            "FFmpeg produced no audio chunks."
        )

    print(
        f"Prepared {len(chunks)} Whisper chunk(s)"
    )

    return (
        chunk_dir,
        chunks,
        total_duration,
    )


def _rebuild_segment_from_words(
    segment: dict,
    words: list[dict],
) -> dict | None:
    """Rebuild segment boundaries/text after removing overlap words."""

    if not words:
        return None

    rebuilt = dict(
        segment
    )

    rebuilt_words = [
        dict(word)
        for word in words
    ]

    rebuilt["words"] = (
        rebuilt_words
    )

    rebuilt["start"] = next(
        (word.get("start") for word in rebuilt_words if word.get("start") is not None),
        segment.get("start"),
    )

    rebuilt["end"] = next(
        (word.get("end") for word in reversed(rebuilt_words) if word.get("end") is not None),
        segment.get("end"),
    )

    rebuilt["text"] = "".join(
        str(
            word.get(
                "word",
                "",
            )
        )
        for word in rebuilt_words
    ).strip()

    return rebuilt


def _trim_segments_at_cut(
    segments: list[dict],
    *,
    cut_time: float,
    keep_before: bool,
    drop_untimestamped_locations: set[tuple[int, int]] | None = None,
) -> list[dict]:
    """Assign every word to exactly one side of a midpoint stitch.

    Previous chunk:
        word midpoint < cut

    Current chunk:
        word midpoint >= cut
    """

    output: list[
        dict
    ] = []

    for segment_index, original in enumerate(segments):
        words = (
            original.get("words")
            or []
        )

        if words:
            kept_words: list[
                dict
            ] = []

            for word_index, word in enumerate(words):
                midpoint = (
                    _word_midpoint(
                        word
                    )
                )

                if midpoint is None:
                    # Untimestamped lexical evidence is preserved by default.
                    # It is removed only when overlap-local sequence/context
                    # analysis positively identifies this exact occurrence as
                    # the duplicate copy owned by the adjacent chunk.
                    location = (segment_index, word_index)
                    if (
                        drop_untimestamped_locations
                        and location in drop_untimestamped_locations
                    ):
                        continue

                    kept_words.append(word)
                    continue

                if keep_before:
                    keep = (
                        midpoint
                        < cut_time
                    )
                else:
                    keep = (
                        midpoint
                        >= cut_time
                    )

                if keep:
                    kept_words.append(
                        word
                    )

            rebuilt = (
                _rebuild_segment_from_words(
                    original,
                    kept_words,
                )
            )

            if rebuilt is not None:
                output.append(
                    rebuilt
                )

            continue

        # Word timestamps are requested, so wordless segments should be rare.
        # Fall back to the segment midpoint rather than duplicating them.
        midpoint = (
            _segment_midpoint(
                original
            )
        )

        if midpoint is None:
            continue

        if keep_before:
            keep = (
                midpoint
                < cut_time
            )
        else:
            keep = (
                midpoint
                >= cut_time
            )

        if keep:
            output.append(
                dict(original)
            )

    return output


def _word_count(
    segments: list[dict],
) -> int:
    return sum(
        len(
            segment.get(
                "words"
            )
            or []
        )
        for segment in segments
    )


def _stitch_chunk_overlap(
    previous_segments: list[dict],
    current_segments: list[dict],
    *,
    chunk_start: float,
) -> tuple[
    list[dict],
    dict,
]:
    """Stitch two overlapping Whisper outputs at the overlap midpoint.

    The text itself does not need to match. This handles real cases where the
    same audio is decoded as e.g. "totally" by one chunk and "really" by the
    next chunk.
    """

    if (
        not previous_segments
        or WHISPER_CHUNK_OVERLAP_SECONDS <= 0
    ):
        return (
            previous_segments
            + current_segments,
            {
                "chunk_start": (
                    chunk_start
                ),
                "cut_time": None,
                "previous_words_removed": 0,
                "current_words_removed": 0,
            },
        )

    cut_time = (
        chunk_start
        + (
            WHISPER_CHUNK_OVERLAP_SECONDS
            / 2.0
        )
    )

    previous_word_count = (
        _word_count(
            previous_segments
        )
    )

    current_word_count = (
        _word_count(
            current_segments
        )
    )

    (
        drop_previous_untimestamped,
        drop_current_untimestamped,
    ) = _untimestamped_overlap_duplicate_locations(
        previous_segments,
        current_segments,
        chunk_start=chunk_start,
    )

    previous_trimmed = (
        _trim_segments_at_cut(
            previous_segments,
            cut_time=cut_time,
            keep_before=True,
            drop_untimestamped_locations=drop_previous_untimestamped,
        )
    )

    current_trimmed = (
        _trim_segments_at_cut(
            current_segments,
            cut_time=cut_time,
            keep_before=False,
            drop_untimestamped_locations=drop_current_untimestamped,
        )
    )

    previous_removed = (
        previous_word_count
        - _word_count(
            previous_trimmed
        )
    )

    current_removed = (
        current_word_count
        - _word_count(
            current_trimmed
        )
    )

    return (
        previous_trimmed
        + current_trimmed,
        {
            "chunk_start": (
                chunk_start
            ),
            "cut_time": (
                cut_time
            ),
            "previous_words_removed": (
                previous_removed
            ),
            "current_words_removed": (
                current_removed
            ),
        },
    )


def _validate_monotonic_timestamps(
    segments: list[dict],
) -> None:
    """Reject metadata whose segment or word timeline moves backwards."""

    previous_segment_start = None
    previous_word_start = None

    for segment in segments:
        start = segment.get(
            "start"
        )

        end = segment.get(
            "end"
        )

        if (
            start is not None
            and end is not None
            and float(end)
            + TIMESTAMP_EPSILON
            < float(start)
        ):
            raise RuntimeError(
                "Whisper segment end precedes its start."
            )

        if start is not None:
            numeric_start = float(
                start
            )

            if (
                previous_segment_start
                is not None
                and numeric_start
                + TIMESTAMP_EPSILON
                < previous_segment_start
            ):
                raise RuntimeError(
                    "Whisper segment timestamps are not monotonic."
                )

            previous_segment_start = (
                numeric_start
            )

        for word in (
            segment.get("words")
            or []
        ):
            word_start = word.get(
                "start"
            )

            word_end = word.get(
                "end"
            )

            if (
                word_start is not None
                and word_end is not None
                and float(word_end)
                + TIMESTAMP_EPSILON
                < float(word_start)
            ):
                raise RuntimeError(
                    "Whisper word end precedes its start."
                )

            if word_start is None:
                continue

            numeric_word_start = float(
                word_start
            )

            if (
                previous_word_start
                is not None
                and numeric_word_start
                + TIMESTAMP_EPSILON
                < previous_word_start
            ):
                raise RuntimeError(
                    "Whisper word timestamps are not monotonic."
                )

            previous_word_start = (
                numeric_word_start
            )


def _audio_fingerprint(audio_path: Path) -> str:
    """Identify the exact downloaded enclosure bytes for budget accounting."""

    digest = hashlib.sha256()
    with audio_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _actual_usd(payload: object) -> Decimal | None:
    usage = payload.get("usage") if isinstance(payload, dict) else None
    cost = usage.get("cost") if isinstance(usage, dict) else None
    if isinstance(cost, bool) or not isinstance(cost, (int, float)):
        return None
    value = Decimal(str(cost))
    if not value.is_finite() or value < 0:
        return None
    return value


def _request_payload(chunk: AudioChunk, language: str | None) -> dict:
    payload = {
        "model": MODEL_ID,
        "input_audio": {
            "data": base64.b64encode(chunk.path.read_bytes()).decode("ascii"),
            "format": AUDIO_FORMAT,
        },
        "response_format": RESPONSE_FORMAT,
        "timestamp_granularities": list(TIMESTAMP_GRANULARITIES),
        "temperature": 0,
        # No ``provider`` routing here: OpenRouter ignores order/only/ignore on
        # transcription requests. The provider is pinned by the OpenRouter
        # guardrail on PODCAST_WHISPER_API_KEY (provider and model allowlists),
        # and a request served above the pinned price fails closed on budget
        # settlement.
    }
    if language:
        payload["language"] = language
    return payload


def _http_status(error: Exception) -> int | None:
    status = getattr(getattr(error, "response", None), "status_code", None)
    return status if isinstance(status, int) else None


def _retryable(error: Exception) -> bool:
    """Transient non-rate-limit failures that earn one more attempt."""

    if isinstance(error, (requests.Timeout, requests.ConnectionError)):
        return True
    status = _http_status(error)
    return status is not None and 500 <= status < 600


def _rate_limit_reason(error: Exception) -> str:
    """Short, secret-free description of a 429 for the run log.

    Records only the provider error message (truncated) and whether
    Retry-After was sent, so repeated limits can be attributed to the key,
    OpenRouter, or the upstream provider.
    """

    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None) or {}
    message = ""
    try:
        body = response.json() if response is not None else None
    except ValueError:
        body = None
    if isinstance(body, dict):
        error_body = body.get("error")
        if isinstance(error_body, dict) and isinstance(error_body.get("message"), str):
            message = error_body["message"]
    message = " ".join(message.split())[:160] or "no message"
    retry_after = "Retry-After" if headers.get("Retry-After") is not None else "no Retry-After"
    return f"{message}; {retry_after}"


def _rate_limit_delay_seconds(error: Exception, retry_index: int) -> float:
    """Wait before retrying an HTTP 429: Retry-After when given, else backoff.

    With the provider pinned by the key's guardrail, OpenRouter can no longer
    route around a rate-limited provider, so a 429 is waited out (bounded by
    the manifest) instead of failing the whole Worker run.
    """

    headers = getattr(getattr(error, "response", None), "headers", None) or {}
    try:
        seconds = float(headers.get("Retry-After"))
    except (TypeError, ValueError):
        seconds = None
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        seconds = float(
            RATE_LIMIT_BACKOFF_SECONDS[min(retry_index, len(RATE_LIMIT_BACKOFF_SECONDS) - 1)]
        )
    return min(seconds, float(RETRY_AFTER_CAP_SECONDS))


def _transcribe_chunk(
    chunk: AudioChunk,
    *,
    api_key: str,
    language: str | None,
    episode_key: str,
    budget_fingerprint: str,
    usd_per_second: Decimal,
) -> dict:
    """Send one budget-admitted chunk request and return the raw response."""

    reserved_usd = derive_audio_reservation_usd(
        max_billable_seconds=Decimal(str(chunk.duration)).quantize(Decimal("1."), rounding="ROUND_CEILING"),
        usd_per_second=usd_per_second,
    )
    body = _request_payload(chunk, language)
    transient_retries = 0
    rate_limit_retries = 0
    while True:
        attempt_id = f"{STAGE_WHISPER_TRANSCRIPTION}-{uuid4().hex}"
        reserve_budget_batch(
            episode_key,
            budget_fingerprint,
            [
                {
                    "attempt_id": attempt_id,
                    "stage": STAGE_WHISPER_TRANSCRIPTION,
                    "reserved_usd": reserved_usd,
                    "third_asr": False,
                }
            ],
        )
        try:
            response = requests.post(
                STT_ENDPOINT,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=body,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as error:
            mark_budget_attempt_uncertain(
                episode_key,
                budget_fingerprint,
                attempt_id,
                reason=type(error).__name__,
            )
            status = _http_status(error)
            if status == 429 and rate_limit_retries < RATE_LIMIT_ATTEMPTS - 1:
                delay = _rate_limit_delay_seconds(error, rate_limit_retries)
                rate_limit_retries += 1
                print(
                    f"Rate limited (HTTP 429); retrying chunk in {delay:.0f}s"
                    f" [{_rate_limit_reason(error)}]"
                )
                time.sleep(delay)
                continue
            if status != 429 and transient_retries < MAX_ATTEMPTS - 1 and _retryable(error):
                transient_retries += 1
                time.sleep(RETRY_DELAY_SECONDS)
                continue
            raise RuntimeError(
                "OpenRouter Whisper transcription failed"
                + (f" with HTTP {status}" if status is not None else "")
                + f" ({type(error).__name__})"
            ) from None
        settle_budget_attempt(
            episode_key,
            budget_fingerprint,
            attempt_id,
            actual_usd=_actual_usd(payload),
        )
        if not isinstance(payload, dict):
            raise RuntimeError("OpenRouter Whisper transcription response is not an object")
        return payload


def transcribe_audio(
    audio_file,
    episode_key,
):
    """Transcribe one local enclosure and store canonical TXT + JSON in GCS.

    Audio, encoded chunks, and upload staging files live in one temporary
    directory for this invocation only. The caller is responsible for making
    the RSS enclosure available locally; a GCS audio-path fallback would make
    temporary audio persistence part of resumability again. A missing
    credential or provider failure raises before anything is published, so
    the episode stays resumable.
    """

    audio_path = Path(audio_file)
    if not audio_path.is_file():
        raise ValueError(
            "Whisper transcription requires a local audio file for this execution."
        )
    api_key = os.getenv(WHISPER_API_KEY_ENV)
    if not api_key:
        raise RuntimeError(f"Missing {WHISPER_API_KEY_ENV} for Whisper transcription")

    producer = current_whisper_producer()
    budget_fingerprint = _audio_fingerprint(audio_path)
    pricing = resolve_audio_endpoint_pricing(
        MODEL_ID,
        provider=MODEL_PROVIDER,
        api_key=api_key,
    )

    with tempfile.TemporaryDirectory(prefix="podcast-whisper-") as directory:
        work_dir = Path(directory)
        chunk_dir, chunks, total_duration = _prepare_audio_chunks(audio_path, work_dir)
        print(f"Transcribing with {MODEL} via OpenRouter ({MODEL_PROVIDER})...")

        metadata_segments: list[dict] = []
        detected_language: str | None = None
        stitch_boundaries: list[dict] = []
        chunk_diagnostics: list[dict] = []
        billed_usd = Decimal("0")
        try:
            for chunk_number, chunk in enumerate(chunks, start=1):
                print(f"Transcribing chunk {chunk_number}/{len(chunks)}: {chunk.path.name}")
                response = _transcribe_chunk(
                    chunk,
                    api_key=api_key,
                    language=detected_language,
                    episode_key=episode_key,
                    budget_fingerprint=budget_fingerprint,
                    usd_per_second=pricing.usd_per_second,
                )
                billed_usd += _actual_usd(response) or Decimal("0")
                if detected_language is None and isinstance(response.get("language"), str):
                    detected_language = response["language"]
                chunk_segments = _openrouter_segment_metadata(
                    response,
                    offset_seconds=chunk.start,
                )
                usable_transcript = _chunk_has_usable_transcript(chunk_segments)
                chunk_diagnostics.append(
                    {
                        "chunk_number": chunk_number,
                        "start": chunk.start,
                        "duration": chunk.duration,
                        "segment_count": len(chunk_segments),
                        "word_count": _word_count(chunk_segments),
                        "has_usable_transcript": usable_transcript,
                    }
                )
                if _suspicious_empty_chunk(
                    chunk,
                    chunk_number=chunk_number,
                    chunk_count=len(chunks),
                    segments=chunk_segments,
                ):
                    raise RuntimeError(
                        "Whisper returned no usable transcript for substantial "
                        f"chunk {chunk_number}/{len(chunks)}; refusing to publish a "
                        "possibly incomplete canonical transcript."
                    )
                if chunk_number == 1:
                    metadata_segments = chunk_segments
                else:
                    metadata_segments, stitch_info = _stitch_chunk_overlap(
                        metadata_segments,
                        chunk_segments,
                        chunk_start=chunk.start,
                    )
                    stitch_boundaries.append(stitch_info)
                    print(
                        f"Stitched overlap at {stitch_info['cut_time']:.2f}s "
                        f"(-{stitch_info['previous_words_removed']} previous words, "
                        f"-{stitch_info['current_words_removed']} current words)"
                    )
                del response, chunk_segments
                gc.collect()
                print(f"Finished chunk {chunk_number}/{len(chunks)}")
        finally:
            shutil.rmtree(chunk_dir, ignore_errors=True)

        for segment_id, segment in enumerate(metadata_segments):
            segment["id"] = segment_id
        _validate_monotonic_timestamps(metadata_segments)
        print("Timestamp validation passed")

        text = " ".join(
            segment["text"] for segment in metadata_segments if segment.get("text")
        ).strip()
        transcript_file = work_dir / "transcript.txt"
        metadata_file = work_dir / "transcript.json"
        transcript_file.write_text(text, encoding="utf-8")

        removed_word_count = sum(
            boundary["previous_words_removed"] + boundary["current_words_removed"]
            for boundary in stitch_boundaries
        )
        metadata = {
            "schema_version": WHISPER_METADATA_SCHEMA_VERSION,
            "source": WHISPER_METADATA_SOURCE,
            "model": MODEL,
            "producer": producer,
            "language": detected_language,
            "duration": total_duration,
            "chunk_seconds": WHISPER_CHUNK_SECONDS,
            "chunk_overlap_seconds": WHISPER_CHUNK_OVERLAP_SECONDS,
            "chunk_count": len(chunks),
            "deduplication": STITCH_METHOD,
            "deduplicated_word_count": removed_word_count,
            "stitch_boundaries": stitch_boundaries,
            "chunk_diagnostics": chunk_diagnostics,
            "audio_fingerprint": budget_fingerprint,
            "billed_usd": str(billed_usd),
            "segments": metadata_segments,
        }
        metadata_file.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        source_paths = paths_for(episode_key)
        text_gcs_path = upload_path_to_gcs(
            transcript_file,
            source_paths["whisper_text"],
        )
        metadata_gcs_path = upload_path_to_gcs(
            metadata_file,
            source_paths["whisper_metadata"],
        )
        print(f"Transcript saved in GCS: {text_gcs_path}")
        return {"text": text_gcs_path, "metadata": metadata_gcs_path}
