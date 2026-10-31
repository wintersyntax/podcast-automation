"""Local audio clips and narrow third-ASR evidence for human review."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
from typing import Any

import requests

from .fingerprint_types import InputFingerprint


OPENROUTER_STT_URL = "https://openrouter.ai/api/v1/audio/transcriptions"
THIRD_ASR_MODEL = os.environ.get("THIRD_ASR_MODEL", "openai/gpt-transcribe")
REVIEW_CLIP_SECONDS = float(os.environ.get("REVIEW_CLIP_SECONDS", "12"))
REVIEW_CLIP_MAX_SECONDS = float(os.environ.get("REVIEW_CLIP_MAX_SECONDS", "15"))
# TASK-126: a card longer than the default clip gets a clip that covers its
# words plus a margin on each side, so the context anchors around the card
# are inside the clip. Speech rate is a conservative estimate for the longer
# of the two source readings.
REVIEW_CLIP_LONG_MAX_SECONDS = float(os.environ.get("REVIEW_CLIP_LONG_MAX_SECONDS", "30"))
REVIEW_CLIP_MARGIN_SECONDS = 3.0
REVIEW_CLIP_WORDS_PER_SECOND = 2.5
_CLIP_WORD = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)
THIRD_ASR_TIMEOUT_SECONDS = int(os.environ.get("THIRD_ASR_TIMEOUT_SECONDS", "120"))


def clip_window(
    start: float | None,
    end: float | None,
    *,
    duration: float | None = None,
    target_seconds: float = REVIEW_CLIP_SECONDS,
    max_seconds: float = REVIEW_CLIP_MAX_SECONDS,
) -> dict[str, float]:
    """Return a bounded, Whisper-timeline clip window around a conflict."""

    if target_seconds <= 0 or max_seconds < target_seconds:
        raise ValueError("Invalid review clip duration configuration")
    point_start = max(0.0, float(start if start is not None else 0.0))
    point_end = max(point_start, float(end if end is not None else point_start))
    span = point_end - point_start
    clip_seconds = min(max_seconds, max(target_seconds, span))
    center = (point_start + point_end) / 2.0
    window_start = max(0.0, center - clip_seconds / 2.0)
    window_end = window_start + clip_seconds
    if duration is not None:
        duration = max(0.0, float(duration))
        window_end = min(window_end, duration)
        window_start = max(0.0, window_end - clip_seconds)
    return {
        "start": round(window_start, 3),
        "end": round(window_end, 3),
        "duration": round(window_end - window_start, 3),
    }


def review_clip_window(item: dict, *, duration: float | None = None) -> dict[str, float]:
    """Clip window for one review card, long enough to hold its context.

    Short cards keep the default clip around the Whisper timestamps. A card
    whose Whisper span, or whose longer source reading at about 2.5 words per
    second, does not fit with a 3 s margin on each side gets a longer clip,
    capped at ``REVIEW_CLIP_LONG_MAX_SECONDS``. Reversed timestamps are read
    as a span.
    """

    start = item.get("whisper_start_timestamp")
    end = item.get("whisper_end_timestamp")
    if start is not None and end is not None and float(end) < float(start):
        start, end = end, start
    words = max(
        len(_CLIP_WORD.findall(value))
        for value in (item.get("apple_text"), item.get("whisper_text"), "")
        if isinstance(value, str)
    )
    span = 0.0
    if start is not None and end is not None:
        span = max(0.0, float(end) - float(start))
    needed = max(span, words / REVIEW_CLIP_WORDS_PER_SECOND) + 2 * REVIEW_CLIP_MARGIN_SECONDS
    if needed <= REVIEW_CLIP_SECONDS:
        return clip_window(start, end, duration=duration)
    # Whole seconds: providers bill whole seconds, and an integral clip keeps
    # the reservation equal to what is billed.
    target = float(min(REVIEW_CLIP_LONG_MAX_SECONDS, math.ceil(max(REVIEW_CLIP_MAX_SECONDS, needed))))
    return clip_window(start, end, duration=duration, target_seconds=target, max_seconds=target)


def extract_audio_clip(
    audio_file: str | Path,
    destination: str | Path,
    window: dict[str, float],
) -> Path:
    """Extract a review-safe mono WAV clip with ffmpeg."""

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-ss", f"{window['start']:.3f}", "-i", str(audio_file),
        "-t", f"{window['duration']:.3f}", "-map", "0:a:0", "-vn",
        "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(destination),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() or error.stdout.strip() or str(error)
        raise RuntimeError(f"FFmpeg review clip extraction failed: {detail}") from error
    return destination


def third_asr_cache_key(*, input_fingerprint: InputFingerprint, item_id: int, window: dict[str, float]) -> str:
    """Identify evidence, not a resolution, for one immutable review input."""

    payload = json.dumps(
        {"fingerprint": input_fingerprint, "item_id": item_id, "window": window, "model": THIRD_ASR_MODEL},
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _audio_request_payload(audio_file: str | Path) -> dict[str, Any]:
    path = Path(audio_file)
    audio_format = path.suffix.removeprefix(".").lower() or "wav"
    return {
        "model": THIRD_ASR_MODEL,
        "input_audio": {"data": base64.b64encode(path.read_bytes()).decode("ascii"), "format": audio_format},
    }


def parse_third_asr_response(payload: dict) -> dict:
    """Keep the useful OpenRouter STT fields and reject malformed results."""

    text = payload.get("text") if isinstance(payload, dict) else None
    if not isinstance(text, str) or not text.strip():
        raise ValueError("OpenRouter STT response does not contain transcript text")
    result = {"text": text.strip()}
    for field in ("language", "duration", "usage", "cost"):
        if field in payload:
            result[field] = payload[field]
    return result


def transcribe_review_clip(audio_file: str | Path) -> dict:
    """Request third-ASR evidence using only the dedicated third-ASR credential."""

    api_key = os.getenv("PODCAST_REVIEW_ASR_API_KEY")
    if not api_key:
        raise RuntimeError("Missing PODCAST_REVIEW_ASR_API_KEY for third-ASR review")
    response = requests.post(
        OPENROUTER_STT_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=_audio_request_payload(audio_file),
        timeout=THIRD_ASR_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    result = parse_third_asr_response(response.json())
    result.update({"model": THIRD_ASR_MODEL, "audio_format": Path(audio_file).suffix.removeprefix(".").lower()})
    return result


# TASK-076 Task 9: bounded failure taxonomy for transcribe_review_clip.
#
# See docs/superpowers/specs/2026-09-16-human-review-evidence-assisted-
# adjudication-design.md, "Retry and failure taxonomy". The caller
# (podcast_engine.human_review.ensure_third_asr) owns retry/budget
# orchestration; this module only classifies what already happened so
# that orchestration never has to re-derive network semantics itself.

THIRD_ASR_FAILURE_PRE_SEND_RETRYABLE = "pre_send_retryable"
THIRD_ASR_FAILURE_POST_SEND_UNCERTAIN = "post_send_uncertain"
THIRD_ASR_FAILURE_PROVIDER_RETRYABLE = "provider_retryable"
THIRD_ASR_FAILURE_PERMANENT = "permanent"

THIRD_ASR_RETRYABLE_FAILURE_CLASSES = frozenset(
    {
        THIRD_ASR_FAILURE_PRE_SEND_RETRYABLE,
        THIRD_ASR_FAILURE_POST_SEND_UNCERTAIN,
        THIRD_ASR_FAILURE_PROVIDER_RETRYABLE,
    }
)


def classify_third_asr_failure(error: BaseException) -> str:
    """Classify one transcribe_review_clip failure into the Task 9 taxonomy.

    Ambiguous outcomes -- anything Python cannot positively prove was
    never sent to the provider -- classify toward ``post_send_uncertain``,
    the budget-conservative default, rather than toward a class that
    could release a reservation for money that may already be spent.
    """

    # requests.exceptions.ConnectTimeout subclasses both ConnectionError
    # and Timeout, so it must be checked before either general case: a
    # timeout while still connecting proves no request reached the
    # provider.
    if isinstance(error, requests.exceptions.ConnectTimeout):
        return THIRD_ASR_FAILURE_PRE_SEND_RETRYABLE
    if isinstance(error, requests.exceptions.ConnectionError):
        return THIRD_ASR_FAILURE_PRE_SEND_RETRYABLE
    if isinstance(error, requests.exceptions.Timeout):
        # A read/ambiguous timeout after the connection was established
        # cannot disprove that the provider received the request.
        return THIRD_ASR_FAILURE_POST_SEND_UNCERTAIN
    if isinstance(error, requests.HTTPError):
        response = getattr(error, "response", None)
        status = getattr(response, "status_code", None)
        if status == 429 or (isinstance(status, int) and 500 <= status < 600):
            return THIRD_ASR_FAILURE_PROVIDER_RETRYABLE
        return THIRD_ASR_FAILURE_PERMANENT
    if isinstance(error, RuntimeError):
        # Missing/misconfigured credential: deterministic, not transient.
        return THIRD_ASR_FAILURE_PERMANENT
    # Malformed-response (ValueError from parse_third_asr_response) and any
    # other unexpected error: the request may already have reached the
    # provider, so stay conservative rather than guessing permanent.
    return THIRD_ASR_FAILURE_POST_SEND_UNCERTAIN
