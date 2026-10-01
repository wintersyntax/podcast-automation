"""Canonical Whisper producer identity and artifact provenance.

TASK-125: the canonical Whisper source is produced by ``openai/whisper-large-v3``
through OpenRouter's transcription endpoint, pinned to one provider, instead of
a bundled local Faster-Whisper ``small`` model. The producer identity covers
every setting that changes the transcript (model, provider, request format,
audio encoding and chunking), so artifacts from another producer -- including
every earlier local Faster-Whisper artifact -- never match and are not reused.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


WHISPER_MODEL_MANIFEST_SCHEMA_VERSION = 2
WHISPER_PRODUCER_SCHEMA_VERSION = 2
WHISPER_METADATA_SCHEMA_VERSION = 3
WHISPER_METADATA_SOURCE = "openrouter-whisper"

MANIFEST_PATH = (
    Path(__file__).resolve().parents[1]
    / "config"
    / "whisper-model.json"
)


def _load_manifest() -> dict:
    manifest = json.loads(
        MANIFEST_PATH.read_text(encoding="utf-8")
    )
    if manifest.get("schema_version") != WHISPER_MODEL_MANIFEST_SCHEMA_VERSION:
        raise RuntimeError(
            "Unsupported Whisper model manifest schema version"
        )
    return manifest


MANIFEST = _load_manifest()

MODEL_ID = MANIFEST["model"]["id"]
MODEL_PROVIDER = MANIFEST["model"]["provider"]

ENGINE_NAME = MANIFEST["engine"]["name"]
STT_ENDPOINT = MANIFEST["engine"]["endpoint"]
RESPONSE_FORMAT = MANIFEST["engine"]["response_format"]
TIMESTAMP_GRANULARITIES = tuple(MANIFEST["engine"]["timestamp_granularities"])

AUDIO_FORMAT = MANIFEST["audio"]["format"]
AUDIO_SAMPLE_RATE = MANIFEST["audio"]["sample_rate"]
AUDIO_CHANNELS = MANIFEST["audio"]["channels"]
AUDIO_BITRATE_KBPS = MANIFEST["audio"]["bitrate_kbps"]

REQUEST_TIMEOUT_SECONDS = MANIFEST["runtime"]["request_timeout_seconds"]
MAX_ATTEMPTS = MANIFEST["runtime"]["max_attempts"]
RATE_LIMIT_ATTEMPTS = MANIFEST["runtime"]["rate_limit_attempts"]
RATE_LIMIT_BACKOFF_SECONDS = tuple(MANIFEST["runtime"]["rate_limit_backoff_seconds"])
RETRY_AFTER_CAP_SECONDS = MANIFEST["runtime"]["retry_after_cap_seconds"]

CHUNK_SECONDS = MANIFEST["chunking"]["chunk_seconds"]
CHUNK_OVERLAP_SECONDS = MANIFEST["chunking"][
    "overlap_seconds"
]
STITCH_METHOD = MANIFEST["chunking"]["stitch_method"]


def _canonical_json(value: dict) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def producer_payload() -> dict:
    """Return semantic producer identity excluding its fingerprint."""

    return {
        "producer_schema_version": WHISPER_PRODUCER_SCHEMA_VERSION,
        "model": {
            "id": MODEL_ID,
            "provider": MODEL_PROVIDER,
        },
        "engine": {
            "name": ENGINE_NAME,
            "endpoint": STT_ENDPOINT,
            "response_format": RESPONSE_FORMAT,
            "timestamp_granularities": list(TIMESTAMP_GRANULARITIES),
        },
        "audio": {
            "format": AUDIO_FORMAT,
            "sample_rate": AUDIO_SAMPLE_RATE,
            "channels": AUDIO_CHANNELS,
            "bitrate_kbps": AUDIO_BITRATE_KBPS,
        },
        "chunking": {
            "chunk_seconds": CHUNK_SECONDS,
            "overlap_seconds": CHUNK_OVERLAP_SECONDS,
            "stitch_method": STITCH_METHOD,
        },
    }


def producer_fingerprint(payload: dict) -> str:
    digest = hashlib.sha256(
        _canonical_json(payload)
    ).hexdigest()
    return f"sha256:{digest}"


def current_whisper_producer() -> dict:
    """Return the exact current canonical producer identity."""

    payload = producer_payload()
    return {
        **payload,
        "fingerprint": producer_fingerprint(payload),
    }


def whisper_metadata_matches_current_producer(
    metadata: object,
) -> bool:
    """Return whether canonical metadata is safe for current reuse."""

    if not isinstance(metadata, dict):
        return False

    if metadata.get("schema_version") != WHISPER_METADATA_SCHEMA_VERSION:
        return False

    if metadata.get("source") != WHISPER_METADATA_SOURCE:
        return False

    if not isinstance(metadata.get("segments"), list):
        return False

    producer = metadata.get("producer")
    if not isinstance(producer, dict):
        return False

    embedded_fingerprint = producer.get("fingerprint")
    if not isinstance(embedded_fingerprint, str):
        return False

    embedded_payload = {
        key: value
        for key, value in producer.items()
        if key != "fingerprint"
    }

    if (
        producer_fingerprint(embedded_payload)
        != embedded_fingerprint
    ):
        return False

    try:
        expected = current_whisper_producer()
    except RuntimeError:
        return False

    return producer == expected
