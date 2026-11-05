"""Shared immutable episode source-generation identity.

TASK-076: this is the single canonical fingerprint of the exact Apple and
Whisper source bytes one episode generation is built from. The Human Review
lifecycle (podcast_engine.compilation) and the episode AI budget ledger both
key their generation identity off this same fingerprint, so a changed
transcript source invalidates both consistently instead of drifting apart.

source_generation_fingerprint(...) must stay byte-for-byte compatible with
the resolver's pre-existing source_fingerprint contract: already-persisted
Human Review records key off this exact payload/hash and must keep matching.
"""

from __future__ import annotations

import hashlib
import json

from .storage import download_gcs_bytes


class EpisodeSourceGenerationError(ValueError):
    """Raised when an episode's canonical source generation cannot be loaded.

    This is a fail-closed rejection, not a best-effort default: callers that
    need the immutable generation identity (Human Review, the budget ledger)
    must not proceed on a guessed or partial identity.
    """


def _bytes_digest(value: bytes | None) -> str | None:
    """Return the content digest used by the durable review-state contract."""

    return (
        "sha256:" + hashlib.sha256(value).hexdigest()
        if value is not None
        else None
    )


def source_generation_fingerprint(
    *,
    apple_text: bytes,
    apple_metadata: bytes | None,
    whisper_text: bytes,
    whisper_metadata: bytes | None,
) -> str:
    """Fingerprint the exact source bytes that identify one episode generation.

    This is the canonical identity payload. It must remain byte-for-byte
    compatible with the resolver's pre-existing source_fingerprint contract.
    """

    payload = {
        "apple_text": _bytes_digest(apple_text),
        "apple_metadata": _bytes_digest(apple_metadata),
        "whisper_text": _bytes_digest(whisper_text),
        "whisper_metadata": _bytes_digest(whisper_metadata),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _source_text_path(source: object) -> str | None:
    if not isinstance(source, dict):
        return None
    path = source.get("text")
    return path if isinstance(path, str) and path else None


def _source_metadata_path(source: object) -> str | None:
    if not isinstance(source, dict):
        return None
    path = source.get("metadata")
    return path if isinstance(path, str) and path else None


def source_generation_fingerprint_from_episode(episode: dict) -> str:
    """Load and fingerprint one episode's canonical Apple/Whisper sources.

    This is the shared entry point for anything that needs the same
    immutable generation identity Human Review uses, without duplicating
    its internals: it downloads the exact canonical source bytes named on
    the episode contract and fingerprints them with
    source_generation_fingerprint. A missing or malformed canonical source
    shape fails closed with EpisodeSourceGenerationError rather than
    silently fingerprinting a partial or guessed identity.
    """

    if not isinstance(episode, dict):
        raise EpisodeSourceGenerationError("episode must be a dict")

    files = episode.get("files")
    sources = files.get("sources") if isinstance(files, dict) else None
    sources = sources if isinstance(sources, dict) else {}

    apple = sources.get("apple")
    whisper = sources.get("whisper")

    apple_text_path = _source_text_path(apple)
    if apple_text_path is None:
        raise EpisodeSourceGenerationError(
            "episode is missing a canonical Apple source text path"
        )

    whisper_text_path = _source_text_path(whisper)
    if whisper_text_path is None:
        raise EpisodeSourceGenerationError(
            "episode is missing a canonical Whisper source text path"
        )

    apple_metadata_path = _source_metadata_path(apple)
    whisper_metadata_path = _source_metadata_path(whisper)

    return source_generation_fingerprint(
        apple_text=download_gcs_bytes(apple_text_path),
        apple_metadata=(
            download_gcs_bytes(apple_metadata_path)
            if apple_metadata_path is not None
            else None
        ),
        whisper_text=download_gcs_bytes(whisper_text_path),
        whisper_metadata=(
            download_gcs_bytes(whisper_metadata_path)
            if whisper_metadata_path is not None
            else None
        ),
    )
