"""Cloud-side acquisition of official Apple Podcasts transcripts.

This module is responsible only for obtaining an Apple transcript and writing
the existing immutable incoming source pair:

    incoming/apple/<episode_key>/apple-transcript.json
    incoming/apple/<episode_key>/apple-transcript.txt

The JSON object is written first. The TXT object is written last because the
existing GCS finalize handler treats it as the completion signal.

This module does not:

- generate Apple bearer tokens
- write episodes.json
- write canonical episode state
- perform macOS cache or GUI fallback

Token generation remains macOS-specific. Canonical state remains owned by
apple_ingest.py.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .apple_transcript import DEFAULT_STOREFRONT, resolve_and_fetch
from .apple_token_status import inspect_bearer_token
from .episode_contract import incoming_apple_paths
from .episode_identity import episode_key_for
from .observability import emit_event
from .storage import BUCKET_NAME, get_bucket


DEFAULT_TOKEN_PATH = Path("/secrets/apple/bearer_token")

# The macOS maintenance agent refreshes daily once five days or less remain.
# Reaching three days therefore means at least two refresh attempts failed:
# warn while the token still works, instead of first learning about it from
# failed acquisitions after expiry (TASK-116).
TOKEN_EARLY_WARNING_SECONDS = 3 * 24 * 60 * 60


def _token_path() -> Path:
    """Return the mounted Apple bearer-token path."""

    configured = os.environ.get("APPLE_PODCASTS_TOKEN_FILE")

    if configured:
        return Path(configured).expanduser()

    return DEFAULT_TOKEN_PATH


def read_bearer_token() -> str:
    """Read the Apple bearer token without logging or exposing it."""

    path = _token_path()

    try:
        token = path.read_text(encoding="utf-8").strip()
    except OSError as error:
        raise RuntimeError(
            f"Could not read Apple bearer token from {path}: {error}"
        ) from error

    if not token:
        emit_event(
            "apple_token_invalid",
            "Mounted Apple bearer token file is empty.",
        )
        raise RuntimeError(
            f"Apple bearer token file is empty: {path}"
        )

    token_status = inspect_bearer_token(token)

    if token_status["status"] in {"invalid", "expired"}:
        emit_event(
            "apple_token_invalid",
            "Mounted Apple bearer token is not usable.",
            token_status=token_status["status"],
        )
        raise RuntimeError("Mounted Apple bearer token is not usable.")

    if token_status["remaining_seconds"] <= TOKEN_EARLY_WARNING_SECONDS:
        emit_event(
            "apple_token_expiring",
            "Mounted Apple bearer token expires soon; the macOS refresh has not replaced it.",
            severity="WARNING",
            remaining_days=round(token_status["remaining_seconds"] / 86_400, 1),
        )

    return token


def _validate_episode_identity(episode: dict) -> str:
    """Validate and return the canonical episode key."""

    episode_key = episode.get("episode_key") or episode.get("id")
    feed_url = episode.get("feed_url")
    rss_guid = episode.get("guid") or episode.get("rss_guid")

    if not episode_key:
        raise ValueError("Episode is missing episode_key.")

    if not feed_url:
        raise ValueError("Episode is missing feed_url.")

    if not rss_guid:
        raise ValueError("Episode is missing RSS GUID.")

    expected_key = episode_key_for(
        feed_url,
        rss_guid,
    )

    if expected_key != episode_key:
        raise ValueError(
            "Episode identity mismatch: "
            f"expected {expected_key}, received {episode_key}."
        )

    return episode_key


def _apple_config(podcast_config: dict) -> dict:
    """Return and validate the Apple Podcasts configuration for one show."""

    apple = podcast_config.get("apple_podcasts")

    if not isinstance(apple, dict):
        raise ValueError(
            "Podcast configuration is missing apple_podcasts."
        )

    show_id = str(apple.get("show_id") or "").strip()

    if not show_id:
        raise ValueError(
            "Podcast configuration is missing apple_podcasts.show_id."
        )

    return {
        "show_id": show_id,
        "show_title": apple.get("show_title"),
        "storefront": (
            apple.get("storefront")
            or DEFAULT_STOREFRONT
        ),
    }


def _source_metadata(
    *,
    podcast_config: dict,
    episode: dict,
    result,
) -> dict:
    """Build the existing Apple incoming JSON contract."""

    metadata = dict(result.metadata or {})

    return {
        "source": (
            metadata.get("source")
            or "apple_podcasts_api"
        ),
        "episode": {
            "episode_key": episode["episode_key"],
            "podcast": (
                episode.get("podcast")
                or podcast_config.get("name")
            ),
            "podcast_id": (
                episode.get("podcast_id")
                or podcast_config.get("id")
            ),
            "feed_url": episode.get("feed_url"),
            "rss_guid": (
                episode.get("guid")
                or episode.get("rss_guid")
            ),
            "show_title": (
                podcast_config.get(
                    "apple_podcasts",
                    {},
                ).get("show_title")
            ),
            "title": episode.get("title"),
            "published": episode.get("published"),
            "link": episode.get("link"),
            "apple_episode_id": metadata.get(
                "apple_episode_id"
            ),
        },
        "metadata": metadata,
        "segments": result.segments or [],
    }


def upload_incoming_apple_source(
    *,
    podcast_config: dict,
    episode: dict,
    result,
) -> dict:
    """Upload the Apple JSON first and TXT last as the completion signal."""

    if not result.ready or result.transcript is None:
        raise ValueError(
            "Only a READY Apple transcript can be uploaded."
        )

    episode_key = _validate_episode_identity(
        episode
    )

    paths = incoming_apple_paths(
        episode_key
    )

    bucket = get_bucket()

    source = _source_metadata(
        podcast_config=podcast_config,
        episode=episode,
        result=result,
    )

    metadata_text = (
        json.dumps(
            source,
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )

    transcript_text = (
        result.transcript.rstrip()
        + "\n"
    )

    # Metadata must exist before the TXT completion signal is finalized.
    bucket.blob(
        paths["metadata"]
    ).upload_from_string(
        metadata_text,
        content_type="application/json",
    )

    print(
        "Uploaded Apple metadata: "
        f"gs://{BUCKET_NAME}/{paths['metadata']}"
    )

    # The existing GCS finalize function listens for this object.
    bucket.blob(
        paths["text"]
    ).upload_from_string(
        transcript_text,
        content_type="text/plain; charset=utf-8",
    )

    print(
        "Uploaded Apple transcript completion object: "
        f"gs://{BUCKET_NAME}/{paths['text']}"
    )

    return {
        "status": "uploaded",
        "episode_key": episode_key,
        "apple_episode_id": (
            source["episode"]["apple_episode_id"]
        ),
        "metadata_path": paths["metadata"],
        "text_path": paths["text"],
        "segment_count": len(
            result.segments or []
        ),
    }


def acquire_apple_transcript(
    podcast_config: dict,
    episode: dict,
    *,
    upload: bool = True,
    token: str | None = None,
) -> dict:
    """Resolve, download, parse, and optionally upload one Apple transcript."""

    episode_key = _validate_episode_identity(
        episode
    )

    apple = _apple_config(
        podcast_config
    )

    bearer_token = (
        token
        if token is not None
        else read_bearer_token()
    )

    result = resolve_and_fetch(
        token=bearer_token,
        apple_show_id=apple["show_id"],
        rss_guid=(
            episode.get("guid")
            or episode.get("rss_guid")
        ),
        title=episode.get("title"),
        storefront=apple["storefront"],
    )

    outcome = {
        "status": result.status,
        "message": result.message,
        "episode_key": episode_key,
        "metadata": result.metadata,
    }

    if not result.ready:
        return outcome

    if not upload:
        return {
            **outcome,
            "segment_count": len(
                result.segments or []
            ),
            "transcript_characters": len(
                result.transcript or ""
            ),
        }

    return upload_incoming_apple_source(
        podcast_config=podcast_config,
        episode=episode,
        result=result,
    )
