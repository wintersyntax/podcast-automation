"""Cloud Storage handler for cloud-owned Apple transcript uploads.

Deploy ``gcs_apple_transcript_finalize`` as a GCS object-finalize function.
It accepts only the Mac's final completion object, validates its metadata,
copies both source files into the canonical episode directory, and atomically
updates the cloud-owned episode index.
"""

from __future__ import annotations

import json
import re

from google.api_core.exceptions import NotFound, PreconditionFailed
from google.cloud import storage

from .episode_contract import (
    ensure_v3_record,
    incoming_apple_paths,
    new_episode_record,
    now_iso,
    paths_for,
)
from .episode_identity import episode_key_for
from .observability import emit_event
from .storage import STORAGE_FILE
from .worker_control import request_worker_run


KEY_PATTERN = re.compile(r"^[0-9a-f]{24}$")
MAX_INDEX_RETRIES = 3


def _event_data(event) -> dict:
    if isinstance(event, dict):
        return event
    data = getattr(event, "data", None)
    return data if isinstance(data, dict) else {}


def _incoming_key(object_name: str) -> str | None:
    pieces = object_name.split("/")
    if (
        len(pieces) == 4
        and pieces[0] == "incoming"
        and pieces[1] == "apple"
        and pieces[3] == "apple-transcript.txt"
        and KEY_PATTERN.fullmatch(pieces[2])
    ):
        return pieces[2]
    return None


def _load_index(bucket) -> tuple[list[dict], int]:
    blob = bucket.blob(STORAGE_FILE)
    if not blob.exists():
        return [], 0
    content = blob.download_as_text(encoding="utf-8")
    blob.reload()
    return json.loads(content), int(blob.generation)


def _save_index(bucket, episodes: list[dict], generation: int) -> None:
    bucket.blob(STORAGE_FILE).upload_from_string(
        json.dumps(episodes, ensure_ascii=False, indent=2) + "\n",
        content_type="application/json",
        if_generation_match=generation,
    )


def _apple_ready_record(record: dict, source: dict, destination: dict[str, str]) -> dict:
    updated = ensure_v3_record(record)
    updated["files"]["sources"]["apple"] = {
        "text": destination["apple_text"],
        "metadata": destination["apple_metadata"],
    }
    updated["status"]["apple_transcript"] = {"state": "ready", "updated_at": now_iso()}
    whisper_ready = bool(updated["files"]["sources"]["whisper"].get("text"))
    updated["status"]["compiler"] = {
        "state": "ready" if whisper_ready else "blocked",
        "updated_at": now_iso(),
    }
    updated["updated_at"] = now_iso()
    return updated


def _record_from_apple_source(source: dict, episode_key: str) -> dict:
    episode = source["episode"]
    return _apple_ready_record(
        new_episode_record(
            episode_key=episode_key,
            podcast=episode.get("podcast") or episode.get("show_title") or "Unknown Podcast",
            podcast_id=episode.get("podcast_id"),
            feed_url=episode["feed_url"],
            rss_guid=episode["rss_guid"],
            title=episode.get("title") or "Untitled episode",
            published=episode.get("published"),
            link=episode.get("link"),
            audio_url=None,
        ),
        source,
        paths_for(episode_key),
    )


def _merge_index(bucket, episode_key: str, source: dict, destination: dict[str, str]) -> dict:
    episode = source["episode"]
    for _ in range(MAX_INDEX_RETRIES):
        episodes, generation = _load_index(bucket)
        match_index = next(
            (
                index
                for index, item in enumerate(episodes)
                if item.get("episode_key") == episode_key
            ),
            None,
        )
        if match_index is None:
            updated = _record_from_apple_source(source, episode_key)
            episodes.append(updated)
        else:
            updated = _apple_ready_record(episodes[match_index], source, destination)
            updated["episode_key"] = episode_key
            updated["feed_url"] = episode.get("feed_url")
            updated["podcast_id"] = episode.get("podcast_id") or updated.get("podcast_id")
            episodes[match_index] = updated
        try:
            _save_index(bucket, episodes, generation)
            return updated
        except PreconditionFailed:
            continue
    raise RuntimeError("episodes.json changed repeatedly while ingesting Apple transcript")


def _already_ingested(bucket, episode_key: str) -> bool:
    episodes, _ = _load_index(bucket)
    record = next(
        (item for item in episodes if item.get("episode_key") == episode_key),
        None,
    )
    if not record:
        return False
    if record.get("status", {}).get("apple_transcript", {}).get("state") != "ready":
        return False
    destination = paths_for(episode_key)
    return all(
        bucket.blob(path).exists()
        for path in (destination["apple_text"], destination["apple_metadata"])
    )


def _delete_incoming_blob(blob, generation: int) -> None:
    """Delete one consumed staging object without deleting a newer replacement."""

    try:
        blob.delete(if_generation_match=generation)
    except (NotFound, PreconditionFailed):
        # A retry may already have cleaned this object, or a new upload may
        # have replaced it.  In either case, leave the current object alone.
        pass


def ingest_apple_transcript(bucket_name: str, object_name: str) -> dict:
    """Ingest one completed Apple source object; safe to retry on duplicate events."""

    episode_key = _incoming_key(object_name)
    if not episode_key:
        return {"status": "ignored", "reason": "not_an_apple_completion_object"}
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    incoming = incoming_apple_paths(episode_key)
    metadata_blob = bucket.blob(incoming["metadata"])
    text_blob = bucket.blob(incoming["text"])
    metadata_exists = metadata_blob.exists()
    text_exists = text_blob.exists()
    if not metadata_exists or not text_exists:
        if _already_ingested(bucket, episode_key):
            return {"status": "already_ingested", "episode_key": episode_key}
        if not metadata_exists:
            raise ValueError("Apple completion arrived without apple-transcript.json")
        raise ValueError("Apple completion object is missing")

    metadata_blob.reload()
    text_blob.reload()
    metadata_generation = int(metadata_blob.generation)
    text_generation = int(text_blob.generation)
    source = json.loads(metadata_blob.download_as_text(encoding="utf-8"))
    episode = source.get("episode") if isinstance(source, dict) else None
    if not isinstance(episode, dict):
        raise ValueError("Apple metadata is missing its episode object")
    if (episode.get("episode_key") or episode.get("source_key")) != episode_key:
        raise ValueError("Apple metadata episode_key does not match its GCS object path")
    expected_key = episode_key_for(episode.get("feed_url", ""), episode.get("rss_guid", ""))
    if expected_key != episode_key:
        raise ValueError("Apple metadata feed_url/RSS GUID does not match episode_key")

    destination = paths_for(episode_key)
    bucket.copy_blob(
        text_blob,
        bucket,
        destination["apple_text"],
        source_generation=text_generation,
    )
    bucket.copy_blob(
        metadata_blob,
        bucket,
        destination["apple_metadata"],
        source_generation=metadata_generation,
    )
    _merge_index(bucket, episode_key, source, destination)
    _delete_incoming_blob(text_blob, text_generation)
    _delete_incoming_blob(metadata_blob, metadata_generation)
    return {"status": "ingested", "episode_key": episode_key}


def _request_worker_resume(episode_key: str) -> dict:
    """Best-effort acceleration after canonical Apple ingest succeeds.

    Failure here must never roll back or invalidate the already durable Apple
    source. Scheduled worker runs remain the recovery path.
    """

    try:
        request = request_worker_run(
            requested_by="apple_ingest",
            episode_key=episode_key,
        )
    except Exception as exc:
        emit_event(
            "apple_ingest_worker_resume_failed",
            "Apple transcript was ingested, but immediate worker resume failed; "
            "the normal scheduler remains the fallback.",
            severity="ERROR",
            episode_key=episode_key,
            error_type=type(exc).__name__,
        )
        return {
            "status": "failed",
            "operation": None,
        }

    emit_event(
        "apple_ingest_worker_resume_requested",
        "Apple transcript was ingested and an immediate worker resume was requested.",
        severity="INFO",
        episode_key=episode_key,
        operation=request.get("operation"),
        attempts=request.get("attempts"),
    )
    return {
        "status": "requested",
        "operation": request.get("operation"),
        "attempts": request.get("attempts"),
    }


def gcs_apple_transcript_finalize(event, context=None):
    """Cloud Functions/Cloud Run function entrypoint for a GCS finalize event."""

    data = _event_data(event)
    result = ingest_apple_transcript(data.get("bucket", ""), data.get("name", ""))
    if result.get("status") != "ingested":
        return result

    return {
        **result,
        "worker_resume": _request_worker_resume(result["episode_key"]),
    }
