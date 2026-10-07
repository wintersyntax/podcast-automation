"""Versioned canonical GCS paths and episode state."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from urllib.parse import urlsplit, urlunsplit


SCHEMA_VERSION = 3


VAULT_SYNC_READY_SUMMARY_STATES = frozenset({"ready", "completed"})


def canonical_episode_url(value: str | None) -> str | None:
    """Normalize only the known broken Libsyn publisher-page URL pattern."""

    if value is None:
        return None
    if not isinstance(value, str):
        return value

    parsed = urlsplit(value)
    if (
        parsed.hostname != "example-strength.libsyn.com"
        or not parsed.path.startswith("/")
        or parsed.path.startswith("/website/")
        or parsed.path.count("/") != 1
        or not parsed.path.strip("/")
    ):
        return value

    return urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            f"/website{parsed.path}",
            parsed.query,
            parsed.fragment,
        )
    )


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def summary_is_vault_sync_ready(episode: dict) -> bool:
    """Return whether Vault Sync should select this episode's summary.

    ``completed`` remains an intentional compatibility state for records that
    predate the current ``ready`` summary-state contract.
    """

    summary = episode.get("status", {}).get("summary", {})
    return (
        isinstance(summary, dict)
        and summary.get("state") in VAULT_SYNC_READY_SUMMARY_STATES
    )


def paths_for(episode_key: str) -> dict[str, str]:
    """Return every canonical Cloud Storage path for one episode."""

    root = f"episodes/{episode_key}"
    return {
        "apple_text": f"{root}/sources/apple/transcript.txt",
        "apple_metadata": f"{root}/sources/apple/transcript.json",
        "whisper_text": f"{root}/sources/whisper/transcript.txt",
        "whisper_metadata": f"{root}/sources/whisper/transcript.json",
        "compiled_text": f"{root}/compiled/transcript.txt",
        "compiler_report": f"{root}/compiled/report.json",
        "resolver_record": f"{root}/review/resolver.json",
        "summary_body": f"{root}/summary/body.md",
        "summary_metadata": f"{root}/summary/metadata.json",
        "summary": f"{root}/summary/summary.md",
    }


def summary_review_artifact_paths(episode_key: str, review_id: str) -> dict[str, str]:
    """Return immutable artifact paths for one summary-review attempt."""

    root = f"episodes/{episode_key}/summary/reviews/{review_id}/"
    return {
        "root": root,
        "draft": root + "draft.md",
        "final": root + "final.md",
        "review": root + "review.json",
        "transcript_index": root + "transcript-span-index.json",
        "draft_index": root + "draft-block-index.json",
        "risk_inventory": root + "risk-inventory.json",
    }


def incoming_apple_paths(episode_key: str) -> dict[str, str]:
    root = f"incoming/apple/{episode_key}"
    return {
        "text": f"{root}/apple-transcript.txt",
        "metadata": f"{root}/apple-transcript.json",
    }


def _state(state: str, timestamp: str, **extra: object) -> dict:
    return {"state": state, "updated_at": timestamp, **extra}


def _state_or_default(value: object, state: str, timestamp: str) -> dict:
    if isinstance(value, dict) and isinstance(value.get("state"), str):
        return value
    return _state(state, timestamp)


def new_episode_record(
    *,
    episode_key: str,
    podcast: str,
    podcast_id: str | None,
    feed_url: str,
    podcast_url: str | None = None,
    rss_guid: str,
    title: str,
    published: str | None,
    link: str | None,
    audio_url: str | None,
    category: str | None = None,
    prompt: str | None = None,
) -> dict:
    """Create one Episode Contract V3 record without legacy aliases."""

    created_at = now_iso()
    return {
        "schema_version": SCHEMA_VERSION,
        "id": episode_key,
        "episode_key": episode_key,
        "podcast": podcast,
        "podcast_id": podcast_id,
        "podcast_url": podcast_url,
        "feed_url": feed_url,
        "category": category,
        "prompt": prompt,
        "guid": rss_guid,
        "title": title,
        "published": published,
        "link": canonical_episode_url(link),
        "audio_url": audio_url,
        "created_at": created_at,
        "updated_at": created_at,
        "apple_transcript_late_notified_at": None,
        "status": {
            "download": _state("pending", created_at),
            "apple_transcript": _state("pending", created_at),
            "whisper": _state("pending", created_at),
            "compiler": _state("blocked", created_at),
            "summary": _state("pending", created_at),
        },
        "files": {
            "sources": {
                "apple": {"text": None, "metadata": None},
                "whisper": {"text": None, "metadata": None},
            },
            "compiled": {"transcript": None, "report": None},
            "summary": {"body": None, "metadata": None, "markdown": None},
        },
    }


def ensure_v3_record(record: dict) -> dict:
    """Return a normalized current Episode Contract V3 record.

    Runtime callers accept only V3 records. Current optional fields and
    incremental nested objects receive safe defaults, but historical schemas
    and aliases are rejected rather than migrated.
    """

    if not isinstance(record, dict):
        raise ValueError("Episode Contract record must be an object")
    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            "Unsupported Episode Contract schema version "
            f"{record.get('schema_version')!r}; expected {SCHEMA_VERSION}"
        )

    upgraded = deepcopy(record)
    timestamp = upgraded.get("updated_at") or upgraded.get("created_at") or now_iso()
    upgraded.setdefault("episode_key", upgraded.get("id"))
    upgraded.setdefault("feed_url", None)
    upgraded.setdefault("podcast_id", None)
    upgraded.setdefault("podcast_url", None)
    upgraded.setdefault("updated_at", timestamp)
    upgraded.setdefault("apple_transcript_late_notified_at", None)
    upgraded["link"] = canonical_episode_url(upgraded.get("link"))

    def object_or_default(container: dict, key: str) -> dict:
        value = container.get(key)
        if value is None:
            value = {}
            container[key] = value
        if not isinstance(value, dict):
            raise ValueError(f"Episode Contract V3 {key} must be an object")
        return value

    status = object_or_default(upgraded, "status")
    files = object_or_default(upgraded, "files")
    for key in (
        "detected",
        "downloaded",
        "transcribed",
        "summarized",
        "exported",
        "export",
    ):
        if key in status:
            raise ValueError(
                f"Episode Contract V3 cannot contain obsolete status field {key!r}"
            )
    for key in ("audio", "export", "transcript", "markdown"):
        if key in files:
            raise ValueError(
                f"Episode Contract V3 cannot contain obsolete file field {key!r}"
            )

    sources = object_or_default(files, "sources")
    apple = object_or_default(sources, "apple")
    whisper = object_or_default(sources, "whisper")
    compiled = object_or_default(files, "compiled")
    summary = object_or_default(files, "summary")

    apple.setdefault("text", None)
    apple.setdefault("metadata", None)
    whisper.setdefault("text", None)
    whisper.setdefault("metadata", None)
    compiled.setdefault("transcript", None)
    compiled.setdefault("report", None)
    summary.setdefault("body", None)
    summary.setdefault("metadata", None)
    summary.setdefault("markdown", None)

    status["download"] = _state_or_default(
        status.get("download"),
        "pending",
        timestamp,
    )
    status["apple_transcript"] = _state_or_default(
        status.get("apple_transcript"),
        "ready" if apple.get("text") else "pending",
        timestamp,
    )
    status["whisper"] = _state_or_default(
        status.get("whisper"),
        "ready" if whisper.get("text") else "pending",
        timestamp,
    )
    compiler_default = "completed" if compiled.get("transcript") else (
        "ready" if apple.get("text") and whisper.get("text") else "blocked"
    )
    status["compiler"] = _state_or_default(
        status.get("compiler"), compiler_default, timestamp
    )
    status["summary"] = _state_or_default(
        status.get("summary"),
        "ready" if summary.get("markdown") else "pending",
        timestamp,
    )
    return upgraded
