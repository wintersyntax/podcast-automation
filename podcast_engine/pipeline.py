"""Cloud-side episode workflow: audio, Whisper, Apple review, and summary."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

from .apple_acquisition import acquire_apple_transcript
from .compilation import compile_episode_sources, review_record_is_current
from .human_review import (
    load_review_record,
    pending_review_items,
    review_queue_fingerprint,
)
from .config import ENABLED_PODCASTS
from .downloader import downloaded_audio
from .episode_contract import (
    ensure_v3_record,
    paths_for,
    summary_is_vault_sync_ready,
)
from .episode_identity import episode_key_for
from .rss import get_latest_episode
from .whisper_provenance import whisper_metadata_matches_current_producer
from .storage import (
    add_episode,
    download_gcs_bytes,
    file_exists_in_gcs,
    get_episode_by_key,
    load_episodes,
    merge_episode_metadata,
    update_episode,
)
from .knowledge import build_knowledge_note, canonical_summary_generated_at
from .notifications import (
    send_email_review_notification,
    send_email_review_refreshed_notification,
    send_slack_review_notification,
    send_slack_review_refreshed_notification,
    send_slack_summary_ready_notification,
)
from .observability import emit_event
from .transcriber import transcribe_audio


APPLE_TRANSCRIPT_DELAY_DAYS = int(
    os.environ.get(
        "APPLE_TRANSCRIPT_DELAY_DAYS",
        "2",
    )
)

APPLE_TRANSCRIPT_LATE_DAYS = 10


DEFAULT_REVIEW_URL = (
    "https://review.example.com"
)

PODCAST_REVIEW_URL = (
    os.environ.get("PODCAST_REVIEW_URL", DEFAULT_REVIEW_URL).strip()
    or DEFAULT_REVIEW_URL
)


def _triage_unavailable_count(review_items: object) -> int:
    """Count pending cards whose advisory triage fell back to unavailable."""

    if not isinstance(review_items, list):
        return 0
    return sum(
        1
        for item in review_items
        if isinstance(item, dict)
        and isinstance(item.get("triage"), dict)
        and item["triage"].get("status") == "unavailable"
    )


def _log_human_review_required(
    episode: dict,
    pending_count: int,
) -> None:
    """Emit one structured Cloud Logging event for a new review queue."""

    emit_event(
        "human_review_required",
        "Podcast human review required",
        severity="NOTICE",
        podcast=episode.get("podcast"),
        title=episode.get("title"),
        episode_key=episode.get("episode_key"),
        pending_count=int(pending_count),
        review_url=PODCAST_REVIEW_URL,
    )


def _log_human_review_refreshed(
    episode: dict,
    previous_review: list[dict],
    review: list[dict],
) -> None:
    """Emit one structured event for a materially changed refreshed queue."""

    emit_event(
        "human_review_refreshed",
        "Podcast human review queue refreshed",
        severity="NOTICE",
        podcast=episode.get("podcast"),
        title=episode.get("title"),
        episode_key=episode.get("episode_key"),
        previous_pending_count=len(previous_review),
        pending_count=len(review),
        previous_queue_fingerprint=review_queue_fingerprint(previous_review),
        queue_fingerprint=review_queue_fingerprint(review),
        review_url=PODCAST_REVIEW_URL,
    )



def _review_gate_state(
    episode: dict,
) -> tuple[bool, list[dict] | None]:
    """Return whether a valid durable review gate still blocks this episode."""

    compiler_state = (
        episode.get("status", {})
        .get("compiler", {})
        .get("state")
    )

    if compiler_state != "review_required":
        return False, None

    episode_key = episode.get("episode_key")

    if not isinstance(episode_key, str) or not episode_key:
        emit_event(
            "human_review_state_inconsistent",
            "Human review state is inconsistent; pipeline is blocked.",
            severity="ERROR",
            episode_key=None,
            podcast=episode.get("podcast"),
            compiler_state="review_required",
            reason="missing_episode_key",
        )

        raise RuntimeError(
            "Episode requires human review but has no valid episode key."
        )

    try:
        record = load_review_record(
            episode_key
        )
    except (
        FileNotFoundError,
        ValueError,
        UnicodeDecodeError,
    ) as error:
        emit_event(
            "human_review_state_inconsistent",
            "Human review state is inconsistent; pipeline is blocked.",
            severity="ERROR",
            episode_key=episode_key,
            podcast=episode.get("podcast"),
            compiler_state="review_required",
            reason=type(error).__name__,
        )

        raise RuntimeError(
            "Episode requires human review but its durable review record "
            "is missing or invalid."
        ) from error

    human_review = record.get("human_review")
    human_decisions = record.get(
        "human_decisions",
        [],
    )

    valid_contract = (
        record.get("episode_key") == episode_key
        and isinstance(human_review, list)
        and isinstance(human_decisions, list)
        and all(
            isinstance(item, dict)
            and isinstance(item.get("id"), int)
            for item in human_review
        )
        and all(
            isinstance(item, dict)
            for item in human_decisions
        )
    )

    if not valid_contract:
        emit_event(
            "human_review_state_inconsistent",
            "Human review state is inconsistent; pipeline is blocked.",
            severity="ERROR",
            episode_key=episode_key,
            podcast=episode.get("podcast"),
            compiler_state="review_required",
            reason="invalid_review_record_contract",
        )

        raise RuntimeError(
            "Episode requires human review but its durable review record "
            "does not match the expected contract."
        )

    return review_record_is_current(episode, record), pending_review_items(record)


def _rss_episode(podcast_config: dict) -> dict | None:
    episode = get_latest_episode(
        podcast_config["rss"],
        podcast_config["name"],
    )

    if not episode or "error" in episode:
        return None

    episode["category"] = podcast_config.get(
        "category"
    )

    episode["prompt"] = podcast_config.get(
        "prompt"
    )

    episode["podcast_id"] = podcast_config.get(
        "id"
    )

    episode["podcast_url"] = podcast_config.get(
        "podcast_url"
    )

    return episode


def discover_latest_rss_episode(
    podcast_config: dict,
) -> dict | None:
    """Discover the current RSS item and return its canonical GCS record."""

    rss_episode = _rss_episode(
        podcast_config
    )

    if rss_episode is None:
        return None

    episode_key = episode_key_for(
        rss_episode["feed_url"],
        rss_episode["guid"],
    )
    episode = get_episode_by_key(episode_key)

    if episode:
        episode = merge_episode_metadata(
            episode_key,
            rss_episode,
        )

        if episode is None:
            raise RuntimeError(
                "Existing episode could not be refreshed"
            )

    else:
        episode = add_episode(
            rss_episode
        )

    return ensure_v3_record(episode)


def _published_datetime(
    value: str | None,
) -> datetime | None:
    """Parse the publication date formats used by RSS feeds."""

    if not value:
        return None

    try:
        parsed = parsedate_to_datetime(value)

        if parsed.tzinfo is None:
            parsed = parsed.replace(
                tzinfo=timezone.utc
            )

        return parsed.astimezone(
            timezone.utc
        )

    except (
        TypeError,
        ValueError,
        OverflowError,
    ):
        pass

    try:
        parsed = datetime.fromisoformat(
            value.replace(
                "Z",
                "+00:00",
            )
        )

        if parsed.tzinfo is None:
            parsed = parsed.replace(
                tzinfo=timezone.utc
            )

        return parsed.astimezone(
            timezone.utc
        )

    except ValueError:
        return None


def _apple_acquisition_eligible_at(
    episode: dict,
) -> datetime | None:
    """Return when Apple acquisition becomes eligible."""

    published = _published_datetime(
        episode.get("published")
    )

    if published is None:
        return None

    return published + timedelta(
        days=APPLE_TRANSCRIPT_DELAY_DAYS
    )


def _apple_acquisition_due(
    episode: dict,
    now: datetime | None = None,
) -> bool:
    """Allow Apple API acquisition after the configured publication delay."""

    eligible_at = (
        _apple_acquisition_eligible_at(
            episode
        )
    )

    # Missing or malformed publication metadata should not permanently block
    # transcript acquisition. The delay is an optimization, not identity.
    if eligible_at is None:
        return True

    current = (
        now
        or datetime.now(
            timezone.utc
        )
    )

    if current.tzinfo is None:
        current = current.replace(
            tzinfo=timezone.utc
        )

    current = current.astimezone(
        timezone.utc
    )

    return current >= eligible_at


def _apple_transcript_late_notification_due(
    episode: dict,
    now: datetime | None = None,
) -> tuple[datetime, datetime, int] | None:
    """Return late-event timing metadata when one notification is due."""

    if episode.get("apple_transcript_late_notified_at"):
        return None

    published = _published_datetime(
        episode.get("published")
    )

    if published is None:
        return None

    current = (
        now
        or datetime.now(
            timezone.utc
        )
    )

    if current.tzinfo is None:
        current = current.replace(
            tzinfo=timezone.utc
        )

    current = current.astimezone(
        timezone.utc
    )
    age = current - published

    if age < timedelta(days=APPLE_TRANSCRIPT_LATE_DAYS):
        return None

    eligible_at = _apple_acquisition_eligible_at(
        episode
    )

    # ``published`` was parsed above, so the configured two-day threshold
    # always yields a timestamp here.
    assert eligible_at is not None

    return current, eligible_at, age.days


def _emit_apple_transcript_late_if_needed(
    episode: dict,
    *,
    now: datetime | None = None,
) -> dict:
    """Persist and emit the one-shot Apple-transcript late notification."""

    timing = _apple_transcript_late_notification_due(
        episode,
        now=now,
    )

    if timing is None:
        return episode

    current, eligible_at, age_days = timing
    updated = update_episode(
        episode["episode_key"],
        apple_transcript_late_notified_at=current.isoformat(),
    )

    if updated is None:
        raise RuntimeError(
            "Could not persist Apple transcript late notification marker."
        )

    emit_event(
        "apple_transcript_late",
        "Apple transcript is still unavailable after ten days.",
        severity="NOTICE",
        podcast=updated.get("podcast"),
        title=updated.get("title"),
        episode_key=updated.get("episode_key"),
        published=updated.get("published"),
        age_days=age_days,
        apple_eligible_at=eligible_at.isoformat(),
    )

    return updated


def _transcribe_if_needed(
    episode: dict,
) -> dict:
    """Reacquire RSS audio locally whenever Whisper is not durable yet."""

    if episode["status"]["whisper"].get(
        "state"
    ) == "ready":
        return episode

    source_paths = paths_for(episode["episode_key"])
    whisper_text = source_paths["whisper_text"]
    whisper_metadata = source_paths["whisper_metadata"]
    if (
        file_exists_in_gcs(whisper_text)
        and file_exists_in_gcs(whisper_metadata)
    ):
        try:
            metadata = json.loads(
                download_gcs_bytes(whisper_metadata).decode("utf-8")
            )
        except (json.JSONDecodeError, UnicodeDecodeError):
            metadata = None
        if (
            isinstance(metadata, dict)
            and whisper_metadata_matches_current_producer(metadata)
        ):
            return update_episode(
                episode["episode_key"],
                transcript_file=whisper_text,
                whisper_metadata_file=whisper_metadata,
            )

    audio_url = episode.get("audio_url")
    if not isinstance(audio_url, str) or not audio_url:
        raise ValueError("Episode has no audio URL yet")

    started = time.time()
    with downloaded_audio(
        audio_url,
        f"{episode['episode_key']}.mp3",
    ) as audio_file:
        episode = update_episode(
            episode["episode_key"],
            download_completed=True,
        )
        whisper_files = transcribe_audio(audio_file, episode["episode_key"])

    print(f"END transcription: {time.time() - started:.2f}s")
    return update_episode(
        episode["episode_key"],
        transcript_file=whisper_files["text"],
        whisper_metadata_file=whisper_files["metadata"],
    )


def resume_tracked_episode(
    podcast_config: dict,
    episode: dict,
) -> dict:
    """Resume one canonical GCS episode without rediscovering RSS history."""

    episode = ensure_v3_record(episode)
    was_vault_sync_ready = summary_is_vault_sync_ready(episode)

    if _episode_is_completed(episode):
        return {
            "status": "skipped",
            "reason": "completed",
            "episode": episode,
        }

    print(
        f"START pipeline: "
        f"{podcast_config['name']} "
        f"({episode.get('episode_key')})"
    )

    episode = _transcribe_if_needed(
        episode
    )

    apple_source = (
        episode["files"]["sources"][
            "apple"
        ].get("text")
    )

    if not apple_source:
        eligible_at = (
            _apple_acquisition_eligible_at(
                episode
            )
        )

        if not _apple_acquisition_due(
            episode
        ):
            return {
                "status": (
                    "waiting_for_apple_transcript"
                ),
                "reason": "delay",
                "apple_eligible_at": (
                    eligible_at.isoformat()
                    if eligible_at
                    else None
                ),
                "episode": episode,
            }

        episode = _emit_apple_transcript_late_if_needed(
            episode
        )

        started = time.time()

        apple_result = (
            acquire_apple_transcript(
                podcast_config,
                episode,
            )
        )

        print(
            "END Apple acquisition: "
            f"{time.time() - started:.2f}s"
        )

        if (
            apple_result.get("status")
            == "uploaded"
        ):
            # apple-transcript.txt has just been finalized. The existing GCS
            # ingest function now owns validation and canonical state updates.
            # Do not bypass that boundary inside this worker invocation.
            return {
                "status": (
                    "waiting_for_apple_transcript"
                ),
                "reason": "ingest_pending",
                "apple": apple_result,
                "episode": episode,
            }

        if apple_result.get("status") == "API_ERROR":
            emit_event(
                "apple_acquisition_failed",
                "Cloud Apple transcript acquisition failed.",
                episode_key=episode.get("episode_key"),
                podcast=episode.get("podcast"),
            )

        return {
            "status": (
                "waiting_for_apple_transcript"
            ),
            "reason": "not_ready",
            "apple": apple_result,
            "episode": episode,
        }

    compiler_status = (
        episode["status"][
            "compiler"
        ].get("state")
    )

    previous_review: list[dict] | None = None

    if compiler_status == "review_required":
        current_review_gate, pending_review = _review_gate_state(
            episode
        )

        if current_review_gate and pending_review:
            return {
                "status": "review_required",
                "episode": episode,
                "pending_count": len(
                    pending_review
                ),
            }

        # The durable record was structurally valid but no longer current.
        # Preserve its pending cards for one comparison after recompilation.
        previous_review = pending_review or []

    if compiler_status != "completed":
        started = time.time()

        compiled = compile_episode_sources(
            episode
        )

        print(
            "END Apple/Whisper compilation: "
            f"{time.time() - started:.2f}s"
        )

        episode = update_episode(
            episode["episode_key"],
            compiled_transcript_file=(
                compiled["transcript"]
            ),
            compiler_report_file=(
                compiled["report"]
            ),
            compiler_review_required=(
                compiled[
                    "review_required"
                ]
            ),
        )

        if compiled[
            "review_required"
        ]:
            if compiler_status != "review_required":
                _log_human_review_required(
                    episode,
                    compiled["review_required"],
                )

                send_slack_review_notification(
                    episode,
                    compiled["review_required"],
                    PODCAST_REVIEW_URL,
                    triage_unavailable=_triage_unavailable_count(
                        compiled.get("review")
                    ),
                )
                send_email_review_notification(
                    episode,
                    compiled["review_required"],
                    PODCAST_REVIEW_URL,
                )
            elif previous_review is not None:
                review = compiled["review"]
                previous_fingerprint = review_queue_fingerprint(previous_review)
                queue_fingerprint = review_queue_fingerprint(review)
                if previous_fingerprint != queue_fingerprint:
                    _log_human_review_refreshed(
                        episode,
                        previous_review,
                        review,
                    )
                    send_slack_review_refreshed_notification(
                        episode,
                        len(previous_review),
                        len(review),
                        PODCAST_REVIEW_URL,
                    )
                    send_email_review_refreshed_notification(
                        episode,
                        len(previous_review),
                        len(review),
                        PODCAST_REVIEW_URL,
                    )

            return {
                "status": "review_required",
                "episode": episode,
                "review": (
                    compiled["review"]
                ),
            }

    # Knowledge owns its own fingerprint/manifest cache.
    started = time.time()
    if build_knowledge_note(episode) is None:
        return {
            "status": "waiting_for_summary_review_activation",
            "reason": "reviewer_not_activated",
            "episode": episode,
        }

    print(
        f"END summary: "
        f"{time.time() - started:.2f}s"
    )
    knowledge_paths = paths_for(episode["episode_key"])
    episode = update_episode(
        episode["episode_key"],
        summary_body_file=knowledge_paths["summary_body"],
        summary_metadata_file=knowledge_paths["summary_metadata"],
        markdown_file=knowledge_paths["summary"],
    )

    if episode is None:
        raise RuntimeError("Could not persist canonical summary state.")

    if (
        not was_vault_sync_ready
        and summary_is_vault_sync_ready(episode)
    ):
        try:
            generated_at = canonical_summary_generated_at(episode["episode_key"])
        except Exception as error:
            emit_event(
                "summary_ready_slack_failed",
                "Slack summary-ready notification could not be prepared.",
                severity="WARNING",
                episode_key=episode.get("episode_key"),
                error_type=type(error).__name__,
            )
            generated_at = None
        if generated_at is not None:
            send_slack_summary_ready_notification(episode, generated_at)

    print(
        "END pipeline"
    )

    return {
        "status": "completed",
        "episode": episode,
    }


def process_episode(
    podcast_config: dict,
) -> dict:
    """Discover and process the newest RSS episode for one podcast."""

    try:
        episode = discover_latest_rss_episode(
            podcast_config
        )
    except RuntimeError as error:
        return {
            "status": "error",
            "message": str(error),
        }

    if episode is None:
        return {
            "status": "error",
            "message": (
                "Could not load the "
                "latest RSS episode"
            ),
        }

    return resume_tracked_episode(
        podcast_config,
        episode,
    )


def _belongs_to_podcast(
    episode: dict,
    podcast_config: dict,
) -> bool:
    """Match a stored record to its configured podcast without RSS fabrication."""

    configured_id = podcast_config.get("id")
    episode_id = episode.get("podcast_id")

    if configured_id and episode_id:
        return episode_id == configured_id

    if episode.get("feed_url"):
        return episode.get("feed_url") == podcast_config.get("rss")

    return episode.get("podcast") == podcast_config.get("name")


def _episode_is_completed(
    episode: dict,
) -> bool:
    """Return whether every worker-owned terminal stage has completed."""

    status = episode.get("status", {})
    return (
        status.get("compiler", {}).get("state") == "completed"
        and status.get("summary", {}).get("state") == "ready"
    )


def _tracked_episodes_to_resume(
    podcast_config: dict,
    *,
    processed_episode_keys: set[str],
) -> list[dict]:
    """Return incomplete GCS records, excluding the RSS record already handled."""

    episodes: list[dict] = []

    for stored_episode in load_episodes():
        episode = ensure_v3_record(stored_episode)
        episode_key = episode.get("episode_key")

        if (
            not episode_key
            or episode_key in processed_episode_keys
            or not _belongs_to_podcast(episode, podcast_config)
            or _episode_is_completed(episode)
        ):
            continue

        episodes.append(episode)

    return episodes


def run_pipeline(
    *,
    target_episode_key: str | None = None,
) -> list[dict]:
    """Discover current RSS episodes or resume one exact stored episode."""

    if target_episode_key is not None:
        episode = get_episode_by_key(target_episode_key)
        if episode is None:
            raise RuntimeError(f"Target episode was not found: {target_episode_key}")
        matching_podcasts = [
            podcast
            for podcast in ENABLED_PODCASTS
            if _belongs_to_podcast(episode, podcast)
        ]
        if len(matching_podcasts) != 1:
            raise RuntimeError("Target episode must match exactly one enabled podcast")
        return [resume_tracked_episode(matching_podcasts[0], episode)]

    outcomes = []

    for podcast in ENABLED_PODCASTS:
        try:
            processed_episode_keys: set[str] = set()

            try:
                latest_episode = discover_latest_rss_episode(
                    podcast
                )
            except RuntimeError as error:
                outcomes.append(
                    {
                        "status": "error",
                        "message": str(error),
                    }
                )
            else:
                if latest_episode is None:
                    outcomes.append(
                        {
                            "status": "error",
                            "message": (
                                "Could not load the "
                                "latest RSS episode"
                            ),
                        }
                    )
                else:
                    processed_episode_keys.add(
                        latest_episode["episode_key"]
                    )
                    outcomes.append(
                        resume_tracked_episode(
                            podcast,
                            latest_episode,
                        )
                    )

            for episode in _tracked_episodes_to_resume(
                podcast,
                processed_episode_keys=(
                    processed_episode_keys
                ),
            ):
                processed_episode_keys.add(
                    episode["episode_key"]
                )
                outcomes.append(
                    resume_tracked_episode(
                        podcast,
                        episode,
                    )
                )
        except Exception as error:
            emit_event(
                "worker_failed",
                "Podcast worker stopped on an unrecoverable pipeline error.",
                podcast=podcast.get("name"),
                error_type=type(error).__name__,
            )
            raise

    return outcomes
