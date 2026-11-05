"""Delete the stored review audio once an episode is finished (TASK-144).

The Worker stores the exact audio each Whisper source was made from
(``sources/whisper/audio.mp3``) so review clips follow the transcript even
when the host inserts ads dynamically. It is needed only while the episode is
in Human Review. User decision: the transcript is what is kept; a later
re-transcription simply downloads the audio again (and stores a new copy for
its own timestamps). So the copy is deleted as soon as the episode is
finished (completed, or recorded as a republished copy); an unfinished
episode keeps it. ``RETAINED_EPISODES`` finished episodes (most recent first)
may keep theirs; the user chose none.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable, Iterable

from google.api_core.exceptions import NotFound

from .episode_contract import now_iso, paths_for
from .episode_duplicates import is_duplicate, published_at

RETAINED_EPISODES = 0
_OLDEST = datetime.min.replace(tzinfo=timezone.utc)


def _finished(episode: dict) -> bool:
    status = episode.get("status") or {}
    completed = (
        (status.get("compiler") or {}).get("state") == "completed"
        and (status.get("summary") or {}).get("state") == "ready"
    )
    return completed or is_duplicate(episode)


def _recency(episode: dict) -> datetime:
    return (
        published_at(episode.get("published"))
        or published_at(episode.get("created_at"))
        or _OLDEST
    )


def retention_candidates(episodes: Iterable[dict], keep: int = RETAINED_EPISODES) -> list[dict]:
    """Finished episodes beyond the latest ``keep`` whose audio is not deleted yet."""

    ranked = sorted(episodes, key=_recency, reverse=True)
    return [
        episode
        for episode in ranked[keep:]
        if _finished(episode) and not episode.get("review_audio_deleted_at")
    ]


def apply_review_audio_retention(
    episodes: Iterable[dict],
    *,
    bucket,
    update_episode: Callable[..., dict | None],
    keep: int = RETAINED_EPISODES,
) -> list[str]:
    """Delete the review audio of older finished episodes; return their keys.

    A missing object (an episode transcribed before TASK-144) is recorded as
    deleted too, so each episode is handled once.
    """

    deleted = []
    for episode in retention_candidates(episodes, keep):
        key = episode["episode_key"]
        try:
            bucket.blob(paths_for(key)["whisper_review_audio"]).delete()
        except NotFound:
            pass
        update_episode(key, review_audio_deleted_at=now_iso())
        deleted.append(key)
    return deleted
