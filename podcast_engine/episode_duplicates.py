"""Recognize an episode that republishes an already tracked conversation (TASK-135).

Shows cross-post the same conversation (for example a "BONUS:" episode on one
feed carrying, trimmed, an episode another feed published days earlier).
Processing both would put the same sentences through Human Review twice and
produce two notes with the same content. Before Apple acquisition, an episode
that is not compiled yet is compared with every earlier-published episode of
the last ``WINDOW_DAYS`` days: when at least ``MIN_CONTAINMENT`` of its
four-word sequences already occur in one source transcript of that earlier
episode, it is recorded as a duplicate and the Worker stops processing it.

The comparison is by transcript content only (titles and show names are
reported, never used to decide). Different ASR models on the same audio still
share most four-word sequences, while different conversations on the same
subject share only common phrases. An operator can overrule a recorded result
with ``scripts/find_duplicate_episodes.py``.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Callable, Iterable


DUPLICATE_POLICY_VERSION = "episode-duplicates-v1"
NGRAM_SIZE = 4
MIN_CONTAINMENT = 0.5
MIN_WORDS = 1000
WINDOW_DAYS = 30
CHECKABLE_COMPILER_STATES = frozenset({"pending", "blocked", "ready"})

_WORD = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)*")


def normalized_words(text: str) -> list[str]:
    """Lowercase word tokens with punctuation removed."""

    return [word.replace("'", "") for word in _WORD.findall(text.lower())]


def ngrams(words: list[str], size: int = NGRAM_SIZE) -> set[tuple[str, ...]]:
    return {
        tuple(words[index:index + size])
        for index in range(len(words) - size + 1)
    }


def containment(candidate_text: str, other_text: str) -> tuple[float, int]:
    """Share of the candidate's word sequences that also occur in ``other_text``.

    Returns ``(containment, candidate_word_count)``.
    """

    words = normalized_words(candidate_text)
    candidate = ngrams(words)
    if not candidate:
        return 0.0, len(words)
    other = ngrams(normalized_words(other_text))
    return len(candidate & other) / len(candidate), len(words)


def published_at(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError, OverflowError):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def recorded_result(episode: dict) -> str | None:
    check = episode.get("duplicate_check")
    if isinstance(check, dict) and check.get("result") in {"duplicate", "not_duplicate"}:
        return check["result"]
    return None


def is_duplicate(episode: dict) -> bool:
    return recorded_result(episode) == "duplicate"


def checkable(episode: dict) -> bool:
    """Whether the Worker should look for an earlier copy of this episode."""

    if recorded_result(episode) is not None:
        return False
    status = episode.get("status") or {}
    compiled = ((episode.get("files") or {}).get("compiled") or {}).get("transcript")
    return (
        not compiled
        and (status.get("compiler") or {}).get("state") in CHECKABLE_COMPILER_STATES
        and (status.get("whisper") or {}).get("state") == "ready"
        and bool(_stored_source_texts(episode, ("whisper",)))
    )


def _stored_source_texts(episode: dict, sources: Iterable[str]) -> list[str]:
    files = ((episode.get("files") or {}).get("sources") or {})
    paths = []
    for source in sources:
        path = (files.get(source) or {}).get("text")
        if isinstance(path, str) and path:
            paths.append(path)
    return paths


def earlier_candidates(episode: dict, episodes: Iterable[dict]) -> list[dict]:
    """Earlier-published, non-duplicate episodes inside the comparison window."""

    published = published_at(episode.get("published"))
    if published is None:
        return []
    earliest = published - timedelta(days=WINDOW_DAYS)
    rows = []
    for other in episodes:
        if other.get("episode_key") == episode.get("episode_key") or is_duplicate(other):
            continue
        other_published = published_at(other.get("published"))
        if other_published is None or not earliest <= other_published < published:
            continue
        if not _stored_source_texts(other, ("apple", "whisper")):
            continue
        rows.append(other)
    return rows


def best_match(
    episode: dict,
    episodes: Iterable[dict],
    read_text: Callable[[str], str],
) -> dict | None:
    """Return the earlier episode whose source best contains this one.

    No threshold is applied; ``None`` means there was nothing to compare.
    """

    candidate_paths = _stored_source_texts(episode, ("whisper",))
    if not candidate_paths:
        return None
    words = normalized_words(read_text(candidate_paths[0]))
    candidate = ngrams(words)
    if not candidate:
        return None
    best: dict | None = None
    for other in earlier_candidates(episode, episodes):
        for source in ("apple", "whisper"):
            paths = _stored_source_texts(other, (source,))
            if not paths:
                continue
            other_ngrams = ngrams(normalized_words(read_text(paths[0])))
            score = len(candidate & other_ngrams) / len(candidate)
            if best is None or score > best["containment"]:
                best = {
                    "original_episode_key": other.get("episode_key"),
                    "original_podcast": other.get("podcast"),
                    "original_title": other.get("title"),
                    "original_published": other.get("published"),
                    "compared_source": source,
                    "containment": round(score, 4),
                    "candidate_words": len(words),
                }
    return best


def is_duplicate_finding(finding: dict | None) -> bool:
    return (
        finding is not None
        and finding["candidate_words"] >= MIN_WORDS
        and finding["containment"] >= MIN_CONTAINMENT
    )


def find_original(
    episode: dict,
    episodes: Iterable[dict],
    read_text: Callable[[str], str],
) -> dict | None:
    """Return the duplicate finding for ``episode``, or ``None``.

    The finding names the earlier episode whose source transcript best
    contains this episode's Whisper transcript, when that containment reaches
    ``MIN_CONTAINMENT`` and the episode has at least ``MIN_WORDS`` words.
    """

    finding = best_match(episode, episodes, read_text)
    return finding if is_duplicate_finding(finding) else None


def duplicate_record(finding: dict, *, checked_at: str, decided_by: str = "worker") -> dict:
    return {
        "result": "duplicate",
        "policy_version": DUPLICATE_POLICY_VERSION,
        "decided_by": decided_by,
        "checked_at": checked_at,
        **finding,
    }


def not_duplicate_record(*, checked_at: str, note: str | None = None) -> dict:
    return {
        "result": "not_duplicate",
        "policy_version": DUPLICATE_POLICY_VERSION,
        "decided_by": "operator",
        "checked_at": checked_at,
        "note": note,
    }
