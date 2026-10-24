"""Episode context shared by the knowledge-note writer and its tooling.

``episode_context`` is the episode identity sent to (and fingerprinted by)
the TASK-118 writer. The worst-case episode, transcript and tag vocabulary
bound the writer's episode-AI-budget reservation; they are never sent to a
provider.
"""

from __future__ import annotations


def episode_context(episode: dict) -> dict:
    """Return only episode data that is relevant to the knowledge note."""

    return {
        "episode_key": episode["episode_key"],
        "podcast": episode.get("podcast"),
        "podcast_id": episode.get("podcast_id"),
        "title": episode.get("title"),
        "published": episode.get("published"),
        "source_url": episode.get("link"),
    }


# TASK-076 Task 5: a realistic long-episode transcript size (24,000 words,
# roughly two and a half hours of speech), used to bound a stage's own
# episode-AI-budget reservation. It is not a hard ceiling on transcript
# length; nothing rejects a longer transcript.
WORST_CASE_TRANSCRIPT_WORDS = 24_000

# A generous per-word stand-in (8 bytes incl. trailing space) -- above
# typical English spoken-word averages (~5-6 bytes/word) so the byte
# count this produces errs toward reserving more, never less, than a
# real transcript of the same word count.
_WORST_CASE_WORD = "wordish "

WORST_CASE_EPISODE = {
    "episode_key": "0" * 24,
    "podcast": "Worst Case Podcast Title For Budget Reservation Bounding",
    "podcast_id": "worst-case-podcast-id-for-budget-reservation-bounding",
    "title": (
        "A Sufficiently Long Worst Case Episode Title For Episode AI "
        "Budget Reservation Bounding Purposes Only, Never Sent To A "
        "Provider"
    ),
    "published": "2026-01-01T00:00:00Z",
    "link": "https://example.com/worst-case-episode-link-for-budget-reservation-bounding",
    "category": "worst-case-category-for-budget-reservation-bounding",
    "prompt": "worst case podcast profile prompt text " * 20,
}


def worst_case_transcript_text() -> str:
    """A synthetic transcript at the documented maximum size (see
    WORST_CASE_TRANSCRIPT_WORDS), for budget reservation bounding only."""

    return (_WORST_CASE_WORD * WORST_CASE_TRANSCRIPT_WORDS).strip()


# TASK-076 Task 5: worst-case tag vocabulary bound. tags.py has no
# documented hard cap on canonical tag/alias count; this stand-in size is
# a realistic-practical bound (same methodology as the worst-case
# transcript above), not a Python-enforced ceiling.
WORST_CASE_TAG_VOCABULARY_SIZE = 200


def worst_case_tag_vocabulary() -> dict:
    """A generous synthetic tag vocabulary for reservation bounding only."""

    return {
        f"worst-case-canonical-tag-{index:04d}": [
            f"worst-case-alias-{index:04d}-a",
            f"worst-case-alias-{index:04d}-b",
        ]
        for index in range(WORST_CASE_TAG_VOCABULARY_SIZE)
    }


__all__ = [
    "WORST_CASE_EPISODE",
    "WORST_CASE_TRANSCRIPT_WORDS",
    "WORST_CASE_TAG_VOCABULARY_SIZE",
    "episode_context",
    "worst_case_tag_vocabulary",
    "worst_case_transcript_text",
]
