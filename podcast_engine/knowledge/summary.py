"""Prompt payload and input identity for Markdown podcast summaries."""

from __future__ import annotations

import copy
import hashlib
import json
import os

from ..episode_contract import now_iso
from ..preset_provenance import PresetProvenance, fetch_current_designated_preset
from .client import post_openrouter
from .markdown_normalize import normalize_markdown_layout
from .models import (
    DEFAULT_SUMMARY_PRESET,
    SUMMARY_POLICY_VERSION,
    SUMMARY_PRESET_ENV,
)
from .summary_guard import normalize_and_validate_summary_body
from .transcript_navigation import (
    MAX_SEGMENTS,
    NAVIGATION_CONTRACT,
    NAVIGATION_METHOD,
    TARGET_WORDS_PER_SEGMENT,
    render_navigated_transcript,
    segment_transcript,
)


def summary_preset() -> str:
    return os.getenv(SUMMARY_PRESET_ENV, DEFAULT_SUMMARY_PRESET)


def episode_context(episode: dict) -> dict:
    """Return only episode data that is relevant to the summary request."""

    return {
        "episode_key": episode["episode_key"],
        "podcast": episode.get("podcast"),
        "podcast_id": episode.get("podcast_id"),
        "title": episode.get("title"),
        "published": episode.get("published"),
        "source_url": episode.get("link"),
    }


def podcast_profile(episode: dict) -> dict:
    """Return the configured podcast-specific context, without a local prompt."""

    return {
        "category": episode.get("category"),
        "profile": episode.get("prompt"),
    }


def input_fingerprint(episode: dict, transcript_bytes: bytes) -> str:
    """Fingerprint every local input that can change a summary body."""

    payload = {
        "policy_version": SUMMARY_POLICY_VERSION,
        "preset": summary_preset(),
        "transcript_sha256": hashlib.sha256(transcript_bytes).hexdigest(),
        "episode": episode_context(episode),
        "podcast_profile": podcast_profile(episode),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def openrouter_payload(episode: dict, transcript: str, provenance: PresetProvenance) -> dict:
    """Build a direct request from one verified current preset snapshot.

    TASK-076 Task 7: the exact resolved config is the base and the verified
    system prompt is placed before the user payload -- never a mutable
    ``@preset/<slug>`` alias -- so the physical request always matches the
    same snapshot the episode-AI-budget reservation was derived from.
    """

    if (
        not provenance.verified
        or not isinstance(provenance.config, dict)
        or not isinstance(provenance.system_prompt, str)
    ):
        raise RuntimeError("Verified OpenRouter preset provenance is required")

    segments = segment_transcript(transcript)
    payload = copy.deepcopy(provenance.config)
    payload.update(
        {
            "messages": [
                {
                    "role": "system",
                    "content": provenance.system_prompt,
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "episode": episode_context(episode),
                            "podcast_profile": podcast_profile(episode),
                            "transcript_navigation_method": NAVIGATION_METHOD,
                            "transcript_navigation_segment_count": len(segments),
                            "transcript_navigation_contract": NAVIGATION_CONTRACT,
                            "compiled_transcript": render_navigated_transcript(segments),
                            "output_contract": "Return Markdown body only. Do not emit YAML frontmatter.",
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            # TASK-076: request OpenRouter's actual per-call cost so budget
            # settlement can use trustworthy provider-reported usage.cost
            # instead of only ever falling back to "uncertain".
            "usage": {"include": True},
        }
    )
    return payload


# TASK-076 Task 5: architecturally-documented maximum transcript size (see
# docs/system-architecture.md and MAX_SEGMENTS/TARGET_WORDS_PER_SEGMENT
# above) used to bound this stage's own episode-AI-budget reservation.
# This is a realistic-long-episode bound, not a Python-enforced hard
# ceiling on transcript length -- nothing in this pipeline currently
# rejects a longer transcript outright. It is the same methodology the
# Task 3 worst-case cost-estimate finding already used and the project accepted
# (see docs/superpowers/specs/2026-09-16-human-review-evidence-assisted-
# adjudication-design.md, "Task 3 worst-case cost-estimate finding").
WORST_CASE_TRANSCRIPT_WORDS = MAX_SEGMENTS * TARGET_WORDS_PER_SEGMENT

# A generous per-word stand-in (8 bytes incl. trailing space) -- above
# typical English spoken-word averages (~5-6 bytes/word) so the byte
# count this produces errs toward reserving more, never less, than a
# real transcript of the same word count.
_WORST_CASE_WORD = "wordish "

_WORST_CASE_EPISODE = {
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
    """A synthetic transcript at the architecture's documented maximum
    size (see WORST_CASE_TRANSCRIPT_WORDS), for conservative episode-AI-
    budget reservation bounding only. Never sent to a provider."""

    return (_WORST_CASE_WORD * WORST_CASE_TRANSCRIPT_WORDS).strip()


def worst_case_openrouter_payload(provenance: PresetProvenance) -> dict:
    """Build the same-shaped summary request this stage actually sends,
    at the architecture's documented maximum transcript size, from the
    given resolved preset snapshot, so podcast_engine.ai_pricing can derive
    a conservative reservation bound for episode-AI-budget downstream-
    reserve planning (Task 5). Never sent to a provider -- this is
    measurement input only."""

    return openrouter_payload(_WORST_CASE_EPISODE, worst_case_transcript_text(), provenance)


def generate(
    episode: dict,
    transcript: str,
    *,
    episode_key: str,
    source_fingerprint: str,
    pricing_transport=None,
    pricing_now=None,
) -> str:
    """Request, repair layout, and validate the Markdown-only summary body."""

    # Imported locally, not at module level: podcast_engine.ai_budget pulls
    # in google-cloud-storage transitively, and this module must stay
    # importable (for its pure input_fingerprint/episode_context helpers)
    # without that dependency. See test_knowledge_import_boundaries.py.
    from ..ai_budget import STAGE_SUMMARY

    api_key = os.getenv("PODCAST_KNOWLEDGE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing PODCAST_KNOWLEDGE_API_KEY for knowledge requests")

    provenance = fetch_current_designated_preset(
        summary_preset(), api_key=api_key, verified_at=now_iso()
    )
    if not provenance.verified:
        raise RuntimeError(
            f"Summary preset could not be resolved: {provenance.status} "
            f"({provenance.reason or 'unknown'})"
        )

    response = post_openrouter(
        openrouter_payload(episode, transcript, provenance),
        episode_key=episode_key,
        source_fingerprint=source_fingerprint,
        stage=STAGE_SUMMARY,
        provenance=provenance,
        pricing_transport=pricing_transport,
        pricing_now=pricing_now,
    )
    content = response.json()["choices"][0]["message"]["content"]
    if isinstance(content, str):
        content = normalize_markdown_layout(content)
    return normalize_and_validate_summary_body(content)
