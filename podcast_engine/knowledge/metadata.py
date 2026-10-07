"""Strict structured metadata extraction for podcast knowledge notes."""

from __future__ import annotations

import copy
import hashlib
import json
import os

from ..ai_budget import STAGE_METADATA
from ..episode_contract import now_iso
from ..preset_provenance import PresetProvenance, fetch_current_designated_preset
from .client import post_openrouter
from .models import (
    DEFAULT_METADATA_PRESET,
    METADATA_POLICY_VERSION,
    METADATA_PRESET_ENV,
    METADATA_RESPONSE_SCHEMA,
    SUMMARY_POLICY_VERSION,
)
from .summary import (
    _WORST_CASE_EPISODE,
    episode_context,
    worst_case_transcript_text,
)


def metadata_preset() -> str:
    return os.getenv(METADATA_PRESET_ENV, DEFAULT_METADATA_PRESET)


def input_fingerprint(
    episode: dict,
    body_bytes: bytes,
    transcript_bytes: bytes,
    tag_vocabulary: dict | None = None,
) -> str:
    """Fingerprint all inputs that can change structured note metadata."""

    payload = {
        "policy_version": METADATA_POLICY_VERSION,
        "summary_policy_version": SUMMARY_POLICY_VERSION,
        "preset": metadata_preset(),
        "summary_body_sha256": hashlib.sha256(body_bytes).hexdigest(),
        "compiled_transcript_sha256": hashlib.sha256(transcript_bytes).hexdigest(),
        "episode": episode_context(episode),
        "tag_vocabulary": tag_vocabulary or {},
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def openrouter_payload(
    episode: dict,
    summary_body: str,
    compiled_transcript: str,
    tag_vocabulary: dict | None = None,
    *,
    provenance: PresetProvenance,
) -> dict:
    """Build a direct request from one verified current preset snapshot,
    with the non-negotiable strict JSON contract.

    TASK-076 Task 7: the exact resolved config is the base and the verified
    system prompt is placed before the user payload -- never a mutable
    ``@preset/<slug>`` alias.
    """

    if (
        not provenance.verified
        or not isinstance(provenance.config, dict)
        or not isinstance(provenance.system_prompt, str)
    ):
        raise RuntimeError("Verified OpenRouter preset provenance is required")

    payload = copy.deepcopy(provenance.config)
    payload.update(
        {
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "podcast_knowledge_metadata",
                    "strict": True,
                    "schema": METADATA_RESPONSE_SCHEMA,
                },
            },
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
                            "summary_body": summary_body,
                            "compiled_transcript": compiled_transcript,
                            "tag_vocabulary": tag_vocabulary or {},
                            "tag_instruction": (
                                "Reuse existing tags whenever reasonably applicable. "
                                "Propose a new candidate only when the concept is important, "
                                "reusable across future notes, and no existing canonical tag "
                                "or alias adequately represents it."
                            ),
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            "usage": {"include": True},
        }
    )
    return payload


# TASK-076 Task 5: worst-case tag vocabulary bound. tags.py has no
# documented hard cap on canonical tag/alias count; this stand-in size is
# a realistic-practical bound (same methodology as summary.py's worst-case
# transcript), not a Python-enforced ceiling.
_WORST_CASE_TAG_VOCABULARY_SIZE = 200


def _worst_case_tag_vocabulary() -> dict:
    return {
        f"worst-case-canonical-tag-{index:04d}": [
            f"worst-case-alias-{index:04d}-a",
            f"worst-case-alias-{index:04d}-b",
        ]
        for index in range(_WORST_CASE_TAG_VOCABULARY_SIZE)
    }


def worst_case_openrouter_payload(provenance: PresetProvenance) -> dict:
    """Build the same-shaped metadata request this stage actually sends,
    at the architecture's documented maximum transcript size and a
    generous worst-case tag vocabulary, from the given resolved preset
    snapshot, so podcast_engine.ai_pricing can derive a conservative
    reservation bound for episode-AI-budget downstream-reserve planning
    (Task 5). The summary_body worst-case stand-in reuses the same
    oversized transcript text: summary output length has no Python-
    enforced ceiling either, and reusing the larger bound here is
    conservative (errs toward reserving more, never less) rather than
    inventing a second, smaller, unproven number. Never sent to a
    provider -- this is measurement input only."""

    transcript = worst_case_transcript_text()
    return openrouter_payload(
        _WORST_CASE_EPISODE,
        summary_body=transcript,
        compiled_transcript=transcript,
        tag_vocabulary=_worst_case_tag_vocabulary(),
        provenance=provenance,
    )


def _parse_response(content: str) -> dict:
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise ValueError("Podcast metadata response must be a JSON object")
    return parsed


def _clean_values(value, limit: int) -> list[str]:
    """Trim and case-insensitively deduplicate model-provided labels only."""

    if not isinstance(value, list):
        return []
    cleaned = []
    seen = set()
    for item in value:
        if not isinstance(item, str):
            continue
        normalized = item.strip()
        key = normalized.casefold()
        if not normalized or key in seen:
            continue
        seen.add(key)
        cleaned.append(normalized)
        if len(cleaned) == limit:
            break
    return cleaned


def normalize_metadata(payload: dict) -> dict:
    """Keep only the response-schema fields and never infer new concepts."""

    return {
        "topics": _clean_values(payload.get("topics"), 8),
        "people": _clean_values(payload.get("people"), 12),
        "existing_tags": _clean_values(payload.get("existing_tags"), 8),
        "new_tag_candidates": _clean_candidates(payload.get("new_tag_candidates")),
    }


def _clean_candidates(value) -> list[dict[str, str]]:
    """Accept only the two structured, reviewable candidate proposals."""

    if not isinstance(value, list):
        return []
    result = []
    seen = set()
    for item in value:
        if not isinstance(item, dict):
            continue
        tag = item.get("tag")
        category = item.get("category")
        if not isinstance(tag, str) or not isinstance(category, str):
            continue
        tag = tag.strip()
        if not tag or tag.casefold() in seen:
            continue
        seen.add(tag.casefold())
        result.append({"tag": tag, "category": category})
        if len(result) == 2:
            break
    return result


def generate(
    episode: dict,
    summary_body: str,
    compiled_transcript: str,
    tag_vocabulary: dict | None = None,
    *,
    episode_key: str,
    source_fingerprint: str,
    pricing_transport=None,
    pricing_now=None,
) -> dict:
    api_key = os.getenv("PODCAST_KNOWLEDGE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing PODCAST_KNOWLEDGE_API_KEY for knowledge requests")

    provenance = fetch_current_designated_preset(
        metadata_preset(), api_key=api_key, verified_at=now_iso()
    )
    if not provenance.verified:
        raise RuntimeError(
            f"Metadata preset could not be resolved: {provenance.status} "
            f"({provenance.reason or 'unknown'})"
        )

    response = post_openrouter(
        openrouter_payload(
            episode,
            summary_body,
            compiled_transcript,
            tag_vocabulary,
            provenance=provenance,
        ),
        episode_key=episode_key,
        source_fingerprint=source_fingerprint,
        stage=STAGE_METADATA,
        provenance=provenance,
        pricing_transport=pricing_transport,
        pricing_now=pricing_now,
    )
    content = response.json()["choices"][0]["message"]["content"]
    if not isinstance(content, str):
        raise ValueError("Podcast metadata response must contain JSON text")
    return normalize_metadata(_parse_response(content))
