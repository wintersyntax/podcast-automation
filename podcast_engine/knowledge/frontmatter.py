"""Deterministic YAML frontmatter for podcast knowledge notes."""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import yaml

from ..episode_contract import canonical_episode_url


def _optional_text(value) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _published_date(value) -> str | None:
    value = _optional_text(value)
    if not value:
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
    return parsed.astimezone(timezone.utc).date().isoformat()


def _episode_number(title) -> int | None:
    title = _optional_text(title)
    if not title:
        return None
    match = re.match(r"^\s*(?:ep\.?|episode)\s*(\d+)\s*[-:|]\s*", title, re.I)
    return int(match.group(1)) if match else None


def _deduplicate(values) -> list[str]:
    result = []
    seen = set()
    for value in values:
        value = _optional_text(value)
        key = value.casefold() if value else None
        if not value or key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def _tag_slug(value) -> str | None:
    """Return a conservative, single-token tag slug or ``None``."""

    value = _optional_text(value)
    if not value:
        return None
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    value = re.sub(r"\s+", "-", value.casefold())
    value = re.sub(r"[^a-z0-9-]+", "-", value)
    return re.sub(r"-+", "-", value).strip("-") or None


def _normalized_tags(values) -> list[str]:
    return _deduplicate(_tag_slug(value) for value in values)


def build_frontmatter(episode: dict, metadata: dict, created_at: str) -> dict:
    """Create the V1 field order without nulls or model-authored YAML."""

    result = {
        "type": "podcast-summary",
        "podcast": episode.get("podcast"),
        "title": episode.get("title"),
        "episode_key": episode.get("episode_key"),
        "topics": _deduplicate(metadata.get("topics", [])),
        "people": _deduplicate(metadata.get("people", [])),
        "tags": _normalized_tags(metadata.get("tags", [])),
        "summary_language": "en",
        "transcript_source": "compiled",
        "created": created_at,
    }
    optional = {
        "podcast_id": episode.get("podcast_id"),
        "episode": _episode_number(episode.get("title")),
        "published": _published_date(episode.get("published")),
        "source_url": canonical_episode_url(episode.get("link")),
        "podcast_url": _optional_text(episode.get("podcast_url")),
    }
    if episode.get("status", {}).get("compiler", {}).get("state") == "completed":
        optional["transcript_reviewed"] = True

    ordered = {}
    for key in ("type", "podcast", "podcast_id", "episode", "title", "published", "episode_key", "source_url", "podcast_url", "topics", "people", "tags", "summary_language", "transcript_source", "transcript_reviewed", "created"):
        value = optional.get(key, result.get(key))
        if value is not None and value != "":
            ordered[key] = value
    return ordered


def render(episode: dict, metadata: dict, created_at: str, body: str) -> str:
    yaml_body = yaml.safe_dump(
        build_frontmatter(episode, metadata, created_at),
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    )
    return f"---\n{yaml_body}---\n\n{body.lstrip()}"
