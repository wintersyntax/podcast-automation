"""Stable, cross-machine identity for one podcast episode."""

from __future__ import annotations

import hashlib


IDENTITY_VERSION = "rss-guid-v1"


def episode_key_for(feed_url: str, rss_guid: str) -> str:
    """Return the deterministic GCS key shared by Mac and cloud workers.

    Titles are deliberately excluded: publishers can edit them after release.
    The feed URL scopes GUIDs that are only unique inside a single RSS feed.
    """

    if not isinstance(feed_url, str) or not feed_url.strip():
        raise ValueError("feed_url must be a non-empty string")
    if not isinstance(rss_guid, str) or not rss_guid.strip():
        raise ValueError("rss_guid must be a non-empty string")
    material = f"{IDENTITY_VERSION}\0{feed_url.strip()}\0{rss_guid.strip()}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]
