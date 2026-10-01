"""Platform-neutral Apple Podcasts transcript acquisition and TTML parsing.

This module deliberately knows nothing about how an Apple bearer token is
created or stored.

macOS code may obtain a token through AppleMediaServices.
Cloud code may read the same token from Secret Manager.

Once a token is supplied here, all work is ordinary HTTPS and XML parsing.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as element_tree
from dataclasses import dataclass
from urllib.parse import urlparse


DEFAULT_STOREFRONT = "us"
API_BASE = "https://amp-api.podcasts.apple.com/v1/catalog"
HTTP_TIMEOUT_SECONDS = 30

TTML_NAMESPACE = "http://www.w3.org/ns/ttml"
PODCASTS_NAMESPACE = "http://podcasts.apple.com/transcript-ttml-internal"


@dataclass(frozen=True)
class EpisodeResolution:
    """Outcome of mapping one RSS episode to an Apple Podcasts episode ID."""

    status: str
    message: str
    apple_episode_id: str | None = None
    metadata: dict | None = None

    @property
    def ready(self) -> bool:
        return self.status == "READY"


@dataclass(frozen=True)
class AppleTranscriptResult:
    """Apple transcript plus its raw TTML representation."""

    status: str
    message: str
    transcript: str | None = None
    segments: list[dict] | None = None
    ttml: bytes | None = None
    metadata: dict | None = None

    @property
    def ready(self) -> bool:
        return self.status == "READY"


def _normalise_title(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "", (value or "").casefold())


def _request_json(
    url: str,
    token: str,
) -> tuple[dict | None, int | None, str | None]:
    request = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Origin": "https://podcasts.apple.com",
            "User-Agent": "podcast-worker/1.0",
        },
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=HTTP_TIMEOUT_SECONDS,
        ) as response:
            return (
                json.loads(response.read().decode("utf-8")),
                response.status,
                None,
            )

    except urllib.error.HTTPError as error:
        body = error.read(2_000).decode("utf-8", errors="replace")
        return None, error.code, body or str(error.reason)

    except (OSError, TimeoutError, json.JSONDecodeError) as error:
        return None, None, str(error)


def resolve_episode_id(
    token: str,
    apple_show_id: str | None,
    rss_guid: str | None,
    title: str | None,
    storefront: str = DEFAULT_STOREFRONT,
) -> EpisodeResolution:
    """Resolve an RSS episode to Apple ID, preferring an exact RSS GUID."""

    if not token:
        return EpisodeResolution(
            "API_ERROR",
            "No Apple bearer token supplied.",
        )

    if not apple_show_id:
        return EpisodeResolution(
            "CONFIG_ERROR",
            "No apple_show_id configured for this show.",
        )

    query = urllib.parse.urlencode(
        {
            "include": "podcast",
            "limit": "50",
            "l": "en-US",
        }
    )

    url = (
        f"{API_BASE}/{storefront}/podcasts/"
        f"{apple_show_id}/episodes?{query}"
    )

    payload, status_code, error = _request_json(url, token)

    if payload is None:
        return EpisodeResolution(
            "API_ERROR",
            (
                "Apple episode catalogue request failed "
                f"({status_code or 'network'}): {error}"
            ),
            metadata={"http_status": status_code},
        )

    episodes = payload.get("data") or []

    guid_matches = [
        item
        for item in episodes
        if rss_guid
        and str((item.get("attributes") or {}).get("guid") or "") == rss_guid
    ]

    if len(guid_matches) == 1:
        item = guid_matches[0]

        return EpisodeResolution(
            "READY",
            "Matched the RSS GUID in the Apple Podcasts catalogue.",
            str(item["id"]),
            {
                "match": "rss_guid",
                "apple_show_id": str(apple_show_id),
            },
        )

    normalised_title = _normalise_title(title)

    title_matches = [
        item
        for item in episodes
        if normalised_title
        and _normalise_title(
            (item.get("attributes") or {}).get("name")
        )
        == normalised_title
    ]

    if len(title_matches) == 1:
        item = title_matches[0]

        return EpisodeResolution(
            "READY",
            "Matched a unique exact episode title in the Apple Podcasts catalogue.",
            str(item["id"]),
            {
                "match": "title",
                "apple_show_id": str(apple_show_id),
            },
        )

    if len(title_matches) > 1:
        return EpisodeResolution(
            "AMBIGUOUS_EPISODE",
            (
                "More than one Apple episode has the same normalised title; "
                "refusing to guess."
            ),
        )

    return EpisodeResolution(
        "EPISODE_NOT_READY",
        "The RSS episode is not yet in Apple's first 50 catalogue episodes.",
    )


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _normalise_text(tokens: list[str]) -> str:
    text = " ".join(token for token in tokens if token)
    return re.sub(r"\s+([,.;:!?])", r"\1", text).strip()


def _seconds(value: str | None) -> float | None:
    """Parse TTML times such as 1:06.060 or 00:01:06.060."""

    if not value:
        return None

    try:
        parts = [float(part) for part in value.split(":")]
    except ValueError:
        return None

    if len(parts) == 1:
        return parts[0]

    if len(parts) == 2:
        return parts[0] * 60 + parts[1]

    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]

    return None


def _parse_ttml_root(
    root: element_tree.Element,
) -> tuple[str, list[dict]]:
    paragraphs: list[str] = []
    segments: list[dict] = []

    agent_attribute = f"{{{TTML_NAMESPACE}}}agent"
    unit_attribute = f"{{{PODCASTS_NAMESPACE}}}unit"

    for paragraph in root.iter():
        if _local_name(paragraph.tag) != "p":
            continue

        words = [
            (node.text or "").strip()
            for node in paragraph.iter()
            if _local_name(node.tag) == "span"
            and node.attrib.get(unit_attribute) == "word"
        ]

        text = _normalise_text(words)

        if not text:
            continue

        paragraphs.append(text)

        segment: dict[str, object] = {"text": text}

        start = _seconds(paragraph.attrib.get("begin"))
        end = _seconds(paragraph.attrib.get("end"))
        speaker = paragraph.attrib.get(agent_attribute)

        if start is not None:
            segment["start"] = start

        if end is not None:
            segment["end"] = end

        if speaker:
            segment["speaker"] = speaker

        segments.append(segment)

    return "\n\n".join(paragraphs), segments


def parse_ttml_bytes(
    ttml: bytes,
) -> tuple[str, list[dict]]:
    """Parse Apple TTML directly from downloaded bytes."""

    root = element_tree.fromstring(ttml)

    if _local_name(root.tag) != "tt":
        raise ValueError("The XML root element is not TTML <tt>.")

    return _parse_ttml_root(root)


def parse_ttml_file(
    path,
) -> tuple[str, list[dict]]:
    """Parse Apple TTML stored in a local file."""

    root = element_tree.parse(path).getroot()

    if _local_name(root.tag) != "tt":
        raise ValueError("The XML root element is not TTML <tt>.")

    return _parse_ttml_root(root)


def fetch_transcript(
    token: str,
    apple_episode_id: str | None,
    storefront: str = DEFAULT_STOREFRONT,
) -> AppleTranscriptResult:
    """Download and parse one Apple-generated podcast transcript."""

    if not token:
        return AppleTranscriptResult(
            "API_ERROR",
            "No Apple bearer token supplied.",
        )

    if not apple_episode_id:
        return AppleTranscriptResult(
            "EPISODE_NOT_READY",
            "Apple episode ID has not been resolved yet.",
        )

    query = urllib.parse.urlencode(
        {
            "fields": "ttmlToken,ttmlAssetUrls",
            "include[podcast-episodes]": "podcast",
            "l": "en-US",
            "with": "entitlements",
        }
    )

    url = (
        f"{API_BASE}/{storefront}/podcast-episodes/"
        f"{apple_episode_id}/transcripts?{query}"
    )

    payload, status_code, error = _request_json(url, token)

    if payload is None:
        status = (
            "TRANSCRIPT_NOT_READY"
            if status_code in {404, 409}
            else "API_ERROR"
        )

        return AppleTranscriptResult(
            status,
            (
                "Apple transcript request failed "
                f"({status_code or 'network'}): {error}"
            ),
            metadata={
                "apple_episode_id": str(apple_episode_id),
                "http_status": status_code,
            },
        )

    data = payload.get("data") or []
    attributes = (data[0].get("attributes") or {}) if data else {}
    asset_urls = attributes.get("ttmlAssetUrls") or {}
    ttml_url = asset_urls.get("ttml")

    if not ttml_url:
        return AppleTranscriptResult(
            "TRANSCRIPT_NOT_READY",
            "Apple knows the episode but has not published a TTML asset yet.",
            metadata={
                "apple_episode_id": str(apple_episode_id),
            },
        )

    try:
        with urllib.request.urlopen(
            str(ttml_url),
            timeout=HTTP_TIMEOUT_SECONDS,
        ) as response:
            ttml = response.read()

    except (OSError, TimeoutError, urllib.error.HTTPError) as error:
        return AppleTranscriptResult(
            "API_ERROR",
            f"Could not download Apple TTML asset: {error}",
            metadata={
                "apple_episode_id": str(apple_episode_id),
            },
        )

    try:
        transcript, segments = parse_ttml_bytes(ttml)

    except (ValueError, element_tree.ParseError) as error:
        return AppleTranscriptResult(
            "API_ERROR",
            f"Could not parse Apple TTML: {error}",
            metadata={
                "apple_episode_id": str(apple_episode_id),
            },
        )

    if not transcript:
        return AppleTranscriptResult(
            "API_ERROR",
            "Apple TTML contained no readable words.",
            metadata={
                "apple_episode_id": str(apple_episode_id),
            },
        )

    return AppleTranscriptResult(
        "READY",
        "Apple Podcasts API transcript downloaded.",
        transcript=transcript,
        segments=segments,
        ttml=ttml,
        metadata={
            "source": "apple_podcasts_api",
            "apple_episode_id": str(apple_episode_id),
            "ttml_asset_host": urlparse(str(ttml_url)).netloc,
            "segment_count": len(segments),
        },
    )


def resolve_and_fetch(
    *,
    token: str,
    apple_show_id: str | None,
    rss_guid: str | None,
    title: str | None,
    storefront: str = DEFAULT_STOREFRONT,
) -> AppleTranscriptResult:
    """Resolve one RSS episode and fetch its Apple transcript."""

    resolution = resolve_episode_id(
        token,
        apple_show_id,
        rss_guid,
        title,
        storefront,
    )

    if not resolution.ready:
        return AppleTranscriptResult(
            resolution.status,
            resolution.message,
            metadata=resolution.metadata,
        )

    result = fetch_transcript(
        token,
        resolution.apple_episode_id,
        storefront,
    )

    if not result.metadata:
        return result

    metadata = dict(result.metadata)
    metadata["episode_match"] = resolution.metadata

    return AppleTranscriptResult(
        result.status,
        result.message,
        transcript=result.transcript,
        segments=result.segments,
        ttml=result.ttml,
        metadata=metadata,
    )
