import feedparser

from .episode_contract import canonical_episode_url


def clean_url(url):
    """
    Remove markdown formatting from URLs.
    """

    if not url:
        return None

    url = str(url)

    if url.startswith("[") and "](" in url:
        url = url.split("](")[1]
        url = url.rstrip(")")

    return url


def get_latest_episode(feed_url, podcast_name=None):
    """
    Get latest episode from RSS feed.
    """

    feed = feedparser.parse(
        feed_url
    )

    if not feed.entries:
        return {
            "error": "No episodes found"
        }

    episode = feed.entries[0]

    audio_url = None

    if "enclosures" in episode:
        audio_url = episode.enclosures[0].get(
            "href"
        )

    guid = (
        episode.get("guid")
        or episode.get("id")
        or episode.get("link")
        or episode.get("title")
    )

    return {
        "podcast": podcast_name or feed.feed.get(
            "title",
            "Unknown Podcast"
        ),

        "feed_url": clean_url(feed_url),

        "guid": guid,

        "title": episode.get(
            "title",
            ""
        ),

        "published": episode.get(
            "published",
            ""
        ),

        "link": canonical_episode_url(
            clean_url(
                episode.get(
                    "link",
                    ""
                )
            )
        ),

        "audio_url": clean_url(
            audio_url
        ),

        "processed": False
    }


def fetch_episode_description(feed_url, guid):
    """Best-effort RSS show notes for one episode; ``None`` on any miss.

    TASK-118: the single-pass note writer reads the show notes as episode
    context (the accepted pilot notes were written with them). Never
    raises: an unreachable feed, a parse error or no matching entry simply
    means the writer runs without a description.
    """

    if not feed_url or not guid:
        return None
    try:
        feed = feedparser.parse(clean_url(feed_url))
        for entry in getattr(feed, "entries", []) or []:
            entry_guid = entry.get("guid") or entry.get("id") or entry.get("link")
            if entry_guid == guid:
                value = entry.get("summary") or entry.get("description")
                return value if isinstance(value, str) and value.strip() else None
    except Exception:  # noqa: BLE001 -- best effort by contract
        return None
    return None
