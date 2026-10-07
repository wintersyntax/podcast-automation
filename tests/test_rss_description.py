"""TASK-118: best-effort RSS show notes for the note writer."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from podcast_engine import rss


class FetchEpisodeDescriptionTests(unittest.TestCase):
    def test_matching_entry_returns_its_summary(self):
        feed = SimpleNamespace(entries=[{"guid": "other", "summary": "no"}, {"guid": "g-1", "summary": "Show notes"}])
        with patch.object(rss.feedparser, "parse", return_value=feed):
            self.assertEqual(rss.fetch_episode_description("https://feed.test/rss", "g-1"), "Show notes")

    def test_misses_and_errors_return_none(self):
        self.assertIsNone(rss.fetch_episode_description(None, "g-1"))
        self.assertIsNone(rss.fetch_episode_description("https://feed.test/rss", None))
        with patch.object(rss.feedparser, "parse", side_effect=RuntimeError("down")):
            self.assertIsNone(rss.fetch_episode_description("https://feed.test/rss", "g-1"))
        with patch.object(rss.feedparser, "parse", return_value=SimpleNamespace(entries=[{"guid": "g-1", "summary": "  "}])):
            self.assertIsNone(rss.fetch_episode_description("https://feed.test/rss", "g-1"))


if __name__ == "__main__":
    unittest.main()
