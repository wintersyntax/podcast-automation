import unittest
from unittest.mock import patch

from podcast_engine.review_web import PAGE, create_review_app


class HumanReviewEpisodeQueueTests(unittest.TestCase):
    def test_episode_dropdown_is_replaced_by_queue_cards(self):
        self.assertIn("id=episode-nav", PAGE)
        self.assertIn("episode-grid", PAGE)
        self.assertIn("Review queue", PAGE)
        self.assertIn("Reviewing", PAGE)

        self.assertNotIn(
            "<select id=episode>",
            PAGE,
        )

    def test_single_episode_uses_compact_header(self):
        self.assertIn(
            "episodeList.length===1",
            PAGE,
        )

        self.assertIn(
            "episode-current",
            PAGE,
        )

    def test_episode_cards_are_responsive(self):
        self.assertIn(
            ".episode-grid{display:grid",
            PAGE,
        )

        self.assertIn(
            "grid-template-columns:1fr",
            PAGE,
        )

    def test_detailed_review_defer_is_explicit_and_non_resolving(self):
        self.assertIn("Defer for this pass", PAGE)
        self.assertNotIn("id=skip", PAGE)
        self.assertIn("deferredIds=new Set()", PAGE)
        self.assertIn("Deferred this pass (${items.length})", PAGE)
        self.assertIn("All remaining cards are deferred", PAGE)
        self.assertIn("remain unresolved and still block Recompile &amp; continue", PAGE)
        self.assertIn("data-resume-deferred", PAGE)
        self.assertIn("cards.filter(c=>!deferredIds.has(String(c.id)))", PAGE)
        self.assertIn("if(!reviewable.length)", PAGE)
        self.assertIn("if(!cards.length)", PAGE)

        defer_body = PAGE.split("function deferCurrent(){", 1)[1].split(
            "function resumeDeferred", 1
        )[0]
        self.assertNotIn("api(", defer_body)
        self.assertNotIn("cards.splice", defer_body)

    def test_episode_api_includes_podcast_name(self):
        episode = {
            "episode_key": "episode-1",
            "podcast": "Example Nutrition Podcast",
            "title": "Example Episode",
            "status": {
                "compiler": {
                    "state": "review_required",
                }
            },
        }

        with (
            patch(
                "podcast_engine.review_web.load_episodes",
                return_value=[episode],
            ),
            patch(
                "podcast_engine.review_web.load_review_record",
                return_value={},
            ),
            patch(
                "podcast_engine.review_web.pending_review_items",
                return_value=[
                    {"id": 1},
                    {"id": 2},
                ],
            ),
        ):
            response = (
                create_review_app()
                .test_client()
                .get("/api/review/episodes")
            )

        self.assertEqual(response.status_code, 200)

        self.assertEqual(
            response.get_json()["episodes"],
            [
                {
                    "episode_key": "episode-1",
                    "podcast": "Example Nutrition Podcast",
                    "title": "Example Episode",
                    "pending_count": 2,
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
