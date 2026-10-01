from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "demo_human_review.py"


def _load_demo_module():
    spec = importlib.util.spec_from_file_location("demo_human_review", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class HumanReviewDemoTests(unittest.TestCase):
    def setUp(self):
        self.demo = _load_demo_module()
        self.client = self.demo.create_demo_app().test_client()

    def test_demo_starts_with_two_synthetic_domain_conflicts(self):
        episodes = self.client.get("/api/review/episodes").get_json()["episodes"]
        self.assertEqual(len(episodes), 1)
        self.assertEqual(episodes[0]["pending_count"], 2)
        response = self.client.get(
            f"/api/review/episodes/{self.demo.EPISODE_KEY}"
        ).get_json()
        self.assertEqual(response["progress"]["total"], 2)
        self.assertEqual(response["progress"]["remaining"], 2)
        self.assertEqual(
            response["cards"][0]["source_choices"],
            {
                "apple": "The trial included 42 participants and lasted twelve weeks.",
                "whisper": "The trial included 40 participants and lasted twelve weeks.",
            },
        )
        self.assertEqual(
            response["cards"][1]["source_choices"]["apple"],
            "Romanian deadlift",
        )

    def test_decisions_advance_progress_and_enable_recompile(self):
        episode = self.demo.EPISODE_KEY
        for difference_id, source, text in (
            (1, "apple", "The trial included 42 participants and lasted twelve weeks."),
            (2, "apple", "Romanian deadlift"),
        ):
            response = self.client.post(
                f"/api/review/episodes/{episode}/items/{difference_id}/decision",
                json={"source": source, "text": text},
            )
            self.assertEqual(response.status_code, 200)

        final = self.client.get(f"/api/review/episodes/{episode}").get_json()
        self.assertEqual(final["progress"], {
            "total": 2,
            "reviewed": 2,
            "remaining": 0,
            "assisted_unprepared": 0,
            "triage_unavailable": 0,
        })

        recompile = self.client.post(f"/api/review/episodes/{episode}/recompile")
        self.assertEqual(recompile.status_code, 202)
        self.assertEqual(
            recompile.get_json()["recompile"]["status"],
            "accepted",
        )

    def test_third_asr_is_synthetic_and_local(self):
        response = self.client.post(
            f"/api/review/episodes/{self.demo.EPISODE_KEY}/items/1/third-asr"
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("forty-two", response.get_json()["evidence"]["text"])


if __name__ == "__main__":
    unittest.main()
