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
        self.episode = self.demo.EPISODE_KEY
        self.generation = "sha256:" + "d" * 64

    def _review(self):
        return self.client.get(
            f"/api/review/episodes/{self.episode}"
        ).get_json()

    def _materiality(self, decisions, **session):
        return self.client.post(
            f"/api/review/episodes/{self.episode}/materiality-decision",
            json={
                "expected_generation_fingerprint": self.generation,
                "decisions": decisions,
                "session": session,
            },
        )

    def _clear_materiality_flow(self):
        quick = self._materiality(
            [
                {"id": 1, "source": "apple", "seconds": 2},
                {"id": 2, "source": "whisper", "seconds": 1},
                {"id": 3, "source": "apple", "seconds": 3},
            ],
            note="Synthetic quick-review session",
        )
        self.assertEqual(quick.status_code, 200)

        settled = self._materiality(
            [
                {"id": 4, "source": "whisper"},
                {"id": 5, "source": "apple"},
            ],
            settled_list_opened=True,
        )
        self.assertEqual(settled.status_code, 200)

    def test_demo_starts_with_current_materiality_first_review_flow(self):
        episodes = self.client.get("/api/review/episodes").get_json()["episodes"]
        self.assertEqual(len(episodes), 1)
        self.assertEqual(episodes[0]["pending_count"], 6)

        response = self._review()
        self.assertEqual(response["progress"]["total"], 6)
        self.assertEqual(response["progress"]["remaining"], 6)

        groups = [card["materiality"]["group"] for card in response["cards"]]
        self.assertEqual(
            groups,
            ["sample", "click_one", "proposal", "settled", "settled", "full"],
        )
        self.assertEqual(
            response["cards"][-1]["source_choices"],
            {
                "apple": "The trial included 42 participants and lasted twelve weeks.",
                "whisper": "The trial included 40 participants and lasted twelve weeks.",
            },
        )
        self.assertTrue(response["cards"][-1]["third_available"])

    def test_materiality_decisions_leave_only_protected_full_review(self):
        self._clear_materiality_flow()

        response = self._review()
        self.assertEqual(response["progress"]["reviewed"], 5)
        self.assertEqual(response["progress"]["remaining"], 1)
        self.assertEqual([card["id"] for card in response["cards"]], [6])
        self.assertEqual(
            response["record"]["materiality_review_log"],
            [
                {
                    "accepted_count": 3,
                    "settled_list_opened": False,
                    "note": "Synthetic quick-review session",
                },
                {
                    "accepted_count": 2,
                    "settled_list_opened": True,
                    "note": None,
                },
            ],
        )

    def test_settled_card_cannot_override_filter_reading(self):
        response = self._materiality([{"id": 4, "source": "apple"}])
        self.assertEqual(response.status_code, 409)
        self.assertIn("filter reading", response.get_json()["error"])

    def test_full_review_then_recompile(self):
        self._clear_materiality_flow()

        decision = self.client.post(
            f"/api/review/episodes/{self.episode}/items/6/decision",
            json={
                "source": "apple",
                "text": "The trial included 42 participants and lasted twelve weeks.",
            },
        )
        self.assertEqual(decision.status_code, 200)

        final = self._review()
        self.assertEqual(
            final["progress"],
            {
                "total": 6,
                "reviewed": 6,
                "remaining": 0,
                "assisted_unprepared": 0,
                "triage_unavailable": 0,
            },
        )

        recompile = self.client.post(
            f"/api/review/episodes/{self.episode}/recompile"
        )
        self.assertEqual(recompile.status_code, 202)
        self.assertEqual(
            recompile.get_json()["recompile"]["status"],
            "accepted",
        )

    def test_third_asr_is_synthetic_and_local(self):
        response = self.client.post(
            f"/api/review/episodes/{self.episode}/items/6/third-asr"
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("forty-two", response.get_json()["evidence"]["text"])


if __name__ == "__main__":
    unittest.main()
