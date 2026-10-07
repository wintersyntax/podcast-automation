"""TASK-133: materiality groups in Human Review and their explicit decisions."""

from __future__ import annotations

import copy
import os
import unittest
from unittest.mock import patch

from compiler.materiality import MATERIALITY_POLICY_VERSION, materiality_inputs
from podcast_engine import human_review
from podcast_engine import materiality_queue as queue
from podcast_engine.materiality_shadow import EVIDENCE_KEY, evidence_identity
from podcast_engine.review_web import PAGE, create_review_app
from tests.test_materiality import card


def judged(difference_id: int, apple: str, whisper: str, **evidence) -> dict:
    item = dict(card("so we said", apple, whisper, "and then"), id=difference_id)
    item[EVIDENCE_KEY] = {
        "policy_version": MATERIALITY_POLICY_VERSION,
        "identity": evidence_identity(materiality_inputs(item)),
        **evidence,
    }
    return item


SETTLED = dict(immaterial=True, route="judge", reason="judge_unanimous", use="whisper", use_step="third_asr")


def settled(difference_id: int) -> dict:
    return judged(difference_id, f"apple {difference_id}x", f"whisper {difference_id}x", **SETTLED)


class GroupTests(unittest.TestCase):
    def test_groups_follow_current_evidence(self):
        items = [
            settled(1),
            judged(2, "a", "b", immaterial=True, reason="judge_unanimous", use=None, use_step="click_one"),
            judged(3, "squat", "squelch", immaterial=False, reason="judge_not_unanimous",
                   proposal="apple", proposal_step="judge"),
            judged(4, "2017", "17", immaterial=False, reason="number", proposal=None, proposal_step=None),
            dict(card("x", "a", "b", "y"), id=5),  # no evidence
        ]
        groups = {k: v["group"] for k, v in queue.materiality_groups({"human_review": items}, "e", items).items()}
        self.assertEqual(groups[2], "click_one")
        self.assertEqual(groups[3], "proposal")
        self.assertEqual((groups[4], groups[5]), ("full", "full"))
        # The only settled card is drawn into the control sample (at least two per episode).
        self.assertEqual(groups[1], "sample")

    def test_stale_evidence_counts_as_no_evidence(self):
        item = settled(1)
        item["whisper_text"] = "changed meanwhile"
        self.assertEqual(queue.base_group(item)["group"], "full")
        old = settled(2)
        old[EVIDENCE_KEY]["policy_version"] = "materiality-v2"
        self.assertEqual(queue.base_group(old)["group"], "full")

    def test_either_uses_the_compiler_suggestion_or_asks(self):
        either = dict(immaterial=True, reason="same_letters", use=None, use_step="either")
        with_suggestion = judged(1, "e-mail", "email", **either)
        with_suggestion["suggestion"] = {"source": "apple"}
        self.assertEqual(queue.base_group(with_suggestion)["group"], "settled")
        self.assertEqual(queue.base_group(with_suggestion)["source"], "apple")
        self.assertEqual(queue.base_group(judged(2, "e-mail", "email", **either))["group"], "click_one")

    def test_control_sample_is_stable_ten_percent_with_a_minimum_and_no_spelling_cards(self):
        items = [settled(i) for i in range(1, 41)]
        spelling = judged(99, "e-mail", "email", immaterial=True, reason="same_letters", use="apple", use_step="fuller")
        record = {"human_review": items + [spelling]}
        sample = queue.control_sample_ids(record, "episode")
        self.assertEqual(len(sample), 4)
        self.assertNotIn(99, sample)
        # Deciding a sampled card does not pull another card into the sample.
        decided_id = sorted(sample)[0]
        decided = next(item for item in items if item["id"] == decided_id)
        after = {
            "human_review": [item for item in record["human_review"] if item["id"] != decided_id],
            "human_decisions": [{"id": decided_id, "review_item": decided}],
        }
        self.assertEqual(queue.control_sample_ids(after, "episode"), sample)
        self.assertEqual(len(queue.control_sample_ids({"human_review": items[:5]}, "e")), 2)

    def test_off_switch_puts_every_card_in_ordinary_review(self):
        items = [settled(1)]
        with patch.dict(os.environ, {queue.QUEUE_ENV: "off"}):
            groups = queue.materiality_groups({"human_review": items}, "e", items)
        self.assertEqual(groups[1]["group"], "full")


class DecisionBatchTests(unittest.TestCase):
    def setUp(self):
        self.items = [settled(i) for i in range(1, 31)] + [
            judged(50, "a", "b", immaterial=True, reason="judge_unanimous", use=None, use_step="click_one"),
            judged(51, "2017", "17", immaterial=False, reason="number", proposal=None, proposal_step=None),
        ]
        self.store = {"human_review": copy.deepcopy(self.items), "human_decisions": [],
                      "human_review_generation_fingerprint": "sha256:g"}
        self.sample = queue.control_sample_ids(self.store, "episode-1")
        self.settled_ids = [i for i in range(1, 31) if i not in self.sample]
        self.saves = []
        for patcher in (
            patch("podcast_engine.human_review.load_review_record_with_generation",
                  side_effect=lambda _key: (copy.deepcopy(self.store), 7)),
            patch("podcast_engine.human_review.save_review_record",
                  side_effect=lambda _key, value, **kwargs: self.saves.append((value, kwargs))),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def record(self, decisions, generation="sha256:g"):
        return human_review.record_materiality_decision_batch(
            "episode-1", decisions, expected_generation_fingerprint=generation)

    def test_settled_cards_are_accepted_as_audited_human_decisions(self):
        updated = self.record([{"id": i, "source": "whisper"} for i in self.settled_ids]
                              + [{"id": 50, "source": "apple"}])
        self.assertEqual(self.saves[0][1], {"if_generation_match": 7})
        decisions = {d["id"]: d for d in updated["human_decisions"]}
        first = decisions[self.settled_ids[0]]
        self.assertEqual((first["reviewed_by"], first["chosen_source"]), ("human", "whisper"))
        self.assertEqual(first["materiality"]["group"], "settled")
        self.assertEqual((first["materiality"]["step"], first["materiality"]["agrees_with_filter"]), ("third_asr", True))
        self.assertEqual(decisions[50]["materiality"]["group"], "click_one")
        self.assertIsNone(decisions[50]["materiality"]["agrees_with_filter"])
        self.assertEqual(sorted(item["id"] for item in updated["human_review"]), sorted(self.sample | {51}))

    def test_settled_card_with_another_reading_and_ordinary_cards_reject_the_batch(self):
        with self.assertRaisesRegex(ValueError, "settled reading changed"):
            self.record([{"id": self.settled_ids[0], "source": "apple"}])
        with self.assertRaisesRegex(ValueError, "not in a materiality group"):
            self.record([{"id": 50, "source": "apple"}, {"id": 51, "source": "apple"}])
        self.assertEqual(self.saves, [])

    def test_sampled_card_may_overrule_the_filter(self):
        sampled = sorted(self.sample)[0]
        updated = self.record([{"id": sampled, "source": "apple"}])
        decision = next(d for d in updated["human_decisions"] if d["id"] == sampled)
        self.assertEqual((decision["materiality"]["group"], decision["materiality"]["agrees_with_filter"]), ("sample", False))

    def test_time_and_session_notes_are_audited(self):
        sampled = sorted(self.sample)[0]
        updated = human_review.record_materiality_decision_batch(
            "episode-1",
            [{"id": sampled, "source": "apple", "seconds": 12}, {"id": 50, "source": "whisper", "seconds": 4}],
            expected_generation_fingerprint="sha256:g",
            session={"started_at": "2026-10-07T10:00:00Z", "settled_list_opened": True, "note": " Card 7 was wrong "},
        )
        decision = next(d for d in updated["human_decisions"] if d["id"] == sampled)
        self.assertEqual(decision["materiality"]["seconds"], 12)
        (entry,) = updated["materiality_review_log"]
        self.assertEqual(entry["by_group"], {"sample": 1, "click_one": 1})
        self.assertEqual(entry["overruled_filter"], {"sample": 1})
        self.assertEqual((entry["seconds"], entry["settled_list_opened"], entry["note"]), (16, True, "Card 7 was wrong"))

    def test_bad_seconds_and_sessions_are_rejected(self):
        for decisions, session in (
            ([{"id": 50, "source": "apple", "seconds": -1}], None),
            ([{"id": 50, "source": "apple", "seconds": True}], None),
            ([{"id": 50, "source": "apple", "extra": 1}], None),
            ([{"id": 50, "source": "apple"}], {"note": "x" * 2001}),
            ([{"id": 50, "source": "apple"}], {"settled_list_opened": "yes"}),
            ([{"id": 50, "source": "apple"}], {"other": 1}),
        ):
            with self.subTest(decisions=decisions, session=session), self.assertRaises(ValueError):
                human_review.record_materiality_decision_batch(
                    "episode-1", decisions, expected_generation_fingerprint="sha256:g", session=session)
        self.assertEqual(self.saves, [])

    def test_report_summarizes_overruled_filter_choices(self):
        from scripts.materiality_review_report import summarize_record

        sampled = sorted(self.sample)[0]
        updated = human_review.record_materiality_decision_batch(
            "episode-1", [{"id": sampled, "source": "apple", "seconds": 9}],
            expected_generation_fingerprint="sha256:g", session={"note": "too many cards"})
        summary = summarize_record(updated)
        self.assertEqual(summary["by_group"]["sample"], 1)
        self.assertEqual([o["id"] for o in summary["sample_overruled"]], [sampled])
        self.assertEqual(summary["sample_overruled"][0]["filter"], "whisper")
        self.assertEqual((summary["quick_seconds"], summary["notes"]), (9, ["too many cards"]))

    def test_malformed_requests_and_stale_generation_are_rejected(self):
        for bad in ([], [{"id": 1}], [{"id": "1", "source": "apple"}], [{"id": 1, "source": "third"}],
                    [{"id": 50, "source": "apple"}, {"id": 50, "source": "whisper"}]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.record(bad)
        with self.assertRaisesRegex(ValueError, "Review generation changed"):
            self.record([{"id": 50, "source": "apple"}], generation="sha256:old")
        self.assertEqual(self.saves, [])


class WebTests(unittest.TestCase):
    def test_review_api_presents_groups_and_counts(self):
        items = [settled(1), judged(3, "squat", "squelch", immaterial=False, reason="judge_not_unanimous",
                                    proposal="apple", proposal_step="judge")]
        record = {"human_review": items, "human_decisions": []}
        with patch("podcast_engine.review_web.load_review_record", return_value=record):
            body = create_review_app().test_client().get("/api/review/episodes/episode-1").get_json()
        groups = {c["id"]: c["materiality"]["group"] for c in body["cards"]}
        context = {c["id"]: c["materiality_context"] for c in body["cards"]}
        self.assertEqual({k: v.strip() for k, v in context[3].items()}, {"left": "so we said", "right": "and then"})
        self.assertEqual(groups, {1: "sample", 3: "proposal"})
        self.assertEqual(body["progress"]["materiality"],
                         {"settled": 0, "sample": 1, "click_one": 0, "proposal": 1, "full": 0})

    def test_endpoint_passes_decisions_and_maps_errors(self):
        client = create_review_app().test_client()
        with patch("podcast_engine.review_web.record_materiality_decision_batch",
                   return_value={"human_review": []}) as record:
            response = client.post("/api/review/episodes/e/materiality-decision",
                                   json={"expected_generation_fingerprint": "g", "decisions": [{"id": 1, "source": "apple"}]})
        self.assertEqual(response.get_json(), {"accepted_count": 1, "ready_to_recompile": True})
        record.assert_called_once_with("e", [{"id": 1, "source": "apple"}], expected_generation_fingerprint="g", session=None)
        for body in ({}, {"decisions": [{"id": 1, "source": "apple"}]}, {"expected_generation_fingerprint": "g"}):
            self.assertEqual(client.post("/api/review/episodes/e/materiality-decision", json=body).status_code, 400)
        with patch("podcast_engine.review_web.record_materiality_decision_batch", side_effect=ValueError("stale")):
            response = client.post("/api/review/episodes/e/materiality-decision",
                                   json={"expected_generation_fingerprint": "g", "decisions": [{"id": 1, "source": "apple"}]})
        self.assertEqual(response.status_code, 409)

    def test_page_has_the_panels_and_shortcuts(self):
        for needle in ("id=materiality ", "id=materiality-settled ", "/materiality-decision", "function quickBack()", "function quickContext(", "In the transcript", "ArrowLeft", "id=feedback-text", "Back to quick cards", "Check &amp; save",
                       "k==='u'||k==='U'", "Accept ${settled.length}", "cards=applyCards(sortByTier(data.cards))"):
            self.assertIn(needle, PAGE)
        # No bulk confirmation for proposals: only settled cards are accepted together.
        self.assertNotIn("Confirm all", PAGE)


if __name__ == "__main__":
    unittest.main()
