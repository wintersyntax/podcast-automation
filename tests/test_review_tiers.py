"""TASK-127: deterministic three-tier Human Review queue."""

from __future__ import annotations

import copy
import unittest
from unittest.mock import patch

from compiler.review_tiers import REVIEW_TIER_POLICY_VERSION, derive_review_tier
from podcast_engine import human_review
from podcast_engine.review_web import create_review_app
from tests.test_third_asr_window import JEWETT_357


def _card(difference_id=1, **overrides):
    card = {
        "id": difference_id,
        "kind": "transcription_difference",
        "severity": "medium",
        "category": "other",
        "apple_text": "aim is to gain",
        "whisper_text": "a missing game",
        "risk_reasons": [],
        "citation_signal": False,
        "anomaly": None,
        "custom_edit": None,
        "third_asr": None,
        "triage": {"status": "advisory", "recommendation": "needs_audio", "confidence": "low"},
    }
    card.update(overrides)
    return card


def _triage(source, confidence="high"):
    return {"status": "advisory", "recommendation": f"recommend_{source}", "confidence": confidence, "source": source}


def _two_of_three(**overrides):
    card = _card(category="negation", risk_reasons=["negation"], **{
        key: copy.deepcopy(value) for key, value in JEWETT_357.items()
    })
    card["third_asr"] = {
        "text": (
            "Look for the stage, as in the competitors you're up against, the "
            "lighting, what the judges want to see. There's lots of errors that "
            "will play. So thinking about that as more of a, like, not a bullseye."
        )
    }
    card.update(overrides)
    return card


class DeriveReviewTierTests(unittest.TestCase):
    def test_high_confidence_triage_on_unprotected_card_is_tier_a(self):
        tier = derive_review_tier(_card(triage=_triage("whisper")))
        self.assertEqual(tier["tier"], "A")
        self.assertEqual((tier["reason"], tier["source"], tier["text"]), ("triage_high", "whisper", "a missing game"))
        self.assertEqual(tier["policy_version"], REVIEW_TIER_POLICY_VERSION)

    def test_low_confidence_or_needs_audio_triage_needs_listening(self):
        for triage in (_triage("apple", "low"), {"status": "advisory", "recommendation": "needs_audio", "confidence": "high"}, None):
            with self.subTest(triage=triage):
                self.assertEqual(derive_review_tier(_card(triage=triage))["reason"], "needs_listening")

    def test_whisper_gap_proposes_apple_for_unprotected_cards(self):
        for whisper in ("", "um, you know", "Uh."):
            with self.subTest(whisper=whisper):
                tier = derive_review_tier(_card(apple_text="That's typically five stars", whisper_text=whisper))
                self.assertEqual((tier["tier"], tier["reason"], tier["source"]), ("A", "whisper_gap", "apple"))

    def test_apple_gap_is_not_a_proposal(self):
        tier = derive_review_tier(_card(apple_text="", whisper_text="20%"))
        self.assertEqual((tier["tier"], tier["reason"]), ("B", "needs_listening"))

    def test_protected_cards_never_enter_tier_a_without_two_of_three(self):
        protected = (
            {"category": "negation"},
            {"category": "protocol_number"},
            {"category": "proper_name"},
            {"risk_reasons": ["number_or_date"]},
            {"citation_signal": True},
        )
        for marker in protected:
            with self.subTest(marker=marker):
                gap = derive_review_tier(_card(apple_text="I'm not the", whisper_text="", **marker))
                triage = derive_review_tier(_card(triage=_triage("apple"), **marker))
                self.assertEqual((gap["tier"], gap["reason"]), ("C", "protected_without_confirmation"))
                self.assertEqual((triage["tier"], triage["reason"]), ("C", "protected_without_confirmation"))

    def test_exact_two_of_three_admits_even_a_protected_card(self):
        tier = derive_review_tier(_two_of_three())
        self.assertEqual((tier["tier"], tier["reason"], tier["source"]), ("A", "two_of_three", "apple"))
        self.assertEqual(tier["text"], JEWETT_357["apple_text"])

    def test_third_voice_with_a_new_reading_is_tier_b(self):
        tier = derive_review_tier(_card(**{key: copy.deepcopy(value) for key, value in JEWETT_357.items()}))
        self.assertEqual((tier["tier"], tier["reason"]), ("B", "third_new_reading"))

    def test_conflicting_signals_never_produce_tier_a(self):
        gap_vs_triage = derive_review_tier(_card(apple_text="five stars", whisper_text="", triage=_triage("whisper")))
        self.assertEqual((gap_vs_triage["tier"], gap_vs_triage["reason"]), ("B", "conflicting_evidence"))
        protected_conflict = derive_review_tier(_two_of_three(triage=_triage("whisper")))
        self.assertEqual((protected_conflict["tier"], protected_conflict["reason"]), ("C", "conflicting_evidence"))
        heard_else = derive_review_tier(_card(
            triage=_triage("apple"), **{key: copy.deepcopy(value) for key, value in JEWETT_357.items()}
        ))
        self.assertEqual((heard_else["tier"], heard_else["reason"]), ("B", "conflicting_evidence"))

    def test_anomalies_and_custom_edits_are_never_tier_a(self):
        for override in ({"anomaly": {"kind": "loop"}}, {"custom_edit": {"text": "x"}}, {"generation_stale": True}):
            with self.subTest(override=override):
                tier = derive_review_tier(_card(triage=_triage("apple"), **override))
                self.assertEqual((tier["tier"], tier["reason"]), ("B", "no_safe_proposal"))

    def test_partial_focus_uses_the_focused_texts(self):
        card = _card(triage=_triage("apple"), focus={"scope": "partial", "apple_text": "aim", "whisper_text": "a miss"})
        tier = derive_review_tier(card)
        self.assertEqual((tier["source"], tier["text"]), ("apple", "aim"))


def _record(cards):
    fingerprint = "sha256:input"
    return {
        "episode_key": "episode-1",
        "input_fingerprint": fingerprint,
        "human_decisions": [],
        "human_review": cards,
        "human_review_queue_fingerprint": human_review.review_queue_fingerprint(cards),
        "human_review_generation_fingerprint": human_review.review_generation_fingerprint(fingerprint, cards),
    }


class RecordTierADecisionBatchTests(unittest.TestCase):
    def setUp(self):
        self.store = _record([
            _card(1, triage=_triage("whisper")),
            _card(2, apple_text="That's typically five stars", whisper_text=""),
            _card(3),
        ])
        self.saves = []
        load = patch(
            "podcast_engine.human_review.load_review_record_with_generation",
            side_effect=lambda _key: (copy.deepcopy(self.store), 1),
        )
        save = patch(
            "podcast_engine.human_review.save_review_record",
            side_effect=lambda _key, value, **kwargs: self.saves.append((copy.deepcopy(value), kwargs)),
        )
        for patcher in (load, save):
            patcher.start()
            self.addCleanup(patcher.stop)

    def _confirm(self, ids, generation=None):
        return human_review.record_tier_a_decision_batch(
            "episode-1",
            ids,
            expected_generation_fingerprint=generation or self.store["human_review_generation_fingerprint"],
        )

    def test_confirmation_records_one_audited_source_decision_per_card(self):
        updated = self._confirm([1, 2])
        self.assertEqual(len(self.saves), 1)
        self.assertEqual(self.saves[0][1], {"if_generation_match": 1})
        decisions = {decision["id"]: decision for decision in updated["human_decisions"]}
        self.assertEqual((decisions[1]["chosen_source"], decisions[1]["chosen_text"]), ("whisper", "a missing game"))
        self.assertEqual((decisions[2]["chosen_source"], decisions[2]["chosen_text"]), ("apple", "That's typically five stars"))
        for decision in decisions.values():
            self.assertEqual(decision["reviewed_by"], "human")
            self.assertEqual(decision["review_tier"]["tier"], "A")
            self.assertIn("routing_provenance", decision)
        self.assertEqual([item["id"] for item in updated["human_review"]], [3])

    def test_a_card_outside_tier_a_rejects_the_whole_confirmation(self):
        with self.assertRaisesRegex(ValueError, "no longer in tier A"):
            self._confirm([1, 3])
        self.assertEqual(self.saves, [])

    def test_stale_generation_and_malformed_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Review generation changed"):
            self._confirm([1], generation="sha256:old")
        for ids in ([], [1, 1], ["1"], [True], "1"):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                self._confirm(ids)
        self.assertEqual(self.saves, [])

    def test_individual_decisions_snapshot_their_tier(self):
        with (
            patch("podcast_engine.human_review.load_review_record", side_effect=lambda _key: copy.deepcopy(self.store)),
            patch("podcast_engine.human_review.save_review_record") as save,
        ):
            decision = human_review.record_human_decision("episode-1", 3, source="apple")
        self.assertEqual(decision["review_tier"]["tier"], "B")
        self.assertEqual(decision["review_tier"]["reason"], "needs_listening")
        save.assert_called_once()


class TierWebTests(unittest.TestCase):
    def test_review_api_presents_tiers_and_counts(self):
        record = _record([
            _card(1, triage=_triage("whisper")),
            _card(2, category="negation", apple_text="I'm not the", whisper_text=""),
            _card(3),
        ])
        with patch("podcast_engine.review_web.load_review_record", return_value=record):
            response = create_review_app().test_client().get("/api/review/episodes/episode-1")
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual([card["review_tier"]["tier"] for card in body["cards"]], ["A", "C", "B"])
        self.assertEqual(body["progress"]["tiers"], {"A": 1, "B": 1, "C": 1})

    def test_tier_a_endpoint_sends_only_ids_to_the_backend(self):
        with patch(
            "podcast_engine.review_web.record_tier_a_decision_batch",
            return_value={"human_review": [{"id": 3}]},
        ) as confirm:
            response = create_review_app().test_client().post(
                "/api/review/episodes/episode-1/tier-a-decision",
                json={"expected_generation_fingerprint": "sha256:g", "ids": [1, 2]},
            )
        self.assertEqual(response.status_code, 200)
        confirm.assert_called_once_with("episode-1", [1, 2], expected_generation_fingerprint="sha256:g")
        self.assertEqual(response.get_json(), {"accepted_count": 2, "ready_to_recompile": False})

    def test_tier_a_endpoint_rejects_missing_input_and_stale_cards(self):
        client = create_review_app().test_client()
        for body in ({}, {"ids": [1]}, {"expected_generation_fingerprint": "g", "ids": []}):
            with self.subTest(body=body):
                self.assertEqual(client.post("/api/review/episodes/e/tier-a-decision", json=body).status_code, 400)
        with patch(
            "podcast_engine.review_web.record_tier_a_decision_batch",
            side_effect=ValueError("Review item 3 is no longer in tier A"),
        ):
            response = client.post(
                "/api/review/episodes/e/tier-a-decision",
                json={"expected_generation_fingerprint": "g", "ids": [3]},
            )
        self.assertEqual(response.status_code, 409)

    def test_page_has_the_tier_a_panel(self):
        from podcast_engine.review_web import PAGE

        self.assertIn("id=tier-a ", PAGE)
        self.assertIn("/tier-a-decision", PAGE)
        self.assertIn("sortByTier(data.cards)", PAGE)
        # TASK-126: the third voice is fetched for tier B and C cards on load.
        self.assertIn("id=third-prefetch ", PAGE)
        self.assertIn("renderAssisted();prefetchThird()}", PAGE)
        self.assertIn("(t==='B'||t==='C')&&(!c.third_asr||c.third_refresh)", PAGE)



class ReviewTiersV2Tests(unittest.TestCase):
    """TASK-126 step 2: agreement on the disputed words only."""

    def _with_window(self, card, text):
        return patch("compiler.review_tiers.anchored_third_asr_window", return_value={"text": text}), card

    def _tier(self, card, window_text):
        with patch("compiler.review_tiers.anchored_third_asr_window", return_value={"text": window_text}):
            return derive_review_tier(card)

    def test_policy_version_is_v4(self):
        self.assertEqual(REVIEW_TIER_POLICY_VERSION, "review-tiers-v4")

    def test_only_colloquial_difference_proposes_the_spoken_form(self):
        # Historical #239 and #374: the reviewer kept Whisper's spoken "gonna";
        # the third voice (which writes "going to") is not evidence here.
        cases = [
            ("going to lead in to", "gonna lead into", "going to lead into"),
            ("going to do some dead lifts, but I'm going to", "gonna do some deadlifts, but I'm gonna",
             "going to do some deadlifts, but I'm going to"),
        ]
        for apple, whisper, third in cases:
            with self.subTest(apple=apple):
                tier = self._tier(_card(apple_text=apple, whisper_text=whisper), third)
                self.assertEqual((tier["tier"], tier["reason"], tier["source"]), ("A", "spoken_colloquial_form", "whisper"))

    def test_colloquial_difference_with_other_words_needs_listening(self):
        # Historical #255 and #399: other words differ too; the reviewer chose
        # Whisper, while a folded comparison would have "confirmed" Apple.
        cases = [
            ("fatigue is going to", "fatigue's gonna", "fatigue is gonna"),
            ("like biohacker guy, who's going to like", "biohacker guy who's gonna", "like, biohacker guy who's gonna, like"),
        ]
        for apple, whisper, third in cases:
            with self.subTest(apple=apple):
                tier = self._tier(_card(apple_text=apple, whisper_text=whisper), third)
                self.assertEqual((tier["tier"], tier["reason"]), ("B", "colloquial_mixed"))

    def test_protected_colloquial_card_is_never_tier_a(self):
        card = _card(category="negation", risk_reasons=["negation"], apple_text="I'm not going to", whisper_text="I'm not gonna")
        self.assertEqual(self._tier(card, "I'm not gonna")["tier"], "C")

    def test_numbers_and_unit_symbols_are_never_dropped(self):
        from compiler.review_tiers import _compare_tokens

        self.assertEqual(_compare_tokens("like 8%"), ["like", "8", "%"])
        self.assertEqual(_compare_tokens("2.5 grams"), ["2.5", "grams"])
        self.assertNotEqual(_compare_tokens("2.5 grams"), _compare_tokens("25 grams"))
        card = _card(category="unit", risk_reasons=["unit"], apple_text="8", whisper_text="like 8%")
        tier = self._tier(card, "like 8")
        self.assertNotEqual(tier["tier"], "A")

    def test_fillers_do_not_block_exact_agreement(self):
        card = _card(apple_text="it'll probably normalize later, you may even gain more once you", whisper_text="you know")
        tier = self._tier(card, "it'll probably normalize later. You may even gain more once you, you know")
        self.assertEqual((tier["tier"], tier["reason"], tier["source"]), ("A", "two_of_three", "apple"))

    def test_disputed_words_decide_even_when_shared_words_differ(self):
        card = _card(
            apple_text="we went to the gym and lifted heavy weights today",
            whisper_text="we went to the gem and lifted heavy weights today",
        )
        # The third voice slips on a word both sources agree on ("weight"),
        # but hears "gym" between the shared neighbours.
        tier = self._tier(card, "we went to the gym and lifted heavy weight today")
        self.assertEqual((tier["tier"], tier["reason"], tier["source"]), ("A", "two_of_three_disputed", "apple"))

    def test_disputed_agreement_must_pick_one_source_everywhere(self):
        card = _card(
            apple_text="we went to the gym and lifted heavy weights today",
            whisper_text="we went to the gem and lifted heavy wheats today",
        )
        tier = self._tier(card, "we went to the gym and lifted heavy wheats today")
        self.assertEqual((tier["tier"], tier["reason"]), ("B", "third_new_reading"))

    def test_disputed_agreement_needs_unambiguous_anchors(self):
        from compiler.review_tiers import _disputed_agreement

        apple = ["the", "gym", "the", "end"]
        whisper = ["the", "gem", "the", "end"]
        self.assertIsNone(_disputed_agreement(apple, whisper, ["the", "the", "gym", "the", "end"]))
        # A one-word anchor that occurs twice in the clip is ambiguous too.
        self.assertIsNone(_disputed_agreement(apple, whisper, ["the", "gym", "the", "end"]))
        self.assertEqual(
            _disputed_agreement(["big", "gym", "the", "end"], ["big", "gem", "the", "end"], ["big", "gym", "the", "end"]),
            "apple",
        )

    def test_protected_card_with_disputed_agreement_still_needs_a_click(self):
        card = _card(
            category="negation",
            risk_reasons=["negation"],
            apple_text="I really do not think that works for most people",
            whisper_text="I really do think that works for most people",
        )
        tier = self._tier(card, "I really do not think that works for many people")
        self.assertEqual((tier["tier"], tier["reason"], tier["source"]), ("A", "two_of_three_disputed", "apple"))

    def test_localization_anomaly_needs_positive_agreement(self):
        anomaly = {"kind": "audio_localization_unreliable", "reason": "test"}
        card = _card(category="negation", risk_reasons=["negation"], anomaly=anomaly, apple_text="No", whisper_text="")
        tier = self._tier(card, "No")
        self.assertEqual((tier["tier"], tier["reason"], tier["source"]), ("A", "two_of_three", "apple"))
        # Agreement with an empty source proves nothing about localization.
        empty = _card(category="negation", risk_reasons=["negation"], anomaly=anomaly, apple_text="isn't", whisper_text="")
        self.assertEqual(self._tier(empty, "")["reason"], "no_safe_proposal")
        # Other anomalies stay blocked.
        other = _card(anomaly={"kind": "something_else"}, apple_text="No", whisper_text="")
        self.assertEqual(self._tier(other, "No")["reason"], "no_safe_proposal")
        # A conflicting high-confidence triage keeps the card out of tier A.
        conflict = _card(anomaly=anomaly, apple_text="No", whisper_text="", triage=_triage("whisper"))
        self.assertEqual(self._tier(conflict, "No")["reason"], "no_safe_proposal")



class ThirdReadingProposalPageTests(unittest.TestCase):
    """The third reading is never pre-selected; keyboard shortcuts stay."""

    def test_page_does_not_preselect_the_third_reading(self):
        from podcast_engine.review_web import PAGE

        # On historical decisions the third reading was right on only about
        # 8 of 26 cards where it differed from both sources.
        self.assertNotIn("proposeThirdReading", PAGE)
        self.assertNotIn("selection.proposed", PAGE)
        self.assertIn("return Boolean(selection)||", PAGE)
        # Shortcuts: 1/2/3 choose a source, Enter confirms, Space plays, D defers;
        # ignored while typing in a field.
        self.assertIn("document.addEventListener('keydown'", PAGE)
        self.assertIn("typingTarget(event.target)", PAGE)
        self.assertIn("two_of_three_disputed:", PAGE)


if __name__ == "__main__":
    unittest.main()
