"""TASK-133: shadow-mode materiality evidence recorded by the Worker."""

from __future__ import annotations

import copy
import json
import os
import unittest
from unittest.mock import patch

from google.api_core.exceptions import PreconditionFailed

from podcast_engine import materiality_shadow as shadow
from podcast_engine import pipeline
from podcast_engine.ai_budget import STAGE_MATERIALITY, BudgetAdmissionError
from podcast_engine.preset_provenance import PresetProvenance
from tests.test_materiality import card

EPISODE = {"episode_key": "b" * 24}
FINGERPRINT = "sha256:" + "1" * 64
ENV = {"PODCAST_TRANSCRIPT_REVIEW_API_KEY": "review-key", "PODCAST_WORKER_MATERIALITY_SHADOW": "1"}

RULE_CARD = dict(card("we got a lovely", "e-mail from someone", "email from someone", "who we had covered"), id=1)
NUMBER_CARD = dict(card("you're in that probably", "200", "2", "to 400 calorie surplus"), id=2)
JUDGE_CARD = dict(card("I'm in the whatever camp where", "creatine", "creating", "coffee fish oil"), id=3)


class FakeStore:
    def __init__(self, items: list[dict]):
        self.record = {"source_fingerprint": FINGERPRINT, "human_review": copy.deepcopy(items)}
        self.generation = 1
        self.saves = 0
        self.conflicts = 0

    def load(self, _key: str):
        return copy.deepcopy(self.record), self.generation

    def save(self, _key: str, record: dict, *, if_generation_match: int):
        if self.conflicts:
            self.conflicts -= 1
            self.generation += 1
            raise PreconditionFailed("moved")
        if if_generation_match != self.generation:
            raise PreconditionFailed("stale")
        self.record = copy.deepcopy(record)
        self.generation += 1
        self.saves += 1
        return record

    def evidence(self, item_id: int) -> dict | None:
        for item in self.record["human_review"]:
            if item["id"] == item_id:
                return item.get(shadow.EVIDENCE_KEY)
        return None


def provenance(prompt: str | None = None) -> PresetProvenance:
    return PresetProvenance(
        status="verified",
        slug="podcast-materiality-judge",
        version_id="v1",
        config={"model": "google/gemini-3.8-flash"},
        system_prompt=shadow.prompt_text() if prompt is None else prompt,
        config_digest="sha256:" + "2" * 64,
    )


def answer(verdict: str, better: str = "unclear") -> dict:
    content = {"verdict": verdict, "better_reading": better, "reason": "r"}
    return {"choices": [{"message": {"content": json.dumps(content)}}]}


class Poster:
    """Answers in order; a tuple is (verdict, better_reading)."""

    def __init__(self, verdicts):
        self.verdicts = list(verdicts)
        self.calls: list[dict] = []

    def __call__(self, payload, **kwargs):
        self.calls.append({"payload": payload, **kwargs})
        verdict = self.verdicts.pop(0)
        if isinstance(verdict, Exception):
            raise verdict
        return answer(*verdict) if isinstance(verdict, tuple) else answer(verdict)


class ShadowSwitchTests(unittest.TestCase):
    def test_disabled_without_key_or_with_off_switch(self):
        with patch.dict(os.environ, dict(ENV, PODCAST_TRANSCRIPT_REVIEW_API_KEY=""), clear=False):
            self.assertFalse(shadow.shadow_enabled())
        with patch.dict(os.environ, dict(ENV, PODCAST_WORKER_MATERIALITY_SHADOW="off"), clear=False):
            self.assertFalse(shadow.shadow_enabled())
        with patch.dict(os.environ, ENV, clear=False):
            self.assertTrue(shadow.shadow_enabled())

    def test_disabled_run_reads_nothing(self):
        with patch.dict(os.environ, dict(ENV, PODCAST_WORKER_MATERIALITY_SHADOW="0"), clear=False):
            summary = shadow.run_materiality_shadow(EPISODE, load_record=self.fail)
        self.assertEqual(summary["status"], "disabled")

    @staticmethod
    def fail(*_args, **_kwargs):
        raise AssertionError("must not be called")


class ShadowRunTests(unittest.TestCase):
    def run_shadow(self, store, *, post, preset=None):
        with patch.dict(os.environ, ENV, clear=False):
            return shadow.run_materiality_shadow(
                EPISODE,
                load_record=store.load,
                save_record=store.save,
                fetch_preset=lambda *a, **k: preset or provenance(),
                post=post,
                workers=1,
                now=lambda: "2026-10-06T12:00:00Z",
            )

    def test_rule_and_number_cards_need_no_judge_and_rerun_is_a_no_op(self):
        store = FakeStore([RULE_CARD, NUMBER_CARD])
        poster = Poster([])
        summary = self.run_shadow(store, post=poster)
        self.assertEqual((summary["rule_immaterial"], summary["reviewer"]), (1, 1))
        self.assertEqual(poster.calls, [])
        self.assertTrue(store.evidence(1)["immaterial"])
        # Same letters, same word count: either reading may be kept.
        self.assertEqual((store.evidence(1)["use"], store.evidence(1)["use_step"]), (None, "either"))
        self.assertEqual(store.evidence(2)["route"], "reviewer")
        # Two different numbers: no proposal, the reviewer decides.
        self.assertEqual((store.evidence(2)["proposal"], store.evidence(2)["proposal_step"]), (None, None))
        saves = store.saves
        again = self.run_shadow(store, post=poster)
        self.assertEqual(again["already_recorded"], 2)
        self.assertEqual(store.saves, saves)

    def test_sponsor_read_is_settled_by_keeping_the_other_reading(self):
        ad = "This episode is brought to you by Acme. " + "Acme makes great things for everyone every day. " * 4
        store = FakeStore([dict(card("so welcome back", ad, "", "to the show today"), id=5)])
        summary = self.run_shadow(store, post=Poster([]))
        self.assertEqual(summary["rule_immaterial"], 1)
        evidence = store.evidence(5)
        self.assertEqual((evidence["reason"], evidence["use"], evidence["immaterial"]), ("advertisement", "whisper", True))
        self.assertEqual(evidence["use_step"], "advertisement")

    def test_same_numbers_written_differently_are_settled_without_a_judge(self):
        store = FakeStore([dict(card("the show starts at midnight, 1", "am or 2am", "a.m. or 2 a.m", "the more"), id=6)])
        poster = Poster([])
        summary = self.run_shadow(store, post=poster)
        self.assertEqual((summary["rule_immaterial"], poster.calls), (1, []))
        evidence = store.evidence(6)
        self.assertEqual((evidence["reason"], evidence["use_step"]), ("number_format", "either"))

    def test_number_card_where_one_source_heard_nothing_gets_a_proposal(self):
        store = FakeStore([dict(card("I train for about", "50 minutes", "", "and then I eat"), id=7)])
        self.run_shadow(store, post=Poster([]))
        evidence = store.evidence(7)
        self.assertEqual((evidence["route"], evidence["immaterial"]), ("reviewer", False))
        self.assertEqual((evidence["proposal"], evidence["proposal_step"]), ("apple", "number_gap"))

    def test_unanimous_judge_marks_the_card_immaterial(self):
        store = FakeStore([JUDGE_CARD])
        poster = Poster([("same_content", "reading_1"), ("same_content", "reading_2"), ("same_content", "unclear")])
        summary = self.run_shadow(store, post=poster)
        self.assertEqual(summary["judge_immaterial"], 1)
        self.assertEqual(len(poster.calls), 3)
        self.assertTrue(all(call["stage"] == STAGE_MATERIALITY for call in poster.calls))
        self.assertTrue(all(call["source_fingerprint"] == FINGERPRINT for call in poster.calls))
        first, second = (json.loads(json.dumps(c["payload"]["messages"][1]["content"])) for c in poster.calls[:2])
        self.assertIn("Reading 1 of the disputed span: creatine", first)
        self.assertIn("Reading 1 of the disputed span: creating", second)
        evidence = store.evidence(3)
        self.assertEqual((evidence["route"], evidence["immaterial"]), ("judge", True))
        self.assertEqual(evidence["preset"]["model"], "google/gemini-3.8-flash")
        self.assertEqual(evidence["votes"], ["apple", "apple", "unclear"])
        # "creatine" is a registry term only the Apple reading names.
        self.assertEqual((evidence["use"], evidence["use_step"]), ("apple", "registry_term"))

    def test_material_card_is_asked_three_times_and_gets_a_unanimous_proposal(self):
        store = FakeStore([JUDGE_CARD])
        # Orders are (Apple, Whisper), (Whisper, Apple), (Apple, Whisper).
        poster = Poster([("changes_content", "reading_1"), ("changes_content", "reading_2"), ("same_content", "reading_1")])
        summary = self.run_shadow(store, post=poster)
        self.assertEqual((summary["judge_immaterial"], summary["reviewer"]), (0, 1))
        self.assertEqual(len(poster.calls), 3)
        evidence = store.evidence(3)
        self.assertEqual(evidence["verdicts"], ["changes_content", "changes_content", "same_content"])
        self.assertEqual((evidence["proposal"], evidence["proposal_step"]), ("apple", "judge"))
        self.assertNotIn("use", evidence)

    def test_split_votes_give_no_proposal(self):
        store = FakeStore([JUDGE_CARD])
        self.run_shadow(store, post=Poster([("changes_content", "reading_1")] * 3))
        evidence = store.evidence(3)
        self.assertEqual(evidence["votes"], ["apple", "whisper", "apple"])
        self.assertEqual((evidence["proposal"], evidence["proposal_step"]), (None, None))

    def test_preset_with_another_prompt_judges_nothing(self):
        store = FakeStore([JUDGE_CARD])
        poster = Poster([])
        summary = self.run_shadow(store, post=poster, preset=provenance("Some other prompt"))
        self.assertEqual((summary["status"], summary["judge_unavailable"]), ("judge_unavailable", 1))
        self.assertIsNone(store.evidence(3))

    def test_budget_refusal_stops_the_run_without_evidence(self):
        store = FakeStore([JUDGE_CARD])
        summary = self.run_shadow(store, post=Poster([BudgetAdmissionError("cap")]))
        self.assertEqual((summary["status"], summary["stopped_reason"]), ("stopped", "BudgetAdmissionError"))
        self.assertIsNone(store.evidence(3))

    def test_malformed_answer_fails_the_card_without_evidence(self):
        store = FakeStore([JUDGE_CARD])

        def post(_payload, **_kwargs):
            return {"choices": [{"message": {"content": "not json"}}]}

        summary = self.run_shadow(store, post=post)
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(summary["failure_reasons"], {"judge:ValueError": 1})
        self.assertIsNone(store.evidence(3))

    def test_judges_run_one_card_at_a_time_by_default(self):
        self.assertEqual(shadow.SHADOW_WORKERS, 1)


class FinalizeEvidenceTests(unittest.TestCase):
    def evidence_for(self, item):
        inputs = shadow.materiality_inputs(item)
        return {"identity": shadow.evidence_identity(inputs), "route": "rule"}

    def test_retries_a_moved_generation_then_records_once(self):
        store = FakeStore([RULE_CARD])
        store.conflicts = 1
        evidence = self.evidence_for(RULE_CARD)
        pauses = []
        self.assertEqual(
            shadow.finalize_evidence("k", 1, evidence, load_record=store.load, save_record=store.save, sleep=pauses.append),
            "recorded",
        )
        self.assertEqual(len(pauses), 1)
        self.assertEqual(shadow.finalize_evidence("k", 1, evidence, load_record=store.load, save_record=store.save), "unchanged")
        self.assertEqual(store.saves, 1)

    def test_changed_card_is_not_overwritten(self):
        store = FakeStore([dict(RULE_CARD, whisper_text="mail from someone")])
        evidence = self.evidence_for(RULE_CARD)
        self.assertEqual(shadow.finalize_evidence("k", 1, evidence, load_record=store.load, save_record=store.save), "stale")
        self.assertEqual(store.saves, 0)


class PayloadTests(unittest.TestCase):
    def test_payload_uses_repository_prompt_and_bounded_completion(self):
        inputs = shadow.materiality_inputs(JUDGE_CARD)
        payload = shadow.judge_payload(inputs, "apple", "whisper", provenance())
        self.assertEqual(payload["model"], "google/gemini-3.8-flash")
        self.assertEqual(payload["messages"][0]["content"], shadow.prompt_text())
        self.assertTrue(str(shadow.PROMPT_PATH).endswith("materiality-judge-v3.md"))
        schema = payload["response_format"]["json_schema"]["schema"]
        self.assertEqual(schema["required"], ["verdict", "better_reading", "reason"])
        self.assertEqual(payload["max_tokens"], shadow.MAX_COMPLETION_TOKENS)
        self.assertTrue(payload["usage"]["include"])

    def test_payload_refuses_a_preset_with_another_prompt(self):
        inputs = shadow.materiality_inputs(JUDGE_CARD)
        with self.assertRaises(RuntimeError):
            shadow.judge_payload(inputs, "apple", "whisper", provenance("different"))

    def test_parse_verdict_is_strict(self):
        self.assertEqual(shadow.parse_verdict(answer("same_content", "reading_2"))[:2], ("same_content", "reading_2"))
        no_better = {"choices": [{"message": {"content": "{\"verdict\": \"same_content\"}"}}]}
        for bad in ({}, {"choices": [{"message": {"content": "{\"verdict\": \"maybe\"}"}}]}, no_better):
            with self.assertRaises(ValueError):
                shadow.parse_verdict(bad)


class PipelineWiringTests(unittest.TestCase):
    def test_failure_never_blocks_the_notification(self):
        with (
            patch.object(pipeline, "run_materiality_shadow", side_effect=RuntimeError("boom")),
            patch("builtins.print"),
        ):
            self.assertEqual(pipeline._materiality_shadow_before_notification(EPISODE), {"status": "failed"})

    def test_summary_is_logged(self):
        summary = {"status": "completed", "rule_immaterial": 2}
        with (
            patch.object(pipeline, "run_materiality_shadow", return_value=summary),
            patch.object(pipeline, "emit_event") as event,
        ):
            self.assertEqual(pipeline._materiality_shadow_before_notification(EPISODE), summary)
        self.assertEqual(event.call_args.kwargs["rule_immaterial"], 2)


class EvaluationScriptTests(unittest.TestCase):
    def test_collects_pending_and_decided_cards_and_scores_leaks(self):
        from scripts import evaluate_materiality_shadow as evaluation

        settled = dict(RULE_CARD, materiality_shadow={"route": "rule", "immaterial": True})
        kept = dict(NUMBER_CARD, materiality_shadow={"route": "reviewer", "immaterial": False})
        record = {"human_review": [kept], "human_decisions": [{"review_item": settled, "chosen_source": "apple"}]}

        class Blob:
            def __init__(self, payload):
                self.payload = payload

            def exists(self):
                return self.payload is not None

            def download_as_bytes(self):
                return json.dumps(self.payload).encode()

        class Bucket:
            def blob(self, name):
                if name == "episodes.json":
                    return Blob([{"episode_key": "c" * 24, "title": "Ep"}])
                return Blob(record)

        cards = evaluation.collect_cards(Bucket())
        self.assertEqual(evaluation.summarize(cards)["Ep"]["would_settle"], 1)
        page, key = evaluation.check_page(cards)
        self.assertEqual(len(key), 2)
        self.assertIn("Treba li ti ova kartica?", page)
        leak = next(entry["n"] for entry in key if entry["immaterial"])
        other = next(entry["n"] for entry in key if not entry["immaterial"])
        result = evaluation.score(f"{leak:02d} DA, {other:02d} NE", key)
        self.assertEqual([entry["id"] for entry in result["leaks"]], [1])


class BackfillScriptTests(unittest.TestCase):
    def test_dry_run_plans_only_queued_episodes(self):
        from scripts import run_materiality_shadow as backfill

        recorded = dict(RULE_CARD, id=4)
        inputs = shadow.materiality_inputs(recorded)
        recorded["materiality_shadow"] = {"identity": shadow.evidence_identity(inputs)}
        records = {
            "a" * 24: {"human_review": [RULE_CARD, NUMBER_CARD, JUDGE_CARD, recorded]},
            "c" * 24: {"human_review": []},
        }

        def load(key):
            if key not in records:
                raise FileNotFoundError(key)
            return records[key]

        episodes = [{"episode_key": "a" * 24}, {"episode_key": "c" * 24}, {"episode_key": "d" * 24}]
        rows = backfill.queued_episodes(episodes, load)
        self.assertEqual([row["episode"]["episode_key"] for row in rows], ["a" * 24])
        self.assertEqual(
            rows[0]["plan"],
            {"rule_immaterial": 1, "reviewer_rule": 1, "for_judge": 1, "already_recorded": 1},
        )


if __name__ == "__main__":
    unittest.main()
