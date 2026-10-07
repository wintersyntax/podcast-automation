"""Tests for TASK-106 Phase A Task 8 (review, targeted fixes, and failure
policy).

Covers `podcast_engine.knowledge.notes_v2.review` (pure `note-review-v1`
verdict validation, the bounded fix-excerpt window, `splice_fixed_units`,
and the §8.2 `decide()` policy) and `scripts.knowledge_eval.review` (the
model-calling orchestration: one review call with no retry, the
`condition_dropped` -> re-extraction routing via Task 5's
`reextract_section`, and the bounded two-fix-round loop), mirroring the
combined-coverage pattern established by `test_notes_v2_sources.py` and
`test_notes_v2_composition.py`.
"""

from __future__ import annotations

import json
import unittest
from dataclasses import dataclass

from podcast_engine.knowledge.notes_v2.review import (
    FIX_EXCERPT_WINDOW_WORDS,
    REASON_DUPLICATE_VERDICT,
    REASON_INVALID_VERDICT,
    REASON_MISSING_VERDICT,
    REASON_UNKNOWN_KEY,
    REASON_UNKNOWN_UNIT_ID,
    build_fix_excerpt,
    decide,
    enumerate_review_units,
    splice_fixed_units,
    validate_review,
)
from scripts.knowledge_eval.review import (
    apply_fix_round,
    review_units,
    route_condition_dropped,
    run_review_and_fix,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _item(item_id: str = "sec01-i01", **overrides) -> dict:
    base = {
        "item_id": item_id,
        "kind": "claim",
        "statement": "the speaker made a claim",
        "value": "normal",
        "conditions": "none_stated",
        "evidence_basis": "research",
        "hedged": False,
        "scope": "core",
        "quantities": [],
        "quote": "some quote text",
        "quote_occurrence": 1,
        "local_id": "i-01",
        "section_id": "sec01",
    }
    base.update(overrides)
    return base


def _extraction_item(local_id: str = "i-01", **overrides) -> dict:
    base = {
        "local_id": local_id,
        "kind": "recommendation",
        "statement": "stmt",
        "quote": "take it before bed",
        "quote_occurrence": 1,
        "quantities": [],
        "negated": False,
        "evidence_basis": "research",
        "hedged": False,
        "conditions": "none_stated",
        "scope": "core",
        "value": "normal",
        "protocol": None,
        "source_mention": None,
        "rationale_for": None,
    }
    base.update(overrides)
    return base


def _extraction_result(section_id: str = "sec01", items: list[dict] | None = None, nothing_relevant: bool = False) -> dict:
    return {
        "section_id": section_id,
        "nothing_relevant": nothing_relevant,
        "nothing_relevant_reason": None,
        "items": items or [],
    }


def _unit(text: str = "A grounded sentence.", item_ids: list[str] | None = None) -> dict:
    return {"text": text, "item_ids": item_ids if item_ids is not None else []}


def _bullet(text: str = "A grounded bullet.", item_ids: list[str] | None = None, primary_item_id: str | None = None, asserts: bool = True) -> dict:
    item_ids = item_ids if item_ids is not None else []
    return {"text": text, "item_ids": item_ids, "primary_item_id": primary_item_id, "asserts": asserts}


def _section(title: str = "A topic", bullets: list[dict] | None = None, protocol_item_ids: list[str] | None = None) -> dict:
    return {
        "title": title,
        "bullets": bullets if bullets is not None else [],
        "protocol_item_ids": protocol_item_ids if protocol_item_ids is not None else [],
    }


def _composition(**overrides) -> dict:
    base = {
        "tldr": [],
        "sections": [],
        "research_discussed": [],
        "also_discussed": [],
        "takeaways": [],
        "follow_up": [],
    }
    base.update(overrides)
    return base


def _verdict(unit_id: str, verdict: str = "supported", reason: str | None = None) -> dict:
    return {"unit_id": unit_id, "verdict": verdict, "reason": reason}


def _review(verdicts: list[dict]) -> dict:
    return {"verdicts": verdicts}


def _codes(result) -> list[str]:
    return [e.code for e in result.errors]


@dataclass
class _FakeCallResult:
    response_payload: dict


class _FakeClient:
    """Returns each entry of `responses` in sequence. An entry that is a
    dict is JSON-encoded as the model's content; `None` simulates an
    unparseable response."""

    def __init__(self, responses: list[dict | None]):
        self._responses = list(responses)
        self.calls: list[list[dict]] = []

    def call(self, *, model, provider, messages, schema):
        self.calls.append(messages)
        response = self._responses.pop(0)
        content = "not valid json {{{" if response is None else json.dumps(response)
        return _FakeCallResult(response_payload={"choices": [{"message": {"content": content}}]})


# ---------------------------------------------------------------------------
# validate_review
# ---------------------------------------------------------------------------


class VerdictValidationTests(unittest.TestCase):
    def test_exactly_one_verdict_per_unit_is_accepted(self):
        units = {"tldr[0]": {"text": "x", "item_ids": ["a"]}}
        result = validate_review(_review([_verdict("tldr[0]")]), units)
        self.assertTrue(result.valid, result.errors)

    def test_missing_verdict_is_rejected(self):
        units = {"tldr[0]": {"text": "x", "item_ids": ["a"]}, "tldr[1]": {"text": "y", "item_ids": ["a"]}}
        result = validate_review(_review([_verdict("tldr[0]")]), units)
        self.assertFalse(result.valid)
        self.assertIn(REASON_MISSING_VERDICT, _codes(result))

    def test_duplicate_verdict_is_rejected(self):
        units = {"tldr[0]": {"text": "x", "item_ids": ["a"]}}
        review = _review([_verdict("tldr[0]"), _verdict("tldr[0]", verdict="scope_drift", reason="r")])
        result = validate_review(review, units)
        self.assertFalse(result.valid)
        self.assertIn(REASON_DUPLICATE_VERDICT, _codes(result))

    def test_unknown_unit_id_is_rejected(self):
        units = {"tldr[0]": {"text": "x", "item_ids": ["a"]}}
        review = _review([_verdict("tldr[0]"), _verdict("takeaways[0]")])
        result = validate_review(review, units)
        self.assertFalse(result.valid)
        self.assertIn(REASON_UNKNOWN_UNIT_ID, _codes(result))

    def test_invalid_verdict_string_is_rejected(self):
        units = {"tldr[0]": {"text": "x", "item_ids": ["a"]}}
        result = validate_review(_review([_verdict("tldr[0]", verdict="totally_fine")]), units)
        self.assertFalse(result.valid)
        self.assertIn(REASON_INVALID_VERDICT, _codes(result))

    def test_unknown_top_level_field_is_rejected(self):
        units = {"tldr[0]": {"text": "x", "item_ids": ["a"]}}
        review = _review([_verdict("tldr[0]")])
        review["extra"] = "surprise"
        result = validate_review(review, units)
        self.assertFalse(result.valid)
        self.assertIn(REASON_UNKNOWN_KEY, _codes(result))

    def test_unknown_verdict_entry_field_is_rejected(self):
        units = {"tldr[0]": {"text": "x", "item_ids": ["a"]}}
        entry = _verdict("tldr[0]")
        entry["extra"] = "surprise"
        result = validate_review(_review([entry]), units)
        self.assertFalse(result.valid)
        self.assertIn(REASON_UNKNOWN_KEY, _codes(result))


# ---------------------------------------------------------------------------
# enumerate_review_units
# ---------------------------------------------------------------------------


class EnumerateReviewUnitsTests(unittest.TestCase):
    def test_tldr_bullets_and_takeaways_are_enumerated(self):
        composition = _composition(
            tldr=[_unit("t", ["a"])],
            sections=[_section(bullets=[_bullet("b", ["a"])])],
            takeaways=[_unit("k", ["a"])],
        )
        units = enumerate_review_units(composition)
        self.assertEqual(set(units), {"tldr[0]", "sections[0].bullets[0]", "takeaways[0]"})

    def test_research_discussed_also_discussed_and_follow_up_are_excluded(self):
        composition = _composition(
            research_discussed=[_unit("r", ["a"])],
            also_discussed=[_unit("ad", ["a"])],
            follow_up=[_unit("f", ["a"])],
        )
        units = enumerate_review_units(composition)
        self.assertEqual(units, {})


# ---------------------------------------------------------------------------
# build_fix_excerpt
# ---------------------------------------------------------------------------


class FixExcerptWindowTests(unittest.TestCase):
    def test_excerpt_captures_up_to_window_words_on_each_side(self):
        transcript = "one two three four five SIX seven eight nine ten"
        quote_start = transcript.index("SIX")
        quote_end = quote_start + len("SIX")
        excerpt = build_fix_excerpt(transcript, quote_start=quote_start, quote_end=quote_end, window_words=2)
        self.assertEqual(excerpt["text"], "four five SIX seven eight")

    def test_window_exceeding_the_bounded_maximum_raises_before_reaching_a_prompt(self):
        transcript = "word " * 200
        with self.assertRaises(ValueError):
            build_fix_excerpt(transcript, quote_start=0, quote_end=4, window_words=FIX_EXCERPT_WINDOW_WORDS + 1)

    def test_negative_window_is_rejected(self):
        with self.assertRaises(ValueError):
            build_fix_excerpt("some text", quote_start=0, quote_end=4, window_words=-1)

    def test_out_of_bounds_quote_span_is_rejected(self):
        with self.assertRaises(ValueError):
            build_fix_excerpt("short", quote_start=0, quote_end=100, window_words=2)

    def test_default_window_is_the_bounded_maximum(self):
        transcript = "word " * 300
        quote_start = transcript.index("word", 150 * 5)
        quote_end = quote_start + 4
        excerpt = build_fix_excerpt(transcript, quote_start=quote_start, quote_end=quote_end)
        self.assertEqual(excerpt["window_words"], FIX_EXCERPT_WINDOW_WORDS)

    def test_window_wider_than_available_text_is_bounded_by_the_text_itself(self):
        transcript = "one two three"
        excerpt = build_fix_excerpt(transcript, quote_start=4, quote_end=7, window_words=10)
        self.assertEqual(excerpt["text"], "one two three")


# ---------------------------------------------------------------------------
# splice_fixed_units
# ---------------------------------------------------------------------------


class SpliceFixedUnitsTests(unittest.TestCase):
    def test_splices_bullet_text_and_item_ids(self):
        composition = _composition(sections=[_section(bullets=[_bullet("old", ["a"], primary_item_id="a")])])
        fixed = {"sections[0].bullets[0]": {"text": "new", "item_ids": ["b"], "primary_item_id": "b", "asserts": False}}
        result = splice_fixed_units(composition, fixed)
        bullet = result["sections"][0]["bullets"][0]
        self.assertEqual(bullet["text"], "new")
        self.assertEqual(bullet["item_ids"], ["b"])
        self.assertEqual(bullet["primary_item_id"], "b")
        self.assertFalse(bullet["asserts"])

    def test_splices_tldr_and_takeaway_text(self):
        composition = _composition(tldr=[_unit("old-t", ["a"])], takeaways=[_unit("old-k", ["a"])])
        fixed = {"tldr[0]": {"text": "new-t", "item_ids": ["b"]}, "takeaways[0]": {"text": "new-k", "item_ids": ["c"]}}
        result = splice_fixed_units(composition, fixed)
        self.assertEqual(result["tldr"][0]["text"], "new-t")
        self.assertEqual(result["takeaways"][0]["text"], "new-k")

    def test_untouched_units_are_preserved(self):
        composition = _composition(sections=[_section(bullets=[_bullet("keep-me", ["a"])])])
        result = splice_fixed_units(composition, {})
        self.assertEqual(result["sections"][0]["bullets"][0]["text"], "keep-me")

    def test_primary_item_id_is_nulled_when_no_longer_among_the_new_item_ids(self):
        composition = _composition(sections=[_section(bullets=[_bullet("old", ["a"], primary_item_id="a")])])
        fixed = {"sections[0].bullets[0]": {"text": "new", "item_ids": ["b"], "primary_item_id": None, "asserts": None}}
        result = splice_fixed_units(composition, fixed)
        self.assertIsNone(result["sections"][0]["bullets"][0]["primary_item_id"])

    def test_original_composition_is_not_mutated(self):
        composition = _composition(sections=[_section(bullets=[_bullet("old", ["a"])])])
        splice_fixed_units(composition, {"sections[0].bullets[0]": {"text": "new", "item_ids": ["b"], "primary_item_id": None, "asserts": None}})
        self.assertEqual(composition["sections"][0]["bullets"][0]["text"], "old")


# ---------------------------------------------------------------------------
# decide() -- design spec §8.2 / §8.1
# ---------------------------------------------------------------------------


class DecidePolicyTests(unittest.TestCase):
    def test_all_supported_publishes_with_nothing_removed(self):
        verdicts = {"tldr[0]": "supported", "sections[0].bullets[0]": "supported"}
        units_by_id = {"tldr[0]": {"item_ids": []}, "sections[0].bullets[0]": {"item_ids": []}}
        decision = decide(verdicts, units_by_id=units_by_id, items_by_id={})
        self.assertEqual(decision.action, "publish")
        self.assertEqual(decision.removed_unit_ids, [])
        self.assertIsNone(decision.footer_line)

    def test_up_to_three_failing_plain_bullets_are_removed_not_held(self):
        verdicts = {f"sections[0].bullets[{i}]": "unsupported_claim" for i in range(3)}
        units_by_id = {uid: {"item_ids": []} for uid in verdicts}
        decision = decide(verdicts, units_by_id=units_by_id, items_by_id={})
        self.assertEqual(decision.action, "publish")
        self.assertEqual(set(decision.removed_unit_ids), set(verdicts))

    def test_more_than_three_failing_plain_bullets_holds(self):
        verdicts = {f"sections[0].bullets[{i}]": "unsupported_claim" for i in range(4)}
        units_by_id = {uid: {"item_ids": []} for uid in verdicts}
        decision = decide(verdicts, units_by_id=units_by_id, items_by_id={})
        self.assertEqual(decision.action, "hold")

    def test_a_failing_tldr_sentence_holds(self):
        decision = decide({"tldr[0]": "unsupported_claim"}, units_by_id={"tldr[0]": {"item_ids": []}}, items_by_id={})
        self.assertEqual(decision.action, "hold")

    def test_a_failing_takeaway_holds(self):
        decision = decide({"takeaways[0]": "scope_drift"}, units_by_id={"takeaways[0]": {"item_ids": []}}, items_by_id={})
        self.assertEqual(decision.action, "hold")

    def test_a_failing_protocol_bullet_holds(self):
        verdicts = {"sections[0].bullets[0]": "unsupported_claim"}
        units_by_id = {"sections[0].bullets[0]": {"item_ids": ["p1"]}}
        items_by_id = {"p1": {"kind": "protocol"}}
        decision = decide(verdicts, units_by_id=units_by_id, items_by_id=items_by_id)
        self.assertEqual(decision.action, "hold")

    def test_footer_line_appears_when_a_high_value_item_stayed_rejected(self):
        decision = decide({}, units_by_id={}, items_by_id={}, rejected_high_value_item_ids=["a", "b"])
        self.assertIsNotNone(decision.footer_line)
        self.assertIn("2", decision.footer_line)

    def test_no_footer_line_when_nothing_was_rejected(self):
        decision = decide({}, units_by_id={}, items_by_id={})
        self.assertIsNone(decision.footer_line)


# ---------------------------------------------------------------------------
# review_units (no retry)
# ---------------------------------------------------------------------------


class ReviewUnitsOrchestrationTests(unittest.TestCase):
    def test_a_valid_review_is_not_held(self):
        units = {"tldr[0]": {"text": "x", "item_ids": ["a"]}}
        client = _FakeClient([_review([_verdict("tldr[0]")])])
        result = review_units(units, {}, client=client, model="m", provider="p", episode_context={}, prompt_text="prompt")
        self.assertFalse(result.held)
        self.assertEqual(result.verdicts_by_unit_id["tldr[0]"]["verdict"], "supported")

    def test_a_missing_verdict_holds_the_run(self):
        units = {"tldr[0]": {"text": "x", "item_ids": ["a"]}, "tldr[1]": {"text": "y", "item_ids": ["a"]}}
        client = _FakeClient([_review([_verdict("tldr[0]")])])
        result = review_units(units, {}, client=client, model="m", provider="p", episode_context={}, prompt_text="prompt")
        self.assertTrue(result.held)
        self.assertEqual(result.failure_reason, "validation_failed")

    def test_a_duplicate_verdict_holds_the_run(self):
        units = {"tldr[0]": {"text": "x", "item_ids": ["a"]}}
        review = _review([_verdict("tldr[0]"), _verdict("tldr[0]", verdict="scope_drift", reason="r")])
        client = _FakeClient([review])
        result = review_units(units, {}, client=client, model="m", provider="p", episode_context={}, prompt_text="prompt")
        self.assertTrue(result.held)

    def test_invalid_json_holds_the_run(self):
        units = {"tldr[0]": {"text": "x", "item_ids": ["a"]}}
        client = _FakeClient([None])
        result = review_units(units, {}, client=client, model="m", provider="p", episode_context={}, prompt_text="prompt")
        self.assertTrue(result.held)
        self.assertEqual(result.failure_reason, "invalid_json_response")

    def test_review_never_retries(self):
        units = {"tldr[0]": {"text": "x", "item_ids": ["a"]}}
        client = _FakeClient([None])
        review_units(units, {}, client=client, model="m", provider="p", episode_context={}, prompt_text="prompt")
        self.assertEqual(len(client.calls), 1)


# ---------------------------------------------------------------------------
# route_condition_dropped
# ---------------------------------------------------------------------------


class RouteConditionDroppedTests(unittest.TestCase):
    def test_a_changed_condition_is_reported_as_changed(self):
        original = _extraction_item(conditions="none_stated", quote="take it", quote_occurrence=1)
        section = {"section_id": "sec01", "text": "take it if you are sleep deprived"}
        new_item = _extraction_item(conditions="acute sleep deprivation", quote="take it", quote_occurrence=1)
        client = _FakeClient([_extraction_result(items=[new_item])])
        result = route_condition_dropped(
            original_item=original, section=section, verdict_reason="dropped a condition",
            client=client, model="m", provider="p", episode_context={},
        )
        self.assertTrue(result.changed)
        self.assertEqual(result.new_item["conditions"], "acute sleep deprivation")

    def test_an_unchanged_condition_and_quote_is_reported_as_unchanged(self):
        original = _extraction_item(conditions="none_stated", quote="take it", quote_occurrence=1)
        section = {"section_id": "sec01", "text": "take it"}
        client = _FakeClient([_extraction_result(items=[_extraction_item(conditions="none_stated", quote="take it", quote_occurrence=1)])])
        result = route_condition_dropped(
            original_item=original, section=section, verdict_reason="reason",
            client=client, model="m", provider="p", episode_context={},
        )
        self.assertFalse(result.changed)
        self.assertIsNone(result.new_item)

    def test_invalid_json_is_reported_as_unchanged(self):
        original = _extraction_item()
        section = {"section_id": "sec01", "text": "text"}
        client = _FakeClient([None])
        result = route_condition_dropped(
            original_item=original, section=section, verdict_reason="reason",
            client=client, model="m", provider="p", episode_context={},
        )
        self.assertFalse(result.changed)

    def test_local_id_not_found_in_reextraction_is_reported_as_unchanged(self):
        original = _extraction_item(local_id="i-01")
        section = {"section_id": "sec01", "text": "text"}
        client = _FakeClient([_extraction_result(items=[_extraction_item(local_id="i-99")])])
        result = route_condition_dropped(
            original_item=original, section=section, verdict_reason="reason",
            client=client, model="m", provider="p", episode_context={},
        )
        self.assertFalse(result.changed)

    def test_never_makes_more_than_one_call(self):
        original = _extraction_item()
        section = {"section_id": "sec01", "text": "text"}
        client = _FakeClient([_extraction_result(items=[_extraction_item(conditions="x")])])
        route_condition_dropped(
            original_item=original, section=section, verdict_reason="reason",
            client=client, model="m", provider="p", episode_context={},
        )
        self.assertEqual(len(client.calls), 1)


# ---------------------------------------------------------------------------
# apply_fix_round
# ---------------------------------------------------------------------------


class ApplyFixRoundTests(unittest.TestCase):
    def test_a_valid_fixed_unit_is_accepted(self):
        items_by_id = {"a": _item("a")}
        flagged = [{"unit_id": "tldr[0]", "text": "old", "item_ids": ["a"], "reason": "unsupported_claim"}]
        client = _FakeClient([{"fixed_units": [{"unit_id": "tldr[0]", "text": "new grounded text", "item_ids": ["a"], "primary_item_id": None, "asserts": None}]}])
        result = apply_fix_round(flagged, items_by_id=items_by_id, client=client, model="m", provider="p", episode_context={}, prompt_text="prompt")
        self.assertIn("tldr[0]", result.fixed_units_by_id)
        self.assertEqual(result.fixed_units_by_id["tldr[0]"]["text"], "new grounded text")
        self.assertEqual(result.invalid_unit_ids, [])

    def test_a_fixed_unit_with_a_free_number_is_invalid(self):
        items_by_id = {"a": _item("a")}
        flagged = [{"unit_id": "tldr[0]", "text": "old", "item_ids": ["a"], "reason": "x"}]
        client = _FakeClient([{"fixed_units": [{"unit_id": "tldr[0]", "text": "take 15 grams", "item_ids": ["a"], "primary_item_id": None, "asserts": None}]}])
        result = apply_fix_round(flagged, items_by_id=items_by_id, client=client, model="m", provider="p", episode_context={}, prompt_text="prompt")
        self.assertIn("tldr[0]", result.invalid_unit_ids)
        self.assertNotIn("tldr[0]", result.fixed_units_by_id)

    def test_a_fix_output_citing_an_unknown_item_id_is_invalid(self):
        items_by_id = {"a": _item("a")}
        flagged = [{"unit_id": "tldr[0]", "text": "old", "item_ids": ["a"], "reason": "x"}]
        client = _FakeClient([{"fixed_units": [{"unit_id": "tldr[0]", "text": "grounded", "item_ids": ["does-not-exist"], "primary_item_id": None, "asserts": None}]}])
        result = apply_fix_round(flagged, items_by_id=items_by_id, client=client, model="m", provider="p", episode_context={}, prompt_text="prompt")
        self.assertIn("tldr[0]", result.invalid_unit_ids)

    def test_a_missing_fixed_unit_in_the_response_is_invalid(self):
        items_by_id = {"a": _item("a")}
        flagged = [
            {"unit_id": "tldr[0]", "text": "old", "item_ids": ["a"], "reason": "x"},
            {"unit_id": "takeaways[0]", "text": "old2", "item_ids": ["a"], "reason": "y"},
        ]
        client = _FakeClient([{"fixed_units": [{"unit_id": "tldr[0]", "text": "grounded", "item_ids": ["a"], "primary_item_id": None, "asserts": None}]}])
        result = apply_fix_round(flagged, items_by_id=items_by_id, client=client, model="m", provider="p", episode_context={}, prompt_text="prompt")
        self.assertIn("takeaways[0]", result.invalid_unit_ids)
        self.assertIn("tldr[0]", result.fixed_units_by_id)

    def test_invalid_json_marks_every_flagged_unit_invalid(self):
        items_by_id = {"a": _item("a")}
        flagged = [{"unit_id": "tldr[0]", "text": "old", "item_ids": ["a"], "reason": "x"}]
        client = _FakeClient([None])
        result = apply_fix_round(flagged, items_by_id=items_by_id, client=client, model="m", provider="p", episode_context={}, prompt_text="prompt")
        self.assertEqual(result.invalid_unit_ids, ["tldr[0]"])
        self.assertEqual(result.failure_reason, "invalid_json_response")

    def test_only_flagged_units_are_sent(self):
        items_by_id = {"a": _item("a")}
        flagged = [{"unit_id": "tldr[0]", "text": "old", "item_ids": ["a"], "reason": "x"}]
        client = _FakeClient([{"fixed_units": [{"unit_id": "tldr[0]", "text": "grounded", "item_ids": ["a"], "primary_item_id": None, "asserts": None}]}])
        apply_fix_round(flagged, items_by_id=items_by_id, client=client, model="m", provider="p", episode_context={}, prompt_text="prompt")
        payload = json.loads(client.calls[0][1]["content"])
        self.assertEqual([u["unit_id"] for u in payload["flagged_units"]], ["tldr[0]"])


# ---------------------------------------------------------------------------
# run_review_and_fix (full orchestration)
# ---------------------------------------------------------------------------


class RunReviewAndFixOrchestrationTests(unittest.TestCase):
    def test_all_supported_on_first_review_needs_no_fix_rounds(self):
        composition = _composition(tldr=[_unit("t", ["a"])])
        items_by_id = {"a": _item("a")}
        review_client = _FakeClient([_review([_verdict("tldr[0]")])])
        fix_client = _FakeClient([])
        result = run_review_and_fix(
            composition, items_by_id,
            review_client=review_client, review_model="rm", review_provider="rp",
            fix_client=fix_client, fix_model="fm", fix_provider="fp",
            episode_context={}, review_prompt_text="r", fix_prompt_text="f",
        )
        self.assertFalse(result.held)
        self.assertEqual(result.rounds_used, 0)
        self.assertEqual(result.decision.action, "publish")

    def test_initial_review_invalid_holds_the_whole_run(self):
        composition = _composition(tldr=[_unit("t", ["a"])])
        items_by_id = {"a": _item("a")}
        review_client = _FakeClient([None])
        fix_client = _FakeClient([])
        result = run_review_and_fix(
            composition, items_by_id,
            review_client=review_client, review_model="rm", review_provider="rp",
            fix_client=fix_client, fix_model="fm", fix_provider="fp",
            episode_context={}, review_prompt_text="r", fix_prompt_text="f",
        )
        self.assertTrue(result.held)
        self.assertEqual(result.rounds_used, 0)
        self.assertIsNone(result.decision)

    def test_a_failing_bullet_fixed_in_round_one_becomes_supported(self):
        composition = _composition(sections=[_section(bullets=[_bullet("old", ["a"])])])
        items_by_id = {"a": _item("a")}
        review_client = _FakeClient([
            _review([_verdict("sections[0].bullets[0]", verdict="unsupported_claim", reason="drift")]),
            _review([_verdict("sections[0].bullets[0]")]),
        ])
        fix_client = _FakeClient([
            {"fixed_units": [{"unit_id": "sections[0].bullets[0]", "text": "grounded now", "item_ids": ["a"], "primary_item_id": "a", "asserts": True}]},
        ])
        result = run_review_and_fix(
            composition, items_by_id,
            review_client=review_client, review_model="rm", review_provider="rp",
            fix_client=fix_client, fix_model="fm", fix_provider="fp",
            episode_context={}, review_prompt_text="r", fix_prompt_text="f",
        )
        self.assertFalse(result.held)
        self.assertEqual(result.rounds_used, 1)
        self.assertEqual(result.final_verdicts["sections[0].bullets[0]"], "supported")
        self.assertEqual(result.composition["sections"][0]["bullets"][0]["text"], "grounded now")

    def test_a_bullet_still_failing_after_two_rounds_stops_at_two_rounds(self):
        composition = _composition(sections=[_section(bullets=[_bullet("old", ["a"])])])
        items_by_id = {"a": _item("a")}
        review_client = _FakeClient([
            _review([_verdict("sections[0].bullets[0]", verdict="unsupported_claim", reason="drift")]),
            _review([_verdict("sections[0].bullets[0]", verdict="unsupported_claim", reason="still drift")]),
            _review([_verdict("sections[0].bullets[0]", verdict="unsupported_claim", reason="still drift 2")]),
        ])
        fix_client = _FakeClient([
            {"fixed_units": [{"unit_id": "sections[0].bullets[0]", "text": "fix attempt alpha", "item_ids": ["a"], "primary_item_id": "a", "asserts": True}]},
            {"fixed_units": [{"unit_id": "sections[0].bullets[0]", "text": "fix attempt beta", "item_ids": ["a"], "primary_item_id": "a", "asserts": True}]},
        ])
        result = run_review_and_fix(
            composition, items_by_id,
            review_client=review_client, review_model="rm", review_provider="rp",
            fix_client=fix_client, fix_model="fm", fix_provider="fp",
            episode_context={}, review_prompt_text="r", fix_prompt_text="f",
        )
        self.assertEqual(result.rounds_used, 2)
        self.assertEqual(result.final_verdicts["sections[0].bullets[0]"], "unsupported_claim")
        self.assertEqual(result.decision.action, "publish")
        self.assertIn("sections[0].bullets[0]", result.decision.removed_unit_ids)

    def test_condition_dropped_routes_to_reextraction_without_consuming_a_fix_round(self):
        composition = _composition(tldr=[_unit("old", ["a"])])
        items_by_id = {"a": _item("a", local_id="i-01", section_id="sec01", conditions="none_stated", quote="take it", quote_occurrence=1)}
        section = {"section_id": "sec01", "text": "take it if you are sleep deprived"}

        review_client = _FakeClient([
            _review([_verdict("tldr[0]", verdict="condition_dropped", reason="missing qualifier")]),
            _review([_verdict("tldr[0]")]),
        ])
        extraction_client = _FakeClient([
            _extraction_result(section_id="sec01", items=[_extraction_item(local_id="i-01", conditions="acute sleep deprivation", quote="take it", quote_occurrence=1)]),
        ])
        fix_client = _FakeClient([
            {"fixed_units": [{"unit_id": "tldr[0]", "text": "take it, if sleep deprived", "item_ids": ["a"], "primary_item_id": None, "asserts": None}]},
        ])
        result = run_review_and_fix(
            composition, items_by_id,
            review_client=review_client, review_model="rm", review_provider="rp",
            fix_client=fix_client, fix_model="fm", fix_provider="fp",
            extraction_client=extraction_client, extraction_model="em", extraction_provider="ep",
            sections_by_id={"sec01": section},
            episode_context={}, review_prompt_text="r", fix_prompt_text="f", extract_prompt_text="e",
        )
        self.assertEqual(result.rounds_used, 0)
        self.assertFalse(result.held)
        self.assertEqual(result.final_verdicts["tldr[0]"], "supported")
        self.assertEqual(len(extraction_client.calls), 1)

    def test_condition_dropped_round_count_equals_an_equivalent_run_with_no_condition_dropped(self):
        # condition_dropped -> re-extraction returns no new info -> falls
        # through to one ordinary fix round that succeeds.
        composition_cd = _composition(tldr=[_unit("old", ["a"])])
        items_by_id_cd = {"a": _item("a", local_id="i-01", section_id="sec01", conditions="none_stated", quote="take it", quote_occurrence=1)}
        section = {"section_id": "sec01", "text": "take it"}
        review_client_cd = _FakeClient([
            _review([_verdict("tldr[0]", verdict="condition_dropped", reason="missing qualifier")]),
            _review([_verdict("tldr[0]")]),
        ])
        extraction_client = _FakeClient([
            _extraction_result(section_id="sec01", items=[_extraction_item(local_id="i-01", conditions="none_stated", quote="take it", quote_occurrence=1)]),
        ])
        fix_client_cd = _FakeClient([
            {"fixed_units": [{"unit_id": "tldr[0]", "text": "fixed via ordinary round", "item_ids": ["a"], "primary_item_id": None, "asserts": None}]},
        ])
        cd_result = run_review_and_fix(
            composition_cd, items_by_id_cd,
            review_client=review_client_cd, review_model="rm", review_provider="rp",
            fix_client=fix_client_cd, fix_model="fm", fix_provider="fp",
            extraction_client=extraction_client, extraction_model="em", extraction_provider="ep",
            sections_by_id={"sec01": section},
            episode_context={}, review_prompt_text="r", fix_prompt_text="f", extract_prompt_text="e",
        )

        # Equivalent run: an ordinary (non-condition_dropped) failing unit
        # fixed in exactly one round.
        composition_ordinary = _composition(tldr=[_unit("old", ["a"])])
        items_by_id_ordinary = {"a": _item("a")}
        review_client_ordinary = _FakeClient([
            _review([_verdict("tldr[0]", verdict="unsupported_claim", reason="x")]),
            _review([_verdict("tldr[0]")]),
        ])
        fix_client_ordinary = _FakeClient([
            {"fixed_units": [{"unit_id": "tldr[0]", "text": "fixed via ordinary round", "item_ids": ["a"], "primary_item_id": None, "asserts": None}]},
        ])
        ordinary_result = run_review_and_fix(
            composition_ordinary, items_by_id_ordinary,
            review_client=review_client_ordinary, review_model="rm", review_provider="rp",
            fix_client=fix_client_ordinary, fix_model="fm", fix_provider="fp",
            episode_context={}, review_prompt_text="r", fix_prompt_text="f",
        )

        self.assertEqual(cd_result.rounds_used, ordinary_result.rounds_used)
        self.assertEqual(cd_result.rounds_used, 1)

    def test_only_non_supported_units_are_sent_to_a_fix(self):
        composition = _composition(tldr=[_unit("t", ["a"])], takeaways=[_unit("k", ["a"])])
        items_by_id = {"a": _item("a")}
        review_client = _FakeClient([
            _review([_verdict("tldr[0]"), _verdict("takeaways[0]", verdict="unsupported_claim", reason="x")]),
            _review([_verdict("takeaways[0]")]),
        ])
        fix_client = _FakeClient([
            {"fixed_units": [{"unit_id": "takeaways[0]", "text": "fixed", "item_ids": ["a"], "primary_item_id": None, "asserts": None}]},
        ])
        run_review_and_fix(
            composition, items_by_id,
            review_client=review_client, review_model="rm", review_provider="rp",
            fix_client=fix_client, fix_model="fm", fix_provider="fp",
            episode_context={}, review_prompt_text="r", fix_prompt_text="f",
        )
        fix_payload = json.loads(fix_client.calls[0][1]["content"])
        self.assertEqual([u["unit_id"] for u in fix_payload["flagged_units"]], ["takeaways[0]"])

    def test_only_changed_units_are_re_reviewed(self):
        composition = _composition(tldr=[_unit("t", ["a"])], takeaways=[_unit("k", ["a"])])
        items_by_id = {"a": _item("a")}
        review_client = _FakeClient([
            _review([_verdict("tldr[0]"), _verdict("takeaways[0]", verdict="unsupported_claim", reason="x")]),
            _review([_verdict("takeaways[0]")]),
        ])
        fix_client = _FakeClient([
            {"fixed_units": [{"unit_id": "takeaways[0]", "text": "fixed", "item_ids": ["a"], "primary_item_id": None, "asserts": None}]},
        ])
        run_review_and_fix(
            composition, items_by_id,
            review_client=review_client, review_model="rm", review_provider="rp",
            fix_client=fix_client, fix_model="fm", fix_provider="fp",
            episode_context={}, review_prompt_text="r", fix_prompt_text="f",
        )
        re_review_payload = json.loads(review_client.calls[1][1]["content"])
        self.assertEqual([u["unit_id"] for u in re_review_payload["units"]], ["takeaways[0]"])


if __name__ == "__main__":
    unittest.main()
