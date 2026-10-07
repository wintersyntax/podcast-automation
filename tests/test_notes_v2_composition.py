"""Tests for TASK-106 Phase A Task 7 (composition and its validation).

Covers `podcast_engine.knowledge.notes_v2.composition` (pure
`note-composition-v1` schema + `validate_composition`, design spec §5.4/§5.5)
and `scripts.knowledge_eval.composition` (the model-calling orchestration:
`build_messages` and the one-bounded-retry-then-hold `compose_note` policy),
mirroring the combined-coverage pattern `test_notes_v2_sources.py` already
uses for `sources.py` + `scripts/knowledge_eval/lookup.py`.
"""

from __future__ import annotations

import json
import unittest
from dataclasses import dataclass

from podcast_engine.knowledge.notes_v2.composition import (
    REASON_ALSO_DISCUSSED_SCOPE,
    REASON_FREE_NUMBER,
    REASON_HIGH_VALUE_NOT_CITED,
    REASON_INVALID_PLACEHOLDER,
    REASON_MISSING_KEY,
    REASON_OUT_OF_SCOPE_CITED,
    REASON_PROTOCOL_NOT_LISTED,
    REASON_UNCITED_UNIT,
    REASON_UNKNOWN_ITEM_ID,
    REASON_UNKNOWN_KEY,
    validate_composition,
)
from scripts.knowledge_eval.composition import (
    CompositionRunResult,
    build_messages,
    compose_note,
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
    }
    base.update(overrides)
    return base


def _unit(text: str = "A grounded sentence.", item_ids: list[str] | None = None) -> dict:
    return {"text": text, "item_ids": item_ids if item_ids is not None else []}


def _bullet(
    text: str = "A grounded bullet.",
    item_ids: list[str] | None = None,
    primary_item_id: str | None = None,
    asserts: bool = True,
) -> dict:
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


def _codes(result) -> list[str]:
    return [e.code for e in result.errors]


# ---------------------------------------------------------------------------
# Free-number detection
# ---------------------------------------------------------------------------


class FreeNumberValidationTests(unittest.TestCase):
    def test_digit_outside_placeholder_is_rejected(self):
        items = [_item()]
        composition = _composition(takeaways=[_unit("Take 15 grams daily.", ["sec01-i01"])])
        result = validate_composition(composition, items)
        self.assertFalse(result.valid)
        self.assertIn(REASON_FREE_NUMBER, _codes(result))

    def test_number_word_outside_placeholder_is_rejected(self):
        items = [_item()]
        composition = _composition(takeaways=[_unit("Take fifteen grams daily.", ["sec01-i01"])])
        result = validate_composition(composition, items)
        self.assertFalse(result.valid)
        self.assertIn(REASON_FREE_NUMBER, _codes(result))

    def test_number_inside_placeholder_is_allowed(self):
        items = [_item(quantities=[{"value": 15, "value_high": None, "unit_as_spoken": "grams"}])]
        composition = _composition(takeaways=[_unit("Take {q:sec01-i01:0} daily.", ["sec01-i01"])])
        result = validate_composition(composition, items)
        self.assertNotIn(REASON_FREE_NUMBER, _codes(result))

    def test_structural_ordinal_word_is_allowed(self):
        items = [_item()]
        composition = _composition(takeaways=[_unit("This is the first point made.", ["sec01-i01"])])
        result = validate_composition(composition, items)
        self.assertNotIn(REASON_FREE_NUMBER, _codes(result))

    def test_vague_quantifier_is_not_flagged(self):
        items = [_item()]
        composition = _composition(takeaways=[_unit("Several factors were discussed.", ["sec01-i01"])])
        result = validate_composition(composition, items)
        self.assertNotIn(REASON_FREE_NUMBER, _codes(result))

    def test_idiomatic_one_is_allowed(self):
        items = [_item()]
        composition = _composition(
            takeaways=[_unit("This was one of the biggest myths addressed.", ["sec01-i01"])]
        )
        result = validate_composition(composition, items)
        self.assertNotIn(REASON_FREE_NUMBER, _codes(result))

    def test_idiomatic_two_is_allowed(self):
        items = [_item()]
        composition = _composition(
            takeaways=[_unit("They flew economy for the two of them.", ["sec01-i01"])]
        )
        result = validate_composition(composition, items)
        self.assertNotIn(REASON_FREE_NUMBER, _codes(result))

    def test_digit_in_alphanumeric_term_is_not_flagged(self):
        items = [_item()]
        composition = _composition(
            takeaways=[_unit("VO2 max looked great this year.", ["sec01-i01"])]
        )
        result = validate_composition(composition, items)
        self.assertNotIn(REASON_FREE_NUMBER, _codes(result))

    def test_digit_in_hyphenated_alphanumeric_term_is_not_flagged(self):
        items = [_item()]
        composition = _composition(
            takeaways=[_unit("A GLP-1 medication reduced food preoccupation.", ["sec01-i01"])]
        )
        result = validate_composition(composition, items)
        self.assertNotIn(REASON_FREE_NUMBER, _codes(result))

    def test_non_allowlisted_number_word_in_hyphenated_compound_is_still_rejected(self):
        items = [_item()]
        composition = _composition(
            takeaways=[_unit("The zone-three protocol was recommended.", ["sec01-i01"])]
        )
        result = validate_composition(composition, items)
        self.assertFalse(result.valid)
        self.assertIn(REASON_FREE_NUMBER, _codes(result))

    def test_allowlisted_number_word_in_hyphenated_compound_is_allowed(self):
        items = [_item()]
        composition = _composition(
            takeaways=[_unit("Zone-two cardio was recommended.", ["sec01-i01"])]
        )
        result = validate_composition(composition, items)
        self.assertNotIn(REASON_FREE_NUMBER, _codes(result))


# ---------------------------------------------------------------------------
# Citation requirements
# ---------------------------------------------------------------------------


class CitationTests(unittest.TestCase):
    def test_bullet_with_no_item_ids_is_rejected(self):
        items = [_item()]
        composition = _composition(sections=[_section(bullets=[_bullet(item_ids=[])])])
        result = validate_composition(composition, items)
        self.assertFalse(result.valid)
        self.assertIn(REASON_UNCITED_UNIT, _codes(result))

    def test_bullet_citing_unknown_item_id_is_rejected(self):
        items = [_item()]
        composition = _composition(sections=[_section(bullets=[_bullet(item_ids=["does-not-exist"])])])
        result = validate_composition(composition, items)
        self.assertFalse(result.valid)
        self.assertIn(REASON_UNKNOWN_ITEM_ID, _codes(result))

    def test_bullet_citing_a_real_item_is_accepted(self):
        items = [_item()]
        composition = _composition(sections=[_section(bullets=[_bullet(item_ids=["sec01-i01"])])])
        result = validate_composition(composition, items)
        self.assertTrue(result.valid, result.errors)

    def test_tldr_entry_with_no_citation_is_rejected(self):
        items = [_item()]
        composition = _composition(tldr=[_unit(item_ids=[])])
        result = validate_composition(composition, items)
        self.assertIn(REASON_UNCITED_UNIT, _codes(result))

    def test_takeaway_with_no_citation_is_rejected(self):
        items = [_item()]
        composition = _composition(takeaways=[_unit(item_ids=[])])
        result = validate_composition(composition, items)
        self.assertIn(REASON_UNCITED_UNIT, _codes(result))

    def test_research_discussed_with_no_citation_is_rejected(self):
        items = [_item()]
        composition = _composition(research_discussed=[_unit(item_ids=[])])
        result = validate_composition(composition, items)
        self.assertIn(REASON_UNCITED_UNIT, _codes(result))

    def test_also_discussed_with_no_citation_is_rejected(self):
        items = [_item(kind="side_topic")]
        composition = _composition(also_discussed=[_unit(item_ids=[])])
        result = validate_composition(composition, items)
        self.assertIn(REASON_UNCITED_UNIT, _codes(result))

    def test_follow_up_with_no_citation_is_rejected(self):
        items = [_item()]
        composition = _composition(follow_up=[_unit(item_ids=[])])
        result = validate_composition(composition, items)
        self.assertIn(REASON_UNCITED_UNIT, _codes(result))


# ---------------------------------------------------------------------------
# Placeholders must reference real quantities
# ---------------------------------------------------------------------------


class PlaceholderValidationTests(unittest.TestCase):
    def test_placeholder_referencing_a_real_quantity_is_accepted(self):
        items = [_item(quantities=[{"value": 2, "value_high": None, "unit_as_spoken": "grams"}])]
        composition = _composition(takeaways=[_unit("Dose is {q:sec01-i01:0}.", ["sec01-i01"])])
        result = validate_composition(composition, items)
        self.assertTrue(result.valid, result.errors)

    def test_placeholder_referencing_an_item_not_cited_by_this_unit_is_rejected(self):
        items = [
            _item("sec01-i01", quantities=[{"value": 2, "value_high": None, "unit_as_spoken": "grams"}]),
            _item("sec01-i02"),
        ]
        composition = _composition(takeaways=[_unit("Dose is {q:sec01-i01:0}.", ["sec01-i02"])])
        result = validate_composition(composition, items)
        self.assertFalse(result.valid)
        self.assertIn(REASON_INVALID_PLACEHOLDER, _codes(result))

    def test_placeholder_referencing_an_unknown_item_is_rejected(self):
        items = [_item()]
        composition = _composition(takeaways=[_unit("Dose is {q:does-not-exist:0}.", ["sec01-i01"])])
        result = validate_composition(composition, items)
        self.assertFalse(result.valid)
        self.assertIn(REASON_INVALID_PLACEHOLDER, _codes(result))

    def test_placeholder_index_out_of_range_is_rejected(self):
        items = [_item(quantities=[{"value": 2, "value_high": None, "unit_as_spoken": "grams"}])]
        composition = _composition(takeaways=[_unit("Dose is {q:sec01-i01:3}.", ["sec01-i01"])])
        result = validate_composition(composition, items)
        self.assertFalse(result.valid)
        self.assertIn(REASON_INVALID_PLACEHOLDER, _codes(result))


# ---------------------------------------------------------------------------
# High-value and protocol coverage
# ---------------------------------------------------------------------------


class HighValueAndProtocolCoverageTests(unittest.TestCase):
    def test_uncited_high_value_item_is_rejected(self):
        items = [_item(value="high")]
        composition = _composition()
        result = validate_composition(composition, items)
        self.assertFalse(result.valid)
        self.assertIn(REASON_HIGH_VALUE_NOT_CITED, _codes(result))

    def test_cited_high_value_item_is_accepted(self):
        items = [_item(value="high")]
        composition = _composition(sections=[_section(bullets=[_bullet(item_ids=["sec01-i01"])])])
        result = validate_composition(composition, items)
        self.assertTrue(result.valid, result.errors)

    def test_high_value_item_listed_only_in_protocol_item_ids_counts_as_cited(self):
        items = [_item(kind="protocol", value="high")]
        composition = _composition(sections=[_section(protocol_item_ids=["sec01-i01"])])
        result = validate_composition(composition, items)
        self.assertNotIn(REASON_HIGH_VALUE_NOT_CITED, _codes(result))

    def test_uncited_protocol_item_is_rejected(self):
        items = [_item(kind="protocol", value="normal")]
        composition = _composition(sections=[_section(bullets=[_bullet(item_ids=["sec01-i01"])])])
        result = validate_composition(composition, items)
        self.assertFalse(result.valid)
        self.assertIn(REASON_PROTOCOL_NOT_LISTED, _codes(result))

    def test_protocol_item_listed_in_protocol_item_ids_is_accepted(self):
        items = [_item(kind="protocol", value="normal")]
        composition = _composition(sections=[_section(protocol_item_ids=["sec01-i01"])])
        result = validate_composition(composition, items)
        self.assertNotIn(REASON_PROTOCOL_NOT_LISTED, _codes(result))

    def test_normal_and_low_value_items_are_not_required_to_be_cited(self):
        items = [_item(value="normal"), _item("sec01-i02", value="low")]
        composition = _composition()
        result = validate_composition(composition, items)
        self.assertTrue(result.valid, result.errors)


# ---------------------------------------------------------------------------
# also_discussed scope
# ---------------------------------------------------------------------------


class AlsoDiscussedScopeTests(unittest.TestCase):
    def test_also_discussed_citing_a_side_topic_item_is_accepted(self):
        items = [_item(kind="side_topic", value="normal")]
        composition = _composition(also_discussed=[_unit(item_ids=["sec01-i01"])])
        result = validate_composition(composition, items)
        self.assertNotIn(REASON_ALSO_DISCUSSED_SCOPE, _codes(result))

    def test_also_discussed_citing_a_low_value_item_is_accepted(self):
        items = [_item(kind="claim", value="low")]
        composition = _composition(also_discussed=[_unit(item_ids=["sec01-i01"])])
        result = validate_composition(composition, items)
        self.assertNotIn(REASON_ALSO_DISCUSSED_SCOPE, _codes(result))

    def test_also_discussed_citing_a_high_value_non_side_topic_item_is_rejected(self):
        items = [_item(kind="claim", value="high")]
        composition = _composition(also_discussed=[_unit(item_ids=["sec01-i01"])])
        result = validate_composition(composition, items)
        self.assertFalse(result.valid)
        self.assertIn(REASON_ALSO_DISCUSSED_SCOPE, _codes(result))


# ---------------------------------------------------------------------------
# Out-of-scope items are never cited
# ---------------------------------------------------------------------------


class OutOfScopeNeverCitedTests(unittest.TestCase):
    def test_bullet_citing_an_out_of_scope_item_is_rejected(self):
        items = [_item(scope="out_of_scope")]
        composition = _composition(sections=[_section(bullets=[_bullet(item_ids=["sec01-i01"])])])
        result = validate_composition(composition, items)
        self.assertFalse(result.valid)
        self.assertIn(REASON_OUT_OF_SCOPE_CITED, _codes(result))

    def test_protocol_item_ids_citing_an_out_of_scope_item_is_rejected(self):
        items = [_item(kind="protocol", scope="out_of_scope")]
        composition = _composition(sections=[_section(protocol_item_ids=["sec01-i01"])])
        result = validate_composition(composition, items)
        self.assertFalse(result.valid)
        self.assertIn(REASON_OUT_OF_SCOPE_CITED, _codes(result))

    def test_also_discussed_citing_an_out_of_scope_item_is_rejected(self):
        items = [_item(kind="side_topic", scope="out_of_scope")]
        composition = _composition(also_discussed=[_unit(item_ids=["sec01-i01"])])
        result = validate_composition(composition, items)
        self.assertFalse(result.valid)
        self.assertIn(REASON_OUT_OF_SCOPE_CITED, _codes(result))

    def test_out_of_scope_item_is_never_required_to_be_cited_even_if_high_value(self):
        items = [_item(value="high", scope="out_of_scope")]
        composition = _composition()
        result = validate_composition(composition, items)
        self.assertNotIn(REASON_HIGH_VALUE_NOT_CITED, _codes(result))


# ---------------------------------------------------------------------------
# Unknown/missing fields
# ---------------------------------------------------------------------------


class UnknownFieldTests(unittest.TestCase):
    def test_unknown_top_level_field_is_rejected(self):
        items = [_item()]
        composition = _composition()
        composition["extra_field"] = "surprise"
        result = validate_composition(composition, items)
        self.assertFalse(result.valid)
        self.assertIn(REASON_UNKNOWN_KEY, _codes(result))

    def test_unknown_section_field_is_rejected(self):
        items = [_item()]
        section = _section()
        section["extra_field"] = "surprise"
        composition = _composition(sections=[section])
        result = validate_composition(composition, items)
        self.assertFalse(result.valid)
        self.assertIn(REASON_UNKNOWN_KEY, _codes(result))

    def test_unknown_bullet_field_is_rejected(self):
        items = [_item()]
        bullet = _bullet(item_ids=["sec01-i01"])
        bullet["extra_field"] = "surprise"
        composition = _composition(sections=[_section(bullets=[bullet])])
        result = validate_composition(composition, items)
        self.assertFalse(result.valid)
        self.assertIn(REASON_UNKNOWN_KEY, _codes(result))

    def test_unknown_unit_field_is_rejected(self):
        items = [_item()]
        entry = _unit(item_ids=["sec01-i01"])
        entry["extra_field"] = "surprise"
        composition = _composition(tldr=[entry])
        result = validate_composition(composition, items)
        self.assertFalse(result.valid)
        self.assertIn(REASON_UNKNOWN_KEY, _codes(result))

    def test_missing_required_top_level_field_is_rejected(self):
        items = [_item()]
        composition = _composition()
        del composition["follow_up"]
        result = validate_composition(composition, items)
        self.assertFalse(result.valid)
        self.assertIn(REASON_MISSING_KEY, _codes(result))

    def test_missing_required_section_field_is_rejected(self):
        items = [_item()]
        section = _section()
        del section["protocol_item_ids"]
        composition = _composition(sections=[section])
        result = validate_composition(composition, items)
        self.assertFalse(result.valid)
        self.assertIn(REASON_MISSING_KEY, _codes(result))


# ---------------------------------------------------------------------------
# primary_item_id
# ---------------------------------------------------------------------------


class PrimaryItemIdTests(unittest.TestCase):
    def test_primary_item_id_not_among_item_ids_is_rejected(self):
        items = [_item()]
        bullet = _bullet(item_ids=["sec01-i01"], primary_item_id="not-in-item-ids")
        composition = _composition(sections=[_section(bullets=[bullet])])
        result = validate_composition(composition, items)
        self.assertFalse(result.valid)
        self.assertIn(REASON_UNKNOWN_ITEM_ID, _codes(result))

    def test_primary_item_id_among_item_ids_is_accepted(self):
        items = [_item()]
        bullet = _bullet(item_ids=["sec01-i01"], primary_item_id="sec01-i01")
        composition = _composition(sections=[_section(bullets=[bullet])])
        result = validate_composition(composition, items)
        self.assertTrue(result.valid, result.errors)

    def test_primary_item_id_null_is_accepted(self):
        items = [_item()]
        bullet = _bullet(item_ids=["sec01-i01"], primary_item_id=None)
        composition = _composition(sections=[_section(bullets=[bullet])])
        result = validate_composition(composition, items)
        self.assertTrue(result.valid, result.errors)


# ---------------------------------------------------------------------------
# A fully valid composition
# ---------------------------------------------------------------------------


class FullyValidCompositionTests(unittest.TestCase):
    def test_a_fully_valid_composition_has_no_errors(self):
        items = [
            _item("sec01-i01", kind="protocol", value="high", quantities=[{"value": 5, "value_high": None, "unit_as_spoken": "grams"}]),
            _item("sec01-i02", kind="claim", value="normal"),
            _item("sec02-i01", kind="side_topic", value="low"),
        ]
        composition = _composition(
            tldr=[_unit("The episode covers dosing.", ["sec01-i01"])],
            sections=[_section(
                title="Creatine dosing",
                bullets=[
                    _bullet("Dose is {q:sec01-i01:0} per day.", ["sec01-i01"], primary_item_id="sec01-i01"),
                    _bullet("A related claim was made.", ["sec01-i02"], primary_item_id="sec01-i02"),
                ],
                protocol_item_ids=["sec01-i01"],
            )],
            also_discussed=[_unit("A tangent was mentioned.", ["sec02-i01"])],
            takeaways=[_unit("Consider the dosing protocol.", ["sec01-i01"])],
        )
        result = validate_composition(composition, items)
        self.assertTrue(result.valid, result.errors)


# ---------------------------------------------------------------------------
# build_messages determinism
# ---------------------------------------------------------------------------


class BuildMessagesTests(unittest.TestCase):
    def test_build_messages_is_deterministic_regardless_of_item_order(self):
        items_a = [_item("sec01-i01"), _item("sec01-i02")]
        items_b = [_item("sec01-i02"), _item("sec01-i01")]
        episode_context = {"title": "Ep", "description": "desc"}
        first = build_messages(items=items_a, episode_context=episode_context, prompt_text="prompt")
        second = build_messages(items=items_b, episode_context=episode_context, prompt_text="prompt")
        self.assertEqual(first, second)

    def test_build_messages_includes_previous_errors_when_given(self):
        items = [_item()]
        messages = build_messages(
            items=items, episode_context={}, prompt_text="prompt",
            previous_errors=["uncited_unit: tldr[0] cites no items"],
        )
        payload = json.loads(messages[1]["content"])
        self.assertIn("previous_errors", payload)
        self.assertEqual(payload["previous_errors"], ["uncited_unit: tldr[0] cites no items"])

    def test_build_messages_omits_previous_errors_when_none_given(self):
        items = [_item()]
        messages = build_messages(items=items, episode_context={}, prompt_text="prompt")
        payload = json.loads(messages[1]["content"])
        self.assertNotIn("previous_errors", payload)


# ---------------------------------------------------------------------------
# compose_note orchestration (fake client)
# ---------------------------------------------------------------------------


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


def _valid_composition_for(items: list[dict]) -> dict:
    item_ids = [item["item_id"] for item in items]
    return _composition(takeaways=[_unit("A grounded takeaway.", item_ids)])


class ComposeNoteOrchestrationTests(unittest.TestCase):
    def test_a_valid_first_attempt_returns_immediately_with_no_retry(self):
        items = [_item()]
        client = _FakeClient([_valid_composition_for(items)])
        result = compose_note(items, client=client, model="m", provider="p", episode_context={}, prompt_text="prompt")
        self.assertIsInstance(result, CompositionRunResult)
        self.assertFalse(result.retried)
        self.assertFalse(result.held)
        self.assertEqual(len(client.calls), 1)

    def test_an_invalid_first_attempt_retries_once_with_the_exact_error_list(self):
        items = [_item(value="high")]
        client = _FakeClient([_composition(), _valid_composition_for(items)])
        result = compose_note(items, client=client, model="m", provider="p", episode_context={}, prompt_text="prompt")
        self.assertTrue(result.retried)
        self.assertFalse(result.held)
        self.assertEqual(len(client.calls), 2)
        retry_payload = json.loads(client.calls[1][1]["content"])
        self.assertIn("previous_errors", retry_payload)
        self.assertTrue(any("high_value_item_not_cited" in err for err in retry_payload["previous_errors"]))

    def test_a_still_invalid_composition_after_retry_is_held(self):
        items = [_item(value="high")]
        client = _FakeClient([_composition(), _composition()])
        result = compose_note(items, client=client, model="m", provider="p", episode_context={}, prompt_text="prompt")
        self.assertTrue(result.retried)
        self.assertTrue(result.held)
        self.assertEqual(result.failure_reason, "validation_failed_after_retry")
        self.assertEqual(len(client.calls), 2)

    def test_invalid_json_on_first_attempt_retries_once(self):
        items = [_item()]
        client = _FakeClient([None, _valid_composition_for(items)])
        result = compose_note(items, client=client, model="m", provider="p", episode_context={}, prompt_text="prompt")
        self.assertTrue(result.retried)
        self.assertFalse(result.held)
        self.assertEqual(len(client.calls), 2)

    def test_invalid_json_on_both_attempts_is_held_with_invalid_json_reason(self):
        items = [_item()]
        client = _FakeClient([None, None])
        result = compose_note(items, client=client, model="m", provider="p", episode_context={}, prompt_text="prompt")
        self.assertTrue(result.held)
        self.assertEqual(result.failure_reason, "invalid_json_response")
        self.assertIsNone(result.composition)
        self.assertEqual(len(client.calls), 2)

    def test_compose_note_never_makes_more_than_two_calls(self):
        items = [_item(value="high")]
        client = _FakeClient([_composition(), _composition()])
        compose_note(items, client=client, model="m", provider="p", episode_context={}, prompt_text="prompt")
        self.assertLessEqual(len(client.calls), 2)


if __name__ == "__main__":
    unittest.main()
