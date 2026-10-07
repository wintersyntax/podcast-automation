"""RED/GREEN tests for TASK-106 Phase A Task 3: `knowledge-items-v1` item
schema validation and the low-condition-confidence heuristic.

Design authority:
docs/superpowers/specs/2026-09-24-grounded-knowledge-note-pipeline-design.md
§5.1, §5.2, §5.2.1, §6.
"""

import re
import unittest

from podcast_engine.knowledge.notes_v2.items import (
    REASON_STATEMENT_QUANTITY_UNDECLARED,
    ExcludedItem,
    RejectedItem,
    ValidatedItem,
    flag_low_condition_confidence,
    validate_items,
)


def _valid_item(**overrides):
    item = {
        "local_id": "i-01",
        "kind": "recommendation",
        "statement": "Take two grams of creatine before bed",
        "quote": "Take two grams of creatine before bed.",
        "quote_occurrence": 1,
        "quantities": [{"value": 2, "value_high": None, "unit_as_spoken": "grams"}],
        "negated": False,
        "evidence_basis": "coaching_experience",
        "hedged": False,
        "conditions": "acute sleep deprivation",
        "scope": "core",
        "value": "normal",
        "protocol": None,
        "source_mention": None,
        "rationale_for": None,
    }
    item.update(overrides)
    return item


def _section(text, section_id="sec-01", start=0):
    return {"section_id": section_id, "start": start, "end": start + len(text), "text": text}


def _extraction(section_id, items=None, nothing_relevant=False, nothing_relevant_reason=None):
    return {
        "section_id": section_id,
        "nothing_relevant": nothing_relevant,
        "nothing_relevant_reason": nothing_relevant_reason,
        "items": items or [],
    }


class SchemaShapeTests(unittest.TestCase):
    def test_unknown_key_is_rejected(self):
        text = "Take two grams of creatine before bed."
        item = _valid_item(quote=text)
        item["not_a_real_field"] = "nope"
        result = validate_items([_section(text)], [_extraction("sec-01", [item])])
        self.assertEqual(len(result.rejected), 1)
        self.assertEqual(result.rejected[0].reason, "unknown_key")

    def test_missing_key_is_rejected(self):
        text = "Take two grams of creatine before bed."
        item = _valid_item(quote=text)
        del item["hedged"]
        result = validate_items([_section(text)], [_extraction("sec-01", [item])])
        self.assertEqual(len(result.rejected), 1)
        self.assertEqual(result.rejected[0].reason, "missing_key")

    def test_invalid_kind_is_rejected(self):
        text = "Take two grams of creatine before bed."
        item = _valid_item(quote=text, kind="not_a_real_kind")
        result = validate_items([_section(text)], [_extraction("sec-01", [item])])
        self.assertEqual(result.rejected[0].reason, "invalid_enum")

    def test_invalid_value_is_rejected(self):
        text = "Take two grams of creatine before bed."
        item = _valid_item(quote=text, value="urgent")
        result = validate_items([_section(text)], [_extraction("sec-01", [item])])
        self.assertEqual(result.rejected[0].reason, "invalid_enum")

    def test_invalid_scope_is_rejected(self):
        text = "Take two grams of creatine before bed."
        item = _valid_item(quote=text, scope="somewhere_else")
        result = validate_items([_section(text)], [_extraction("sec-01", [item])])
        self.assertEqual(result.rejected[0].reason, "invalid_enum")

    def test_invalid_evidence_basis_is_rejected(self):
        text = "Take two grams of creatine before bed."
        item = _valid_item(quote=text, evidence_basis="vibes")
        result = validate_items([_section(text)], [_extraction("sec-01", [item])])
        self.assertEqual(result.rejected[0].reason, "invalid_enum")

    def test_valid_item_is_accepted(self):
        text = "Take two grams of creatine before bed."
        item = _valid_item(quote=text)
        result = validate_items([_section(text)], [_extraction("sec-01", [item])])
        self.assertEqual(len(result.accepted), 1)
        self.assertEqual(result.rejected, [])
        self.assertEqual(result.excluded, [])
        self.assertIsInstance(result.accepted[0], ValidatedItem)


class QuoteLocationTests(unittest.TestCase):
    def test_quote_inside_section_at_the_right_occurrence_is_accepted(self):
        text = "Filler. Take two grams of creatine before bed. More filler."
        quote = "Take two grams of creatine before bed."
        item = _valid_item(quote=quote, quote_occurrence=1)
        result = validate_items([_section(text)], [_extraction("sec-01", [item])])
        self.assertEqual(len(result.accepted), 1)

    def test_quote_absent_from_section_is_rejected(self):
        text = "This section never mentions creatine at all."
        item = _valid_item(quote="Take two grams of creatine before bed.", quote_occurrence=1)
        result = validate_items([_section(text)], [_extraction("sec-01", [item])])
        self.assertEqual(result.rejected[0].reason, "quote_not_found")

    def test_wrong_occurrence_is_rejected(self):
        text = "Take two grams of creatine before bed. Only said once here."
        item = _valid_item(quote="Take two grams of creatine before bed.", quote_occurrence=2)
        result = validate_items([_section(text)], [_extraction("sec-01", [item])])
        self.assertEqual(result.rejected[0].reason, "quote_not_found")

    def test_second_occurrence_is_located_correctly(self):
        text = "Take two grams of creatine before bed. Take two grams of creatine before bed."
        item = _valid_item(quote="Take two grams of creatine before bed.", quote_occurrence=2)
        result = validate_items([_section(text)], [_extraction("sec-01", [item])])
        self.assertEqual(len(result.accepted), 1)


class QuantityInQuoteTests(unittest.TestCase):
    def test_quantity_present_in_quote_is_accepted(self):
        text = "Take two grams of creatine before bed."
        item = _valid_item(
            quote=text,
            quantities=[{"value": 2, "value_high": None, "unit_as_spoken": "grams"}],
        )
        result = validate_items([_section(text)], [_extraction("sec-01", [item])])
        self.assertEqual(len(result.accepted), 1)

    def test_quantity_absent_from_quote_is_rejected(self):
        text = "Take two grams of creatine before bed."
        item = _valid_item(
            quote=text,
            quantities=[{"value": 5, "value_high": None, "unit_as_spoken": "grams"}],
        )
        result = validate_items([_section(text)], [_extraction("sec-01", [item])])
        self.assertEqual(result.rejected[0].reason, "quantity_not_in_quote")

    def test_range_quantity_must_have_both_ends_in_quote(self):
        text = "Take fifteen to thirty grams of protein per meal."
        item = _valid_item(
            quote=text,
            quantities=[{"value": 15, "value_high": 45, "unit_as_spoken": "grams"}],
        )
        result = validate_items([_section(text)], [_extraction("sec-01", [item])])
        self.assertEqual(result.rejected[0].reason, "quantity_not_in_quote")


class StatementQuantityFidelityTests(unittest.TestCase):
    """2026-09-27, Task 12 Step 6 dev-tuning: the complementary direction to
    `QuantityInQuoteTests` above. Modeled directly on a real dev-set item
    (episode e680cd486749195b6ae058df, section sec02, item i09) whose quote
    cited "half is set a la, Pelland, Remmert and Robinson" and whose
    statement said "...an indirect contribution counts as half a set..." but
    whose `quantities` array held four unrelated entries, none covering
    "half" -- composition correctly refused to cite a number absent from the
    item's own `quantities`, so the note silently dropped the fact.
    """

    def test_half_a_set_not_declared_in_quantities_is_rejected(self):
        text = (
            "Pelland, Remmert, and Robinson published research showing that "
            "twenty sets per week is optimal, but an indirect contribution "
            "only counts as half a set."
        )
        item = _valid_item(
            statement=(
                "An indirect contribution only counts as half a set toward the "
                "twenty-set weekly target, per Pelland, Remmert, and Robinson."
            ),
            quote=text,
            quantities=[{"value": 20, "value_high": None, "unit_as_spoken": "sets"}],
        )
        result = validate_items([_section(text)], [_extraction("sec-01", [item])])
        self.assertEqual(result.rejected[0].reason, REASON_STATEMENT_QUANTITY_UNDECLARED)

    def test_number_backed_by_a_declared_quantity_is_accepted(self):
        text = "Take two grams of creatine before bed."
        item = _valid_item(
            statement="Take two grams of creatine before bed",
            quote=text,
            quantities=[{"value": 2, "value_high": None, "unit_as_spoken": "grams"}],
        )
        result = validate_items([_section(text)], [_extraction("sec-01", [item])])
        self.assertEqual(len(result.accepted), 1)

    def test_number_backed_by_a_declared_range_high_end_is_accepted(self):
        text = "Take fifteen to thirty grams of protein per meal."
        item = _valid_item(
            statement="Take up to thirty grams of protein per meal",
            quote=text,
            quantities=[{"value": 15, "value_high": 30, "unit_as_spoken": "grams"}],
        )
        result = validate_items([_section(text)], [_extraction("sec-01", [item])])
        self.assertEqual(len(result.accepted), 1)

    def test_structural_word_in_statement_is_not_flagged(self):
        text = "This was the first point made about recovery."
        item = _valid_item(
            statement="This was the first point made about recovery",
            quote=text,
            quantities=[],
        )
        result = validate_items([_section(text)], [_extraction("sec-01", [item])])
        self.assertEqual(len(result.accepted), 1)

    def test_vague_quantifier_in_statement_is_not_flagged(self):
        text = "Several factors were discussed regarding sleep."
        item = _valid_item(
            statement="Several factors were discussed regarding sleep",
            quote=text,
            quantities=[],
        )
        result = validate_items([_section(text)], [_extraction("sec-01", [item])])
        self.assertEqual(len(result.accepted), 1)

    def test_digit_in_alphanumeric_term_in_statement_is_not_flagged(self):
        text = "VO2 max improved after the block."
        item = _valid_item(
            statement="VO2 max improved after the block",
            quote=text,
            quantities=[],
        )
        result = validate_items([_section(text)], [_extraction("sec-01", [item])])
        self.assertEqual(len(result.accepted), 1)

    def test_comma_grouped_number_in_statement_backed_by_the_declared_value_is_accepted(self):
        # 2026-09-27: a real dev-set item ("step count of 6,000 to 8,000")
        # exposed a tokenizer bug where a thousands-grouped number split into
        # separate digit tokens ("6", "000") at the comma, none of which
        # matched the correctly-declared 6000 entry -- a pure false positive,
        # fixed by making the tokenizer comma/decimal-aware.
        text = "My step count lands between 6,000 to 8,000 on a day to day basis."
        item = _valid_item(
            statement="Daily step count is between 6,000 and 8,000.",
            quote=text,
            quantities=[{"value": 6000, "value_high": 8000, "unit_as_spoken": "steps"}],
        )
        result = validate_items([_section(text)], [_extraction("sec-01", [item])])
        self.assertEqual(len(result.accepted), 1)

    def test_decimal_number_in_statement_backed_by_the_declared_value_is_accepted(self):
        text = "Three and a half times per week for major muscle groups."
        item = _valid_item(
            statement="Major muscle groups are trained 3.5 times per week.",
            quote=text,
            quantities=[{"value": 3.5, "value_high": None, "unit_as_spoken": "times per week"}],
        )
        result = validate_items([_section(text)], [_extraction("sec-01", [item])])
        self.assertEqual(len(result.accepted), 1)

    def test_noun_and_a_half_idiom_backed_by_the_declared_combined_value_is_accepted(self):
        # 2026-09-27: a real dev-set item ("a scoop and a half of whey")
        # exposed a limitation where the bare word "half" parses to 0.5, but
        # the phrase "a scoop and a half" is a single combined claim of 1.5
        # -- already correctly declared -- and should not be judged as an
        # undeclared bare 0.5.
        text = "I might do a scoop and a half of whey post-training."
        item = _valid_item(
            statement="Have a scoop and a half of whey after training.",
            quote=text,
            quantities=[{"value": 1.5, "value_high": None, "unit_as_spoken": "scoop of whey"}],
        )
        result = validate_items([_section(text)], [_extraction("sec-01", [item])])
        self.assertEqual(len(result.accepted), 1)

    def test_noun_and_a_half_idiom_not_backed_by_any_declared_value_is_rejected(self):
        text = "I might do a scoop and a half of whey post-training."
        item = _valid_item(
            statement="Have a scoop and a half of whey after training.",
            quote=text,
            quantities=[],
        )
        result = validate_items([_section(text)], [_extraction("sec-01", [item])])
        self.assertEqual(result.rejected[0].reason, REASON_STATEMENT_QUANTITY_UNDECLARED)

    def test_clock_time_in_statement_backed_by_the_declared_concatenated_value_is_accepted(self):
        # 2026-09-27: a real dev-set item stored a spoken time as the single
        # concatenated digits `530` in `quantities` (this codebase's own
        # extraction convention), but the statement wrote it back out as
        # "5:30" -- a pure formatting mismatch, not an undeclared number.
        text = "I'm taking it at like 530 AM."
        item = _valid_item(
            statement="Caffeine is taken at 5:30 AM.",
            quote=text,
            quantities=[{"value": 530, "value_high": None, "unit_as_spoken": "AM"}],
        )
        result = validate_items([_section(text)], [_extraction("sec-01", [item])])
        self.assertEqual(len(result.accepted), 1)


class NegationAgreementTests(unittest.TestCase):
    def test_negated_false_with_no_negation_token_is_accepted(self):
        text = "Take two grams of creatine before bed."
        item = _valid_item(quote=text, negated=False)
        result = validate_items([_section(text)], [_extraction("sec-01", [item])])
        self.assertEqual(len(result.accepted), 1)

    def test_negated_true_with_no_negation_token_is_rejected(self):
        text = "Take two grams of creatine before bed."
        item = _valid_item(quote=text, negated=True)
        result = validate_items([_section(text)], [_extraction("sec-01", [item])])
        self.assertEqual(result.rejected[0].reason, "negation_mismatch")

    def test_negated_true_with_a_negation_token_is_accepted(self):
        text = "Do not take creatine on an empty stomach."
        item = _valid_item(quote=text, negated=True, quantities=[])
        result = validate_items([_section(text)], [_extraction("sec-01", [item])])
        self.assertEqual(len(result.accepted), 1)

    def test_negated_false_with_a_negation_token_is_rejected(self):
        text = "Do not take creatine on an empty stomach."
        item = _valid_item(quote=text, negated=False, quantities=[])
        result = validate_items([_section(text)], [_extraction("sec-01", [item])])
        self.assertEqual(result.rejected[0].reason, "negation_mismatch")


class ConditionsRequirementTests(unittest.TestCase):
    def test_recommendation_missing_conditions_key_is_rejected_with_its_own_reason(self):
        text = "Take two grams of creatine before bed."
        item = _valid_item(quote=text, kind="recommendation")
        del item["conditions"]
        result = validate_items([_section(text)], [_extraction("sec-01", [item])])
        self.assertEqual(result.rejected[0].reason, "missing_conditions")

    def test_recommendation_null_conditions_is_rejected_with_its_own_reason(self):
        text = "Take two grams of creatine before bed."
        item = _valid_item(quote=text, kind="recommendation", conditions=None)
        result = validate_items([_section(text)], [_extraction("sec-01", [item])])
        self.assertEqual(result.rejected[0].reason, "missing_conditions")

    def test_missing_conditions_reason_is_distinct_from_generic_schema_violation(self):
        text = "Take two grams of creatine before bed."
        missing_conditions_item = _valid_item(quote=text, kind="recommendation", conditions=None)
        bad_enum_item = _valid_item(quote=text, kind="not_a_kind")
        result = validate_items(
            [_section(text)],
            [_extraction("sec-01", [missing_conditions_item, bad_enum_item])],
        )
        reasons = {rejected.reason for rejected in result.rejected}
        self.assertIn("missing_conditions", reasons)
        self.assertIn("invalid_enum", reasons)

    def test_none_stated_conditions_is_accepted(self):
        text = "Take two grams of creatine before bed."
        item = _valid_item(quote=text, kind="protocol", conditions="none_stated")
        result = validate_items([_section(text)], [_extraction("sec-01", [item])])
        self.assertEqual(len(result.accepted), 1)

    def test_real_qualifying_text_is_accepted(self):
        text = "Take two grams of creatine before bed."
        item = _valid_item(quote=text, kind="claim", conditions="beginners only")
        result = validate_items([_section(text)], [_extraction("sec-01", [item])])
        self.assertEqual(len(result.accepted), 1)

    def test_null_conditions_is_fine_for_a_kind_that_does_not_require_it(self):
        text = "Creatine works by increasing phosphocreatine stores."
        item = _valid_item(quote=text, kind="mechanism", conditions=None, quantities=[])
        result = validate_items([_section(text)], [_extraction("sec-01", [item])])
        self.assertEqual(len(result.accepted), 1)


class ScopeExclusionTests(unittest.TestCase):
    def test_out_of_scope_item_is_excluded_not_rejected_or_accepted(self):
        text = "Check out our sponsor for ten percent off your order."
        item = _valid_item(
            quote=text,
            kind="side_topic",
            scope="out_of_scope",
            conditions=None,
            quantities=[],
        )
        result = validate_items([_section(text)], [_extraction("sec-01", [item])])
        self.assertEqual(result.accepted, [])
        self.assertEqual(result.rejected, [])
        self.assertEqual(len(result.excluded), 1)
        self.assertIsInstance(result.excluded[0], ExcludedItem)


class NothingRelevantTests(unittest.TestCase):
    def test_valid_nothing_relevant_produces_no_items(self):
        text = "The hosts spent five minutes on an unrelated story."
        extraction = _extraction(
            "sec-01", items=[], nothing_relevant=True,
            nothing_relevant_reason="entire section was an unrelated personal story",
        )
        result = validate_items([_section(text)], [extraction])
        self.assertEqual(result.accepted, [])
        self.assertEqual(result.rejected, [])
        self.assertEqual(result.excluded, [])

    def test_nothing_relevant_without_a_reason_is_rejected(self):
        text = "The hosts spent five minutes on an unrelated story."
        extraction = _extraction("sec-01", items=[], nothing_relevant=True, nothing_relevant_reason=None)
        result = validate_items([_section(text)], [extraction])
        self.assertEqual(len(result.rejected), 1)
        self.assertEqual(result.rejected[0].reason, "nothing_relevant_inconsistent")

    def test_nothing_relevant_with_an_in_scope_item_is_rejected(self):
        text = "Take two grams of creatine before bed."
        item = _valid_item(quote=text)
        extraction = _extraction(
            "sec-01", items=[item], nothing_relevant=True,
            nothing_relevant_reason="mostly unrelated",
        )
        result = validate_items([_section(text)], [extraction])
        self.assertEqual(len(result.rejected), 1)
        self.assertEqual(result.rejected[0].reason, "nothing_relevant_inconsistent")


class OverlapMergeTests(unittest.TestCase):
    def test_overlapping_sections_merge_the_duplicate_into_one_item(self):
        text = "Filler before it. Take two grams of creatine before bed for better sleep. Tail words."
        quote = "Take two grams of creatine before bed for better sleep."
        item_a = _valid_item(quote=quote, local_id="i-01")
        item_b = _valid_item(quote=quote, local_id="i-01")

        # Section B starts 5 characters into section A's transcript range, so
        # the same phrase in both sections' own text resolves to overlapping
        # *global* offsets -- the scenario §5.2 describes for the ~150-word
        # section overlap.
        sections = [
            _section(text, section_id="sec-01", start=0),
            _section(text, section_id="sec-02", start=5),
        ]
        extractions = [
            _extraction("sec-01", [item_a]),
            _extraction("sec-02", [item_b]),
        ]

        result = validate_items(sections, extractions)
        self.assertEqual(len(result.accepted), 1)
        self.assertRegex(result.accepted[0].item_id, r"^sec\d+-i\d+$")

    def test_non_overlapping_items_are_both_kept(self):
        text_a = "Take two grams of creatine before bed."
        text_b = "Sleep for eight hours a night."
        item_a = _valid_item(quote=text_a, local_id="i-01")
        item_b = _valid_item(
            quote=text_b, local_id="i-01", quantities=[{"value": 8, "value_high": None, "unit_as_spoken": "hours"}],
        )
        sections = [
            _section(text_a, section_id="sec-01", start=0),
            _section(text_b, section_id="sec-02", start=10_000),
        ]
        extractions = [
            _extraction("sec-01", [item_a]),
            _extraction("sec-02", [item_b]),
        ]
        result = validate_items(sections, extractions)
        self.assertEqual(len(result.accepted), 2)
        item_ids = {item.item_id for item in result.accepted}
        self.assertEqual(item_ids, {"sec01-i01", "sec02-i01"})


class EveryRejectionHasAReasonCodeTests(unittest.TestCase):
    def test_every_rejected_item_carries_a_non_empty_reason(self):
        text = "Take two grams of creatine before bed."
        bad_kind = _valid_item(quote=text, kind="nope")
        bad_conditions = _valid_item(quote=text, kind="protocol", conditions=None)
        bad_quote = _valid_item(quote="never appears in the section")
        bad_quantity = _valid_item(
            quote=text, quantities=[{"value": 999, "value_high": None, "unit_as_spoken": "grams"}],
        )
        bad_negation = _valid_item(quote=text, negated=True)
        result = validate_items(
            [_section(text)],
            [_extraction("sec-01", [bad_kind, bad_conditions, bad_quote, bad_quantity, bad_negation])],
        )
        self.assertEqual(len(result.rejected), 5)
        for rejected in result.rejected:
            self.assertIsInstance(rejected, RejectedItem)
            self.assertTrue(rejected.reason)


class LowConditionConfidenceTests(unittest.TestCase):
    def test_none_stated_with_nearby_hedge_word_is_flagged(self):
        text = "This is usually true for most people. Take two grams of creatine before bed. It works well."
        quote = "Take two grams of creatine before bed."
        start = text.index(quote)
        end = start + len(quote)
        flagged = flag_low_condition_confidence(
            kind="recommendation",
            conditions="none_stated",
            quote_start=start,
            quote_end=end,
            section_text=text,
        )
        self.assertTrue(flagged)

    def test_explicit_condition_is_never_flagged_even_with_a_nearby_hedge_word(self):
        text = "This is usually true for most people. Take two grams of creatine before bed. It works well."
        quote = "Take two grams of creatine before bed."
        start = text.index(quote)
        end = start + len(quote)
        flagged = flag_low_condition_confidence(
            kind="recommendation",
            conditions="beginners only",
            quote_start=start,
            quote_end=end,
            section_text=text,
        )
        self.assertFalse(flagged)

    def test_no_hedge_word_in_window_is_not_flagged(self):
        text = "This is a plain statement. Take two grams of creatine before bed. Nothing hedgy here."
        quote = "Take two grams of creatine before bed."
        start = text.index(quote)
        end = start + len(quote)
        flagged = flag_low_condition_confidence(
            kind="recommendation",
            conditions="none_stated",
            quote_start=start,
            quote_end=end,
            section_text=text,
        )
        self.assertFalse(flagged)

    def test_kind_outside_protocol_or_recommendation_is_never_flagged(self):
        text = "This is usually true for most people. Creatine works by raising phosphocreatine. Fine."
        quote = "Creatine works by raising phosphocreatine."
        start = text.index(quote)
        end = start + len(quote)
        flagged = flag_low_condition_confidence(
            kind="claim",
            conditions="none_stated",
            quote_start=start,
            quote_end=end,
            section_text=text,
        )
        self.assertFalse(flagged)

    def test_flag_never_changes_acceptance(self):
        text = "This is usually true for most people. Take two grams of creatine before bed. It works well."
        quote = "Take two grams of creatine before bed."
        item = _valid_item(quote=quote, kind="recommendation", conditions="none_stated")
        result = validate_items([_section(text)], [_extraction("sec-01", [item])])
        # Acceptance never depends on the low-condition-confidence heuristic;
        # it is a separate, later pass.
        self.assertEqual(len(result.accepted), 1)


if __name__ == "__main__":
    unittest.main()
