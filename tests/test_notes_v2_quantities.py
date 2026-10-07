"""RED/GREEN tests for TASK-106 Phase A Task 2: quantity parsing, in-quote
validation, and SI conversion.

Design authority:
docs/superpowers/specs/2026-09-24-grounded-knowledge-note-pipeline-design.md
§4.5 (SI conversion) and §6 (numeric validation).
"""

import unittest
from decimal import Decimal

from podcast_engine.knowledge.notes_v2.quantities import (
    QuantityConversion,
    convert_quantity,
    parse_number,
    parse_range,
    quantity_in_quote,
)


class ParseNumberDigitsTests(unittest.TestCase):
    def test_parses_a_plain_integer(self):
        self.assertEqual(parse_number("200"), Decimal("200"))

    def test_parses_a_decimal(self):
        self.assertEqual(parse_number("12.5"), Decimal("12.5"))

    def test_parses_a_thousands_separator(self):
        self.assertEqual(parse_number("1,234"), Decimal("1234"))

    def test_parses_a_thousands_separator_with_a_decimal(self):
        self.assertEqual(parse_number("1,234.5"), Decimal("1234.5"))

    def test_returns_none_for_non_numeric_text(self):
        self.assertIsNone(parse_number("creatine"))

    def test_returns_none_for_empty_text(self):
        self.assertIsNone(parse_number("   "))


class ParseNumberWordsTests(unittest.TestCase):
    def test_parses_a_cardinal_word(self):
        self.assertEqual(parse_number("two"), Decimal("2"))

    def test_parses_one_and_a_half(self):
        self.assertEqual(parse_number("one and a half"), Decimal("1.5"))

    def test_parses_a_noun_and_a_half_as_one_and_a_half(self):
        self.assertEqual(parse_number("a gram and a half"), Decimal("1.5"))

    def test_parses_half_a_noun_as_one_half(self):
        self.assertEqual(parse_number("half a gram"), Decimal("0.5"))

    def test_a_couple_is_not_a_number(self):
        self.assertIsNone(parse_number("a couple"))

    def test_a_few_is_not_a_number(self):
        self.assertIsNone(parse_number("a few"))

    def test_several_is_not_a_number(self):
        self.assertIsNone(parse_number("several"))


class ParseRangeTests(unittest.TestCase):
    def test_parses_a_to_separated_range(self):
        self.assertEqual(parse_range("15 to 30"), (Decimal("15"), Decimal("30")))

    def test_parses_an_en_dash_range(self):
        self.assertEqual(parse_range("15–30"), (Decimal("15"), Decimal("30")))

    def test_parses_an_ascii_hyphen_range(self):
        self.assertEqual(parse_range("15-30"), (Decimal("15"), Decimal("30")))

    def test_parses_a_word_number_range(self):
        self.assertEqual(parse_range("two to three"), (Decimal("2"), Decimal("3")))

    def test_returns_none_for_a_non_range(self):
        self.assertIsNone(parse_range("thirty"))

    def test_returns_none_when_one_end_is_not_a_number(self):
        self.assertIsNone(parse_range("15 to creatine"))


class QuantityInQuoteTests(unittest.TestCase):
    def test_true_when_the_digit_value_appears_in_the_quote(self):
        self.assertTrue(
            quantity_in_quote(Decimal("200"), None, "take 200 milligrams before bed")
        )

    def test_true_when_the_word_value_appears_in_the_quote(self):
        self.assertTrue(
            quantity_in_quote(Decimal("2"), None, "take two grams of creatine")
        )

    def test_true_when_both_range_ends_appear_in_the_quote(self):
        self.assertTrue(
            quantity_in_quote(
                Decimal("15"), Decimal("30"), "somewhere in the 15 to 30 gram range"
            )
        )

    def test_true_when_both_range_ends_appear_via_en_dash(self):
        self.assertTrue(
            quantity_in_quote(
                Decimal("15"), Decimal("30"), "a dose of 15–30 grams"
            )
        )

    def test_false_when_the_value_is_absent(self):
        self.assertFalse(
            quantity_in_quote(Decimal("500"), None, "take 200 milligrams before bed")
        )

    def test_false_when_only_the_low_end_of_a_range_appears(self):
        self.assertFalse(
            quantity_in_quote(Decimal("15"), Decimal("30"), "take 15 grams daily")
        )

    def test_false_when_value_is_none(self):
        self.assertFalse(quantity_in_quote(None, None, "take 200 milligrams"))


class ConvertQuantityConversionTableTests(unittest.TestCase):
    def test_converts_pounds_to_kilograms(self):
        result = convert_quantity(Decimal("200"), None, "lb")
        self.assertEqual(result.converted_value, Decimal("91"))
        self.assertEqual(result.si_unit, "kg")
        self.assertFalse(result.unknown_unit)
        self.assertTrue(result.rendered.startswith("~91 kg"))

    def test_converts_grams_per_pound_to_grams_per_kilogram(self):
        result = convert_quantity(Decimal("0.2"), None, "g/lb")
        self.assertEqual(result.converted_value, Decimal("0.44"))
        self.assertEqual(result.si_unit, "g/kg")

    def test_converts_ounces_to_grams(self):
        result = convert_quantity(Decimal("8"), None, "oz")
        self.assertEqual(result.converted_value, Decimal("230"))
        self.assertEqual(result.si_unit, "g")

    def test_converts_miles_to_kilometers(self):
        result = convert_quantity(Decimal("3"), None, "mile")
        self.assertEqual(result.converted_value, Decimal("4.8"))
        self.assertEqual(result.si_unit, "km")

    def test_converts_feet_to_meters(self):
        result = convert_quantity(Decimal("6"), None, "ft")
        self.assertEqual(result.converted_value, Decimal("1.8"))
        self.assertEqual(result.si_unit, "m")

    def test_converts_inches_to_centimeters(self):
        result = convert_quantity(Decimal("10"), None, "in")
        self.assertEqual(result.converted_value, Decimal("25"))
        self.assertEqual(result.si_unit, "cm")

    def test_converts_fahrenheit_to_celsius(self):
        result = convert_quantity(Decimal("98.6"), None, "F")
        self.assertEqual(result.converted_value, Decimal("37"))
        self.assertEqual(result.si_unit, "°C")

    def test_converts_both_ends_of_a_range(self):
        result = convert_quantity(Decimal("150"), Decimal("200"), "lb")
        self.assertEqual(result.converted_value, Decimal("68"))
        self.assertEqual(result.converted_value_high, Decimal("91"))
        self.assertIn("68", result.rendered)
        self.assertIn("91", result.rendered)


class ConvertQuantityPassThroughTests(unittest.TestCase):
    def _assert_passes_through(self, unit):
        result = convert_quantity(Decimal("5"), None, unit)
        self.assertEqual(result.converted_value, Decimal("5"))
        self.assertEqual(result.si_unit, unit)
        self.assertFalse(result.unknown_unit)

    def test_kcal_passes_through(self):
        self._assert_passes_through("kcal")

    def test_grams_pass_through(self):
        self._assert_passes_through("g")

    def test_kilograms_pass_through(self):
        self._assert_passes_through("kg")

    def test_percent_passes_through(self):
        self._assert_passes_through("%")

    def test_reps_pass_through(self):
        self._assert_passes_through("reps")

    def test_sets_pass_through(self):
        self._assert_passes_through("sets")

    def test_rir_passes_through(self):
        self._assert_passes_through("RIR")

    def test_hours_pass_through(self):
        self._assert_passes_through("hours")


class ConvertQuantityUnknownUnitTests(unittest.TestCase):
    def test_unknown_unit_renders_as_spoken_and_is_flagged(self):
        result = convert_quantity(Decimal("2"), None, "stone")
        self.assertTrue(result.unknown_unit)
        self.assertEqual(result.si_unit, "stone")
        self.assertEqual(result.rendered, "2 stone")

    def test_unknown_unit_value_is_unchanged(self):
        result = convert_quantity(Decimal("2"), None, "stone")
        self.assertEqual(result.converted_value, Decimal("2"))


class QuantityConversionIsComparableTests(unittest.TestCase):
    def test_equal_conversions_compare_equal(self):
        first = convert_quantity(Decimal("200"), None, "lb")
        second = convert_quantity(Decimal("200"), None, "lb")
        self.assertEqual(first, second)
        self.assertIsInstance(first, QuantityConversion)


if __name__ == "__main__":
    unittest.main()
