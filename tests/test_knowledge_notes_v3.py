"""TASK-118: single-pass knowledge-note library (schema, SI, checks, render)."""

import copy
import unittest
from unittest.mock import patch

from podcast_engine.knowledge.notes_v3 import checks, render, schema, units

TRANSCRIPT = (
    "Okay, welcome back. Today we talk about creatine and sleep. "
    "The 2024 study used 0.35 grams per kilogram and cognition was rescued by about ten to fifteen percent. "
    "For heavier guys Nordic curls are brutal, so I put a bench in front of me to catch my chest and shorten the range. "
    "I finished the diet at 89 kilos at six foot and I'm up to like 92 now. "
    "This episode is brought to you by Elite FTS, use code MRR10 for ten percent off. "
    "Personally I think creatine might explain why elite lifters get by on less sleep, but I'm not sure."
)


def _bullet(text, anchor, basis="none", hedged=False):
    return {"text": text, "basis": basis, "hedged": hedged, "anchor": anchor}


def _note():
    return {
        "tldr": ["Creatine can offset some effects of sleep loss.", "Nordic curls can be shortened with a bench."],
        "sections": [
            {
                "title": "Creatine and sleep",
                "bottom_line": "A high acute dose rescued cognition after sleep loss.",
                "bullets": [
                    _bullet("The 2024 study used 0.35 g/kg and rescued cognition by about 10-15%.",
                            "the 2024 study used 0.35 grams per kilogram and cognition was rescued", "research"),
                    _bullet("Creatine may partly explain why elite lifters need less sleep.",
                            "creatine might explain why elite lifters get by on less sleep", "opinion", True),
                ],
                "protocols": [],
            },
            {
                "title": "Nordic curls for heavier lifters",
                "bottom_line": "Shorten the range with a bench to make them trainable.",
                "bullets": [
                    _bullet("Geoff puts a bench in front to catch his chest and shorten the range of motion.",
                            "so I put a bench in front of me to catch my chest", "personal_experience"),
                    _bullet("Warm-up regression for Nordic curls.", "put a bench in front of me"),
                    _bullet("He finished his diet at 89 kg at {{6 feet}} and is now about 92 kg.",
                            "I finished the diet at 89 kilos at six foot", "personal_experience"),
                ],
                "protocols": [],
            },
        ],
        "research_discussed": [{
            "authors_year": "Gordji-Nejad 2024", "design": "RCT", "sample": "not stated", "duration": "not stated",
            "result": "0.35 g/kg rescued cognition after sleep deprivation.", "host_comment": "Dose matters.",
            "basis": "research", "hedged": False,
            "anchor": "The 2024 study used 0.35 grams per kilogram",
        }],
        "numbers": [{"topic": "Acute creatine", "value": "0.35 g/kg", "basis": "research"}],
        "takeaways": ["A higher acute creatine dose may help after a bad night."],
        "sources_mentioned": [],
        "also_discussed": [{"text": "Elite FTS discount, use code MRR10.", "anchor": "use code MRR10 for ten percent off"}],
        "follow_up": [],
        "topics": ["creatine", "sleep"],
        "people": ["Eric Helms"],
        "existing_tags": [],
        "new_tag_candidates": [],
    }


class SchemaTests(unittest.TestCase):
    def test_valid_note_has_no_errors(self):
        self.assertEqual(schema.validate_note(_note()), [])

    def test_missing_field_unexpected_field_and_bad_enum_are_reported(self):
        note = _note()
        del note["takeaways"]
        note["extra"] = 1
        note["sections"][0]["bullets"][0]["basis"] = "vibes"
        errors = schema.validate_note(note)
        self.assertTrue(any("takeaways: missing" in e for e in errors))
        self.assertTrue(any("extra: unexpected field" in e for e in errors))
        self.assertTrue(any("'vibes'" in e for e in errors))

    def test_note_needs_a_section_and_anchors(self):
        note = _note()
        note["sections"] = []
        self.assertIn("note.sections: at least one section is required", schema.validate_note(note))
        note = _note()
        note["sections"][0]["bullets"][0]["anchor"] = "  "
        self.assertTrue(any(e.endswith(".anchor: empty") for e in schema.validate_note(note)))

    def test_response_format_is_strict_and_has_no_item_bounds(self):
        fmt = schema.RESPONSE_FORMAT["json_schema"]
        self.assertTrue(fmt["strict"])
        self.assertNotIn("maxItems", repr(fmt["schema"]))


class UnitsTests(unittest.TestCase):
    def convert(self, text):
        report = units.UnitReport()
        return units.to_si(text, report), report

    def test_braced_quantities_are_converted_with_the_spoken_original(self):
        self.assertEqual(self.convert("gained {{50 lb}}")[0], "gained ~23 kg (said: 50 lb)")
        self.assertEqual(self.convert("over {{200-230 lb}}")[0], "over ~91–100 kg (said: 200–230 lb)")
        self.assertEqual(self.convert("{{1 g/lb}} daily")[0], "~2.2 g/kg (said: 1 g/lb) daily")
        self.assertEqual(self.convert("{{0.8 g per pound of bodyweight}}")[0], "~1.8 g/kg (said: 0.8 g per pound of bodyweight)")
        self.assertEqual(self.convert("{{6 feet}} tall")[0], "~1.8 m (said: 6 feet) tall")
        self.assertEqual(self.convert("{{1 gallon}} of water")[0], "~3.8 L (said: 1 gallon) of water")

    def test_compound_heights_are_converted_to_centimetre_precision(self):
        self.assertEqual(self.convert("{{6 feet 4 inches}} tall")[0], "~1.93 m (said: 6 feet 4 inches) tall")

    def test_fractions_and_mixed_numbers_convert_as_one_quantity(self):
        self.assertEqual(self.convert("lost {{5/8 inch}}")[0], "lost ~1.6 cm (said: 5/8 inch)")
        self.assertEqual(self.convert("{{5 5/8 inches}}")[0], "~14 cm (said: 5 5/8 inches)")
        text, report = self.convert("neck lost about 5/8 inch")
        self.assertEqual(text, "neck lost about 1.6 cm (said: 5/8 inch)")
        self.assertEqual(report.unbraced_imperial, ["5/8 inch"])
        self.assertNotIn("/~", self.convert("about 5/8 inch or {{5/8 inch}}")[0])

    def test_calories_pass_through_as_kcal_and_approximation_is_not_doubled(self):
        self.assertEqual(self.convert("a {{200-400 calorie}} surplus")[0], "a 200–400 kcal surplus")
        self.assertEqual(self.convert("lost about {{5/8 inch}}")[0], "lost about 1.6 cm (said: 5/8 inch)")
        self.assertEqual(self.convert("roughly {{50 lb}}")[0], "roughly 23 kg (said: 50 lb)")

    def test_si_units_in_braces_pass_through(self):
        self.assertEqual(self.convert("{{5 kilos}}")[0], "5 kg")

    def test_unbraced_imperial_is_converted_and_reported(self):
        text, report = self.convert("held 93-95 kg (about 205 lb) for a year")
        self.assertEqual(text, "held 93-95 kg (about 93 kg (said: 205 lb)) for a year")
        self.assertEqual(report.unbraced_imperial, ["205 lb"])

    def test_unknown_units_are_kept_as_spoken_and_reported(self):
        text, report = self.convert("{{3 stone}}")
        self.assertEqual(text, "3 stone")
        self.assertEqual(report.unknown_units, ["{{3 stone}}"])


class ChecksTests(unittest.TestCase):
    def test_ads_stubs_and_unanchored_units_are_dropped_and_reported(self):
        note = _note()
        note["sections"][0]["bullets"].append(_bullet("A confident claim that nobody on the episode made.", "this sentence is not in the transcript at all"))
        # The seven-unit fixture is far smaller than a real note (~60 units).
        with patch.object(checks, "MAX_DROPPED_FRACTION", 0.5):
            checked, report = checks.apply_checks(note, TRANSCRIPT)
        reasons = sorted(d.reason for d in report.dropped)
        self.assertEqual(reasons, ["ad_or_sponsor", "anchor_not_in_transcript", "label_only_bullet"])
        texts = [b["text"] for s in checked["sections"] for b in s["bullets"]]
        self.assertNotIn("Warm-up regression for Nordic curls.", texts)
        self.assertEqual(checked["also_discussed"], [])
        self.assertEqual(len(texts), 4)

    def test_small_copy_slips_in_an_anchor_are_tolerated(self):
        index = checks.TranscriptIndex(TRANSCRIPT)
        offset, fuzzy = index.locate("so I put the bench in front of me to catch my chest and shorten the range")
        self.assertIsNotNone(offset)
        self.assertTrue(fuzzy)

    def test_input_note_is_not_mutated(self):
        note = _note()
        before = copy.deepcopy(note)
        checks.apply_checks(note, TRANSCRIPT)
        self.assertEqual(note, before)

    def test_note_losing_too_many_units_is_rejected(self):
        note = _note()
        for section in note["sections"]:
            for bullet in section["bullets"]:
                bullet["anchor"] = "words that never occur in this transcript anywhere"
        with self.assertRaises(checks.NoteRejectedError):
            checks.apply_checks(note, TRANSCRIPT)


class RenderTests(unittest.TestCase):
    def body(self):
        checked, _ = checks.apply_checks(_note(), TRANSCRIPT)
        return render.render_body(checked)

    def test_evidence_scale_labels_asserting_bullets_only(self):
        body = self.body()
        self.assertIn("- **▰▰▰ Study** — The **2024** study used **0.35 g/kg**", body)
        self.assertIn("- **▱▱▱ Opinion · unsure** — Creatine may partly explain", body)
        self.assertIn("- **▰▱▱ One person** — Geoff puts a bench in front", body)
        self.assertNotIn("🟢", body)

    def test_si_is_rendered_by_python_with_the_original_unbolded(self):
        self.assertIn("**89 kg** at **~1.8 m** (said: 6 feet) and is now about **92 kg**.", self.body())

    def test_sections_have_bottom_lines_and_studies_follow_the_fixed_order(self):
        body = self.body()
        self.assertIn("## Creatine and sleep\n*A high acute dose rescued cognition after sleep loss.*", body)
        self.assertIn("- **▰▰▰ Study** — **Gordji-Nejad 2024** — RCT → **0.35 g/kg** rescued cognition after sleep deprivation. *Hosts:* Dose matters.", body)

    def test_bold_does_not_swallow_commas_or_mixed_terms(self):
        report = units.UnitReport()
        self.assertEqual(render.format_text("sets of 60, 70, 80 kg", report), "sets of **60**, **70**, **80 kg**")
        self.assertEqual(render.format_text("GLP-1 and VO2 max", report), "GLP-1 and VO2 max")
        self.assertEqual(render.format_text("aiming for 2:55 on the rower", report), "aiming for **2:55** on the rower")
        self.assertEqual(render.format_text("1:30-2:00 rest", report), "**1:30-2:00** rest")

    def test_rendering_is_deterministic_and_has_no_frontmatter(self):
        first, second = self.body(), self.body()
        self.assertEqual(first, second)
        self.assertTrue(first.startswith("## TL;DR\n"))
        self.assertNotIn("\n# ", "\n" + first)


if __name__ == "__main__":
    unittest.main()
