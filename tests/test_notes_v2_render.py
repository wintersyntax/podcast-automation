"""TASK-106 Phase A Task 9: golden-file tests for `notes_v2.render`.

Design authority:
docs/superpowers/specs/2026-09-24-grounded-knowledge-note-pipeline-design.md
§4 (note format v2), §7 (rendering), §8.1-§8.2 (failure-policy rendering).
"""

import unittest

from podcast_engine.knowledge.notes_v2.render import (
    LEGEND_LINE,
    render_note,
    render_placeholders,
)
from podcast_engine.knowledge.notes_v2.sources import (
    STATUS_AS_HEARD,
    STATUS_AUTO_MATCHED,
    STATUS_CORRECTED,
    STATUS_SHOW_NOTES,
    SourceResolution,
)

FRONTMATTER = "---\ntype: podcast-summary\ntitle: Example Episode\n---"


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
        "quote": "the exact quoted transcript text",
        "quote_occurrence": 1,
        "protocol": None,
        "source_mention": None,
    }
    base.update(overrides)
    return base


def _bullet(text: str = "A grounded bullet.", item_ids=None, primary_item_id=None, asserts: bool = True) -> dict:
    return {
        "text": text,
        "item_ids": item_ids if item_ids is not None else [],
        "primary_item_id": primary_item_id,
        "asserts": asserts,
    }


def _unit(text: str = "A grounded sentence.", item_ids=None) -> dict:
    return {"text": text, "item_ids": item_ids if item_ids is not None else []}


def _section(title: str = "Topic one", bullets=None, protocol_item_ids=None) -> dict:
    return {
        "title": title,
        "bullets": bullets if bullets is not None else [],
        "protocol_item_ids": protocol_item_ids if protocol_item_ids is not None else [],
    }


def _composition(**overrides) -> dict:
    base = {
        "tldr": [_unit("The episode covers X and Y.")],
        "sections": [],
        "research_discussed": [],
        "also_discussed": [],
        "takeaways": [_unit("Do the thing.")],
        "follow_up": [],
    }
    base.update(overrides)
    return base


class PlaceholderSubstitutionTests(unittest.TestCase):
    def test_converted_unit_bolds_only_the_si_value(self):
        items = {"a": _item("a", quantities=[{"value": 150, "value_high": None, "unit_as_spoken": "lb"}])}
        result = render_placeholders("Take {q:a:0} before bed.", items)
        self.assertIn("**~68 kg**", result)
        self.assertIn("(said: 150 lb)", result)
        self.assertNotIn("**~68 kg** (said: 150 lb)**", result)

    def test_passthrough_unit_bolds_the_whole_value_with_no_said_clause(self):
        items = {"a": _item("a", quantities=[{"value": 20, "value_high": None, "unit_as_spoken": "g"}])}
        result = render_placeholders("Take {q:a:0} daily.", items)
        self.assertEqual(result, "Take **20 g** daily.")

    def test_range_quantity_converts_both_ends(self):
        items = {"a": _item("a", quantities=[{"value": 0.2, "value_high": 0.35, "unit_as_spoken": "grams per pound of bodyweight"}])}
        result = render_placeholders("Dose: {q:a:0}.", items)
        self.assertIn("(said: 0.2-0.35 grams per pound of bodyweight)", result)
        self.assertIn("g/kg", result)

    def test_multiple_placeholders_in_one_string_all_resolve(self):
        items = {
            "a": _item("a", quantities=[{"value": 5, "value_high": None, "unit_as_spoken": "g"}]),
            "b": _item("b", quantities=[{"value": 10, "value_high": None, "unit_as_spoken": "g"}]),
        }
        result = render_placeholders("{q:a:0} then {q:b:0}.", items)
        self.assertEqual(result, "**5 g** then **10 g**.")

    def test_unknown_item_id_raises(self):
        with self.assertRaises(ValueError):
            render_placeholders("{q:missing:0}", {})

    def test_out_of_range_index_raises(self):
        items = {"a": _item("a", quantities=[{"value": 5, "value_high": None, "unit_as_spoken": "g"}])}
        with self.assertRaises(ValueError):
            render_placeholders("{q:a:1}", items)

    def test_text_with_no_placeholder_is_unchanged(self):
        self.assertEqual(render_placeholders("plain text", {}), "plain text")


class EvidenceMarkerTests(unittest.TestCase):
    def test_descriptive_bullet_carries_no_marker(self):
        items = {"a": _item("a", evidence_basis="research")}
        composition = _composition(sections=[_section(bullets=[_bullet("Descriptive.", ["a"], "a", asserts=False)])])
        note = render_note(FRONTMATTER, composition, items)
        self.assertIn("- Descriptive.\n", note)

    def test_asserting_bullet_carries_its_basis_marker(self):
        items = {"a": _item("a", evidence_basis="coaching_experience")}
        composition = _composition(sections=[_section(bullets=[_bullet("Recommends X.", ["a"], "a", asserts=True)])])
        note = render_note(FRONTMATTER, composition, items)
        self.assertIn("- Recommends X. 👥\n", note)

    def test_hedged_item_appends_hedge_mark(self):
        items = {"a": _item("a", evidence_basis="opinion", hedged=True)}
        composition = _composition(sections=[_section(bullets=[_bullet("Might help.", ["a"], "a", asserts=True)])])
        note = render_note(FRONTMATTER, composition, items)
        self.assertIn("- Might help. 💭❔\n", note)

    def test_mixed_basis_bullet_uses_the_primary_items_basis(self):
        items = {
            "a": _item("a", evidence_basis="research"),
            "b": _item("b", evidence_basis="opinion"),
        }
        composition = _composition(sections=[_section(bullets=[_bullet("Mixed.", ["a", "b"], "b", asserts=True)])])
        note = render_note(FRONTMATTER, composition, items)
        self.assertIn("- Mixed. 💭\n", note)

    def test_no_primary_item_falls_back_to_the_first_cited_item(self):
        items = {
            "a": _item("a", evidence_basis="personal_experience"),
            "b": _item("b", evidence_basis="opinion"),
        }
        composition = _composition(sections=[_section(bullets=[_bullet("No primary.", ["a", "b"], None, asserts=True)])])
        note = render_note(FRONTMATTER, composition, items)
        self.assertIn("- No primary. 👤\n", note)

    def test_bullet_is_hedged_when_any_cited_item_is_hedged_even_if_not_primary(self):
        items = {
            "a": _item("a", evidence_basis="research", hedged=False),
            "b": _item("b", evidence_basis="opinion", hedged=True),
        }
        composition = _composition(sections=[_section(bullets=[_bullet("Mixed hedge.", ["a", "b"], "a", asserts=True)])])
        note = render_note(FRONTMATTER, composition, items)
        self.assertIn("- Mixed hedge. 🔬❔\n", note)


class ProtocolBlockTests(unittest.TestCase):
    def test_fully_specified_protocol_renders_every_field_verbatim(self):
        items = {
            "a": _item(
                "a", kind="protocol", statement="Creatine loading",
                protocol={"what": "Creatine", "dose": "20 g/day", "when_for_whom": "first week only", "caveat": "GI upset possible"},
            ),
        }
        composition = _composition(sections=[_section(protocol_item_ids=["a"])])
        note = render_note(FRONTMATTER, composition, items)
        self.assertIn("> **📋 Protocol: Creatine loading**", note)
        self.assertIn("> **What:** Creatine", note)
        self.assertIn("> **Dose / parameters:** 20 g/day", note)
        self.assertIn("> **When / for whom:** first week only", note)
        self.assertIn("> **Caveat:** GI upset possible", note)

    def test_unstated_when_for_whom_renders_not_specified(self):
        items = {"a": _item("a", kind="protocol", protocol={"what": "X", "dose": "Y", "when_for_whom": None, "caveat": None})}
        composition = _composition(sections=[_section(protocol_item_ids=["a"])])
        note = render_note(FRONTMATTER, composition, items)
        self.assertIn("> **When / for whom:** not specified", note)

    def test_unstated_caveat_renders_none_stated_not_not_specified(self):
        items = {"a": _item("a", kind="protocol", protocol={"what": "X", "dose": "Y", "when_for_whom": "Z", "caveat": None})}
        composition = _composition(sections=[_section(protocol_item_ids=["a"])])
        note = render_note(FRONTMATTER, composition, items)
        self.assertIn("> **Caveat:** none stated", note)
        self.assertNotIn("> **Caveat:** not specified", note)

    def test_protocol_fields_are_never_bolded_as_quantities_even_with_numbers(self):
        items = {"a": _item("a", kind="protocol", protocol={"what": "X", "dose": "20 g/day", "when_for_whom": None, "caveat": None})}
        composition = _composition(sections=[_section(protocol_item_ids=["a"])])
        note = render_note(FRONTMATTER, composition, items)
        self.assertIn("> **Dose / parameters:** 20 g/day", note)
        self.assertNotIn("**20 g/day**", note)


class NumbersTableTests(unittest.TestCase):
    def test_quantity_bearing_item_gets_a_row(self):
        items = {"a": _item("a", quantities=[{"value": 5, "value_high": None, "unit_as_spoken": "g"}])}
        composition = _composition(sections=[_section("Dosing", bullets=[_bullet("Take {q:a:0}.", ["a"], "a")])])
        note = render_note(FRONTMATTER, composition, items)
        self.assertIn("## Numbers & protocols", note)
        self.assertIn("| Dosing | **5 g** | 🔬 |", note)

    def test_protocol_item_gets_a_condensed_row(self):
        items = {"a": _item("a", kind="protocol", protocol={"what": "Creatine", "dose": "20 g/day", "when_for_whom": "always", "caveat": None})}
        composition = _composition(sections=[_section("Dosing", protocol_item_ids=["a"])])
        note = render_note(FRONTMATTER, composition, items)
        self.assertIn("| Dosing | Creatine — 20 g/day, always | 🔬 |", note)

    def test_item_with_no_quantities_and_not_a_protocol_gets_no_row(self):
        items = {"a": _item("a", quantities=[])}
        composition = _composition(sections=[_section("Topic", bullets=[_bullet("Plain.", ["a"], "a")])])
        note = render_note(FRONTMATTER, composition, items)
        self.assertNotIn("## Numbers & protocols", note)

    def test_same_item_cited_twice_in_a_section_gets_one_row(self):
        items = {"a": _item("a", quantities=[{"value": 5, "value_high": None, "unit_as_spoken": "g"}])}
        composition = _composition(sections=[_section("Dosing", bullets=[
            _bullet("First {q:a:0}.", ["a"], "a"),
            _bullet("Second {q:a:0}.", ["a"], "a"),
        ])])
        note = render_note(FRONTMATTER, composition, items)
        self.assertEqual(note.count("| Dosing |"), 1)

    def test_table_omitted_when_no_section_has_a_numeric_or_protocol_item(self):
        items = {"a": _item("a")}
        composition = _composition(sections=[_section("Topic", bullets=[_bullet("Plain.", ["a"], "a")])])
        note = render_note(FRONTMATTER, composition, items)
        self.assertNotIn("Numbers & protocols", note)


class SourcesSectionTests(unittest.TestCase):
    def _source_item(self, item_id="src01"):
        return _item(
            item_id, kind="study_description", evidence_basis="research",
            source_mention={"authors_as_heard": "Smith", "year_as_heard": "2020", "title_as_heard": "Some Study", "type": "study"},
        )

    def test_show_notes_label(self):
        items = {"a": self._source_item()}
        resolutions = {"a": SourceResolution(status=STATUS_SHOW_NOTES, title="Smith et al. 2020", url="http://x", doi=None)}
        composition = _composition(research_discussed=[_unit("Discussed.", ["a"])])
        note = render_note(FRONTMATTER, composition, items, resolutions_by_item_id=resolutions)
        self.assertIn("- Smith et al. 2020 — ✔︎ show notes", note)

    def test_auto_matched_label(self):
        items = {"a": self._source_item()}
        resolutions = {"a": SourceResolution(status=STATUS_AUTO_MATCHED, title="Smith et al. 2020", url="http://x", doi="10.1/x")}
        composition = _composition(research_discussed=[_unit("Discussed.", ["a"])])
        note = render_note(FRONTMATTER, composition, items, resolutions_by_item_id=resolutions)
        self.assertIn("✔︎ auto-matched", note)

    def test_corrected_label(self):
        items = {"a": self._source_item()}
        resolutions = {"a": SourceResolution(status=STATUS_CORRECTED, title="Corrected Title", url=None, doi=None)}
        composition = _composition(research_discussed=[_unit("Discussed.", ["a"])])
        note = render_note(FRONTMATTER, composition, items, resolutions_by_item_id=resolutions)
        self.assertIn("- Corrected Title — ✎ corrected", note)

    def test_as_heard_label_with_no_resolution_supplied(self):
        items = {"a": self._source_item()}
        composition = _composition(research_discussed=[_unit("Discussed.", ["a"])])
        note = render_note(FRONTMATTER, composition, items)
        self.assertIn("- Some Study — as heard", note)

    def test_sources_section_omitted_when_no_source_items_cited(self):
        items = {"a": _item("a")}
        composition = _composition(sections=[_section(bullets=[_bullet("Plain.", ["a"], "a")])])
        note = render_note(FRONTMATTER, composition, items)
        self.assertNotIn("Sources mentioned", note)

    def test_research_discussed_bullet_prefixes_the_resolved_reference(self):
        items = {"a": self._source_item()}
        resolutions = {"a": SourceResolution(status=STATUS_AS_HEARD, title=None, url=None, doi=None)}
        composition = _composition(research_discussed=[_unit("small sample, host was skeptical.", ["a"])])
        note = render_note(FRONTMATTER, composition, items, resolutions_by_item_id=resolutions)
        self.assertIn("- **Some Study** — small sample, host was skeptical.", note)


class RemovedBulletTests(unittest.TestCase):
    def test_removed_bullet_becomes_a_collapsible_details_block(self):
        items = {"a": _item("a", quote="the exact removed quote")}
        composition = _composition(sections=[_section(bullets=[_bullet("Unverified.", ["a"], "a")])])
        note = render_note(FRONTMATTER, composition, items, removed_unit_ids=["sections[0].bullets[0]"])
        self.assertIn("<details>", note)
        self.assertIn("<summary>Removed 1 unverified point</summary>", note)
        self.assertIn("> the exact removed quote", note)
        self.assertNotIn("- Unverified.", note)

    def test_non_removed_bullet_in_the_same_section_still_renders_normally(self):
        items = {"a": _item("a"), "b": _item("b")}
        composition = _composition(sections=[_section(bullets=[
            _bullet("Removed one.", ["a"], "a"),
            _bullet("Kept one.", ["b"], "b"),
        ])])
        note = render_note(FRONTMATTER, composition, items, removed_unit_ids=["sections[0].bullets[0]"])
        self.assertIn("- Kept one.", note)
        self.assertIn("<details>", note)


class FooterLineTests(unittest.TestCase):
    def test_footer_line_appears_when_supplied(self):
        items = {"a": _item("a")}
        composition = _composition(sections=[_section(bullets=[_bullet("X.", ["a"], "a")])])
        note = render_note(FRONTMATTER, composition, items, footer_line="1 key point(s) could not be verified — see PodcastOps.")
        self.assertTrue(note.rstrip().endswith("1 key point(s) could not be verified — see PodcastOps."))

    def test_no_footer_line_when_not_supplied(self):
        items = {"a": _item("a")}
        composition = _composition(sections=[_section(bullets=[_bullet("X.", ["a"], "a")])])
        note = render_note(FRONTMATTER, composition, items)
        self.assertNotIn("could not be verified", note)


class SectionOrderAndOmissionTests(unittest.TestCase):
    def test_full_section_order(self):
        items = {
            "a": _item("a"),
            "src": _item("src", kind="study_description", source_mention={"authors_as_heard": "S", "year_as_heard": "2020", "title_as_heard": "Study", "type": "study"}),
        }
        composition = _composition(
            sections=[_section("Topic one", bullets=[_bullet("Bullet.", ["a"], "a")])],
            research_discussed=[_unit("Discussed.", ["src"])],
            also_discussed=[_unit("Side topic.")],
            follow_up=[_unit("Open question.")],
        )
        note = render_note(FRONTMATTER, composition, items)
        for marker in ["## TL;DR", "## Topic one", "## Research discussed", "## Also discussed", "## Takeaways", "## Sources mentioned", "## Follow up", "Basis: 🔬"]:
            self.assertIn(marker, note)
        order = [note.index(m) for m in ["## TL;DR", "## Topic one", "## Research discussed", "## Also discussed", "## Takeaways", "## Sources mentioned", "## Follow up"]]
        self.assertEqual(order, sorted(order))

    def test_research_discussed_omitted_when_empty(self):
        items = {"a": _item("a")}
        composition = _composition(sections=[_section(bullets=[_bullet("X.", ["a"], "a")])])
        note = render_note(FRONTMATTER, composition, items)
        self.assertNotIn("Research discussed", note)

    def test_also_discussed_omitted_when_empty(self):
        items = {"a": _item("a")}
        composition = _composition(sections=[_section(bullets=[_bullet("X.", ["a"], "a")])])
        note = render_note(FRONTMATTER, composition, items)
        self.assertNotIn("Also discussed", note)

    def test_follow_up_omitted_when_empty(self):
        items = {"a": _item("a")}
        composition = _composition(sections=[_section(bullets=[_bullet("X.", ["a"], "a")])])
        note = render_note(FRONTMATTER, composition, items)
        self.assertNotIn("Follow up", note)

    def test_takeaways_always_renders_even_when_it_has_content(self):
        items = {"a": _item("a")}
        composition = _composition(sections=[_section(bullets=[_bullet("X.", ["a"], "a")])])
        note = render_note(FRONTMATTER, composition, items)
        self.assertIn("## Takeaways", note)


class FrontmatterPreservationTests(unittest.TestCase):
    def test_existing_frontmatter_is_reproduced_verbatim(self):
        items = {"a": _item("a")}
        composition = _composition(sections=[_section(bullets=[_bullet("X.", ["a"], "a")])])
        custom_frontmatter = "---\ntype: podcast-summary\ntitle: My Special Title\ntags: [a, b]\n---"
        note = render_note(custom_frontmatter, composition, items)
        self.assertTrue(note.startswith(custom_frontmatter))


class DeterminismTests(unittest.TestCase):
    def test_rendering_the_same_input_twice_gives_identical_bytes(self):
        items = {
            "a": _item("a", quantities=[{"value": 150, "value_high": None, "unit_as_spoken": "lb"}]),
            "b": _item("b", kind="protocol", protocol={"what": "X", "dose": "Y", "when_for_whom": None, "caveat": None}),
            "src": _item("src", kind="study_description", source_mention={"authors_as_heard": "S", "year_as_heard": "2020", "title_as_heard": "Study", "type": "study"}),
        }
        composition = _composition(
            sections=[_section("Topic", bullets=[_bullet("Take {q:a:0}.", ["a"], "a")], protocol_item_ids=["b"])],
            research_discussed=[_unit("Discussed.", ["src"])],
            also_discussed=[_unit("Side.")],
            follow_up=[_unit("Open question.")],
        )
        first = render_note(FRONTMATTER, composition, items, footer_line="1 key point(s) could not be verified — see PodcastOps.")
        second = render_note(FRONTMATTER, composition, items, footer_line="1 key point(s) could not be verified — see PodcastOps.")
        self.assertEqual(first, second)
        self.assertIsInstance(first, str)


class LegendLineTests(unittest.TestCase):
    def test_legend_line_constant_matches_spec_text(self):
        self.assertIn("🔬 research cited", LEGEND_LINE)
        self.assertIn("❔ speaker hedged", LEGEND_LINE)

    def test_legend_line_appears_in_rendered_note(self):
        items = {"a": _item("a")}
        composition = _composition(sections=[_section(bullets=[_bullet("X.", ["a"], "a")])])
        note = render_note(FRONTMATTER, composition, items)
        self.assertIn(LEGEND_LINE, note)


if __name__ == "__main__":
    unittest.main()
