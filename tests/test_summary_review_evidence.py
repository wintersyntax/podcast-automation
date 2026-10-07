import copy
import unittest

from podcast_engine.knowledge.summary_review_evidence import (
    build_draft_block_index,
    build_review_context,
    build_risk_inventory,
    build_transcript_span_index,
    draft_blocks_by_id,
    normalize_newlines,
    risks_by_id,
    transcript_spans_by_id,
    validate_review_context,
)


class TranscriptSpanTests(unittest.TestCase):
    def test_transcript_spans_are_stable_and_bounded(self):
        transcript = "\r\n".join(f"word{i}" for i in range(1, 98))
        normalized = normalize_newlines(transcript)
        index = build_transcript_span_index(transcript)

        self.assertEqual(
            [span["span_id"] for span in index["spans"]],
            ["S0001", "S0002", "S0003"],
        )
        self.assertEqual(
            [len(span["text"].split()) for span in index["spans"]],
            [48, 48, 1],
        )
        for span in index["spans"]:
            self.assertEqual(
                normalized[span["start_char"] : span["end_char"]],
                span["text"],
            )

        self.assertEqual(build_transcript_span_index(transcript), index)

    def test_transcript_source_change_changes_source_and_index_hashes(self):
        first = build_transcript_span_index("alpha beta gamma")
        second = build_transcript_span_index("alpha beta gammA")

        self.assertNotEqual(first["source_sha256"], second["source_sha256"])
        self.assertNotEqual(first["index_sha256"], second["index_sha256"])


class DraftBlockTests(unittest.TestCase):
    def test_headings_list_items_and_paragraphs_are_stable_source_slices(self):
        draft = (
            "## TL;DR\r\n\r\n"
            "- First item\r\n  continued line\r\n"
            "- Second item\r\n\r\n"
            "## Key Ideas\r\n\r\n"
            "### Topic\r\n\r\n"
            "A prose paragraph\r\ncontinues here.\r\n\r\n"
            "Final paragraph.\r\n"
        )
        normalized = normalize_newlines(draft)
        index = build_draft_block_index(draft)

        self.assertEqual(
            [block["block_id"] for block in index["blocks"]],
            [f"D{number:04d}" for number in range(1, 8)],
        )
        self.assertEqual(
            [block["text"] for block in index["blocks"]],
            [
                "## TL;DR",
                "- First item\n  continued line",
                "- Second item",
                "## Key Ideas",
                "### Topic",
                "A prose paragraph\ncontinues here.",
                "Final paragraph.",
            ],
        )
        for block in index["blocks"]:
            self.assertEqual(
                normalized[block["start_char"] : block["end_char"]],
                block["text"],
            )
        self.assertEqual(build_draft_block_index(draft), index)


class RiskInventoryTests(unittest.TestCase):
    def test_risks_follow_draft_block_and_surface_order(self):
        draft = (
            "Performance reached 99%.\n\n"
            "They used Concept2.\n\n"
            "This is definitely effective.\n\n"
            "People should try it.\n\n"
            "It always helps.\n"
        )
        draft_index = build_draft_block_index(draft)
        inventory = build_risk_inventory(draft, draft_index)

        self.assertEqual(
            [item["risk_id"] for item in inventory["risks"]],
            ["R0001", "R0002", "R0003", "R0004", "R0005"],
        )
        self.assertEqual(
            {item["kind"] for item in inventory["risks"]},
            {
                "numeric_precision",
                "named_or_technical_term",
                "certainty_or_causality",
                "recommendation_language",
                "absolute_quantifier",
            },
        )
        self.assertEqual(
            [item["surface"] for item in inventory["risks"]],
            ["99%", "Concept2", "definitely", "should", "always"],
        )
        for item in inventory["risks"]:
            self.assertEqual(
                draft[item["start_char"] : item["end_char"]],
                item["surface"],
            )

    def test_detector_rules_cover_committed_bounded_surfaces(self):
        blocks = (
            "99% 2.5 kg 5 hours 120 mg\n\n"
            "Concept2 VO2max GLP1 mTOR AeroFlux-X9\n\n"
            "definitely certainly proves proven guarantees causes caused leads to results in\n\n"
            "should must need to needs to recommend recommended best to\n\n"
            "always never all none every everyone nobody no one\n"
        )
        inventory = build_risk_inventory(blocks, build_draft_block_index(blocks))
        by_kind = {}
        for item in inventory["risks"]:
            by_kind.setdefault(item["kind"], []).append(item["surface"])

        self.assertEqual(
            by_kind["numeric_precision"],
            ["99%", "2.5 kg", "5 hours", "120 mg"],
        )
        self.assertEqual(
            by_kind["named_or_technical_term"],
            ["Concept2", "VO2max", "GLP1", "mTOR", "AeroFlux-X9"],
        )
        self.assertEqual(
            by_kind["certainty_or_causality"],
            [
                "definitely",
                "certainly",
                "proves",
                "proven",
                "guarantees",
                "causes",
                "caused",
                "leads to",
                "results in",
            ],
        )
        self.assertEqual(
            by_kind["recommendation_language"],
            [
                "should",
                "must",
                "need to",
                "needs to",
                "recommend",
                "recommended",
                "best to",
            ],
        )
        self.assertEqual(
            by_kind["absolute_quantifier"],
            ["always", "never", "all", "none", "every", "everyone", "nobody", "no one"],
        )

    def test_headings_and_questions_do_not_create_common_word_risks(self):
        draft = (
            "## All Topics\n\n"
            "## Should You Act?\n\n"
            "Should you act?\n\n"
            "- Should everyone act?\n\n"
            "- You should act.\n\n"
            "Every person benefits.\n"
        )
        inventory = build_risk_inventory(draft, build_draft_block_index(draft))

        self.assertEqual(
            [
                (risk["kind"], risk["surface"], risk["draft_block_id"])
                for risk in inventory["risks"]
            ],
            [
                ("recommendation_language", "should", "D0005"),
                ("absolute_quantifier", "Every", "D0006"),
            ],
        )

    def test_headings_and_questions_preserve_numeric_and_technical_risks(self):
        draft = (
            "## All Concept2 Results 99%\n\n"
            "Should Concept2 reach 98%?\n"
        )
        inventory = build_risk_inventory(draft, build_draft_block_index(draft))

        observed = [
            (risk["kind"], risk["surface"], risk["draft_block_id"])
            for risk in inventory["risks"]
        ]

        self.assertIn(("named_or_technical_term", "Concept2", "D0001"), observed)
        self.assertIn(("numeric_precision", "99%", "D0001"), observed)
        self.assertIn(("named_or_technical_term", "Concept2", "D0002"), observed)
        self.assertIn(("numeric_precision", "98%", "D0002"), observed)

        self.assertNotIn(("absolute_quantifier", "All", "D0001"), observed)
        self.assertNotIn(("recommendation_language", "Should", "D0002"), observed)

    def test_plain_capitalized_proper_noun_mid_sentence_is_flagged_as_technical_term(self):
        draft = (
            "The draft misattributes this warning to Ramon Limacher instead "
            "of Eric Helms."
        )
        inventory = build_risk_inventory(draft, build_draft_block_index(draft))

        observed = {risk["surface"] for risk in inventory["risks"]}

        for name in ("Ramon", "Limacher", "Eric", "Helms"):
            self.assertIn(name, observed)

    def test_sentence_initial_capitalization_is_not_flagged_as_technical_term(self):
        draft = (
            "The guest explained the protocol. This was straightforward. "
            "It helped a lot."
        )
        inventory = build_risk_inventory(draft, build_draft_block_index(draft))

        observed = {
            risk["surface"]
            for risk in inventory["risks"]
            if risk["kind"] == "named_or_technical_term"
        }

        self.assertEqual(observed, set())

    def test_first_word_after_list_marker_is_not_flagged_as_technical_term(self):
        draft = "- The measured result reached 99%."
        inventory = build_risk_inventory(draft, build_draft_block_index(draft))

        observed = {
            risk["surface"]
            for risk in inventory["risks"]
            if risk["kind"] == "named_or_technical_term"
        }

        self.assertEqual(observed, set())

    def test_all_caps_acronym_is_flagged_as_technical_term(self):
        draft = "The clinic used a DEXA scan to confirm body composition."
        inventory = build_risk_inventory(draft, build_draft_block_index(draft))

        observed = {
            risk["surface"]
            for risk in inventory["risks"]
            if risk["kind"] == "named_or_technical_term"
        }

        self.assertIn("DEXA", observed)

    def test_heading_text_does_not_generate_capitalization_noise(self):
        lines = ["## Key Ideas", "", "The protocol remained unchanged."]
        draft = chr(10).join(lines)
        inventory = build_risk_inventory(draft, build_draft_block_index(draft))

        observed = [
            risk["surface"]
            for risk in inventory["risks"]
            if risk["kind"] == "named_or_technical_term"
        ]

        self.assertEqual(observed, [])

    def test_mixed_assertion_and_question_is_not_treated_as_purely_interrogative(self):
        draft = "You should take 5 mg. Should everyone?\n"
        inventory = build_risk_inventory(draft, build_draft_block_index(draft))

        observed = [
            (risk["kind"], risk["surface"])
            for risk in inventory["risks"]
        ]

        self.assertIn(("recommendation_language", "should"), observed)
        self.assertIn(("numeric_precision", "5 mg"), observed)

    def test_equivalent_surfaces_collapse_and_preserve_all_occurrences(self):
        draft = "Concept2 and concept2 and CONCEPT2."
        inventory = build_risk_inventory(draft, build_draft_block_index(draft))

        self.assertEqual(len(inventory["risks"]), 1)
        risk = inventory["risks"][0]
        self.assertEqual(risk["surface"], "Concept2")
        self.assertEqual(len(risk["occurrences"]), 3)
        self.assertEqual(
            [draft[item["start_char"] : item["end_char"]] for item in risk["occurrences"]],
            ["Concept2", "concept2", "CONCEPT2"],
        )

    def test_exactly_64_risks_succeed_and_65_fail_closed(self):
        valid_draft = " ".join(f"{number}%" for number in range(1, 65))
        valid = build_risk_inventory(
            valid_draft,
            build_draft_block_index(valid_draft),
        )
        self.assertEqual(len(valid["risks"]), 64)

        overflow_draft = " ".join(f"{number}%" for number in range(1, 66))
        with self.assertRaisesRegex(ValueError, "risk_inventory_overflow"):
            build_risk_inventory(
                overflow_draft,
                build_draft_block_index(overflow_draft),
            )


class ReviewContextTests(unittest.TestCase):
    def test_context_builds_exact_indexes_and_id_maps(self):
        transcript = "One supported sentence. Another supported sentence."
        draft = "## TL;DR\n\n- Concept2 reached 99%.\n"
        context = build_review_context(transcript, draft)

        self.assertEqual(
            set(context),
            {"transcript_span_index", "draft_block_index", "risk_inventory"},
        )
        self.assertEqual(
            transcript_spans_by_id(context),
            {span["span_id"]: span for span in context["transcript_span_index"]["spans"]},
        )
        self.assertEqual(
            draft_blocks_by_id(context),
            {block["block_id"]: block for block in context["draft_block_index"]["blocks"]},
        )
        self.assertEqual(
            risks_by_id(context),
            {risk["risk_id"]: risk for risk in context["risk_inventory"]["risks"]},
        )
        self.assertEqual(validate_review_context(context, transcript, draft), context)

    def test_validation_rebuilds_every_index_and_rejects_tampering(self):
        transcript = "A source sentence."
        draft = "## TL;DR\n\n- It always works.\n"
        context = build_review_context(transcript, draft)

        mutations = (
            ("transcript_span_index", "source_sha256"),
            ("draft_block_index", "index_sha256"),
            ("risk_inventory", "index_sha256"),
        )
        for section, field in mutations:
            with self.subTest(section=section, field=field):
                tampered = copy.deepcopy(context)
                tampered[section][field] = "sha256:" + "0" * 64
                with self.assertRaisesRegex(ValueError, "Review context does not match"):
                    validate_review_context(tampered, transcript, draft)

        with self.assertRaisesRegex(ValueError, "Review context does not match"):
            validate_review_context(context, transcript + " changed", draft)


if __name__ == "__main__":
    unittest.main()
