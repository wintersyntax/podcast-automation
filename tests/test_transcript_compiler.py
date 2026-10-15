import json
import tempfile
import unittest
from pathlib import Path

from compiler.transcript import (
    _align_source_metadata,
    _alignment_opcodes,
    _classify_difference,
    _numbers,
    build_resolver_batch,
    clean_transcript,
    compile_transcripts,
    main,
    tokenize,
    validate_resolver_response,
    write_outputs,
)


class TranscriptCompilerTests(unittest.TestCase):
    def test_clean_transcript_removes_apple_timestamps(self):
        source = "0:04\nHello world.\n1:02:03\nNext sentence."
        self.assertEqual(
            clean_transcript(source, remove_timestamps=True),
            "Hello world. Next sentence.",
        )

    def test_formatting_and_number_words_are_equivalent(self):
        result = compile_transcripts(
            "We tested twelve people. This works!",
            "we tested 12 people this works",
        )
        self.assertEqual(result.status, "pass")
        self.assertEqual(result.similarity, 1.0)
        self.assertEqual(result.differences, [])

    def test_sentence_boundaries_do_not_create_alignment_drift(self):
        result = compile_transcripts(
            "First thought. Second thought continues. The ending is stable.",
            "First thought second thought. Continues the ending is stable.",
            merge_gap=2,
        )
        self.assertGreater(result.similarity, 0.9)
        self.assertLessEqual(result.medium_risk, 1)

    def test_number_and_negation_changes_are_high_risk(self):
        result = compile_transcripts(
            "Take 5 grams because it does not reduce strength.",
            "Take 15 grams because it does reduce strength.",
            merge_gap=2,
        )
        kinds = {difference.kind for difference in result.differences}
        self.assertIn("number_mismatch", kinds)
        self.assertIn("negation_mismatch", kinds)
        self.assertEqual(result.high_risk, 2)

    def test_nocebo_is_not_treated_as_negation(self):
        result = compile_transcripts(
            "This can create a nocebo effect.",
            "This can create a no-cebo effect.",
        )
        self.assertFalse(
            any(item.kind == "negation_mismatch" for item in result.differences)
        )

    def test_attached_letters_and_numbers_are_tokenized_separately(self):
        result = compile_transcripts(
            "Read Review10 before continuing.",
            "Read Review 10 before continuing.",
        )
        self.assertEqual(result.similarity, 1.0)
        self.assertEqual(result.high_risk, 0)

    def test_long_omission_requires_review(self):
        missing = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima"
        result = compile_transcripts(
            f"Shared opening {missing} shared ending",
            "Shared opening shared ending",
        )
        self.assertEqual(result.medium_risk, 1)
        self.assertEqual(result.differences[0].kind, "whisper_omission")

    def test_consensus_preserves_content_only_whisper_contains(self):
        result = compile_transcripts(
            "The study was useful.",
            "The brand new clinically relevant study was useful.",
            primary="apple",
        )
        self.assertEqual(
            result.compiled_transcript,
            "The brand new clinically relevant study was useful.",
        )
        self.assertGreaterEqual(result.compiler_edits, 1)
        self.assertTrue(
            any(item.merge_action == "preserved_whisper_content" for item in result.differences)
        )

    def test_consensus_preserves_ordinary_short_whisper_addition(self):
        result = compile_transcripts(
            "With one study, publication bias is possible.",
            "You with one study, publication bias is possible.",
            primary="apple",
        )
        self.assertEqual(
            result.compiled_transcript,
            "You With one study, publication bias is possible.",
        )
        self.assertTrue(
            any(
                item.merge_action == "preserved_whisper_content"
                for item in result.differences
            )
        )

    def test_consensus_keeps_content_only_apple_contains(self):
        result = compile_transcripts(
            "The complete study was very useful.",
            "The study was useful.",
            primary="apple",
        )
        self.assertEqual(
            result.compiled_transcript, "The complete study was very useful."
        )

    def test_consensus_can_complete_a_whisper_primary_transcript(self):
        result = compile_transcripts(
            "The complete clinically relevant study was useful.",
            "The study was useful.",
            primary="whisper",
        )
        self.assertEqual(
            result.compiled_transcript,
            "The complete clinically relevant study was useful.",
        )
        self.assertGreaterEqual(result.compiler_edits, 1)

    def test_single_source_only_repetition_is_preserved_without_review_spam(self):
        result = compile_transcripts(
            "This is is a useful result.",
            "This is a useful result.",
            primary="apple",
        )
        self.assertEqual(result.compiled_transcript, "This is is a useful result.")
        self.assertTrue(
            any(
                item.merge_action == "preserved_apple_content"
                for item in result.differences
            )
        )

    def test_short_source_only_repetitions_are_preserved_without_review_spam(self):
        result = compile_transcripts(
            "Yes. Yes. Continue. Sorry, sorry, Alex.",
            "Yes. Continue. Sorry, Alex.",
            primary="apple",
        )
        self.assertEqual(
            result.compiled_transcript, "Yes. Yes. Continue. Sorry, sorry, Alex."
        )
        self.assertEqual(result.review_required, 0)

    def test_multiword_whisper_repetition_is_reviewed_not_declared_hallucination(self):
        result = compile_transcripts("Start finish.", "Start thank you thank you thank you finish.")
        difference = next(item for item in result.differences if item.source_only)
        self.assertEqual(
            difference.merge_action, "review_source_only_suspected_repetition"
        )
        self.assertTrue(difference.review_required)
        self.assertNotIn("confirmed_hallucination", result.to_dict().__str__())

    def test_filler_only_whisper_addition_is_not_merged(self):
        result = compile_transcripts(
            "I think this works.",
            "I um uh hmm think this works.",
            primary="whisper",
        )
        self.assertEqual(result.compiled_transcript, "I think this works.")
        self.assertTrue(
            any(
                item.merge_action == "discarded_whisper_confirmed_filler"
                and item.preservation_class == "discarded_confirmed_filler"
                for item in result.differences
            )
        )

    def test_source_only_high_risk_evidence_is_symmetric_and_awaits_review(self):
        spans = (
            "5 grams", "50%", "2024", "0.5%", "10 mg", "not recommended",
            "Eric Helms", "creatine monohydrate",
        )
        for source in ("apple", "whisper"):
            for primary in ("apple", "whisper", "auto"):
                for span in spans:
                    with self.subTest(source=source, primary=primary, span=span):
                        with_evidence = f"start {span} end"
                        without_evidence = "start end"
                        apple, whisper = (
                            (with_evidence, without_evidence)
                            if source == "apple"
                            else (without_evidence, with_evidence)
                        )
                        result = compile_transcripts(apple, whisper, primary=primary)
                        difference = next(item for item in result.differences if item.source_only)
                        self.assertEqual(difference.source_only_source, source)
                        self.assertEqual(difference.severity, "high")
                        self.assertTrue(difference.risk_reasons)
                        self.assertTrue(difference.review_required)
                        self.assertEqual(
                            difference.merge_action, "review_source_only_semantic_risk"
                        )
                        self.assertEqual(difference.preservation_class, "awaiting_review")
                        self.assertTrue(
                            span in result.compiled_transcript or difference.review_required
                        )

    def test_ordinary_short_source_only_evidence_is_symmetric_and_preserved(self):
        spans = ("for beginners", "in women", "at rest", "after training", "in practice")
        for source in ("apple", "whisper"):
            for primary in ("apple", "whisper", "auto"):
                for span in spans:
                    with self.subTest(source=source, primary=primary, span=span):
                        with_evidence = f"start {span} end"
                        without_evidence = "start end"
                        apple, whisper = (
                            (with_evidence, without_evidence)
                            if source == "apple"
                            else (without_evidence, with_evidence)
                        )
                        result = compile_transcripts(apple, whisper, primary=primary)
                        difference = next(item for item in result.differences if item.source_only)
                        self.assertEqual(difference.source_only_source, source)
                        self.assertEqual(difference.preservation_class, "preserved")
                        self.assertFalse(difference.review_required)
                        self.assertIn(span, result.compiled_transcript)
                        self.assertEqual(result.high_risk, 0)
                        self.assertEqual(result.medium_risk, 0)

    def test_whisper_source_only_respects_existing_acoustic_confidence_thresholds(self):
        text = "start ordinary evidence end"
        def metadata(avg_logprob, no_speech_prob):
            return {"segments": [{
                "avg_logprob": avg_logprob,
                "no_speech_prob": no_speech_prob,
                "words": [{"word": word} for word in text.split()],
            }]}

        confident = compile_transcripts(
            "start end", text, primary="apple",
            whisper_metadata=metadata(-0.2, 0.01),
        )
        confident_difference = next(item for item in confident.differences if item.source_only)
        self.assertEqual(confident_difference.preservation_class, "preserved")
        self.assertFalse(confident_difference.review_required)
        self.assertIn("ordinary evidence", confident.compiled_transcript)

        low_confidence = compile_transcripts(
            "start end", text, primary="apple",
            whisper_metadata=metadata(-1.2, 0.9),
        )
        low_difference = next(item for item in low_confidence.differences if item.source_only)
        self.assertEqual(
            low_difference.preservation_class,
            "preserved_low_whisper_confidence",
        )
        self.assertFalse(low_difference.review_required)
        self.assertIn("ordinary evidence", low_confidence.compiled_transcript)
        self.assertIn("low_whisper_acoustic_confidence", low_difference.risk_reasons)
        self.assertEqual(
            low_confidence.source_only_summary["low_confidence_only_preserved"],
            1,
        )
        self.assertEqual(
            build_resolver_batch(low_confidence, "low-confidence-source-only")[
                "batch"
            ]["diff_items"],
            [],
        )

    def test_numeric_and_unit_formatting_equivalence_is_proven_against_counterpart(self):
        for apple, whisper in (
            ("3,481", "34 81"),
            ("3481", "34 81"),
            ("1355 kcal", "13 55 kcals"),
            ("kcal", "kcals"),
            ("1760 to 2112", "seventeen sixty to twenty one twelve"),
        ):
            with self.subTest(apple=apple, whisper=whisper):
                result = compile_transcripts(apple, whisper)
                difference = result.differences[0]
                self.assertEqual(difference.kind, "numeric_unit_equivalent")
                self.assertFalse(difference.review_required)
                self.assertIn(result.compiled_transcript, {apple, whisper})

    def test_percentage_expression_alignment_is_deterministically_equivalent(self):
        for apple, whisper in (
            ("fat by 55 percent", "fat by 55%"),
            ("fat by 55 percentage", "fat by 55%"),
            ("fat by 55%", "fat by 55 percent"),
        ):
            with self.subTest(apple=apple, whisper=whisper):
                result = compile_transcripts(apple, whisper)
                self.assertEqual(result.review_required, 0)
                self.assertEqual(result.differences, [])

    def test_percentage_alignment_preserves_exact_selected_and_audit_source_text(self):
        apple = "fat by 55 percent"
        whisper = "fat by 55%"
        self.assertEqual(
            compile_transcripts(apple, whisper, primary="apple").compiled_transcript,
            apple,
        )
        self.assertEqual(
            compile_transcripts(apple, whisper, primary="whisper").compiled_transcript,
            whisper,
        )

        audited = compile_transcripts(
            "fat by 55 percent but 1", "fat by 55% but 2"
        )
        difference = audited.differences[0]
        self.assertIn("55 percent", difference.apple_context)
        self.assertIn("55%", difference.whisper_context)

    def test_percentage_equivalence_keeps_numeric_unit_and_neighbor_conflicts_safe(self):
        unsafe = (
            ("fat by 55", "fat by 55%"),
            ("fat by 55%", "fat by 60%"),
            ("dose is 25 kg", "dose is 25 lb"),
        )
        for apple, whisper in unsafe:
            with self.subTest(apple=apple, whisper=whisper):
                result = compile_transcripts(apple, whisper)
                self.assertGreater(result.review_required, 0)
        kilograms = compile_transcripts("dose is 25 kg", "dose is 25 lb")
        self.assertEqual(kilograms.differences[0].kind, "unit_mismatch")
        nearby = compile_transcripts(
            "fat by 55 percent today", "fat by 55% yesterday"
        )
        self.assertTrue(nearby.differences)
        self.assertTrue(any("today" in item.apple_text for item in nearby.differences))

    def test_ambiguous_numeric_grouping_and_residual_number_remain_reviewable(self):
        for apple, whisper in (
            ("800 to 960", "eight hundred and nine sixty"),
            ("day month is 3481", "30 day month is 3 4 8 1"),
        ):
            with self.subTest(apple=apple, whisper=whisper):
                difference = compile_transcripts(apple, whisper).differences[0]
                self.assertEqual(difference.kind, "number_mismatch")
                self.assertTrue(difference.review_required)

    def test_high_risk_low_confidence_whisper_only_content_is_reviewed(self):
        text = "start 5 grams end"
        result = compile_transcripts(
            "start end", text, primary="apple",
            whisper_metadata={"segments": [{
                "avg_logprob": -1.2,
                "no_speech_prob": 0.9,
                "words": [{"word": word} for word in text.split()],
            }]},
        )
        difference = next(item for item in result.differences if item.source_only)
        self.assertEqual(difference.severity, "high")
        self.assertTrue(difference.review_required)
        self.assertEqual(difference.preservation_class, "awaiting_review")

    def test_apple_only_ordinary_content_is_preserved(self):
        result = compile_transcripts(
            "start useful context end", "start end", primary="whisper"
        )
        difference = next(item for item in result.differences if item.source_only)
        self.assertEqual(difference.source_only_source, "apple")
        self.assertEqual(difference.preservation_class, "preserved")
        self.assertIn("useful context", result.compiled_transcript)

    def test_only_strong_hesitation_fillers_are_discarded(self):
        for span, discarded in (
            ("you know", False), ("well", False), ("I think", False),
            ("right", False), ("okay", False), ("uh", True), ("um", True),
            ("uh yeah", False), ("ER", False), ("MM", False), ("Ah", False),
        ):
            with self.subTest(span=span):
                result = compile_transcripts(
                    "start end", f"start {span} end", primary="apple"
                )
                difference = next(item for item in result.differences if item.source_only)
                if discarded:
                    self.assertEqual(
                        difference.preservation_class, "discarded_confirmed_filler"
                    )
                    self.assertNotIn(span, result.compiled_transcript)
                else:
                    self.assertEqual(difference.preservation_class, "preserved")
                    self.assertIn(span.casefold(), result.compiled_transcript.casefold())

    def test_high_risk_repetition_is_not_silently_removed_as_hallucination(self):
        for span in (
            "5 grams 5 grams 5 grams",
            "creatine monohydrate creatine monohydrate creatine monohydrate",
        ):
            with self.subTest(span=span):
                result = compile_transcripts("start end", f"start {span} end")
                difference = next(item for item in result.differences if item.source_only)
                self.assertTrue(difference.risk_reasons)
                self.assertEqual(difference.preservation_class, "awaiting_review")
                self.assertNotEqual(
                    difference.merge_action, "removed_whisper_hallucination"
                )

    def test_repetition_requires_review_except_strong_filler(self):
        filler = compile_transcripts("start end", "start uh uh uh end")
        filler_difference = next(item for item in filler.differences if item.source_only)
        self.assertEqual(filler_difference.preservation_class, "discarded_confirmed_filler")
        self.assertFalse(filler_difference.review_required)

        for span in (
            "that's the point that's the point that's the point",
            "this is important this is important this is important",
            "five grams five grams five grams",
            "creatine monohydrate creatine monohydrate creatine monohydrate",
        ):
            with self.subTest(span=span):
                result = compile_transcripts("start end", f"start {span} end")
                difference = next(item for item in result.differences if item.source_only)
                self.assertTrue(difference.review_required)
                self.assertEqual(difference.preservation_class, "awaiting_review")
                self.assertIn("suspected_repetition", difference.risk_reasons)

    def test_source_only_repetition_is_symmetric_between_apple_and_whisper(self):
        span = "this is important this is important this is important"
        for source in ("apple", "whisper"):
            with self.subTest(source=source):
                apple, whisper = (
                    (f"start {span} end", "start end")
                    if source == "apple" else ("start end", f"start {span} end")
                )
                difference = next(
                    item for item in compile_transcripts(apple, whisper).differences
                    if item.source_only
                )
                self.assertTrue(difference.review_required)
                self.assertEqual(difference.source_only_source, source)

    def test_known_and_unknown_proper_names_are_high_risk_source_only_evidence(self):
        for name in ("Eric Helms", "Dr. Smith", "Unlisted Person"):
            with self.subTest(name=name):
                result = compile_transcripts("start end", f"start {name} end")
                difference = next(item for item in result.differences if item.source_only)
                self.assertTrue(difference.review_required)
                self.assertIn("proper_name_or_entity", difference.risk_reasons)
                self.assertEqual(
                    difference.resolver_category,
                    "citation" if name == "Eric Helms" else "proper_name",
                )

    def test_representative_medical_terms_are_reviewed_but_unknown_terms_preserve(self):
        for term in ("metformin", "semaglutide", "testosterone", "insulin", "hypertension"):
            with self.subTest(term=term):
                result = compile_transcripts("start end", f"start {term} end")
                difference = next(item for item in result.differences if item.source_only)
                self.assertTrue(difference.review_required)
                self.assertEqual(difference.resolver_category, "scientific_medical_term")

        unknown = compile_transcripts("start end", "start unfamiliarbiofactor end")
        difference = next(item for item in unknown.differences if item.source_only)
        self.assertEqual(difference.preservation_class, "preserved")
        self.assertFalse(difference.review_required)

    def test_resolved_difference_is_not_left_marked_for_review(self):
        initial = compile_transcripts("Use 5 grams.", "Use 15 grams.")
        item = initial.differences[0]
        resolved = compile_transcripts(
            "Use 5 grams.", "Use 15 grams.",
            resolver_resolutions=[{"id": item.id, "source": "apple", "text": item.apple_text}],
        )
        self.assertFalse(resolved.differences[0].review_required)
        self.assertEqual(resolved.review_required, 0)
        self.assertEqual(resolved.status, "pass")

    def test_source_only_report_summary_and_difference_fields_are_auditable(self):
        result = compile_transcripts(
            "start uh bridge end", "start bridge 5 grams end",
            primary="apple", merge_gap=0,
        )
        payload = result.to_dict()
        self.assertEqual(payload["terminology_registry"]["schema_version"], 1)
        self.assertEqual(
            payload["terminology_registry"]["decision_effect"],
            "legacy_domain_glossary_compatibility_only",
        )
        self.assertFalse(payload["terminology_registry"]["authoritative"])
        self.assertEqual(payload["source_only_summary"]["apple_only"], 1)
        self.assertEqual(payload["source_only_summary"]["whisper_only"], 1)
        self.assertEqual(payload["source_only_summary"]["discarded_confirmed_noise"], 1)
        self.assertEqual(payload["source_only_summary"]["high_risk"], 1)
        self.assertEqual(payload["source_only_summary"]["review_required"], 1)
        self.assertTrue(all("preservation_class" in item for item in payload["differences"]))

    def test_spoken_numbers_cover_years_scales_decimals_and_fractions(self):
        self.assertEqual(_numbers([token.value for token in tokenize("twenty twenty four")]), ["2024"])
        self.assertEqual(_numbers([token.value for token in tokenize("two hundred and fifty")]), ["250"])
        self.assertEqual(_numbers([token.value for token in tokenize("a hundred and fifty")]), ["150"])
        self.assertEqual(_numbers([token.value for token in tokenize("ten point five")]), ["10.5"])
        self.assertEqual(_numbers([token.value for token in tokenize("two thirds")]), ["0.666667"])
        self.assertEqual(_numbers([token.value for token in tokenize("three and a half")]), ["3.5"])
        self.assertEqual(
            _numbers([token.value for token in tokenize("twenty two and a half")]),
            ["22.5"],
        )
        kind, severity, reason = _classify_difference(
            [token.value for token in tokenize("three and a half")],
            [token.value for token in tokenize("four and a half")],
        )
        self.assertEqual((kind, severity), ("number_mismatch", "high"))
        self.assertIn("['3.5']", reason)
        self.assertIn("['4.5']", reason)

    def test_auto_caption_punctuation_does_not_reduce_quality(self):
        apple = " ".join(f"word{index}" for index in range(20))
        whisper = " ".join(f"word{index}" for index in range(19)) + "."
        result = compile_transcripts(apple, whisper, apple_has_timestamps=False)
        self.assertEqual(result.recommended_source, "apple")

    def test_whisper_confidence_breaks_an_otherwise_ambiguous_tie(self):
        metadata = {
            "segments": [{
                "avg_logprob": -0.2,
                "no_speech_prob": 0.01,
                "words": [{"word": word} for word in "The blue result works".split()],
            }]
        }
        result = compile_transcripts(
            "The red result works.",
            "The blue result works.",
            primary="apple",
            whisper_metadata=metadata,
        )
        self.assertEqual(result.compiled_transcript, "The blue result works.")
        self.assertTrue(
            any(item.merge_action == "preferred_confident_whisper" for item in result.differences)
        )

    def test_measurement_unit_mismatch_is_high_risk(self):
        result = compile_transcripts(
            "Use 25 kilograms for this lift.",
            "Use 25 pounds for this lift.",
        )
        difference = result.differences[0]
        self.assertEqual(difference.kind, "unit_mismatch")
        self.assertEqual(difference.severity, "high")
        self.assertEqual(difference.resolver_category, "unit")

    def test_domain_glossary_promotes_exercise_name_difference(self):
        result = compile_transcripts(
            "Use a Bulgarian split squat for the accessory work.",
            "Use a Bulgarian split squot for the accessory work.",
        )
        difference = result.differences[0]
        self.assertEqual(difference.kind, "domain_term_mismatch")
        self.assertEqual(difference.severity, "medium")
        self.assertIn("Bulgarian split squat", difference.domain_terms)
        self.assertEqual(difference.resolver_category, "exercise_name")

    def test_citation_year_difference_is_routed_to_citation_review(self):
        result = compile_transcripts(
            "The 2019 study supports the intervention.",
            "The 2020 study supports the intervention.",
        )
        difference = result.differences[0]
        self.assertTrue(difference.citation_signal)
        self.assertEqual(difference.resolver_category, "citation")

    def test_researcher_name_near_miss_is_routed_to_citation_review(self):
        result = compile_transcripts(
            "Brad Schoenfeld recommends sufficient training volume.",
            "Brad Schoonfeld recommends sufficient training volume.",
        )
        difference = result.differences[0]
        self.assertEqual(difference.resolver_category, "citation")
        self.assertIn("Brad Schoenfeld", difference.domain_terms)

    def test_nearby_supplement_does_not_reclassify_unrelated_contraction(self):
        result = compile_transcripts(
            "Creatine works because it is inexpensive.",
            "Creatine works cause it is inexpensive.",
        )

        difference = result.differences[0]

        self.assertEqual(
            difference.resolver_category,
            "other",
        )

        self.assertNotEqual(
            difference.kind,
            "domain_term_mismatch",
        )


    def test_nearby_citation_does_not_reclassify_unrelated_wording(self):
        result = compile_transcripts(
            "The 2017 study was going to say this.",
            "The 2017 study was gonna say this.",
        )

        difference = result.differences[0]

        self.assertEqual(
            difference.resolver_category,
            "other",
        )

        self.assertFalse(
            difference.citation_signal
        )


    def test_next_one_is_not_treated_as_protocol_number(self):
        self.assertEqual(
            _numbers(
                [
                    "i",
                    "will",
                    "see",
                    "you",
                    "in",
                    "the",
                    "next",
                    "1",
                ]
            ),
            [],
        )

        self.assertEqual(
            _numbers(
                [
                    "take",
                    "1",
                    "gram",
                ]
            ),
            ["1"],
        )


    def test_resolver_batch_includes_only_anonymous_source_quality_metadata(self):
        metadata = {
            "segments": [{
                "start": 1.0,
                "end": 2.0,
                "avg_logprob": -0.2,
                "no_speech_prob": 0.01,
                "words": [
                    {
                        "word": "Use",
                        "start": 1.0,
                        "end": 1.1,
                        "probability": 0.99,
                    },
                    {
                        "word": "25",
                        "start": 1.1,
                        "end": 1.2,
                        "probability": 0.99,
                    },
                    {
                        "word": "pounds",
                        "start": 1.2,
                        "end": 2.0,
                        "probability": 0.99,
                    },
                    {"word": "The"},
                    {"word": "preferred"},
                    {"word": "unit"},
                    {"word": "remains"},
                    {"word": "kilograms"},
                ],
            }],
        }

        result = compile_transcripts(
            "Use 25 kilograms. The preferred unit remains kilograms.",
            "Use 25 pounds. The preferred unit remains kilograms.",
            whisper_metadata=metadata,
        )

        payload = build_resolver_batch(
            result,
            "episode-metadata",
        )

        item = payload["batch"]["diff_items"][0]

        self.assertIn("severity", item)
        self.assertIn("changed_source_a_words", item)
        self.assertIn("changed_source_b_words", item)
        self.assertIn("source_a_metadata", item)
        self.assertIn("source_b_metadata", item)

        # Provider identity is retained only in the local source_mappings table,
        # never in model-facing field names or risk labels.
        serialized_item = json.dumps(item)
        for provider_key in (
            "whisper_avg_logprob",
            "whisper_no_speech_prob",
            "whisper_start_timestamp",
            "whisper_end_timestamp",
            "apple_start_timestamp",
            "apple_end_timestamp",
            "low_whisper_acoustic_confidence",
        ):
            self.assertNotIn(provider_key, serialized_item)

        mapping = payload["source_mappings"][str(item["id"])]
        whisper_side = (
            "source_a" if mapping["source_a"] == "whisper" else "source_b"
        )
        apple_side = "source_b" if whisper_side == "source_a" else "source_a"

        self.assertEqual(item[f"{whisper_side}_metadata"]["start_timestamp"], 1.2)
        self.assertEqual(item[f"{whisper_side}_metadata"]["end_timestamp"], 2.0)
        self.assertIsNone(item[f"{apple_side}_metadata"]["start_timestamp"])
        self.assertIsNone(item[f"{apple_side}_metadata"]["end_timestamp"])
        self.assertNotIn("avg_logprob", item[f"{whisper_side}_metadata"])
        self.assertNotIn("no_speech_prob", item[f"{whisper_side}_metadata"])


    def test_resolver_focuses_domain_term_inside_larger_difference(self):
        apple = (
            "Sleep loss means you are going to have messed up RPE today. "
            "The measure remains RPE."
        )

        whisper = (
            "Sleep loss means you are gonna have messed up RP today. "
            "The measure remains RPE."
        )

        result = compile_transcripts(
            apple,
            whisper,
            primary="whisper",
        )

        payload = build_resolver_batch(
            result,
            "focus-test",
        )

        item = next(
            item
            for item in payload["batch"]["diff_items"]
            if item["category"] == "training_term"
        )

        self.assertEqual(
            item["focus_scope"],
            "partial",
        )

        self.assertEqual(
            item["focus_source_a_text"],
            "RPE",
        )

        self.assertEqual(
            item["focus_source_b_text"],
            "RP",
        )

        validated = validate_resolver_response(
            payload,
            {
                "resolutions": [
                    {
                        "id": item["id"],
                        "resolved_value": "source_a",
                        "corrected_text": "RPE",
                        "category": "training_term",
                        "confidence": "high",
                        "flag_for_human": False,
                    }
                ]
            },
        )

        self.assertEqual(
            len(
                validated["accepted"]
            ),
            1,
        )

        recompiled = compile_transcripts(
            apple,
            whisper,
            primary="whisper",
            resolver_resolutions=(
                validated["accepted"]
            ),
        )

        self.assertIn(
            "gonna have messed up RPE",
            recompiled.compiled_transcript,
        )

        self.assertNotIn(
            "going to have messed up RPE",
            recompiled.compiled_transcript,
        )

        self.assertTrue(
            any(
                difference.merge_action
                == "resolver_validated_focus_choice"
                for difference
                in recompiled.differences
            )
        )


    def test_is_contraction_domain_difference_is_not_sent_to_resolver(self):
        result = compile_transcripts(
            "Creatine is useful here.",
            "Creatine's useful here.",
        )

        payload = build_resolver_batch(
            result,
            "contraction-test",
        )

        self.assertFalse(
            any(
                item["category"]
                == "supplement"
                for item
                in payload["batch"]["diff_items"]
            )
        )


    def test_review_context_is_wider_than_classification_window(self):
        prefix = (
            "alpha bravo charlie delta echo foxtrot golf hotel india juliet "
            "kilo lima mike november oscar papa quebec romeo sierra tango "
            "uniform victor whiskey xray yankee"
        )

        apple = (
            f"{prefix} "
            "you are going to have messed up RPE today and RPE again"
        )

        whisper = (
            f"{prefix} "
            "you are gonna have messed up RP today and RPE again"
        )

        result = compile_transcripts(
            apple,
            whisper,
        )

        payload = build_resolver_batch(
            result,
            "context-test",
        )

        item = next(
            item
            for item in payload["batch"]["diff_items"]
            if item["category"] == "training_term"
        )

        self.assertIn(
            "alpha",
            item["source_a_review_context"],
        )

        self.assertIn(
            "alpha",
            item["source_b_review_context"],
        )

        self.assertIn(
            "RPE",
            item["source_a_review_context"],
        )

        self.assertIn(
            "RP",
            item["source_b_review_context"],
        )

        self.assertEqual(
            item["category"],
            "training_term",
        )


    def test_resolver_batch_is_limited_to_high_value_differences(self):
        result = compile_transcripts(
            "Use 25 kilograms for this Bulgarian split squat. Bulgarian split squat uses kilograms.",
            "Use 25 pounds for this Bulgarian split squot. Bulgarian split squat uses kilograms.",
        )
        payload = build_resolver_batch(result, "episode-42")
        items = payload["batch"]["diff_items"]
        self.assertEqual(payload["batch"]["episode_id"], "episode-42")
        self.assertEqual({item["category"] for item in items}, {"unit", "exercise_name"})
        self.assertTrue(all("compiled_transcript" not in item for item in items))

    def test_high_risk_source_only_evidence_bypasses_ai_without_corroboration(self):
        result = compile_transcripts(
            "Start the protocol now.",
            "Start the 5 grams protocol now.",
        )

        payload = build_resolver_batch(result, "source-only-dose")

        self.assertEqual(payload["batch"]["diff_items"], [])
        difference = next(item for item in result.differences if item.source_only)
        self.assertTrue(difference.review_required)
        self.assertEqual(
            payload["bypassed_items"],
            [{
                "id": difference.id,
                "reason": "structurally_ineligible_no_acceptance_path",
            }],
        )

    def test_resolver_preflight_keeps_every_source_choice_with_an_acceptance_path(self):
        result = compile_transcripts(
            "Start use 5 grams now. The protocol uses 5 grams.",
            "Start use 6 grams now. The protocol uses 5 grams.",
        )
        payload = build_resolver_batch(result, "corroborated-dose")

        self.assertEqual(payload["bypassed_items"], [])
        self.assertEqual(len(payload["batch"]["diff_items"]), 1)
        item = payload["batch"]["diff_items"][0]
        accepted_side = item["textually_supported_sides"][0]
        accepted = validate_resolver_response(
            payload,
            {
                "resolutions": [{
                    "id": item["id"],
                    "resolved_value": accepted_side,
                    "corrected_text": item[f"focus_{accepted_side}_text"],
                    "category": item["category"],
                    "confidence": "high",
                    "flag_for_human": False,
                }],
            },
        )

        self.assertEqual(len(accepted["accepted"]), 1)
        self.assertEqual(accepted["review"], [])

    def test_episode_local_memory_shadow_finds_prior_aligned_confirmation(self):
        result = compile_transcripts(
            "Eric Helms explained the model. Later Eric Holmes discussed volume.",
            "Eric Helms explained the model. Later Eric Helms discussed volume.",
            merge_gap=0,
        )

        memory = result.episode_local_memory
        self.assertEqual(memory["mode"], "shadow")
        self.assertFalse(memory["authoritative"])
        self.assertEqual(memory["decision_effect"], "none")
        self.assertEqual(memory["one_sided_signal_count"], 1)
        signal = memory["signals"][0]
        self.assertEqual(signal["category"], "citation")
        self.assertEqual(signal["signal"], "one_side_confirmed")
        self.assertEqual(signal["confirmed_sources"], ["whisper"])
        whisper_form = next(
            form for form in signal["forms"] if form["source"] == "whisper"
        )
        self.assertTrue(whisper_form["episode_confirmed"])
        self.assertTrue(whisper_form["prior_confirmed"])
        self.assertEqual(whisper_form["occurrences"][0]["relation"], "before")

    def test_episode_local_memory_shadow_does_not_change_canonical_semantics(self):
        args = (
            "Use 5 grams for the protocol. Later use 6 grams for the protocol.",
            "Use 5 grams for the protocol. Later use 5 grams for the protocol.",
        )
        shadow = compile_transcripts(*args, merge_gap=0)
        disabled = compile_transcripts(
            *args,
            merge_gap=0,
            episode_local_memory_shadow=False,
        )

        self.assertEqual(shadow.compiled_transcript, disabled.compiled_transcript)
        self.assertEqual(shadow.status, disabled.status)
        self.assertEqual(shadow.review_required, disabled.review_required)
        self.assertEqual(
            [
                (item.id, item.selected_source, item.selected_text, item.merge_action, item.review_required)
                for item in shadow.differences
            ],
            [
                (item.id, item.selected_source, item.selected_text, item.merge_action, item.review_required)
                for item in disabled.differences
            ],
        )
        self.assertEqual(disabled.episode_local_memory["mode"], "disabled")

    def test_famous_name_source_only_evidence_bypasses_ai(self):
        result = compile_transcripts(
            "The guest joined us.",
            "Eric Helms the guest joined us.",
        )

        payload = build_resolver_batch(result, "source-only-name")

        self.assertEqual(payload["batch"]["diff_items"], [])

    def test_glossary_correction_must_be_item_allowed_and_two_sided(self):
        batch = {
            "source_mappings": {},
            "batch": {
                "diff_items": [{
                    "id": 7,
                    "category": "supplement",
                    "focus_scope": "full",
                    "source_a_text": "creatin monohydrate",
                    "source_b_text": "creatine monohydrat",
                    "allowed_glossary_candidates": ["creatine monohydrate"],
                    "textually_supported_glossary_candidates": ["creatine monohydrate"],
                    "requires_textual_corroboration": False,
                    "source_only": False,
                }],
            },
        }

        accepted = validate_resolver_response(
            batch,
            {"resolutions": [{
                "id": 7,
                "resolved_value": "neither",
                "corrected_text": "creatine monohydrate",
                "category": "supplement",
                "confidence": "high",
                "flag_for_human": False,
            }]},
        )
        invented = validate_resolver_response(
            batch,
            {"resolutions": [{
                "id": 7,
                "resolved_value": "neither",
                "corrected_text": "creatine hydrochloride",
                "category": "supplement",
                "confidence": "high",
                "flag_for_human": False,
            }]},
        )

        self.assertEqual(accepted["accepted"][0]["source"], "glossary")
        self.assertEqual(invented["review"][0]["reason"], "unresolved_or_unsupported")

    def test_anonymous_source_mapping_preserves_the_actual_choice(self):
        batch = {
            "source_mappings": {"8": {"source_a": "whisper", "source_b": "apple"}},
            "batch": {
                "diff_items": [{
                    "id": 8,
                    "category": "training_term",
                    "focus_scope": "full",
                    "source_a_text": "RP",
                    "source_b_text": "RPE",
                    "allowed_glossary_candidates": [],
                    "textually_supported_glossary_candidates": [],
                    "requires_textual_corroboration": False,
                    "source_only": False,
                }],
            },
        }

        resolved = validate_resolver_response(
            batch,
            {"resolutions": [{
                "id": 8,
                "resolved_value": "source_b",
                "corrected_text": "RPE",
                "category": "training_term",
                "confidence": "high",
                "flag_for_human": False,
            }]},
        )

        self.assertEqual(resolved["accepted"][0]["source"], "apple")

    def test_resolver_response_rejects_unsupported_quantitative_choice(self):
        payload = {
            "source_mappings": {"7": {"source_a": "apple", "source_b": "whisper"}},
            "batch": {"diff_items": [{
                "id": 7,
                "category": "unit",
                "focus_scope": "full",
                "source_a_text": "25 kilograms",
                "source_b_text": "25 pounds",
                "requires_textual_corroboration": True,
                "textually_supported_sides": [],
                "source_only": False,
                "allowed_glossary_candidates": [],
                "textually_supported_glossary_candidates": [],
            }]},
        }
        rejected = validate_resolver_response(
            payload,
            {
                "resolutions": [{
                    "id": 7,
                    "resolved_value": "source_a",
                    "corrected_text": "25 kilograms",
                    "category": "unit",
                    "confidence": "high",
                    "flag_for_human": False,
                }]
            },
        )
        self.assertEqual(
            rejected["review"][0]["reason"],
            "unsupported_textual_evidence",
        )

    def test_difference_records_apple_timestamp_hints(self):
        result = compile_transcripts(
            "0:10\nStart stable.\n0:20\nWrong word here.",
            "Start stable. Right word here.",
            primary="apple",
        )
        difference = result.differences[0]
        self.assertEqual(difference.apple_start_timestamp, 20.0)
        self.assertEqual(difference.apple_end_timestamp, 20.0)

    def test_difference_records_whisper_word_timestamp_hints(self):
        metadata = {
            "segments": [{
                "start": 10.0,
                "end": 13.0,
                "words": [
                    {"word": "Start", "start": 10.0, "end": 10.4},
                    {"word": "right", "start": 10.4, "end": 10.8},
                    {"word": "result", "start": 10.8, "end": 11.3},
                ],
            }],
        }
        result = compile_transcripts(
            "Start wrong result.",
            "Start right result.",
            whisper_metadata=metadata,
        )
        difference = result.differences[0]
        self.assertEqual(difference.whisper_start_timestamp, 10.4)
        self.assertEqual(difference.whisper_end_timestamp, 10.8)

    def test_faster_whisper_metadata_shape_supplies_confidence_and_timestamps(self):
        metadata = {
            "source": "faster-whisper",
            "segments": [{
                "start": 10.0,
                "end": 13.0,
                "avg_logprob": -0.2,
                "no_speech_prob": 0.01,
                "words": [
                    {"word": "Start", "start": 10.0, "end": 10.4, "probability": 0.99},
                    {"word": "right", "start": 10.4, "end": 10.8, "probability": 0.99},
                    {"word": "result", "start": 10.8, "end": 11.3, "probability": 0.99},
                ],
            }],
        }
        result = compile_transcripts(
            "Start wrong result.",
            "Start right result.",
            primary="apple",
            whisper_metadata=metadata,
        )
        difference = result.differences[0]
        self.assertEqual(difference.whisper_start_timestamp, 10.4)
        self.assertEqual(difference.whisper_end_timestamp, 10.8)
        self.assertEqual(difference.whisper_avg_logprob, -0.2)

    def test_openrouter_whisper_metadata_shape_uses_segment_confidence(self):
        # TASK-125: OpenRouter words carry no per-word probability, so the
        # compiler must fall back to the owning segment's avg_logprob.
        metadata = {
            "source": "openrouter-whisper",
            "segments": [{
                "start": 10.0,
                "end": 13.0,
                "avg_logprob": -0.35,
                "no_speech_prob": 0.02,
                "words": [
                    {"word": " Start", "start": 10.0, "end": 10.4},
                    {"word": " right", "start": 10.4, "end": 10.8},
                    {"word": " result.", "start": 10.8, "end": 11.3},
                ],
            }],
        }
        result = compile_transcripts(
            "Start wrong result.",
            "Start right result.",
            primary="apple",
            whisper_metadata=metadata,
        )
        difference = result.differences[0]
        self.assertEqual(difference.whisper_start_timestamp, 10.4)
        self.assertEqual(difference.whisper_end_timestamp, 10.8)
        self.assertEqual(difference.whisper_avg_logprob, -0.35)
        self.assertEqual(difference.whisper_no_speech_prob, 0.02)

    def test_metadata_alignment_merges_decimal_word_splits(self):
        metadata = {
            "segments": [{
                "words": [
                    {
                        "word": "Dose",
                        "start": 0.0,
                        "end": 0.4,
                        "probability": 0.99,
                    },
                    {
                        "word": " 1",
                        "start": 0.4,
                        "end": 0.6,
                        "probability": 0.98,
                    },
                    {
                        "word": ".4",
                        "start": 0.6,
                        "end": 0.8,
                        "probability": 0.97,
                    },
                    {
                        "word": " grams",
                        "start": 0.8,
                        "end": 1.2,
                        "probability": 0.99,
                    },
                ],
            }],
        }

        tokens = tokenize(
            "Dose 1.4 grams"
        )

        (
            logprobs,
            no_speech,
            starts,
            ends,
        ) = _align_source_metadata(
            tokens,
            metadata,
        )

        self.assertEqual(
            len(logprobs),
            len(tokens),
        )

        self.assertIsNotNone(
            logprobs[1]
        )

        self.assertEqual(
            starts[1],
            0.4,
        )

        self.assertEqual(
            ends[1],
            0.8,
        )

        self.assertEqual(
            starts[2],
            0.8,
        )


    def test_metadata_alignment_keeps_good_spans_around_one_mismatch(self):
        transcript_words = (
            "alpha bravo charlie delta echo foxtrot golf hotel india juliet "
            "kilo lima mike november oscar papa quebec romeo sierra tango"
        ).split()

        metadata_words = (
            transcript_words.copy()
        )

        metadata_words[10] = (
            "wrong"
        )

        metadata = {
            "segments": [{
                "words": [
                    {
                        "word": word,
                        "start": float(index),
                        "end": float(index) + 0.5,
                    }
                    for index, word
                    in enumerate(
                        metadata_words
                    )
                ],
            }],
        }

        tokens = tokenize(
            " ".join(
                transcript_words
            )
        )

        (
            _,
            _,
            starts,
            ends,
        ) = _align_source_metadata(
            tokens,
            metadata,
        )

        self.assertEqual(
            starts[0],
            0.0,
        )

        self.assertIsNone(
            starts[10]
        )

        self.assertIsNone(
            ends[10]
        )

        self.assertEqual(
            starts[11],
            11.0,
        )

        self.assertEqual(
            ends[-1],
            19.5,
        )


    def test_metadata_alignment_discards_low_coverage_mapping(self):
        metadata = {
            "segments": [{
                "words": [
                    {
                        "word": word,
                        "start": index,
                        "end": index + 0.5,
                    }
                    for index, word
                    in enumerate(
                        "one two three four five".split()
                    )
                ],
            }],
        }

        tokens = tokenize(
            "alpha bravo charlie delta echo"
        )

        (
            logprobs,
            no_speech,
            starts,
            ends,
        ) = _align_source_metadata(
            tokens,
            metadata,
        )

        self.assertTrue(
            all(
                value is None
                for value in logprobs
            )
        )

        self.assertTrue(
            all(
                value is None
                for value in no_speech
            )
        )

        self.assertTrue(
            all(
                value is None
                for value in starts
            )
        )

        self.assertTrue(
            all(
                value is None
                for value in ends
            )
        )


    def test_long_transcripts_use_anchor_window_alignment(self):
        apple = [f"token{index}" for index in range(6_100)]
        whisper = apple[:3_000] + ["inserted", "words"] + apple[3_000:]
        opcodes = _alignment_opcodes(apple, whisper)
        self.assertTrue(any(tag == "insert" for tag, *_ in opcodes))

    def test_consensus_splits_obvious_run_on_word(self):
        result = compile_transcripts(
            "This is a healthpromoting intervention.",
            "This is a health promoting intervention.",
            primary="apple",
        )
        self.assertEqual(
            result.compiled_transcript, "This is a health promoting intervention."
        )
        self.assertTrue(
            any(item.merge_action == "split_run_on_word" for item in result.differences)
        )

    def test_consensus_does_not_guess_number_conflicts(self):
        result = compile_transcripts(
            "Participants took 5 grams daily.",
            "Participants took 15 grams daily.",
            primary="apple",
        )
        self.assertEqual(result.compiled_transcript, "Participants took 5 grams daily.")
        difference = result.differences[0]
        self.assertEqual(difference.merge_action, "review_kept_primary")
        self.assertEqual(difference.selected_source, "apple")

    def test_outputs_include_json_and_primary_transcript(self):
        result = compile_transcripts(
            "Apple has the complete sentence here.",
            "Apple has complete sentence here.",
            primary="apple",
        )
        with tempfile.TemporaryDirectory() as directory:
            paths = write_outputs(
                result,
                output_dir=directory,
                apple_path="apple.txt",
                whisper_path="whisper.txt",
            )
            payload = json.loads(paths["json"].read_text(encoding="utf-8"))
            self.assertEqual(payload["recommended_source"], "apple")
            self.assertIn("differences", payload)
            self.assertIn("source_only_summary", payload)
            self.assertEqual(
                paths["transcript"].read_text(encoding="utf-8").strip(),
                "Apple has the complete sentence here.",
            )

    def test_cli_returns_two_when_fail_on_high_is_enabled(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            apple = root / "apple.txt"
            whisper = root / "whisper.txt"
            apple.write_text("Use 5 grams for this test.", encoding="utf-8")
            whisper.write_text("Use 50 grams for this test.", encoding="utf-8")
            exit_code = main([
                str(apple), str(whisper), "--output-dir", str(root / "out"),
                "--fail-on", "high",
            ])
            self.assertEqual(exit_code, 2)


if __name__ == "__main__":
    unittest.main()
