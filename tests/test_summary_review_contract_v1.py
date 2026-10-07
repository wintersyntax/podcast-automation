import json
import unittest
from pathlib import Path

from podcast_engine.knowledge.summary_review_contract_v1 import (
    MAX_EVIDENCE_CHARS,
    REVIEW_CONTRACT_ID,
    SUMMARY_REVIEW_RESPONSE_SCHEMA,
    parse_review_content,
    validate_review_result,
    validate_review_shape,
)

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_CONFIG = ROOT / "config" / "summary-review-benchmark-v1.json"
BENCHMARK_PROMPT = ROOT / "config" / "summary-review-benchmark-v1-prompt.txt"

DRAFT = """## TL;DR\n\n- Supported point.\n\n## Key Ideas\n\n### Topic\n\nSupported point.\n"""
TRANSCRIPT = "The guest said this supported point and explained why it mattered."
REVISED = DRAFT.replace("Supported point.", "Supported point with context.", 1)


class FrozenV1ContractTests(unittest.TestCase):
    def test_frozen_v1_contract_identity(self):
        self.assertEqual(REVIEW_CONTRACT_ID, "summary-review-v1")


def issue(
    issue_type="other",
    *,
    severity="minor",
    draft_excerpt="Supported point.",
    transcript_evidence=None,
    resolution="Corrected the reviewed draft.",
):
    return {
        "type": issue_type,
        "severity": severity,
        "draft_excerpt": draft_excerpt,
        "transcript_evidence": transcript_evidence,
        "resolution": resolution,
    }


class SummaryReviewContractTests(unittest.TestCase):
    def test_schema_is_strict(self):
        self.assertFalse(SUMMARY_REVIEW_RESPONSE_SCHEMA["additionalProperties"])
        self.assertEqual(
            SUMMARY_REVIEW_RESPONSE_SCHEMA["required"],
            ["status", "issues_found", "final_markdown"],
        )

    def test_pass_is_issue_free_and_model_final_is_null(self):
        self.assertEqual(
            validate_review_result(
                {"status": "pass", "issues_found": [], "final_markdown": None},
                DRAFT,
                TRANSCRIPT,
            )["status"],
            "pass",
        )
        with self.assertRaisesRegex(ValueError, "pass.*null"):
            validate_review_result(
                {
                    "status": "pass",
                    "issues_found": [],
                    "final_markdown": DRAFT,
                },
                DRAFT,
                TRANSCRIPT,
            )
        with self.assertRaisesRegex(ValueError, "pass.*zero issues"):
            validate_review_result(
                {
                    "status": "pass",
                    "issues_found": [issue()],
                    "final_markdown": None,
                },
                DRAFT,
                TRANSCRIPT,
            )

    def test_revised_requires_changed_final_and_at_least_one_issue(self):
        result = validate_review_result(
            {
                "status": "revised",
                "issues_found": [issue()],
                "final_markdown": REVISED,
            },
            DRAFT,
            TRANSCRIPT,
        )
        self.assertEqual(result["status"], "revised")

        with self.assertRaisesRegex(ValueError, "revised.*at least one issue"):
            validate_review_result(
                {"status": "revised", "issues_found": [], "final_markdown": REVISED},
                DRAFT,
                TRANSCRIPT,
            )
        with self.assertRaisesRegex(ValueError, "revised.*changed"):
            validate_review_result(
                {
                    "status": "revised",
                    "issues_found": [issue()],
                    "final_markdown": DRAFT,
                },
                DRAFT,
                TRANSCRIPT,
            )

    def test_fail_requires_null_final(self):
        self.assertEqual(
            validate_review_result(
                {"status": "fail", "issues_found": [], "final_markdown": None},
                DRAFT,
                TRANSCRIPT,
            )["status"],
            "fail",
        )
        with self.assertRaisesRegex(ValueError, "fail.*null"):
            validate_review_result(
                {"status": "fail", "issues_found": [], "final_markdown": DRAFT},
                DRAFT,
                TRANSCRIPT,
            )

    def test_parser_requires_json_object(self):
        parsed = parse_review_content(
            '{"status":"fail","issues_found":[],"final_markdown":null}'
        )
        self.assertEqual(parsed["status"], "fail")
        with self.assertRaises((json.JSONDecodeError, ValueError)):
            parse_review_content("not json")
        with self.assertRaisesRegex(ValueError, "JSON object"):
            parse_review_content("[]")

    def test_shape_validation_allows_cross_field_semantic_failure(self):
        payload = {
            "status": "pass",
            "issues_found": [],
            "final_markdown": DRAFT,
        }

        self.assertIs(validate_review_shape(payload), payload)

        with self.assertRaisesRegex(ValueError, "final_markdown"):
            validate_review_shape(
                {
                    "status": "pass",
                    "issues_found": [],
                    "final_markdown": 42,
                }
            )

        with self.assertRaisesRegex(ValueError, "pass.*null"):
            validate_review_result(payload, DRAFT, TRANSCRIPT)

    def test_coverage_omission_requires_transcript_evidence_but_not_draft_excerpt(self):
        value = issue(
            "coverage_omission",
            draft_excerpt=None,
            transcript_evidence="explained why it mattered",
        )
        validate_review_result(
            {"status": "revised", "issues_found": [value], "final_markdown": REVISED},
            DRAFT,
            TRANSCRIPT,
        )
        value["transcript_evidence"] = None
        with self.assertRaisesRegex(ValueError, "transcript_evidence"):
            validate_review_result(
                {"status": "revised", "issues_found": [value], "final_markdown": REVISED},
                DRAFT,
                TRANSCRIPT,
            )

    def test_semantic_issues_require_both_exact_excerpts(self):
        for issue_type in (
            "unsupported_claim",
            "unsupported_precision",
            "unsupported_term_normalization",
            "epistemic_drift",
            "causal_overstatement",
            "recommendation_drift",
        ):
            with self.subTest(issue_type=issue_type):
                validate_review_result(
                    {
                        "status": "revised",
                        "issues_found": [
                            issue(
                                issue_type,
                                transcript_evidence="supported point",
                            )
                        ],
                        "final_markdown": REVISED,
                    },
                    DRAFT,
                    TRANSCRIPT,
                )

                bad = issue(issue_type, transcript_evidence="not in transcript")
                with self.assertRaisesRegex(ValueError, "transcript_evidence"):
                    validate_review_result(
                        {"status": "revised", "issues_found": [bad], "final_markdown": REVISED},
                        DRAFT,
                        TRANSCRIPT,
                    )

    def test_unsupported_claim_absent_requires_draft_excerpt_and_null_transcript_evidence(self):
        absent = issue(
            "unsupported_claim_absent",
            draft_excerpt="Supported point.",
            transcript_evidence=None,
            resolution="Removed a claim with no support anywhere in the transcript.",
        )
        validate_review_result(
            {"status": "revised", "issues_found": [absent], "final_markdown": REVISED},
            DRAFT,
            TRANSCRIPT,
        )

        for field, value in (
            ("draft_excerpt", None),
            ("transcript_evidence", "supported point"),
        ):
            with self.subTest(field=field):
                invalid = dict(absent)
                invalid[field] = value
                with self.assertRaisesRegex(ValueError, field):
                    validate_review_result(
                        {
                            "status": "revised",
                            "issues_found": [invalid],
                            "final_markdown": REVISED,
                        },
                        DRAFT,
                        TRANSCRIPT,
                    )

    def test_structural_issues_require_draft_excerpt_only(self):
        for issue_type in ("redundancy", "structure_violation"):
            with self.subTest(issue_type=issue_type):
                validate_review_result(
                    {
                        "status": "revised",
                        "issues_found": [issue(issue_type)],
                        "final_markdown": REVISED,
                    },
                    DRAFT,
                    TRANSCRIPT,
                )
                bad = issue(issue_type, draft_excerpt=None)
                with self.assertRaisesRegex(ValueError, "draft_excerpt"):
                    validate_review_result(
                        {"status": "revised", "issues_found": [bad], "final_markdown": REVISED},
                        DRAFT,
                        TRANSCRIPT,
                    )

    def test_other_requires_at_least_one_evidence_excerpt(self):
        validate_review_result(
            {
                "status": "revised",
                "issues_found": [issue("other")],
                "final_markdown": REVISED,
            },
            DRAFT,
            TRANSCRIPT,
        )
        with self.assertRaisesRegex(ValueError, "at least one"):
            validate_review_result(
                {
                    "status": "revised",
                    "issues_found": [issue("other", draft_excerpt=None, transcript_evidence=None)],
                    "final_markdown": REVISED,
                },
                DRAFT,
                TRANSCRIPT,
            )

    def test_excerpt_membership_uses_normalized_newlines(self):
        draft = DRAFT.replace("\n", "\r\n")
        transcript = "First line.\r\nSecond line."
        final = REVISED.replace("\n", "\r\n")
        validate_review_result(
            {
                "status": "revised",
                "issues_found": [
                    issue(
                        "unsupported_claim",
                        draft_excerpt="Supported point.",
                        transcript_evidence="First line.\nSecond line.",
                    )
                ],
                "final_markdown": final,
            },
            draft,
            transcript,
        )

    def test_excerpt_length_is_bounded(self):
        oversized = "x" * (MAX_EVIDENCE_CHARS + 1)
        draft = oversized + "\n" + DRAFT
        with self.assertRaisesRegex(ValueError, str(MAX_EVIDENCE_CHARS)):
            validate_review_result(
                {
                    "status": "revised",
                    "issues_found": [issue("other", draft_excerpt=oversized)],
                    "final_markdown": REVISED,
                },
                draft,
                TRANSCRIPT,
            )

    def test_issue_fields_type_severity_and_resolution_are_strict(self):
        unknown_field = issue()
        unknown_field["extra"] = True
        with self.assertRaisesRegex(ValueError, "issue fields"):
            validate_review_result(
                {"status": "revised", "issues_found": [unknown_field], "final_markdown": REVISED},
                DRAFT,
                TRANSCRIPT,
            )
        with self.assertRaisesRegex(ValueError, "issue type"):
            validate_review_result(
                {
                    "status": "revised",
                    "issues_found": [issue("invented")],
                    "final_markdown": REVISED,
                },
                DRAFT,
                TRANSCRIPT,
            )
        with self.assertRaisesRegex(ValueError, "severity"):
            validate_review_result(
                {
                    "status": "revised",
                    "issues_found": [issue(severity="critical")],
                    "final_markdown": REVISED,
                },
                DRAFT,
                TRANSCRIPT,
            )
        with self.assertRaisesRegex(ValueError, "resolution"):
            validate_review_result(
                {
                    "status": "revised",
                    "issues_found": [issue(resolution="   ")],
                    "final_markdown": REVISED,
                },
                DRAFT,
                TRANSCRIPT,
            )

    def test_final_markdown_rejects_unknown_duplicate_or_out_of_order_h2(self):
        invalid_values = (
            DRAFT + "\n## Practical Takeaways\n\nNo.\n",
            DRAFT + "\n## Key Ideas\n\nDuplicate.\n",
            "## Key Ideas\n\nWrong.\n\n## TL;DR\n\nWrong.\n",
            "## TL;DR\n\nOkay.\n\n## Research & Evidence\n\nToo early.\n\n## Key Ideas\n\nWrong.\n",
        )
        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    validate_review_result(
                        {
                            "status": "revised",
                            "issues_found": [issue("structure_violation")],
                            "final_markdown": value,
                        },
                        DRAFT,
                        TRANSCRIPT,
                    )

    def test_final_markdown_accepts_optional_sections_in_order(self):
        final = (
            "## TL;DR\n\nShort.\n\n"
            "## Key Ideas\n\n### Topic\n\nBody.\n\n"
            "## Research & Evidence\n\nEvidence.\n\n"
            "## Follow Up\n\nQuestion.\n"
        )
        validate_review_result(
            {
                "status": "revised",
                "issues_found": [issue("structure_violation")],
                "final_markdown": final,
            },
            DRAFT,
            TRANSCRIPT,
        )

    def test_revised_final_also_satisfies_the_canonical_summary_guard(self):
        invalid_values = (
            "---\ntitle: Model-authored\n---\n" + REVISED,
            "```markdown\n" + REVISED + "```\n",
        )
        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    validate_review_result(
                        {
                            "status": "revised",
                            "issues_found": [issue("structure_violation")],
                            "final_markdown": value,
                        },
                        DRAFT,
                        TRANSCRIPT,
                    )


class SummaryReviewBenchmarkConfigTests(unittest.TestCase):
    def test_benchmark_config_locks_exact_candidates_runs_and_weights(self):
        config = json.loads(BENCHMARK_CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(
            config["models"],
            [
                "openai/gpt-5-mini",
                "anthropic/claude-haiku-4.5",
                "anthropic/claude-sonnet-4.6",
                "qwen/qwen3-235b-a22b-2507",
                "qwen/qwen3.6-35b-a3b",
            ],
        )
        self.assertEqual(config["runs_per_model"], 3)
        self.assertEqual(sum(config["rubric_weights"].values()), 100)

    def test_benchmark_prompt_does_not_leak_fixture_answers(self):
        prompt = BENCHMARK_PROMPT.read_text(encoding="utf-8").casefold()
        for forbidden in (
            "99%",
            "concept2",
            "yates style",
            "gait style",
            "geoffrey",
            "relationship/lifestyle",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden.casefold(), prompt)


if __name__ == "__main__":
    unittest.main()
