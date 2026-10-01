import unittest

from podcast_engine.knowledge.summary_review_contract import (
    REVIEW_CONTRACT_ID,
    SUMMARY_REVIEW_AUDIT_SCHEMA,
    SUMMARY_REVIEW_EDIT_SCHEMA,
    accepted_final_markdown,
    build_edit_obligations,
    validate_audit_result,
    validate_edit_result,
)
from podcast_engine.knowledge.summary_review_evidence import build_review_context


DRAFT = (
    "## TL;DR\n\n"
    "- The measured result reached 99%.\n\n"
    "## Key Ideas\n\n"
    "### Measurement\n\n"
    "The measured result reached 99%.\n"
)
TRANSCRIPT = "The measured result reached 99% in the reported test."


def context():
    return build_review_context(TRANSCRIPT, DRAFT)


def supported_assessments(review_context):
    return [
        {
            "risk_id": risk["risk_id"],
            "disposition": "supported",
            "issue_type": None,
            "severity": None,
            "draft_block_ids": [risk["draft_block_id"]],
            "transcript_span_ids": ["S0001"],
            "resolution": "The transcript states the same bounded claim.",
        }
        for risk in review_context["risk_inventory"]["risks"]
    ]


class SummaryReviewV2SchemaTests(unittest.TestCase):
    def test_contract_identity_and_disjoint_top_level_schemas(self):
        self.assertEqual(REVIEW_CONTRACT_ID, "summary-review-v2")
        self.assertEqual(
            set(SUMMARY_REVIEW_AUDIT_SCHEMA["required"]),
            {"status", "risk_assessments", "additional_issues"},
        )
        self.assertEqual(
            set(SUMMARY_REVIEW_EDIT_SCHEMA["required"]),
            {"resolved_issue_ids", "final_markdown"},
        )
        self.assertFalse(SUMMARY_REVIEW_AUDIT_SCHEMA["additionalProperties"])
        self.assertFalse(SUMMARY_REVIEW_EDIT_SCHEMA["additionalProperties"])

        self.assertNotIn("final_markdown", SUMMARY_REVIEW_AUDIT_SCHEMA["properties"])
        self.assertNotIn("status", SUMMARY_REVIEW_EDIT_SCHEMA["properties"])
        self.assertNotIn("issues_found", SUMMARY_REVIEW_AUDIT_SCHEMA["properties"])
        self.assertNotIn("draft_excerpt", repr(SUMMARY_REVIEW_AUDIT_SCHEMA))
        self.assertNotIn("transcript_evidence", repr(SUMMARY_REVIEW_AUDIT_SCHEMA))

    def test_pass_audit_requires_complete_exact_risk_adjudication(self):
        review_context = context()
        assessments = supported_assessments(review_context)

        validated = validate_audit_result(
            {
                "status": "pass",
                "risk_assessments": assessments,
                "additional_issues": [],
            },
            TRANSCRIPT,
            review_context,
        )
        self.assertEqual(validated["status"], "pass")

        missing = {
            "status": "pass",
            "risk_assessments": [],
            "additional_issues": [],
        }
        with self.assertRaises(ValueError):
            validate_audit_result(missing, TRANSCRIPT, review_context)

        duplicate = {
            "status": "pass",
            "risk_assessments": assessments + [dict(assessments[0])],
            "additional_issues": [],
        }
        with self.assertRaises(ValueError):
            validate_audit_result(duplicate, TRANSCRIPT, review_context)

        invented = dict(assessments[0])
        invented["risk_id"] = "R9999"
        with self.assertRaises(ValueError):
            validate_audit_result(
                {
                    "status": "pass",
                    "risk_assessments": [invented] + assessments[1:],
                    "additional_issues": [],
                },
                TRANSCRIPT,
                review_context,
            )

    def test_pass_audit_proceeds_to_byte_preserving_noop_edit(self):
        review_context = context()
        audit = validate_audit_result(
            {
                "status": "pass",
                "risk_assessments": supported_assessments(review_context),
                "additional_issues": [],
            },
            TRANSCRIPT,
            review_context,
        )

        obligations = build_edit_obligations(audit)
        self.assertEqual(obligations, [])

        edit = validate_edit_result(
            {
                "resolved_issue_ids": [],
                "final_markdown": DRAFT,
            },
            audit,
            DRAFT,
            TRANSCRIPT,
            review_context,
        )
        self.assertEqual(
            accepted_final_markdown(audit, edit, DRAFT),
            DRAFT,
        )

        with self.assertRaises(ValueError):
            validate_edit_result(
                {
                    "resolved_issue_ids": [],
                    "final_markdown": DRAFT.replace("99%", "98%", 1),
                },
                audit,
                DRAFT,
                TRANSCRIPT,
                review_context,
            )

    def test_audit_and_edit_reject_v1_or_cross_phase_fields(self):
        review_context = context()
        assessments = supported_assessments(review_context)

        with self.assertRaises(ValueError):
            validate_audit_result(
                {
                    "status": "pass",
                    "risk_assessments": assessments,
                    "additional_issues": [],
                    "final_markdown": DRAFT,
                },
                TRANSCRIPT,
                review_context,
            )

        audit = validate_audit_result(
            {
                "status": "pass",
                "risk_assessments": assessments,
                "additional_issues": [],
            },
            TRANSCRIPT,
            review_context,
        )

        with self.assertRaises(ValueError):
            validate_edit_result(
                {
                    "resolved_issue_ids": [],
                    "final_markdown": DRAFT,
                    "status": "pass",
                },
                audit,
                DRAFT,
                TRANSCRIPT,
                review_context,
            )

    def test_dispositions_enforce_exact_evidence_semantics(self):
        review_context = context()

        supported = supported_assessments(review_context)
        validate_audit_result(
            {
                "status": "pass",
                "risk_assessments": supported,
                "additional_issues": [],
            },
            TRANSCRIPT,
            review_context,
        )

        bad_supported = [dict(item) for item in supported]
        bad_supported[0]["transcript_span_ids"] = []
        with self.assertRaises(ValueError):
            validate_audit_result(
                {
                    "status": "pass",
                    "risk_assessments": bad_supported,
                    "additional_issues": [],
                },
                TRANSCRIPT,
                review_context,
            )

        not_applicable = [dict(item) for item in supported]
        not_applicable[0] = {
            **not_applicable[0],
            "disposition": "not_applicable",
            "issue_type": None,
            "severity": None,
            "transcript_span_ids": [],
            "resolution": "This detector hit is not a semantic assertion.",
        }
        validate_audit_result(
            {
                "status": "pass",
                "risk_assessments": not_applicable,
                "additional_issues": [],
            },
            TRANSCRIPT,
            review_context,
        )

        invalid_not_applicable = [dict(item) for item in not_applicable]
        invalid_not_applicable[0]["transcript_span_ids"] = ["S0001"]
        with self.assertRaises(ValueError):
            validate_audit_result(
                {
                    "status": "pass",
                    "risk_assessments": invalid_not_applicable,
                    "additional_issues": [],
                },
                TRANSCRIPT,
                review_context,
            )

        issue = [dict(item) for item in supported]
        issue[0] = {
            **issue[0],
            "disposition": "issue",
            "issue_type": "unsupported_precision",
            "severity": "material",
            "resolution": "The precision is not sufficiently supported.",
        }
        validated = validate_audit_result(
            {
                "status": "revised",
                "risk_assessments": issue,
                "additional_issues": [],
            },
            TRANSCRIPT,
            review_context,
        )
        self.assertEqual(validated["status"], "revised")

        invalid_issue = [dict(item) for item in issue]
        invalid_issue[0]["issue_type"] = None
        with self.assertRaises(ValueError):
            validate_audit_result(
                {
                    "status": "revised",
                    "risk_assessments": invalid_issue,
                    "additional_issues": [],
                },
                TRANSCRIPT,
                review_context,
            )

    def test_evidence_ids_must_be_python_owned(self):
        review_context = context()
        assessments = supported_assessments(review_context)

        bad_draft = [dict(item) for item in assessments]
        bad_draft[0]["draft_block_ids"] = ["D9999"]
        with self.assertRaises(ValueError):
            validate_audit_result(
                {
                    "status": "pass",
                    "risk_assessments": bad_draft,
                    "additional_issues": [],
                },
                TRANSCRIPT,
                review_context,
            )

        bad_span = [dict(item) for item in assessments]
        bad_span[0]["transcript_span_ids"] = ["S9999"]
        with self.assertRaises(ValueError):
            validate_audit_result(
                {
                    "status": "pass",
                    "risk_assessments": bad_span,
                    "additional_issues": [],
                },
                TRANSCRIPT,
                review_context,
            )

    def test_additional_issue_uses_id_evidence_and_drives_revised_status(self):
        review_context = context()
        audit = validate_audit_result(
            {
                "status": "revised",
                "risk_assessments": supported_assessments(review_context),
                "additional_issues": [
                    {
                        "issue_type": "coverage_omission",
                        "severity": "moderate",
                        "draft_block_ids": [],
                        "transcript_span_ids": ["S0001"],
                        "resolution": "A supported transcript point is omitted.",
                    }
                ],
            },
            TRANSCRIPT,
            review_context,
        )
        self.assertEqual(audit["status"], "revised")

        invalid = {
            "status": "revised",
            "risk_assessments": supported_assessments(review_context),
            "additional_issues": [
                {
                    "issue_type": "coverage_omission",
                    "severity": "moderate",
                    "draft_block_ids": ["D0001"],
                    "transcript_span_ids": [],
                    "resolution": "Invalid coverage evidence.",
                }
            ],
        }
        with self.assertRaises(ValueError):
            validate_audit_result(invalid, TRANSCRIPT, review_context)

    def test_pass_edit_is_byte_preserving_not_only_newline_equivalent(self):
        review_context = context()
        audit = validate_audit_result(
            {
                "status": "pass",
                "risk_assessments": supported_assessments(review_context),
                "additional_issues": [],
            },
            TRANSCRIPT,
            review_context,
        )

        crlf_variant = DRAFT.replace("\n", "\r\n")
        self.assertNotEqual(crlf_variant, DRAFT)

        with self.assertRaises(ValueError):
            validate_edit_result(
                {
                    "resolved_issue_ids": [],
                    "final_markdown": crlf_variant,
                },
                audit,
                DRAFT,
                TRANSCRIPT,
                review_context,
            )

    def test_fail_audit_is_terminal_and_has_no_edit_obligations(self):
        review_context = context()
        assessments = supported_assessments(review_context)

        audit = validate_audit_result(
            {
                "status": "fail",
                "risk_assessments": assessments,
                "additional_issues": [],
            },
            TRANSCRIPT,
            review_context,
        )

        self.assertEqual(audit["status"], "fail")
        with self.assertRaises(ValueError):
            build_edit_obligations(audit)

        self.assertIsNone(accepted_final_markdown(audit, None, DRAFT))

    def test_revised_obligations_preserve_risk_ids_then_assign_stable_additional_ids(self):
        review_context = context()
        assessments = supported_assessments(review_context)
        assessments[0] = {
            **assessments[0],
            "disposition": "issue",
            "issue_type": "unsupported_precision",
            "severity": "material",
            "resolution": "The exact precision is unsupported.",
        }

        audit = validate_audit_result(
            {
                "status": "revised",
                "risk_assessments": assessments,
                "additional_issues": [
                    {
                        "issue_type": "coverage_omission",
                        "severity": "moderate",
                        "draft_block_ids": [],
                        "transcript_span_ids": ["S0001"],
                        "resolution": "A developed point is omitted.",
                    },
                    {
                        "issue_type": "structure_violation",
                        "severity": "minor",
                        "draft_block_ids": ["D0003"],
                        "transcript_span_ids": [],
                        "resolution": "The section placement is invalid.",
                    },
                ],
            },
            TRANSCRIPT,
            review_context,
        )

        obligations = build_edit_obligations(audit)
        self.assertEqual(
            [item["issue_id"] for item in obligations],
            ["R0001", "I0001", "I0002"],
        )

    def test_evidence_ids_are_deduplicated_and_canonicalized_in_source_order(self):
        review_context = context()
        assessments = supported_assessments(review_context)

        assessments[0] = {
            **assessments[0],
            "draft_block_ids": ["D0002", "D0002"],
            "transcript_span_ids": ["S0001", "S0001"],
        }

        audit = validate_audit_result(
            {
                "status": "pass",
                "risk_assessments": assessments,
                "additional_issues": [],
            },
            TRANSCRIPT,
            review_context,
        )

        first = audit["risk_assessments"][0]
        self.assertEqual(first["draft_block_ids"], ["D0002"])
        self.assertEqual(first["transcript_span_ids"], ["S0001"])

    def test_approved_replacement_text_is_optional_and_defaults_to_none(self):
        review_context = context()
        assessments = supported_assessments(review_context)

        audit = validate_audit_result(
            {
                "status": "pass",
                "risk_assessments": assessments,
                "additional_issues": [],
            },
            TRANSCRIPT,
            review_context,
        )

        for assessment in audit["risk_assessments"]:
            self.assertIsNone(assessment["approved_replacement_text"])

    def test_approved_replacement_text_allowed_for_unsupported_precision_issue(self):
        review_context = context()
        assessments = supported_assessments(review_context)
        assessments[0] = {
            **assessments[0],
            "disposition": "issue",
            "issue_type": "unsupported_precision",
            "severity": "material",
            "resolution": "The exact precision is unsupported.",
            "approved_replacement_text": "97%",
        }

        audit = validate_audit_result(
            {
                "status": "revised",
                "risk_assessments": assessments,
                "additional_issues": [],
            },
            TRANSCRIPT,
            review_context,
        )

        obligations = build_edit_obligations(audit)
        self.assertEqual(obligations[0]["approved_replacement_text"], "97%")

    def test_approved_replacement_text_rejected_for_disallowed_issue_type(self):
        review_context = context()
        assessments = supported_assessments(review_context)
        assessments[0] = {
            **assessments[0],
            "disposition": "issue",
            "issue_type": "redundancy",
            "severity": "minor",
            "resolution": "This restates an earlier point.",
            "approved_replacement_text": "some replacement",
        }

        with self.assertRaisesRegex(ValueError, "approved_replacement_text"):
            validate_audit_result(
                {
                    "status": "revised",
                    "risk_assessments": assessments,
                    "additional_issues": [],
                },
                TRANSCRIPT,
                review_context,
            )

    def test_approved_replacement_text_rejected_for_supported_disposition(self):
        review_context = context()
        assessments = supported_assessments(review_context)
        assessments[0] = {
            **assessments[0],
            "approved_replacement_text": "97%",
        }

        with self.assertRaisesRegex(ValueError, "approved_replacement_text"):
            validate_audit_result(
                {
                    "status": "pass",
                    "risk_assessments": assessments,
                    "additional_issues": [],
                },
                TRANSCRIPT,
                review_context,
            )

    def test_approved_replacement_text_rejects_non_string_and_oversized_values(self):
        review_context = context()
        assessments = supported_assessments(review_context)

        for invalid_value in (123, "", "x" * 201):
            candidate = list(assessments)
            candidate[0] = {
                **candidate[0],
                "disposition": "issue",
                "issue_type": "unsupported_precision",
                "severity": "material",
                "resolution": "The exact precision is unsupported.",
                "approved_replacement_text": invalid_value,
            }
            with self.assertRaisesRegex(ValueError, "approved_replacement_text"):
                validate_audit_result(
                    {
                        "status": "revised",
                        "risk_assessments": candidate,
                        "additional_issues": [],
                    },
                    TRANSCRIPT,
                    review_context,
                )

    def test_additional_issue_approved_replacement_text_allowed_for_term_normalization(self):
        review_context = context()
        assessments = supported_assessments(review_context)

        audit = validate_audit_result(
            {
                "status": "revised",
                "risk_assessments": assessments,
                "additional_issues": [
                    {
                        "issue_type": "unsupported_term_normalization",
                        "severity": "minor",
                        "draft_block_ids": ["D0003"],
                        "transcript_span_ids": ["S0001"],
                        "resolution": "Use the transcript's own wording.",
                        "approved_replacement_text": "foot-to-bar",
                    }
                ],
            },
            TRANSCRIPT,
            review_context,
        )

        self.assertEqual(
            audit["additional_issues"][0]["approved_replacement_text"],
            "foot-to-bar",
        )

    def test_additional_issue_approved_replacement_text_rejected_for_disallowed_issue_type(self):
        review_context = context()
        assessments = supported_assessments(review_context)

        with self.assertRaisesRegex(ValueError, "approved_replacement_text"):
            validate_audit_result(
                {
                    "status": "revised",
                    "risk_assessments": assessments,
                    "additional_issues": [
                        {
                            "issue_type": "structure_violation",
                            "severity": "minor",
                            "draft_block_ids": ["D0003"],
                            "transcript_span_ids": [],
                            "resolution": "The section placement is invalid.",
                            "approved_replacement_text": "some replacement",
                        }
                    ],
                },
                TRANSCRIPT,
                review_context,
            )


if __name__ == "__main__":
    unittest.main()
