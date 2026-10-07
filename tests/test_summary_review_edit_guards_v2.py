import unittest

from podcast_engine.knowledge.summary_review_contract import (
    build_edit_obligations,
    validate_audit_result,
    validate_edit_result,
)
from podcast_engine.knowledge.summary_review_evidence import build_review_context


DRAFT = (
    "## TL;DR\n\n"
    "- The reported rate was 99%.\n\n"
    "## Key Ideas\n\n"
    "### Context\n\n"
    "The exercise protocol remained unchanged.\n"
)
TRANSCRIPT = (
    "The reported rate was 95% in the measured trial. "
    "The exercise protocol remained unchanged."
)


def _context_and_audit(approved_replacement_text=None):
    review_context = build_review_context(TRANSCRIPT, DRAFT)
    assessments = []
    for risk in review_context["risk_inventory"]["risks"]:
        if risk["surface"] == "99%":
            assessments.append(
                {
                    "risk_id": risk["risk_id"],
                    "disposition": "issue",
                    "issue_type": "unsupported_precision",
                    "severity": "material",
                    "draft_block_ids": [risk["draft_block_id"]],
                    "transcript_span_ids": ["S0001"],
                    "resolution": "Replace the unsupported exact rate with the measured rate.",
                    "approved_replacement_text": approved_replacement_text,
                }
            )
        else:
            assessments.append(
                {
                    "risk_id": risk["risk_id"],
                    "disposition": "supported",
                    "issue_type": None,
                    "severity": None,
                    "draft_block_ids": [risk["draft_block_id"]],
                    "transcript_span_ids": ["S0001"],
                    "resolution": "The transcript supports this bounded draft surface.",
                }
            )

    audit = validate_audit_result(
        {
            "status": "revised",
            "risk_assessments": assessments,
            "additional_issues": [],
        },
        TRANSCRIPT,
        review_context,
    )
    return review_context, audit



RECOMMENDATION_DRAFT = (
    "## TL;DR\n\n"
    "- The guest said people should not change their routine.\n\n"
    "## Key Ideas\n\n"
    "### Context\n\n"
    "The exercise protocol remained unchanged.\n"
)
RECOMMENDATION_TRANSCRIPT = (
    "The guest did not recommend changing anyone's routine from this observation alone. "
    "The exercise protocol remained unchanged."
)

HYPHEN_DRAFT = (
    "## TL;DR\n\n"
    "- The routine includes toes-to-bar movements on rings.\n\n"
    "## Key Ideas\n\n"
    "### Context\n\n"
    "The exercise protocol remained unchanged.\n"
)
HYPHEN_TRANSCRIPT = (
    "The routine includes foot to bar movements on rings. "
    "The exercise protocol remained unchanged."
)


def _context_and_audit_for(draft, transcript, issue_surface, resolution):
    review_context = build_review_context(transcript, draft)
    assessments = []
    for risk in review_context["risk_inventory"]["risks"]:
        if risk["surface"] == issue_surface:
            assessments.append(
                {
                    "risk_id": risk["risk_id"],
                    "disposition": "issue",
                    "issue_type": "unsupported_term_normalization",
                    "severity": "minor",
                    "draft_block_ids": [risk["draft_block_id"]],
                    "transcript_span_ids": ["S0001"],
                    "resolution": resolution,
                }
            )
        else:
            assessments.append(
                {
                    "risk_id": risk["risk_id"],
                    "disposition": "supported",
                    "issue_type": None,
                    "severity": None,
                    "draft_block_ids": [risk["draft_block_id"]],
                    "transcript_span_ids": ["S0001"],
                    "resolution": "The transcript supports this bounded draft surface.",
                }
            )

    audit = validate_audit_result(
        {
            "status": "revised",
            "risk_assessments": assessments,
            "additional_issues": [],
        },
        transcript,
        review_context,
    )
    return review_context, audit


class SummaryReviewV2EditGuardTests(unittest.TestCase):
    def test_revised_edit_accepts_supported_correction_and_preserves_other_blocks(self):
        review_context, audit = _context_and_audit()
        final = DRAFT.replace("99%", "95%")

        validated = validate_edit_result(
            {
                "resolved_issue_ids": ["R0001"],
                "final_markdown": final,
            },
            audit,
            DRAFT,
            TRANSCRIPT,
            review_context,
        )

        self.assertEqual(validated["final_markdown"], final)

    def test_revised_edit_rejects_claiming_resolution_while_target_surface_remains(self):
        review_context, audit = _context_and_audit()
        final = DRAFT.replace(
            "The reported rate was 99%.",
            "The reported rate was 99% in the measured trial.",
        )

        with self.assertRaisesRegex(ValueError, "targeted obligation"):
            validate_edit_result(
                {
                    "resolved_issue_ids": ["R0001"],
                    "final_markdown": final,
                },
                audit,
                DRAFT,
                TRANSCRIPT,
                review_context,
            )

    def test_revised_edit_rejects_new_risk_surface_absent_from_draft_and_transcript(self):
        review_context, audit = _context_and_audit()
        final = DRAFT.replace("99%", "98%")

        with self.assertRaisesRegex(ValueError, "unsupported new material"):
            validate_edit_result(
                {
                    "resolved_issue_ids": ["R0001"],
                    "final_markdown": final,
                },
                audit,
                DRAFT,
                TRANSCRIPT,
                review_context,
            )

    def test_revised_edit_rejects_removing_untargeted_draft_block(self):
        review_context, audit = _context_and_audit()
        final = DRAFT.replace("99%", "95%").replace(
            "\n\nThe exercise protocol remained unchanged.\n",
            "\n",
        )

        with self.assertRaisesRegex(ValueError, "non-regression"):
            validate_edit_result(
                {
                    "resolved_issue_ids": ["R0001"],
                    "final_markdown": final,
                },
                audit,
                DRAFT,
                TRANSCRIPT,
                review_context,
            )


    def test_revised_edit_tolerates_recommendation_inflection_supported_by_transcript(self):
        review_context, audit = _context_and_audit_for(
            RECOMMENDATION_DRAFT,
            RECOMMENDATION_TRANSCRIPT,
            "should",
            "Transcript shows this as an explicit non-recommendation; rephrase accordingly.",
        )
        final = RECOMMENDATION_DRAFT.replace(
            "The guest said people should not change their routine.",
            "The guest said people's routine was not recommended as a change.",
        )

        validated = validate_edit_result(
            {
                "resolved_issue_ids": ["R0001"],
                "final_markdown": final,
            },
            audit,
            RECOMMENDATION_DRAFT,
            RECOMMENDATION_TRANSCRIPT,
            review_context,
        )

        self.assertEqual(validated["final_markdown"], final)

    def test_revised_edit_tolerates_hyphenated_term_matching_transcript_spacing(self):
        review_context, audit = _context_and_audit_for(
            HYPHEN_DRAFT,
            HYPHEN_TRANSCRIPT,
            "toes-to-bar",
            "Transcript says 'foot to bar', not 'toes-to-bar'; use the transcript's wording.",
        )
        final = HYPHEN_DRAFT.replace("toes-to-bar", "foot-to-bar")

        validated = validate_edit_result(
            {
                "resolved_issue_ids": ["R0001"],
                "final_markdown": final,
            },
            audit,
            HYPHEN_DRAFT,
            HYPHEN_TRANSCRIPT,
            review_context,
        )

        self.assertEqual(validated["final_markdown"], final)

    def test_revised_edit_still_rejects_hyphenated_term_absent_after_dehyphenation(self):
        review_context, audit = _context_and_audit_for(
            HYPHEN_DRAFT,
            HYPHEN_TRANSCRIPT,
            "toes-to-bar",
            "Transcript says 'foot to bar', not 'toes-to-bar'; use the transcript's wording.",
        )
        final = HYPHEN_DRAFT.replace("toes-to-bar", "hand-to-bar")

        with self.assertRaisesRegex(ValueError, "unsupported new material"):
            validate_edit_result(
                {
                    "resolved_issue_ids": ["R0001"],
                    "final_markdown": final,
                },
                audit,
                HYPHEN_DRAFT,
                HYPHEN_TRANSCRIPT,
                review_context,
            )


    def test_revised_edit_accepts_approved_replacement_text_as_third_grounding_source(self):
        review_context, audit = _context_and_audit(approved_replacement_text="97%")
        final = DRAFT.replace("99%", "97%")

        validated = validate_edit_result(
            {
                "resolved_issue_ids": ["R0001"],
                "final_markdown": final,
            },
            audit,
            DRAFT,
            TRANSCRIPT,
            review_context,
        )

        self.assertEqual(validated["final_markdown"], final)

    def test_revised_edit_still_rejects_surface_not_matching_declared_approved_replacement_text(self):
        review_context, audit = _context_and_audit(approved_replacement_text="97%")
        final = DRAFT.replace("99%", "98%")

        with self.assertRaisesRegex(ValueError, "unsupported new material"):
            validate_edit_result(
                {
                    "resolved_issue_ids": ["R0001"],
                    "final_markdown": final,
                },
                audit,
                DRAFT,
                TRANSCRIPT,
                review_context,
            )


    def test_revised_edit_rejects_misattributed_name_introduced_by_edit(self):
        draft = chr(10).join([
            "## TL;DR",
            "",
            "- The supplement regulation gap warning came from Ramon Limacher.",
            "",
            "## Key Ideas",
            "",
            "### Regulation",
            "",
            "The exercise protocol remained unchanged.",
        ])
        transcript = (
            "The supplement regulation gap warning came from Eric Helms. "
            "The exercise protocol remained unchanged."
        )
        review_context = build_review_context(transcript, draft)
        assessments = []
        for risk in review_context["risk_inventory"]["risks"]:
            if risk["surface"] in ("Ramon", "Limacher"):
                assessments.append(
                    {
                        "risk_id": risk["risk_id"],
                        "disposition": "issue",
                        "issue_type": "unsupported_claim",
                        "severity": "material",
                        "draft_block_ids": [risk["draft_block_id"]],
                        "transcript_span_ids": ["S0001"],
                        "resolution": "Attribute the warning to the transcript's own speaker.",
                    }
                )
            else:
                assessments.append(
                    {
                        "risk_id": risk["risk_id"],
                        "disposition": "supported",
                        "issue_type": None,
                        "severity": None,
                        "draft_block_ids": [risk["draft_block_id"]],
                        "transcript_span_ids": ["S0001"],
                        "resolution": "The transcript supports this bounded draft surface.",
                    }
                )

        audit = validate_audit_result(
            {
                "status": "revised",
                "risk_assessments": assessments,
                "additional_issues": [],
            },
            transcript,
            review_context,
        )

        obligations = build_edit_obligations(audit)
        resolved_issue_ids = [item["issue_id"] for item in obligations]
        final = draft.replace("Ramon Limacher", "Marco Weber")

        with self.assertRaisesRegex(ValueError, "unsupported new material"):
            validate_edit_result(
                {
                    "resolved_issue_ids": resolved_issue_ids,
                    "final_markdown": final,
                },
                audit,
                draft,
                transcript,
                review_context,
            )


if __name__ == "__main__":
    unittest.main()
