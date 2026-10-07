import json
import unittest

from podcast_engine.knowledge import summary_review_contract as contract


VALID_MARKDOWN = (
    "## TL;DR\n\n"
    "- Supported point.\n\n"
    "## Key Ideas\n\n"
    "### Topic\n\n"
    "Supported point.\n"
)


class SummaryReviewContractTests(unittest.TestCase):
    def test_current_contract_identity_and_phase_schemas_are_strict(self):
        self.assertEqual(contract.REVIEW_CONTRACT_ID, "summary-review-v2")
        self.assertEqual(contract.MAX_REPAIR_TURNS_PER_CASE, 1)

        self.assertFalse(contract.SUMMARY_REVIEW_AUDIT_SCHEMA["additionalProperties"])
        self.assertEqual(
            contract.SUMMARY_REVIEW_AUDIT_SCHEMA["required"],
            ["status", "risk_assessments", "additional_issues"],
        )
        self.assertFalse(contract.SUMMARY_REVIEW_EDIT_SCHEMA["additionalProperties"])
        self.assertEqual(
            contract.SUMMARY_REVIEW_EDIT_SCHEMA["required"],
            ["resolved_issue_ids", "final_markdown"],
        )

    def test_parser_requires_json_object(self):
        parsed = contract.parse_review_content(
            '{"status":"pass","risk_assessments":[],"additional_issues":[]}'
        )
        self.assertEqual(parsed["status"], "pass")

        with self.assertRaises((json.JSONDecodeError, ValueError)):
            contract.parse_review_content("not json")
        with self.assertRaisesRegex(ValueError, "JSON object"):
            contract.parse_review_content("[]")

    def test_completion_metadata_sums_usage_and_preserves_final_identity(self):
        result = contract.aggregate_completion_metadata(
            [
                {
                    "completion_id": "first",
                    "served_model": "model/a",
                    "served_provider": "Provider A",
                    "prompt_tokens": 10,
                    "completion_tokens": 3,
                    "total_tokens": 13,
                    "cost": 0.01,
                },
                {
                    "completion_id": "second",
                    "served_model": "model/a",
                    "served_provider": "Provider A",
                    "prompt_tokens": 20,
                    "completion_tokens": 7,
                    "total_tokens": 27,
                    "cost": 0.02,
                },
            ],
            attempt_count=2,
        )

        self.assertEqual(result["attempt_count"], 2)
        self.assertEqual(result["completion_id"], "second")
        self.assertEqual(result["served_model"], "model/a")
        self.assertEqual(result["served_provider"], "Provider A")
        self.assertEqual(result["prompt_tokens"], 30)
        self.assertEqual(result["completion_tokens"], 10)
        self.assertEqual(result["total_tokens"], 40)
        self.assertAlmostEqual(result["cost"], 0.03)

        for invalid in (0, -1, True):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    contract.aggregate_completion_metadata([], attempt_count=invalid)

    def test_final_markdown_accepts_optional_sections_in_order(self):
        final = (
            "## TL;DR\n\nShort.\n\n"
            "## Key Ideas\n\n### Topic\n\nBody.\n\n"
            "## Research & Evidence\n\nEvidence.\n\n"
            "## Follow Up\n\nQuestion.\n"
        )
        contract.validate_final_markdown(final)

    def test_final_markdown_rejects_invalid_architecture_and_canonical_guard_failures(self):
        invalid_values = (
            VALID_MARKDOWN + "\n## Practical Takeaways\n\nNo.\n",
            VALID_MARKDOWN + "\n## Key Ideas\n\nDuplicate.\n",
            "## Key Ideas\n\nWrong.\n\n## TL;DR\n\nWrong.\n",
            "---\ntitle: Model-authored\n---\n" + VALID_MARKDOWN,
            "```markdown\n" + VALID_MARKDOWN + "```\n",
        )
        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    contract.validate_final_markdown(value)


if __name__ == "__main__":
    unittest.main()
