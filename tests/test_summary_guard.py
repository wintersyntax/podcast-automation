import unittest

from podcast_engine.knowledge.summary_guard import normalize_and_validate_summary_body


class SummaryGuardTests(unittest.TestCase):
    def test_preserves_existing_output_normalization(self):
        body = "  ## Overview  \r\n\r\nLine one.  \r\n"
        self.assertEqual(
            normalize_and_validate_summary_body(body),
            "## Overview  \r\n\r\nLine one.\n",
        )

    def test_rejects_non_text_and_empty_output(self):
        for value in (None, 3, {}, "   \n\t"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    normalize_and_validate_summary_body(value)

    def test_rejects_yaml_frontmatter(self):
        with self.assertRaisesRegex(ValueError, "YAML frontmatter"):
            normalize_and_validate_summary_body("---\ntitle: nope\n---\n\n## Notes\nBody")

    def test_allows_leading_horizontal_rule_without_frontmatter_block(self):
        body = "---\n\n## Notes\nBody\n"
        self.assertEqual(normalize_and_validate_summary_body(body), body)

    def test_allows_internal_code_fence_and_ignores_heading_like_code(self):
        body = (
            "## Notes\n\n```text\n## Duplicate-looking code\n"
            "## Duplicate-looking code\n```\n\nBody\n"
        )
        self.assertEqual(normalize_and_validate_summary_body(body), body)

    def test_rejects_wrapped_markdown_code_fence(self):
        with self.assertRaisesRegex(ValueError, "code fence"):
            normalize_and_validate_summary_body("```markdown\n## Notes\nBody\n```")

    def test_rejects_unclosed_markdown_code_fence(self):
        with self.assertRaisesRegex(ValueError, "unclosed Markdown code fence"):
            normalize_and_validate_summary_body("## Notes\n\n```text\nexample")

    def test_rejects_empty_heading(self):
        with self.assertRaisesRegex(ValueError, "empty Markdown heading"):
            normalize_and_validate_summary_body("## Notes\nBody\n\n###   \n")

    def test_rejects_duplicate_level_two_sections(self):
        with self.assertRaisesRegex(ValueError, "duplicate level-2 section"):
            normalize_and_validate_summary_body(
                "## Key Ideas\nFirst\n\n##   key   ideas   \nSecond"
            )

    def test_rejects_duplicate_level_two_sections_with_optional_closing_hashes(self):
        with self.assertRaisesRegex(ValueError, "duplicate level-2 section"):
            normalize_and_validate_summary_body(
                "## Key Ideas ##\nFirst\n\n## key ideas\nSecond"
            )

    def test_allows_same_level_three_heading_under_different_sections(self):
        body = "## A\n### Evidence\nOne\n\n## B\n### Evidence\nTwo\n"
        self.assertEqual(normalize_and_validate_summary_body(body), body)

    def test_does_not_forbid_optional_or_legacy_section_names(self):
        body = (
            "## Practical Takeaways\nOne\n\n"
            "## Notable Insights\nTwo\n\n"
            "## Worth Remembering\nThree\n"
        )
        self.assertEqual(normalize_and_validate_summary_body(body), body)


if __name__ == "__main__":
    unittest.main()
