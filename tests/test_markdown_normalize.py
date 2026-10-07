import unittest

from podcast_engine.knowledge.markdown_normalize import normalize_markdown_layout


class MarkdownNormalizeTests(unittest.TestCase):
    def test_repairs_accidental_inline_headings_and_bullets(self):
        messy = (
            "   ## TL;DR   \n"
            "        - First point.          - Second point.   \n"
            "Paragraph text.            ### Key Idea   \n"
            "Body.   \n"
        )

        self.assertEqual(
            normalize_markdown_layout(messy),
            "## TL;DR\n\n"
            "- First point.\n"
            "- Second point.\n\n"
            "Paragraph text.\n\n"
            "### Key Idea\n\n"
            "Body.\n",
        )

    def test_preserves_nested_research_bullets(self):
        markdown = (
            "## Research & Evidence\n\n"
            "- Study one\n"
            "  - Method: crossover design\n"
            "  - Finding: directional effect\n\n"
            "Paragraph after list.\n"
        )

        self.assertEqual(normalize_markdown_layout(markdown), markdown)

    def test_normalizes_shifted_list_base_while_preserving_relative_nesting(self):
        messy = (
            "## Research & Evidence\n\n"
            "        - Study one\n"
            "          - Method: crossover design\n"
            "        - Study two\n"
        )

        self.assertEqual(
            normalize_markdown_layout(messy),
            "## Research & Evidence\n\n"
            "- Study one\n"
            "  - Method: crossover design\n"
            "- Study two\n",
        )

    def test_fenced_code_content_is_opaque(self):
        markdown = (
            "## Example\n\n"
            "```text\n"
            "    ## not a heading   \n"
            "value.          - not a bullet   \n"
            "```\n"
        )

        self.assertEqual(normalize_markdown_layout(markdown), markdown)

    def test_normalization_is_idempotent(self):
        messy = (
            "      ## TL;DR\n"
            "        - One.          - Two.\n"
            "Text.           ### Details\n"
            "More text.   \n"
        )
        once = normalize_markdown_layout(messy)
        twice = normalize_markdown_layout(once)
        self.assertEqual(twice, once)


if __name__ == "__main__":
    unittest.main()
