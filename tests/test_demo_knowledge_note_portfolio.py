"""Keep the public note example and viewer aligned with the real renderer."""

import unittest
from pathlib import Path

from scripts.demo_knowledge_note import build_demo_markdown, create_demo_app


class SyntheticNotePortfolioTests(unittest.TestCase):
    def test_published_markdown_is_exactly_regenerated(self):
        current = build_demo_markdown()
        published = (
            Path(__file__).resolve().parents[1] / "examples" / "demo-knowledge-note.md"
        ).read_text(encoding="utf-8")
        self.assertEqual(published, current)
        self.assertIn("# Protein Timing, Hypertrophy Research & Exercise Selection", current)
        self.assertIn("🔬 **Study**", current)
        self.assertIn("🏋️ **Coaches**", current)
        self.assertNotIn("▰", current)
        self.assertNotIn("▱", current)

    def test_local_preview_uses_icon_legend_and_one_title(self):
        page = create_demo_app().test_client().get("/")
        self.assertEqual(page.status_code, 200)
        html = page.get_data(as_text=True)
        for label in ("🔬 Study", "🏋️ Coaches", "👤 One person", "💬 Opinion"):
            self.assertIn(label, html)
        self.assertNotIn("▰", html)
        self.assertNotIn("▱", html)
        self.assertEqual(
            html.count("<h1>Protein Timing, Hypertrophy Research &amp; Exercise Selection</h1>"),
            1,
        )


if __name__ == "__main__":
    unittest.main()
