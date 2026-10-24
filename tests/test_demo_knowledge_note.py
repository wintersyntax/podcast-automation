from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "demo_knowledge_note.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("demo_knowledge_note", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class KnowledgeNoteDemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.demo = _load_module()

    def test_demo_uses_production_renderer_and_domain_specific_sections(self):
        markdown = self.demo.build_demo_markdown()
        self.assertIn("## TL;DR", markdown)
        self.assertIn("## Research discussed", markdown)
        self.assertIn("**42** resistance-trained adults", markdown)
        self.assertIn("12 weeks", markdown)
        self.assertIn("1.6 g/kg", markdown)
        self.assertIn("## Romanian deadlift execution", markdown)
        self.assertIn("## Numbers & protocols", markdown)
        self.assertIn("## Sources mentioned", markdown)
        self.assertIn("Demo et al. 2025 (synthetic study)", markdown)

    def test_demo_is_explicitly_synthetic(self):
        markdown = self.demo.build_demo_markdown()
        self.assertIn("Example Strength & Nutrition Podcast", markdown)
        self.assertIn("Demo et al. 2025 (synthetic)", markdown)

    def test_browser_preview_and_raw_markdown_are_available(self):
        client = self.demo.create_demo_app().test_client()
        page = client.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertIn(b"Synthetic portfolio demo", page.data)
        self.assertIn(b"Romanian deadlift", page.data)

        raw = client.get("/raw.md")
        self.assertEqual(raw.status_code, 200)
        self.assertIn(b"## Research discussed", raw.data)


if __name__ == "__main__":
    unittest.main()
