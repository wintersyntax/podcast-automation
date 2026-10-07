import os
import unittest
from unittest.mock import Mock, patch

from podcast_engine.knowledge import summary
from podcast_engine.knowledge.models import SUMMARY_POLICY_VERSION
from podcast_engine.preset_provenance import PresetProvenance


def _provenance() -> PresetProvenance:
    return PresetProvenance(
        status="verified",
        slug="podcast-summary",
        preset_id="preset-id",
        version_id="version-id-1",
        version=1,
        config={"model": "test/summary-model", "max_tokens": 900},
        system_prompt="Summary system prompt.",
        config_digest="sha256:config",
        system_prompt_digest="sha256:prompt",
        verified_at="2026-09-16T00:00:00Z",
    )


class SummaryGenerationTests(unittest.TestCase):
    def _episode(self):
        return {
            "episode_key": "episode-1",
            "podcast": "Example Strength Podcast",
            "podcast_id": "example-strength",
            "category": "exercise_strength",
            "prompt": "strength",
            "title": "Generation test",
            "published": "2026-09-04T00:00:00Z",
            "link": "https://example.test/episode",
        }

    def _response(self, content):
        response = Mock()
        response.json.return_value = {
            "choices": [{"message": {"content": content}}],
        }
        return response

    def test_summary_policy_version_is_bumped_for_complete_reviewer_chain(self):
        self.assertEqual(SUMMARY_POLICY_VERSION, "summary-v6")

    def test_generate_repairs_layout_before_structural_validation(self):
        content = (
            "      ## TL;DR\n"
            "        - First point.          - Second point.\n"
            "Text.            ## Key Ideas\n"
            "Body.\n"
        )
        with patch.dict(
            os.environ, {"PODCAST_KNOWLEDGE_API_KEY": "knowledge-key"}, clear=False
        ), patch(
            "podcast_engine.knowledge.summary.fetch_current_designated_preset",
            return_value=_provenance(),
        ), patch(
            "podcast_engine.knowledge.summary.post_openrouter",
            return_value=self._response(content),
        ):
            generated = summary.generate(
                self._episode(),
                "Compiled transcript",
                episode_key="a" * 24,
                source_fingerprint="sha256:" + "c" * 64,
            )

        self.assertEqual(
            generated,
            "## TL;DR\n\n"
            "- First point.\n"
            "- Second point.\n\n"
            "Text.\n\n"
            "## Key Ideas\n\n"
            "Body.\n",
        )

    def test_generate_rejects_duplicate_section_revealed_by_layout_repair(self):
        content = (
            "## TL;DR\n\n"
            "- First point.          ## TL;DR\n"
            "- Duplicate section.\n"
        )
        with patch.dict(
            os.environ, {"PODCAST_KNOWLEDGE_API_KEY": "knowledge-key"}, clear=False
        ), patch(
            "podcast_engine.knowledge.summary.fetch_current_designated_preset",
            return_value=_provenance(),
        ), patch(
            "podcast_engine.knowledge.summary.post_openrouter",
            return_value=self._response(content),
        ):
            with self.assertRaisesRegex(ValueError, "duplicate level-2 section"):
                summary.generate(
                    self._episode(),
                    "Compiled transcript",
                    episode_key="a" * 24,
                    source_fingerprint="sha256:" + "c" * 64,
                )


if __name__ == "__main__":
    unittest.main()
