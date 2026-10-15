from decimal import Decimal
import json
import os
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from podcast_engine.ai_pricing import ConservativeTextPricingBound, ResolvedPreset
from podcast_engine.knowledge.orchestration import (
    build_knowledge_note,
    canonical_summary_generated_at,
)
from podcast_engine.knowledge import client
from podcast_engine.knowledge.frontmatter import build_frontmatter, render
from podcast_engine.knowledge.notes_v3.schema import normalize_metadata
from podcast_engine.preset_provenance import PresetProvenance


def _provenance(*, config=None, system_prompt="Verified system prompt.") -> PresetProvenance:
    return PresetProvenance(
        status="verified",
        slug="test-preset",
        preset_id="preset-id",
        version_id="version-id-1",
        version=1,
        config=config if config is not None else {"model": "test/model", "max_tokens": 500},
        system_prompt=system_prompt,
        config_digest="sha256:config",
        system_prompt_digest="sha256:prompt",
        verified_at="2026-09-16T00:00:00Z",
    )


def _resolved_preset(provenance: PresetProvenance) -> ResolvedPreset:
    return ResolvedPreset(
        slug=provenance.slug,
        preset_id=provenance.preset_id,
        version_id=provenance.version_id,
        version=provenance.version,
        config=provenance.config,
        system_prompt=provenance.system_prompt,
        config_digest=provenance.config_digest,
        system_prompt_digest=provenance.system_prompt_digest,
        pricing_bound=ConservativeTextPricingBound(
            model_ids=("test/model",),
            prompt_usd_per_token=Decimal("0.000001"),
            completion_usd_per_token=Decimal("0.000002"),
            context_length=200000,
            pricings=(),
        ),
    )


class KnowledgePayloadTests(unittest.TestCase):
    def _episode(self):
        return {
            "id": "episode-1",
            "episode_key": "episode-1",
            "podcast": "Example Strength Podcast",
            "podcast_id": "example-strength-podcast",
            "category": "exercise_strength",
            "prompt": "strength",
            "title": "Ep 386 - Training Principles",
            "published": "Tue, 12 Aug 2026 14:00:00 +0000",
            "link": "https://example.test/episode",
            "status": {"compiler": {"state": "completed"}},
        }

    def test_metadata_normalization_never_invents_values(self):
        self.assertEqual(
            normalize_metadata(
                {
                    "topics": [" Training ", "training", "", 4, "Nutrition"],
                    "people": ["Alice", " alice ", "Bob"],
                    "existing_tags": ["RPE", "rpe", "  "],
                    "new_tag_candidates": [{"tag": "Lengthened partials", "category": "training"}],
                }
            ),
            {
                "topics": ["Training", "Nutrition"],
                "people": ["Alice", "Bob"],
                "existing_tags": ["RPE"],
                "new_tag_candidates": [{"tag": "Lengthened partials", "category": "training"}],
            },
        )

    def _budget_plumbing(self, provenance):
        return (
            patch(
                "podcast_engine.ai_pricing.resolve_preset_model_and_pricing",
                return_value=_resolved_preset(provenance),
            ),
            patch("podcast_engine.ai_budget.reserve_budget_batch"),
            patch("podcast_engine.ai_budget.settle_budget_attempt"),
            patch("podcast_engine.ai_budget.mark_budget_attempt_uncertain"),
        )

    def test_client_uses_knowledge_key_without_workload_fallback(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {}
        provenance = _provenance()
        p1, p2, p3, p4 = self._budget_plumbing(provenance)
        with patch.dict(
            os.environ,
            {
                "PODCAST_KNOWLEDGE_API_KEY": "knowledge-key",
                "PODCAST_TRANSCRIPT_REVIEW_API_KEY": "review-key",
                "OPENROUTER_API_KEY": "other-key",
            },
            clear=True,
        ), patch("podcast_engine.knowledge.client.requests.post", return_value=response) as post, p1, p2, p3, p4:
            client.post_openrouter(
                dict(provenance.config),
                episode_key="a" * 24,
                source_fingerprint="sha256:" + "c" * 64,
                stage="note_writer",
                provenance=provenance,
            )

        self.assertEqual(post.call_args.kwargs["headers"]["Authorization"], "Bearer knowledge-key")

        self.assertEqual(
            post.call_args.kwargs["json"]["provider"]["max_price"],
            {"prompt": 1, "completion": 2},
        )

        with patch.dict(os.environ, {"PODCAST_TRANSCRIPT_REVIEW_API_KEY": "review-key"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "PODCAST_KNOWLEDGE_API_KEY"):
                client.post_openrouter(
                    dict(provenance.config),
                    episode_key="a" * 24,
                    source_fingerprint="sha256:" + "c" * 64,
                    stage="note_writer",
                    provenance=provenance,
                )

    def test_client_rejects_conflicting_completion_ceiling_before_post(self):
        provenance = _provenance(
            config={"model": "test/model", "max_tokens": 500, "max_completion_tokens": 1000}
        )
        p1, p2, p3, p4 = self._budget_plumbing(provenance)
        with patch.dict(os.environ, {"PODCAST_KNOWLEDGE_API_KEY": "knowledge-key"}, clear=True), patch(
            "podcast_engine.knowledge.client.requests.post"
        ) as post, p1, p2, p3, p4:
            with self.assertRaises(RuntimeError):
                client.post_openrouter(
                    dict(provenance.config),
                    episode_key="a" * 24,
                    source_fingerprint="sha256:" + "c" * 64,
                    stage="note_writer",
                    provenance=provenance,
                )
        post.assert_not_called()

    def test_client_reports_bounded_sanitized_http_error_body(self):
        response = Mock(status_code=400)
        response.text = (
            "OpenRouter rejected request: Summary body; Eric Helms transcript; "
            "Bearer knowledge-key; "
            + "x" * 3000
        )
        response.raise_for_status.side_effect = client.requests.HTTPError(
            "400 Client Error",
            response=response,
        )
        payload = {
            "model": "test/model",
            "max_tokens": 500,
            "summary_body": "Summary body",
            "compiled_transcript": "Eric Helms transcript",
        }
        provenance = _provenance()
        p1, p2, p3, p4 = self._budget_plumbing(provenance)

        with patch.dict(
            os.environ,
            {"PODCAST_KNOWLEDGE_API_KEY": "knowledge-key"},
            clear=True,
        ), patch("podcast_engine.knowledge.client.requests.post", return_value=response), p1, p2, p3 as settle, p4 as uncertain:
            with self.assertRaisesRegex(RuntimeError, "HTTP 400") as raised:
                client.post_openrouter(
                    payload,
                    episode_key="a" * 24,
                    source_fingerprint="sha256:" + "c" * 64,
                    stage="note_writer",
                    provenance=provenance,
                )

        # TASK-128: a 400 proves nothing was generated, so it settles at zero.
        settle.assert_called_once()
        self.assertEqual(settle.call_args.kwargs["actual_usd"], Decimal("0"))
        uncertain.assert_not_called()

        message = str(raised.exception)
        self.assertIn("OpenRouter rejected request", message)
        self.assertNotIn("Summary body", message)
        self.assertNotIn("Eric Helms transcript", message)
        self.assertNotIn("knowledge-key", message)
        self.assertLessEqual(
            len(message),
            len("OpenRouter knowledge request failed (HTTP 400): ") + client.ERROR_BODY_LIMIT,
        )




class KnowledgeEntryPointTests(unittest.TestCase):
    def test_canonical_summary_generated_at_uses_the_manifest_summary_instant(self):
        manifest = {
            "schema_version": 1,
            "episode_key": "episode-1",
            "summary": {"generated_at": "2026-08-26T10:00:00+00:00"},
        }
        fake_bucket = _FakeBucket(
            {"episodes/episode-1/summary/metadata.json": json.dumps(manifest)}
        )

        with patch("podcast_engine.knowledge.orchestration.get_bucket", return_value=fake_bucket):
            generated_at = canonical_summary_generated_at("episode-1")

        self.assertEqual(generated_at, "2026-08-26T10:00:00+00:00")

    def test_missing_writer_preset_fails_closed_before_any_storage_access(self):
        for value in (None, "", "   "):
            environ = {} if value is None else {"PODCAST_KNOWLEDGE_WRITER_PRESET": value}
            with self.subTest(value=value), patch.dict(os.environ, environ, clear=True), patch(
                "podcast_engine.knowledge.orchestration.get_bucket"
            ) as get_bucket, patch(
                "podcast_engine.knowledge.orchestration.download_file_from_gcs"
            ) as download:
                with self.assertRaisesRegex(RuntimeError, "PODCAST_KNOWLEDGE_WRITER_PRESET"):
                    build_knowledge_note({"episode_key": "e" * 24})
            get_bucket.assert_not_called()
            download.assert_not_called()


class FrontmatterTests(unittest.TestCase):
    def test_tags_are_slugged_and_deduplicated_without_changing_topics_or_people(self):
        episode = {"podcast": "Example Strength Podcast", "podcast_id": "example-strength-podcast"}
        metadata_values = {
            "topics": ["Sleep Quality"],
            "people": ["Eric Helms"],
            "tags": ["Example Strength Podcast", "example-strength-podcast", " Sleep ", "creatine", "CREATINE", "", "  "],
        }

        frontmatter = build_frontmatter(episode, metadata_values, "date")

        self.assertEqual(frontmatter["tags"], ["example-strength-podcast", "sleep", "creatine"])
        self.assertEqual(frontmatter["topics"], ["Sleep Quality"])
        self.assertEqual(frontmatter["people"], ["Eric Helms"])

    def test_valid_tag_is_preserved_and_slug_collisions_are_deduplicated(self):
        frontmatter = build_frontmatter(
            {"podcast_id": "podcast"},
            {"topics": [], "people": [], "tags": ["creatine", "Sleep Quality", "sleep-quality"]},
            "date",
        )

        self.assertEqual(frontmatter["tags"], ["creatine", "sleep-quality"])

    def test_frontmatter_uses_canonical_source_url(self):
        rendered = render(
            {
                "link": "https://example-strength.libsyn.com/demo-sleep-hypertrophy-topic",
            },
            {"topics": [], "people": [], "tags": []},
            "date",
            "# Summary\n",
        )

        self.assertIn(
            "source_url: https://example-strength.libsyn.com/website/demo-sleep-hypertrophy-topic",
            rendered,
        )

    def test_frontmatter_uses_podcast_url_without_inventing_source_url(self):
        rendered = render(
            {
                "podcast": "Example Fitness Podcast",
                "podcast_id": "example-fitness-podcast",
                "podcast_url": "https://example.test/podcast/",
                "link": None,
            },
            {"topics": [], "people": [], "tags": []},
            "date",
            "# Summary\n",
        )

        self.assertIn(
            "podcast_url: https://example.test/podcast/",
            rendered,
        )
        self.assertNotIn("source_url:", rendered)

    def test_frontmatter_omits_missing_podcast_url(self):
        rendered = render(
            {
                "podcast": "Example Strength Podcast",
                "podcast_id": "example-strength-podcast",
                "link": "https://example-strength.libsyn.com/website/episode",
            },
            {"topics": [], "people": [], "tags": []},
            "date",
            "# Summary\n",
        )

        self.assertNotIn("podcast_url:", rendered)

    def test_frontmatter_is_deterministic_and_omits_missing_optional_fields(self):
        episode = {
            "episode_key": "key-1",
            "podcast": "Example Strength Podcast",
            "title": "Ep 386 - Training Principles",
            "published": "Tue, 12 Aug 2026 14:00:00 +0000",
            "status": {"compiler": {"state": "completed"}},
        }
        rendered = render(
            episode,
            {"topics": ["Training"], "people": [], "tags": ["RPE", "rpe"]},
            "2026-08-26T10:00:00+00:00",
            "# Summary\n",
        )
        self.assertIn("episode: 386", rendered)
        self.assertIn("published: '2026-08-12'", rendered)
        self.assertIn("transcript_reviewed: true", rendered)
        self.assertNotIn("- example-strength-podcast", rendered)
        self.assertNotIn("podcast_id:", rendered)
        self.assertNotIn("source_url:", rendered)
        self.assertEqual(
            build_frontmatter(
                episode,
                {"topics": [], "people": [], "tags": []},
                "date",
            )["tags"],
            [],
        )


    def test_note_opens_with_the_episode_title_as_h1(self):
        episode = {"episode_key": "key-1", "podcast": "Example Fitness Podcast",
                   "title": "Your Body Changes How You Should Train -  Jonathan Warren"}
        rendered = render(episode, {"topics": [], "people": [], "tags": []}, "date", "\n## TL;DR\n\nText.\n")
        frontmatter_end = rendered.index("\n---\n") + len("\n---\n")
        self.assertEqual(
            rendered[frontmatter_end:],
            "\n# Your Body Changes How You Should Train - Jonathan Warren\n\n## TL;DR\n\nText.\n",
        )
        # A body that already has its own H1 is not given a second one;
        # a missing title adds nothing.
        self.assertEqual(rendered.count("\n# "), 1)
        own = render(episode, {"topics": [], "people": [], "tags": []}, "date", "# Summary\n")
        self.assertEqual(own.count("\n# "), 1)
        untitled = render({"episode_key": "key-1"}, {"topics": [], "people": [], "tags": []}, "date", "## TL;DR\n")
        self.assertNotIn("\n# ", untitled)


class _FakeBlob:
    def __init__(self, bucket, name):
        self.bucket = bucket
        self.name = name

    def exists(self):
        return self.name in self.bucket.objects

    @property
    def generation(self):
        return self.bucket.generations.get(self.name, 0)

    def reload(self):
        if not self.exists():
            from google.api_core.exceptions import NotFound

            raise NotFound("missing")

    def download_as_text(self, encoding="utf-8", if_generation_match=None):
        if (
            if_generation_match is not None
            and self.generation != if_generation_match
        ):
            from google.api_core.exceptions import PreconditionFailed

            raise PreconditionFailed("generation mismatch")
        return self.bucket.objects[self.name]

    def download_as_bytes(self):
        return self.bucket.objects[self.name].encode("utf-8")

    def upload_from_filename(self, filename, content_type=None):
        self.bucket.objects[self.name] = Path(filename).read_text(encoding="utf-8")
        self.bucket.generations[self.name] = self.generation + 1
        self.bucket.uploads.append(self.name)

    def upload_from_string(self, content, content_type=None, if_generation_match=None):
        if if_generation_match is not None and self.generation != if_generation_match:
            from google.api_core.exceptions import PreconditionFailed

            raise PreconditionFailed("generation mismatch")
        self.bucket.objects[self.name] = content
        self.bucket.generations[self.name] = self.generation + 1
        self.bucket.uploads.append(self.name)


class _FakeBucket:
    def __init__(self, objects):
        self.objects = dict(objects)
        self.generations = {name: 1 for name in objects}
        self.uploads = []

    def blob(self, name):
        return _FakeBlob(self, name)


if __name__ == "__main__":
    unittest.main()
