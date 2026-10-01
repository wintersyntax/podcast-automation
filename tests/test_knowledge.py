from decimal import Decimal
import json
import os

# Legacy-path tests: a developer .env (loaded by knowledge.client via
# python-dotenv) may activate the TASK-118 note writer. Keep these tests
# hermetic -- the writer path has its own tests.
os.environ["PODCAST_KNOWLEDGE_WRITER_PRESET"] = ""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import ANY, Mock, call, patch

from podcast_engine.ai_pricing import ConservativeTextPricingBound, ResolvedPreset
from podcast_engine.knowledge.orchestration import (
    _metadata_matches,
    _summary_matches,
    build_knowledge_note,
    canonical_summary_generated_at,
)
from podcast_engine.knowledge import client, metadata, summary, summary_review
from podcast_engine.knowledge.frontmatter import build_frontmatter, render
from podcast_engine.knowledge.models import (
    METADATA_POLICY_VERSION,
    METADATA_RESPONSE_SCHEMA,
    SUMMARY_POLICY_VERSION,
    SUMMARY_REVIEW_POLICY_VERSION,
)
from podcast_engine.knowledge.summary_review_contract import validate_audit_result
from podcast_engine.knowledge.summary_review_evidence import build_review_context
from podcast_engine.knowledge.summary_review_storage import persist_review_attempt
from podcast_engine.knowledge.tags import initial_registry, semantic_projection
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
            "podcast_id": "example-strength",
            "category": "exercise_strength",
            "prompt": "strength",
            "title": "Ep 386 - Training Principles",
            "published": "Tue, 12 Aug 2026 14:00:00 +0000",
            "link": "https://example.test/episode",
            "status": {"compiler": {"state": "completed"}},
        }

    def test_summary_payload_uses_only_the_verified_preset_snapshot(self):
        provenance = _provenance(
            config={"model": "test/summary-model", "max_tokens": 900},
            system_prompt="Summary reviewer system prompt.",
        )
        payload = summary.openrouter_payload(self._episode(), "Compiled text", provenance)

        self.assertEqual(payload["model"], "test/summary-model")
        self.assertEqual(payload["max_tokens"], 900)
        self.assertNotIn("temperature", payload)
        self.assertNotIn("provider", payload)
        self.assertEqual(payload["messages"][0], {"role": "system", "content": "Summary reviewer system prompt."})
        content = json.loads(payload["messages"][1]["content"])
        self.assertEqual(content["podcast_profile"]["profile"], "strength")
        self.assertEqual(content["compiled_transcript"], "Compiled text")

    def test_summary_payload_requires_verified_provenance(self):
        unverified = _provenance()
        object.__setattr__(unverified, "status", "invalid")
        with self.assertRaises(RuntimeError):
            summary.openrouter_payload(self._episode(), "Compiled text", unverified)

    def test_metadata_payload_has_strict_schema_and_uses_the_verified_preset_snapshot(self):
        provenance = _provenance(
            config={"model": "test/metadata-model", "max_tokens": 700},
            system_prompt="Metadata system prompt.",
        )
        payload = metadata.openrouter_payload(
            self._episode(),
            "Summary body",
            "Eric Helms and Eric Trexler introduce the episode.",
            provenance=provenance,
        )

        self.assertEqual(payload["model"], "test/metadata-model")
        self.assertEqual(payload["max_tokens"], 700)
        schema = payload["response_format"]["json_schema"]
        self.assertTrue(schema["strict"])
        self.assertIs(schema["schema"], METADATA_RESPONSE_SCHEMA)
        self.assertFalse(METADATA_RESPONSE_SCHEMA["additionalProperties"])
        self.assertEqual(
            METADATA_RESPONSE_SCHEMA["required"],
            ["topics", "people", "existing_tags", "new_tag_candidates"],
        )
        self.assertEqual(METADATA_RESPONSE_SCHEMA["properties"]["topics"]["maxItems"], 8)
        self.assertEqual(METADATA_RESPONSE_SCHEMA["properties"]["people"]["maxItems"], 12)
        self.assertEqual(METADATA_RESPONSE_SCHEMA["properties"]["existing_tags"]["maxItems"], 8)
        self.assertEqual(METADATA_RESPONSE_SCHEMA["properties"]["new_tag_candidates"]["maxItems"], 2)
        self.assertEqual(payload["messages"][0], {"role": "system", "content": "Metadata system prompt."})
        content = json.loads(payload["messages"][1]["content"])
        self.assertEqual(
            content["compiled_transcript"],
            "Eric Helms and Eric Trexler introduce the episode.",
        )
        self.assertIn("Eric Helms", content["compiled_transcript"])
        self.assertIn("Eric Trexler", content["compiled_transcript"])

    def test_metadata_normalization_never_invents_values(self):
        self.assertEqual(
            metadata.normalize_metadata(
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

    def test_summary_policy_version_invalidates_metadata_fingerprint(self):
        episode = self._episode()
        original = metadata.input_fingerprint(
            episode,
            b"Summary body",
            b"Compiled transcript",
        )
        with patch("podcast_engine.knowledge.metadata.SUMMARY_POLICY_VERSION", "summary-policy-test-version"):
            changed = metadata.input_fingerprint(
                episode,
                b"Summary body",
                b"Compiled transcript",
            )
        self.assertNotEqual(original, changed)

    def test_changed_transcript_invalidates_metadata_fingerprint(self):
        episode = self._episode()
        original = metadata.input_fingerprint(
            episode,
            b"Summary body",
            b"First compiled transcript",
        )
        changed = metadata.input_fingerprint(
            episode,
            b"Summary body",
            b"Changed compiled transcript",
        )
        self.assertNotEqual(original, changed)

    def test_current_summary_policy_invalidates_summary_v1_cache(self):
        episode = self._episode()
        fingerprint = summary.input_fingerprint(episode, b"Compiled transcript")
        manifest = {
            "schema_version": 1,
            "summary": {
                "policy_version": "summary-v1",
                "preset": summary.summary_preset(),
                "input_fingerprint": fingerprint,
                "generated_at": "2026-08-26T10:00:00+00:00",
            },
        }
        self.assertFalse(
            _summary_matches(
                manifest,
                fingerprint,
                summary.summary_preset(),
            )
        )

    def test_current_metadata_policy_invalidates_metadata_v1_cache(self):
        episode = self._episode()
        fingerprint = metadata.input_fingerprint(
            episode,
            b"Summary body",
            b"Compiled transcript",
        )
        manifest = {
            "metadata": {
                "policy_version": "metadata-v1",
                "preset": metadata.metadata_preset(),
                "input_fingerprint": fingerprint,
                "generated_at": "2026-08-26T10:00:00+00:00",
                "topics": [],
                "people": [],
                "tags": [],
                "tag_candidates": [],
            },
        }
        self.assertFalse(
            _metadata_matches(
                manifest,
                fingerprint,
                metadata.metadata_preset(),
            )
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
                stage="summary",
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
                    stage="summary",
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
                    stage="summary",
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
        ), patch("podcast_engine.knowledge.client.requests.post", return_value=response), p1, p2, p3, p4:
            with self.assertRaisesRegex(RuntimeError, "HTTP 400") as raised:
                client.post_openrouter(
                    payload,
                    episode_key="a" * 24,
                    source_fingerprint="sha256:" + "c" * 64,
                    stage="summary",
                    provenance=provenance,
                )

        message = str(raised.exception)
        self.assertIn("OpenRouter rejected request", message)
        self.assertNotIn("Summary body", message)
        self.assertNotIn("Eric Helms transcript", message)
        self.assertNotIn("knowledge-key", message)
        self.assertLessEqual(
            len(message),
            len("OpenRouter knowledge request failed (HTTP 400): ") + client.ERROR_BODY_LIMIT,
        )


class KnowledgeCacheTests(unittest.TestCase):
    def _episode(self):
        return {
            "id": "episode-1",
            "episode_key": "episode-1",
            "podcast": "Example Strength Podcast",
            "podcast_id": "example-strength",
            "category": "exercise_strength",
            "prompt": "strength",
            "title": "Ep 386 - Training Principles",
            "published": "2026-08-12T14:00:00Z",
            "link": "https://example.test/episode",
            "status": {"compiler": {"state": "completed"}},
            "files": {
                "sources": {
                    "apple": {"text": "episodes/episode-1/sources/apple.txt"},
                    "whisper": {"text": "episodes/episode-1/sources/whisper.txt"},
                }
            },
        }

    @staticmethod
    def _body(text="Cached point."):
        return (
            "## TL;DR\n\n"
            f"- {text}\n\n"
            "## Key Ideas\n\n"
            "### Topic\n\n"
            f"{text}\n"
        )

    def _review_output(self, transcript, draft, review_context):
        assessments = [
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
        audit = validate_audit_result(
            {
                "status": "pass",
                "risk_assessments": assessments,
                "additional_issues": [],
            },
            transcript,
            review_context,
        )
        edit = {"resolved_issue_ids": [], "final_markdown": draft}
        return {
            "audit_result": audit,
            "edit_result": edit,
            "accepted_final_markdown": draft,
            "completion_metadata": {
                "completion_id": "edit-1",
                "served_model": "anthropic/claude-sonnet-4.6",
                "served_provider": "Anthropic",
                "attempt_count": 2,
                "attempts": [
                    {
                        "phase": "audit",
                        "phase_attempt": 1,
                        "completion_id": "audit-1",
                        "served_model": "anthropic/claude-sonnet-4.6",
                        "served_provider": "Anthropic",
                    },
                    {
                        "phase": "edit",
                        "phase_attempt": 1,
                        "completion_id": "edit-1",
                        "served_model": "anthropic/claude-sonnet-4.6",
                        "served_provider": "Anthropic",
                    },
                ],
            },
            "review_context": review_context,
        }

    def _fixture(self, episode, body, transcript):
        review_context = build_review_context(transcript, body)
        output = self._review_output(transcript, body, review_context)
        bucket = _FakeBucket({})
        record = persist_review_attempt(
            bucket=bucket,
            episode_key=episode["episode_key"],
            transcript=transcript,
            draft=body,
            review_context=review_context,
            audit_result=output["audit_result"],
            edit_result=output["edit_result"],
            accepted_final=body,
            failure=None,
            summary_policy_version=SUMMARY_POLICY_VERSION,
            summary_preset="podcast-summary",
            review_policy_version=SUMMARY_REVIEW_POLICY_VERSION,
            review_preset="podcast-summary-review",
            completion_metadata=output["completion_metadata"],
            reviewed_at="2026-08-26T10:00:00+00:00",
        )
        manifest = {
            "schema_version": 1,
            "episode_key": episode["episode_key"],
            "summary": {
                "policy_version": SUMMARY_POLICY_VERSION,
                "preset": "podcast-summary",
                "input_fingerprint": summary.input_fingerprint(episode, transcript.encode()),
                "generated_at": "2026-08-26T10:00:00+00:00",
            },
            "summary_review": {
                "policy_version": SUMMARY_REVIEW_POLICY_VERSION,
                "preset": "podcast-summary-review",
                "review_id": record["review_id"],
                "artifact_root": record["artifact_root"],
                "status": "pass",
                "final_sha256": record["final_sha256"],
            },
            "metadata": {
                "policy_version": METADATA_POLICY_VERSION,
                "preset": "podcast-metadata",
                "input_fingerprint": metadata.input_fingerprint(
                    episode,
                    body.encode(),
                    transcript.encode(),
                    semantic_projection(initial_registry()),
                ),
                "generated_at": "2026-08-26T10:01:00+00:00",
                "topics": ["Training"],
                "people": ["Alice"],
                "tags": ["RPE"],
                "tag_candidates": [],
                "unknown_existing_tags": [],
            },
        }
        return manifest, dict(bucket.objects)

    def test_canonical_summary_generated_at_uses_the_manifest_summary_instant(self):
        episode = self._episode()
        body = self._body()
        manifest, _ = self._fixture(episode, body, "Compiled transcript")
        fake_bucket = _FakeBucket(
            {
                "episodes/episode-1/summary/metadata.json": json.dumps(manifest),
            }
        )

        with patch("podcast_engine.knowledge.orchestration.get_bucket", return_value=fake_bucket):
            generated_at = canonical_summary_generated_at("episode-1")

        self.assertEqual(generated_at, "2026-08-26T10:00:00+00:00")

    def _run(
        self,
        episode,
        transcript,
        body,
        manifest,
        review_objects,
        final_summary=None,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            compiled = root / "compiled.txt"
            compiled.write_text(transcript, encoding="utf-8")
            objects = {
                **review_objects,
                "episodes/episode-1/summary/body.md": body,
                "episodes/episode-1/summary/metadata.json": json.dumps(
                    manifest,
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
            }
            if final_summary is not None:
                objects["episodes/episode-1/summary/summary.md"] = final_summary
            fake_bucket = _FakeBucket(objects)
            with patch.dict(
                os.environ,
                {
                    "PODCAST_SUMMARY_REVIEW_PRESET": "podcast-summary-review",
                    "PODCAST_SUMMARY_REVIEW_API_KEY": "test-review-key",
                },
                clear=False,
            ), patch(
                "podcast_engine.knowledge.orchestration.RUNTIME_DIR",
                root / "knowledge",
            ), patch(
                "podcast_engine.knowledge.orchestration.download_file_from_gcs",
                return_value=str(compiled),
            ), patch(
                "podcast_engine.knowledge.orchestration.get_bucket",
                return_value=fake_bucket,
            ), patch(
                "podcast_engine.episode_generation.download_gcs_bytes",
                return_value=b"fixture canonical source bytes",
            ):
                path = build_knowledge_note(episode)
                return Path(path).read_text(encoding="utf-8"), fake_bucket

    def _rendered_summary(self, episode, body, manifest):
        metadata_values = {
            field: manifest["metadata"][field]
            for field in ("topics", "people", "tags")
        }
        return render(
            episode,
            metadata_values,
            manifest["summary"]["generated_at"],
            body,
        )

    def test_matching_summary_review_and_metadata_reuse_all_ai_results(self):
        episode = self._episode()
        transcript = "Compiled transcript"
        body = self._body()
        manifest, review_objects = self._fixture(episode, body, transcript)
        with patch("builtins.print") as printed, patch(
            "podcast_engine.knowledge.summary.generate"
        ) as summary_generate, patch(
            "podcast_engine.knowledge.summary_review.generate"
        ) as review_generate, patch(
            "podcast_engine.knowledge.metadata.generate"
        ) as metadata_generate:
            final, _ = self._run(
                episode,
                transcript,
                body,
                manifest,
                review_objects,
            )

        summary_generate.assert_not_called()
        review_generate.assert_not_called()
        metadata_generate.assert_not_called()
        self.assertEqual(
            printed.call_args_list,
            [
                call("Knowledge summary cache hit"),
                call("Knowledge metadata cache hit"),
            ],
        )
        self.assertIn("Cached point.", final)
        self.assertIn("created: '2026-08-26T10:00:00+00:00'", final)

    def test_matching_cache_does_not_rewrite_identical_knowledge_artifacts(self):
        episode = self._episode()
        transcript = "Compiled transcript"
        body = self._body()
        manifest, review_objects = self._fixture(episode, body, transcript)
        final = self._rendered_summary(episode, body, manifest)
        manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"

        with patch("builtins.print"), patch(
            "podcast_engine.knowledge.summary.generate"
        ) as summary_generate, patch(
            "podcast_engine.knowledge.summary_review.generate"
        ) as review_generate, patch(
            "podcast_engine.knowledge.metadata.generate"
        ) as metadata_generate:
            _, fake_bucket = self._run(
                episode,
                transcript,
                body,
                manifest,
                review_objects,
                final,
            )

        summary_generate.assert_not_called()
        review_generate.assert_not_called()
        metadata_generate.assert_not_called()
        self.assertEqual(
            [path for path in fake_bucket.uploads if path.startswith("episodes/")],
            [],
        )
        self.assertEqual(fake_bucket.objects["episodes/episode-1/summary/body.md"], body)
        self.assertEqual(fake_bucket.objects["episodes/episode-1/summary/summary.md"], final)
        self.assertEqual(fake_bucket.objects["episodes/episode-1/summary/metadata.json"], manifest_text)

    def test_changed_compiled_bytes_invalidates_summary_review_and_metadata(self):
        episode = self._episode()
        old_transcript = "Old transcript"
        old_body = self._body("Old point.")
        manifest, review_objects = self._fixture(episode, old_body, old_transcript)
        new_body = self._body("New point.")

        def review_generate(_episode, transcript, draft, review_context, **_kwargs):
            return self._review_output(transcript, draft, review_context)

        with patch(
            "podcast_engine.knowledge.summary.generate",
            return_value=new_body,
        ) as summary_generate, patch(
            "podcast_engine.knowledge.summary_review.generate",
            side_effect=review_generate,
        ) as review_generate, patch(
            "podcast_engine.knowledge.metadata.generate",
            return_value={"topics": [], "people": [], "tags": []},
        ) as metadata_generate:
            final, fake_bucket = self._run(
                episode,
                "Changed transcript",
                old_body,
                manifest,
                review_objects,
            )

        summary_generate.assert_called_once_with(
            episode,
            "Changed transcript",
            episode_key=ANY,
            source_fingerprint=ANY,
        )
        review_generate.assert_called_once()
        metadata_generate.assert_called_once()
        self.assertEqual(metadata_generate.call_args.args[2], "Changed transcript")
        self.assertIn("New point.", final)
        self.assertIn("episodes/episode-1/summary/body.md", fake_bucket.uploads)

    def test_tampered_canonical_body_invalidates_the_whole_accepted_review_chain(self):
        episode = self._episode()
        transcript = "Compiled transcript"
        original_body = self._body("Original point.")
        tampered_body = self._body("Changed externally.")
        manifest, review_objects = self._fixture(episode, original_body, transcript)
        regenerated_body = self._body("Regenerated point.")

        def review_generate(_episode, request_transcript, draft, review_context, **_kwargs):
            return self._review_output(request_transcript, draft, review_context)

        with patch(
            "podcast_engine.knowledge.summary.generate",
            return_value=regenerated_body,
        ) as summary_generate, patch(
            "podcast_engine.knowledge.summary_review.generate",
            side_effect=review_generate,
        ) as review_generate, patch(
            "podcast_engine.knowledge.metadata.generate",
            return_value={"topics": ["Topic"], "people": [], "tags": []},
        ) as metadata_generate:
            final, _ = self._run(
                episode,
                transcript,
                tampered_body,
                manifest,
                review_objects,
            )

        summary_generate.assert_called_once()
        review_generate.assert_called_once()
        metadata_generate.assert_called_once()
        self.assertIn("Regenerated point.", final)


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
                "link": "https://example-strength.libsyn.com/ep-386-sleep-hypertrophy-and-creatine",
            },
            {"topics": [], "people": [], "tags": []},
            "date",
            "# Summary\n",
        )

        self.assertIn(
            "source_url: https://example-strength.libsyn.com/website/ep-386-sleep-hypertrophy-and-creatine",
            rendered,
        )

    def test_frontmatter_uses_podcast_url_without_inventing_source_url(self):
        rendered = render(
            {
                "podcast": "Example Nutrition Podcast",
                "podcast_id": "example-nutrition",
                "podcast_url": "https://example.com/nutrition-podcast/",
                "link": None,
            },
            {"topics": [], "people": [], "tags": []},
            "date",
            "# Summary\n",
        )

        self.assertIn(
            "podcast_url: https://example.com/nutrition-podcast/",
            rendered,
        )
        self.assertNotIn("source_url:", rendered)

    def test_frontmatter_omits_missing_podcast_url(self):
        rendered = render(
            {
                "podcast": "Example Strength Podcast",
                "podcast_id": "example-strength",
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
        self.assertNotIn("- example-strength", rendered)
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
