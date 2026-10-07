import json
import os

# Legacy-path tests: a developer .env (loaded by knowledge.client via
# python-dotenv) may activate the TASK-118 note writer. Keep these tests
# hermetic -- the writer path has its own tests.
os.environ["PODCAST_KNOWLEDGE_WRITER_PRESET"] = ""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import ANY, Mock, patch

import podcast_engine.knowledge.orchestration as knowledge
from podcast_engine.episode_contract import summary_review_artifact_paths
from podcast_engine.knowledge.summary_review_contract import validate_audit_result
from podcast_engine.knowledge.summary_review_evidence import build_review_context
from podcast_engine.knowledge.summary_review_storage import persist_review_attempt
from podcast_engine.knowledge.tags import initial_registry, semantic_projection


DRAFT = (
    "## TL;DR\n\n"
    "- The measured result reached 99%.\n\n"
    "## Key Ideas\n\n"
    "### Measurement\n\n"
    "The measured result reached 99%.\n"
)
TRANSCRIPT = "The measured result reached 99% in the reported test."


def _episode() -> dict:
    return {
        "id": "episode-1",
        "episode_key": "episode-1",
        "podcast": "Example Strength Podcast",
        "podcast_id": "example-strength",
        "category": "exercise_strength",
        "prompt": "strength",
        "title": "Generation test",
        "published": "2026-09-04T00:00:00Z",
        "link": "https://example.test/episode",
        "status": {"compiler": {"state": "completed"}},
        "files": {
            "sources": {
                "apple": {"text": "episodes/episode-1/sources/apple.txt"},
                "whisper": {"text": "episodes/episode-1/sources/whisper.txt"},
            }
        },
    }


def _pass_audit(review_context: dict) -> dict:
    return validate_audit_result(
        {
            "status": "pass",
            "risk_assessments": [
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
            ],
            "additional_issues": [],
        },
        TRANSCRIPT,
        review_context,
    )


class _Blob:
    def __init__(self, bucket, name: str):
        self.bucket = bucket
        self.name = name
        self.generation = 1

    def exists(self):
        return self.name in self.bucket.objects

    def reload(self):
        if not self.exists():
            from google.api_core.exceptions import NotFound

            raise NotFound("missing")

    def download_as_bytes(self):
        return self.bucket.objects[self.name]

    def download_as_text(self, encoding="utf-8", if_generation_match=None):
        if (
            if_generation_match is not None
            and self.generation != if_generation_match
        ):
            from google.api_core.exceptions import PreconditionFailed

            raise PreconditionFailed("generation mismatch")
        return self.bucket.objects[self.name].decode(encoding)

    def upload_from_string(self, content, *, content_type=None, if_generation_match=None):
        if if_generation_match == 0 and self.exists():
            from google.api_core.exceptions import PreconditionFailed

            raise PreconditionFailed("exists")
        data = content.encode("utf-8") if isinstance(content, str) else bytes(content)
        self.bucket.objects[self.name] = data
        self.bucket.uploads.append(self.name)

    def upload_from_filename(self, filename, *, content_type=None):
        self.bucket.objects[self.name] = Path(filename).read_bytes()
        self.bucket.uploads.append(self.name)


class _Bucket:
    def __init__(self, objects=None):
        self.objects = {
            name: value.encode("utf-8") if isinstance(value, str) else bytes(value)
            for name, value in (objects or {}).items()
        }
        self.uploads = []

    def blob(self, name: str):
        return _Blob(self, name)


class _Registry:
    def __init__(self, *, bucket=None):
        self.bucket = bucket

    def load(self):
        return initial_registry()

    def resolve_and_record(self, episode_key, extracted):
        return {
            "topics": list(extracted.get("topics", [])),
            "people": list(extracted.get("people", [])),
            "tags": list(extracted.get("tags", [])),
            "tag_candidates": [],
            "unknown_existing_tags": [],
        }


class SummaryReviewKnowledgeOrchestrationV2Tests(unittest.TestCase):
    def _run_cache_miss(self, *, persist_side_effect=None):
        episode = _episode()
        review_context = build_review_context(TRANSCRIPT, DRAFT)
        audit = _pass_audit(review_context)
        edit = {"resolved_issue_ids": [], "final_markdown": DRAFT}
        review_output = {
            "audit_result": audit,
            "edit_result": edit,
            "accepted_final_markdown": DRAFT,
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

        bucket = _Bucket()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            compiled = root / "compiled.txt"
            compiled.write_text(TRANSCRIPT, encoding="utf-8")

            def persisted(**kwargs):
                self.assertIs(kwargs["review_context"], review_context)
                self.assertNotIn("episodes/episode-1/summary/body.md", bucket.uploads)
                if persist_side_effect is not None:
                    raise persist_side_effect
                return {
                    "review_id": "sr-test",
                    "artifact_root": "episodes/episode-1/summary/reviews/sr-test/",
                    "status": "pass",
                    "final_sha256": knowledge._sha256_bytes(DRAFT.encode("utf-8")),
                }

            with patch.dict(
                os.environ,
                {
                    "PODCAST_SUMMARY_REVIEW_PRESET": "podcast-summary-review",
                    "PODCAST_SUMMARY_REVIEW_API_KEY": "test-review-key",
                },
                clear=False,
            ), patch.object(knowledge, "RUNTIME_DIR", root / "knowledge"), patch.object(
                knowledge,
                "download_file_from_gcs",
                return_value=str(compiled),
            ), patch.object(knowledge, "get_bucket", return_value=bucket), patch(
                "podcast_engine.episode_generation.download_gcs_bytes",
                return_value=b"fixture canonical source bytes",
            ), patch.object(
                knowledge,
                "TagRegistry",
                _Registry,
            ), patch.object(
                knowledge.summary,
                "generate",
                return_value=DRAFT,
            ) as summary_generate, patch.object(
                knowledge,
                "build_review_context",
                return_value=review_context,
                create=True,
            ) as context_builder, patch.object(
                knowledge.summary_review,
                "generate",
                return_value=review_output,
                create=True,
            ) as review_generate, patch.object(
                knowledge,
                "persist_review_attempt",
                side_effect=persisted,
                create=True,
            ) as persist, patch.object(
                knowledge.metadata,
                "generate",
                return_value={"topics": [], "people": [], "tags": []},
            ) as metadata_generate:
                if persist_side_effect is not None:
                    with self.assertRaisesRegex(RuntimeError, "immutable persistence"):
                        knowledge.build_knowledge_note(episode)
                    return {
                        "bucket": bucket,
                        "metadata_generate": metadata_generate,
                    }
                path = knowledge.build_knowledge_note(episode)

        return {
            "path": path,
            "bucket": bucket,
            "summary_generate": summary_generate,
            "context_builder": context_builder,
            "review_generate": review_generate,
            "persist": persist,
            "metadata_generate": metadata_generate,
            "review_context": review_context,
        }

    def test_cache_miss_builds_context_once_and_persists_same_object_before_canonical_body(self):
        result = self._run_cache_miss()
        result["summary_generate"].assert_called_once_with(
            _episode(),
            TRANSCRIPT,
            episode_key=ANY,
            source_fingerprint=ANY,
        )
        result["context_builder"].assert_called_once_with(TRANSCRIPT, DRAFT)
        result["review_generate"].assert_called_once_with(
            _episode(),
            TRANSCRIPT,
            DRAFT,
            result["review_context"],
            episode_key=ANY,
            source_fingerprint=ANY,
        )
        self.assertEqual(result["persist"].call_count, 1)
        self.assertIn("episodes/episode-1/summary/body.md", result["bucket"].uploads)
        result["metadata_generate"].assert_called_once()

    def test_storage_failure_occurs_before_body_upload_and_metadata_generation(self):
        result = self._run_cache_miss(
            persist_side_effect=RuntimeError("immutable persistence failed")
        )
        self.assertNotIn("episodes/episode-1/summary/body.md", result["bucket"].uploads)
        result["metadata_generate"].assert_not_called()

    def test_accepted_cache_requires_all_six_immutable_artifacts_and_exact_revalidation(self):
        self.assertTrue(
            hasattr(knowledge, "_accepted_review_artifacts_match"),
            "V2 cache reuse requires immutable review evidence revalidation",
        )
        episode = _episode()
        review_context = build_review_context(TRANSCRIPT, DRAFT)
        audit = _pass_audit(review_context)
        edit = {"resolved_issue_ids": [], "final_markdown": DRAFT}
        bucket = _Bucket()
        record = persist_review_attempt(
            bucket=bucket,
            episode_key=episode["episode_key"],
            transcript=TRANSCRIPT,
            draft=DRAFT,
            review_context=review_context,
            audit_result=audit,
            edit_result=edit,
            accepted_final=DRAFT,
            failure=None,
            summary_policy_version="summary-v6",
            summary_preset="podcast-summary",
            review_policy_version="summary-review-v2",
            review_preset="podcast-summary-review",
            completion_metadata={
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
            reviewed_at="2026-09-05T15:30:00+00:00",
        )
        manifest = {
            "schema_version": 1,
            "episode_key": episode["episode_key"],
            "summary": {
                "policy_version": "summary-v6",
                "preset": "podcast-summary",
                "input_fingerprint": knowledge.summary.input_fingerprint(
                    episode,
                    TRANSCRIPT.encode("utf-8"),
                ),
                "generated_at": "2026-09-05T15:30:00+00:00",
            },
            "summary_review": {
                "policy_version": "summary-review-v2",
                "preset": "podcast-summary-review",
                "review_id": record["review_id"],
                "artifact_root": record["artifact_root"],
                "status": "pass",
                "final_sha256": record["final_sha256"],
            },
        }

        self.assertTrue(
            knowledge._accepted_review_artifacts_match(
                bucket,
                episode["episode_key"],
                manifest,
                DRAFT.encode("utf-8"),
                TRANSCRIPT.encode("utf-8"),
            )
        )

        paths = summary_review_artifact_paths(episode["episode_key"], record["review_id"])
        removed = bucket.objects.pop(paths["risk_inventory"])
        self.assertFalse(
            knowledge._accepted_review_artifacts_match(
                bucket,
                episode["episode_key"],
                manifest,
                DRAFT.encode("utf-8"),
                TRANSCRIPT.encode("utf-8"),
            )
        )
        bucket.objects[paths["risk_inventory"]] = removed
        bucket.objects[paths["draft_index"]] = b"{}\n"
        self.assertFalse(
            knowledge._accepted_review_artifacts_match(
                bucket,
                episode["episode_key"],
                manifest,
                DRAFT.encode("utf-8"),
                TRANSCRIPT.encode("utf-8"),
            )
        )


if __name__ == "__main__":
    unittest.main()
