import importlib
import json
import unittest
from pathlib import Path

from podcast_engine import episode_contract
from podcast_engine.knowledge.summary_review_contract import validate_audit_result
from podcast_engine.knowledge.summary_review_evidence import build_review_context


ROOT = Path(__file__).resolve().parents[1]
STORAGE_PATH = ROOT / "podcast_engine" / "knowledge" / "summary_review_storage.py"

DRAFT = (
    "## TL;DR\n\n"
    "- The measured result reached 99%.\n\n"
    "## Key Ideas\n\n"
    "### Measurement\n\n"
    "The measured result reached 99%.\n"
)
TRANSCRIPT = "The measured result reached 99% in the reported test."
REVIEWED_AT = "2026-09-05T15:30:00+00:00"


def _context() -> dict:
    return build_review_context(TRANSCRIPT, DRAFT)


def _paths(episode_key: str, review_id: str) -> dict[str, str]:
    return episode_contract.summary_review_artifact_paths(episode_key, review_id)


def _history_index_path(episode_key: str) -> str:
    return f"episodes/{episode_key}/summary/reviews/index.json"


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


class _FakeBlob:
    def __init__(self, bucket, name: str):
        self.bucket = bucket
        self.name = name

    def exists(self):
        return self.name in self.bucket.objects

    @property
    def generation(self):
        return self.bucket.generations.get(self.name)

    def reload(self):
        from google.api_core.exceptions import NotFound

        if not self.exists():
            raise NotFound("missing")

    def upload_from_string(self, content, *, content_type=None, if_generation_match=None):
        from google.api_core.exceptions import PreconditionFailed

        current_generation = self.bucket.generations.get(self.name, 0)
        if (
            if_generation_match is not None
            and current_generation != if_generation_match
        ):
            raise PreconditionFailed("generation mismatch")

        data = content.encode("utf-8") if isinstance(content, str) else bytes(content)
        self.bucket.objects[self.name] = data
        self.bucket.generations[self.name] = current_generation + 1
        self.bucket.uploads.append(self.name)
        self.bucket.events.append(("upload", self.name))

    def download_as_bytes(self, *, if_generation_match=None):
        from google.api_core.exceptions import PreconditionFailed

        current_generation = self.bucket.generations.get(self.name, 0)
        if (
            if_generation_match is not None
            and current_generation != if_generation_match
        ):
            raise PreconditionFailed("generation mismatch")

        self.bucket.events.append(("download", self.name))
        return self.bucket.objects[self.name]

    def download_as_text(self, encoding="utf-8", *, if_generation_match=None):
        return self.download_as_bytes(
            if_generation_match=if_generation_match
        ).decode(encoding)


class _FakeBucket:
    def __init__(self):
        self.objects = {}
        self.generations = {}
        self.uploads = []
        self.events = []

    def blob(self, name: str):
        return _FakeBlob(self, name)


class SummaryReviewStorageV2Tests(unittest.TestCase):
    def test_artifact_paths_include_all_v2_evidence_indexes(self):
        self.assertTrue(
            hasattr(episode_contract, "summary_review_artifact_paths"),
            "V2 storage requires immutable summary-review artifact paths",
        )
        paths = _paths("episode-1", "sr-test")
        root = "episodes/episode-1/summary/reviews/sr-test/"
        self.assertEqual(
            paths,
            {
                "root": root,
                "draft": root + "draft.md",
                "final": root + "final.md",
                "review": root + "review.json",
                "transcript_index": root + "transcript-span-index.json",
                "draft_index": root + "draft-block-index.json",
                "risk_inventory": root + "risk-inventory.json",
            },
        )

    def test_v2_storage_module_exists(self):
        self.assertTrue(
            STORAGE_PATH.exists(),
            "Task 2→3 migration requires V2 immutable review storage",
        )

    @unittest.skipUnless(STORAGE_PATH.exists(), "v2 storage not implemented yet")
    def test_persistence_registers_review_only_after_immutable_evidence_verification(self):
        storage = importlib.import_module(
            "podcast_engine.knowledge.summary_review_storage"
        )
        bucket = _FakeBucket()
        review_context = _context()

        record = storage.persist_review_attempt(
            bucket=bucket,
            episode_key="episode-1",
            transcript=TRANSCRIPT,
            draft=DRAFT,
            review_context=review_context,
            audit_result=None,
            edit_result=None,
            accepted_final=None,
            failure={"stage": "request", "code": "request_failed"},
            summary_policy_version="summary-v6",
            summary_preset="podcast-summary",
            review_policy_version="summary-review-v2",
            review_preset="podcast-summary-review",
            completion_metadata={},
            reviewed_at=REVIEWED_AT,
        )

        paths = _paths("episode-1", record["review_id"])
        index_path = _history_index_path("episode-1")

        self.assertIn(index_path, bucket.objects)
        self.assertEqual(bucket.uploads[-1], index_path)
        self.assertEqual(
            json.loads(bucket.objects[index_path]),
            {
                "schema_version": 1,
                "review_ids": [record["review_id"]],
            },
        )

        index_upload = bucket.events.index(("upload", index_path))
        for evidence_path in (
            paths["draft"],
            paths["transcript_index"],
            paths["draft_index"],
            paths["risk_inventory"],
            paths["review"],
        ):
            with self.subTest(path=evidence_path):
                verification_reads = [
                    position
                    for position, event in enumerate(bucket.events)
                    if event == ("download", evidence_path)
                ]
                self.assertTrue(
                    verification_reads,
                    f"{evidence_path} must be verified before history registration",
                )
                self.assertLess(max(verification_reads), index_upload)

    @unittest.skipUnless(STORAGE_PATH.exists(), "v2 storage not implemented yet")
    def test_accepted_pass_persists_context_before_final_and_binds_hashes_in_envelope(self):
        storage = importlib.import_module("podcast_engine.knowledge.summary_review_storage")
        bucket = _FakeBucket()
        review_context = _context()
        audit = _pass_audit(review_context)
        edit = {"resolved_issue_ids": [], "final_markdown": DRAFT}

        record = storage.persist_review_attempt(
            bucket=bucket,
            episode_key="episode-1",
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
                "total_tokens": 240,
                "cost": 0.02,
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
            reviewed_at=REVIEWED_AT,
        )

        paths = _paths("episode-1", record["review_id"])
        self.assertEqual(
            bucket.uploads,
            [
                paths["draft"],
                paths["transcript_index"],
                paths["draft_index"],
                paths["risk_inventory"],
                paths["final"],
                paths["review"],
                _history_index_path("episode-1"),
            ],
        )
        envelope = json.loads(bucket.objects[paths["review"]])
        self.assertEqual(envelope["schema_version"], 2)
        self.assertEqual(envelope["audit_result"], audit)
        self.assertEqual(envelope["edit_result"], edit)
        self.assertEqual(
            envelope["evidence_context"],
            {
                "transcript_span_algorithm": review_context["transcript_span_index"]["algorithm"],
                "transcript_span_index_sha256": review_context["transcript_span_index"]["index_sha256"],
                "draft_block_algorithm": review_context["draft_block_index"]["algorithm"],
                "draft_block_index_sha256": review_context["draft_block_index"]["index_sha256"],
                "risk_inventory_algorithm": review_context["risk_inventory"]["algorithm"],
                "risk_inventory_sha256": review_context["risk_inventory"]["index_sha256"],
            },
        )
        self.assertEqual(envelope["completion_metadata"]["attempt_count"], 2)
        self.assertEqual(len(envelope["completion_metadata"]["attempts"]), 2)
        serialized = bucket.objects[paths["review"]].decode("utf-8")
        self.assertNotIn(TRANSCRIPT, serialized)
        self.assertNotIn("PODCAST_SUMMARY_REVIEW_API_KEY", serialized)
        self.assertNotIn("Authorization", serialized)

    @unittest.skipUnless(STORAGE_PATH.exists(), "v2 storage not implemented yet")
    def test_identical_persistence_replay_is_an_immutable_noop(self):
        storage = importlib.import_module("podcast_engine.knowledge.summary_review_storage")
        bucket = _FakeBucket()
        review_context = _context()
        audit = _pass_audit(review_context)
        edit = {"resolved_issue_ids": [], "final_markdown": DRAFT}
        kwargs = dict(
            bucket=bucket,
            episode_key="episode-1",
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
                "completion_id": "edit-replay-1",
                "served_model": "anthropic/claude-sonnet-4.6",
                "served_provider": "Anthropic",
                "attempt_count": 2,
                "attempts": [
                    {
                        "phase": "audit",
                        "phase_attempt": 1,
                        "completion_id": "audit-replay-1",
                        "served_model": "anthropic/claude-sonnet-4.6",
                        "served_provider": "Anthropic",
                    },
                    {
                        "phase": "edit",
                        "phase_attempt": 1,
                        "completion_id": "edit-replay-1",
                        "served_model": "anthropic/claude-sonnet-4.6",
                        "served_provider": "Anthropic",
                    },
                ],
            },
            reviewed_at=REVIEWED_AT,
        )

        first = storage.persist_review_attempt(**kwargs)
        first_uploads = list(bucket.uploads)
        second = storage.persist_review_attempt(**kwargs)

        self.assertEqual(second["review_id"], first["review_id"])
        self.assertEqual(bucket.uploads, first_uploads)
        self.assertTrue(second["writes"])
        self.assertTrue(all(value == "unchanged" for value in second["writes"].values()))

    @unittest.skipUnless(STORAGE_PATH.exists(), "v2 storage not implemented yet")
    def test_failed_attempt_still_persists_all_python_context_artifacts_but_no_final(self):
        storage = importlib.import_module("podcast_engine.knowledge.summary_review_storage")
        bucket = _FakeBucket()
        review_context = _context()

        record = storage.persist_review_attempt(
            bucket=bucket,
            episode_key="episode-1",
            transcript=TRANSCRIPT,
            draft=DRAFT,
            review_context=review_context,
            audit_result=None,
            edit_result=None,
            accepted_final=None,
            failure={"stage": "request", "code": "request_failed"},
            summary_policy_version="summary-v6",
            summary_preset="podcast-summary",
            review_policy_version="summary-review-v2",
            review_preset="podcast-summary-review",
            completion_metadata={},
            reviewed_at=REVIEWED_AT,
        )

        paths = _paths("episode-1", record["review_id"])
        self.assertEqual(
            bucket.uploads,
            [
                paths["draft"],
                paths["transcript_index"],
                paths["draft_index"],
                paths["risk_inventory"],
                paths["review"],
                _history_index_path("episode-1"),
            ],
        )
        self.assertNotIn(paths["final"], bucket.objects)

    @unittest.skipUnless(STORAGE_PATH.exists(), "v2 storage not implemented yet")
    def test_completion_attempt_count_allows_three_but_rejects_four(self):
        storage = importlib.import_module("podcast_engine.knowledge.summary_review_storage")
        review_context = _context()
        audit = _pass_audit(review_context)
        edit = {"resolved_issue_ids": [], "final_markdown": DRAFT}

        base = dict(
            bucket=_FakeBucket(),
            episode_key="episode-1",
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
            reviewed_at=REVIEWED_AT,
        )
        storage.persist_review_attempt(
            **base,
            completion_metadata={"attempt_count": 3},
        )
        with self.assertRaises(ValueError):
            storage.persist_review_attempt(
                **{**base, "bucket": _FakeBucket()},
                completion_metadata={"attempt_count": 4},
            )


if __name__ == "__main__":
    unittest.main()
