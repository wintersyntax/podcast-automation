from decimal import Decimal
import importlib
import json
import os
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from podcast_engine.preset_provenance import PresetProvenance
from podcast_engine.knowledge.summary_review_contract import (
    SUMMARY_REVIEW_AUDIT_SCHEMA,
    SUMMARY_REVIEW_EDIT_SCHEMA,
    build_edit_obligations,
    validate_audit_result,
)
from podcast_engine.knowledge.summary_review_evidence import build_review_context


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "podcast_engine" / "knowledge" / "summary_review.py"

DRAFT = (
    "## TL;DR\n\n"
    "- The measured result reached 99%.\n\n"
    "## Key Ideas\n\n"
    "### Measurement\n\n"
    "The measured result reached 99%.\n"
)
TRANSCRIPT = "The measured result reached 99% in the reported test."
EPISODE = {
    "episode_key": "episode-1",
    "podcast": "Example Strength Podcast",
    "podcast_id": "example-strength",
    "category": "exercise_strength",
    "prompt": "strength",
    "title": "Generation test",
    "published": "2026-09-04T00:00:00Z",
    "link": "https://example.test/episode",
}
EPISODE_KEY = "a" * 24
SOURCE_FINGERPRINT = "sha256:" + "c" * 64


def _provenance() -> PresetProvenance:
    return PresetProvenance(
        status="verified",
        slug="podcast-summary-review",
        preset_id="preset-id",
        version_id="version-id-1",
        version=1,
        config={"model": "test/reviewer-model", "max_tokens": 2000},
        system_prompt="Verified summary review system prompt.",
        config_digest="sha256:config",
        system_prompt_digest="sha256:prompt",
        verified_at="2026-09-16T00:00:00Z",
    )


def _supported_audit(review_context: dict) -> dict:
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
    return validate_audit_result(
        {
            "status": "pass",
            "risk_assessments": assessments,
            "additional_issues": [],
        },
        TRANSCRIPT,
        review_context,
    )


def _raw_supported_audit(review_context: dict) -> dict:
    return {
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
    }


def _response(content: dict, *, completion_id: str, cost: float = 0.01) -> Mock:
    response = Mock()
    response.json.return_value = {
        "id": completion_id,
        "model": "anthropic/claude-sonnet-4.6",
        "openrouter_metadata": {
            "endpoints": {
                "available": [
                    {"provider": "Anthropic", "selected": True},
                ]
            }
        },
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "cost": cost,
        },
        "choices": [{"message": {"content": json.dumps(content)}}],
    }
    return response


class SummaryReviewGenerationV2Tests(unittest.TestCase):
    def setUp(self):
        # Every generate() call in this file now resolves a verified preset
        # snapshot and reserves episode AI budget before each physical call
        # (TASK-076 Task 7); these tests are about the AUDIT/EDIT state
        # machine and repair-prompt content, not preset resolution or
        # budget mechanics (see test_knowledge_ai_budget.py for those), so
        # the resolution and ledger side effects are patched away here for
        # every test in this class.
        env_patcher = patch.dict(
            os.environ, {"PODCAST_SUMMARY_REVIEW_API_KEY": "review-key"}, clear=False
        )
        env_patcher.start()
        self.addCleanup(env_patcher.stop)

        fetch_patcher = patch(
            "podcast_engine.knowledge.summary_review.fetch_current_designated_preset",
            return_value=_provenance(),
        )
        fetch_patcher.start()
        self.addCleanup(fetch_patcher.stop)

        pricing_patcher = patch(
            "podcast_engine.knowledge.summary_review.resolve_preset_model_and_pricing"
        )
        resolved_mock = pricing_patcher.start()
        resolved_mock.return_value.config = _provenance().config
        resolved_mock.return_value.system_prompt = _provenance().system_prompt
        resolved_mock.return_value.pricing_bound.prompt_usd_per_token = Decimal("0.000001")
        resolved_mock.return_value.pricing_bound.completion_usd_per_token = Decimal("0.000002")
        resolved_mock.return_value.pricing_bound.context_length = 200000
        self.addCleanup(pricing_patcher.stop)

        for name in (
            "reserve_budget_batch",
            "settle_budget_attempt",
            "mark_budget_attempt_uncertain",
        ):
            budget_patcher = patch(f"podcast_engine.knowledge.summary_review.{name}")
            budget_patcher.start()
            self.addCleanup(budget_patcher.stop)

    def test_v2_production_reviewer_module_exists(self):
        self.assertTrue(
            MODULE_PATH.exists(),
            "Task 2→3 migration requires the V2 production reviewer consumer",
        )

    @unittest.skipUnless(MODULE_PATH.exists(), "production reviewer not implemented yet")
    def test_audit_payload_uses_v2_schema_and_only_grounded_sources_plus_python_context(self):
        summary_review = importlib.import_module("podcast_engine.knowledge.summary_review")
        review_context = build_review_context(TRANSCRIPT, DRAFT)

        payload = summary_review.openrouter_audit_payload(
            EPISODE,
            TRANSCRIPT,
            DRAFT,
            review_context,
        )

        self.assertTrue(payload["response_format"]["json_schema"]["strict"])
        self.assertIs(
            payload["response_format"]["json_schema"]["schema"],
            SUMMARY_REVIEW_AUDIT_SCHEMA,
        )
        self.assertEqual(len(payload["messages"]), 1)
        content = json.loads(payload["messages"][0]["content"])
        self.assertEqual(
            set(content),
            {
                "episode",
                "podcast_profile",
                "compiled_transcript",
                "draft_summary",
                "python_review_context",
            },
        )
        self.assertEqual(content["compiled_transcript"], TRANSCRIPT)
        self.assertEqual(content["draft_summary"], DRAFT)
        self.assertEqual(content["python_review_context"], review_context)

        serialized = payload["messages"][0]["content"].casefold()
        for forbidden in (
            "apple_transcript",
            "whisper",
            "audio_input",
            "evaluator_expectation",
            "expected_gate",
        ):
            self.assertNotIn(forbidden, serialized)

    @unittest.skipUnless(MODULE_PATH.exists(), "production reviewer not implemented yet")
    def test_edit_payload_is_separate_and_receives_validated_audit_and_immutable_obligations(self):
        summary_review = importlib.import_module("podcast_engine.knowledge.summary_review")
        review_context = build_review_context(TRANSCRIPT, DRAFT)
        audit = _supported_audit(review_context)
        obligations = build_edit_obligations(audit)

        payload = summary_review.openrouter_edit_payload(
            EPISODE,
            TRANSCRIPT,
            DRAFT,
            review_context,
            audit,
            obligations,
        )

        self.assertTrue(payload["response_format"]["json_schema"]["strict"])
        self.assertIs(
            payload["response_format"]["json_schema"]["schema"],
            SUMMARY_REVIEW_EDIT_SCHEMA,
        )
        content = json.loads(payload["messages"][0]["content"])
        self.assertEqual(
            set(content),
            {
                "episode",
                "podcast_profile",
                "compiled_transcript",
                "draft_summary",
                "python_review_context",
                "validated_audit",
                "edit_obligations",
            },
        )
        self.assertEqual(content["compiled_transcript"], TRANSCRIPT)
        self.assertEqual(content["draft_summary"], DRAFT)
        self.assertEqual(content["python_review_context"], review_context)
        self.assertEqual(content["validated_audit"], audit)
        self.assertEqual(content["edit_obligations"], obligations)

    @unittest.skipUnless(MODULE_PATH.exists(), "production reviewer not implemented yet")
    def test_pass_audit_always_calls_separate_edit_and_accepts_byte_preserving_draft(self):
        summary_review = importlib.import_module("podcast_engine.knowledge.summary_review")
        review_context = build_review_context(TRANSCRIPT, DRAFT)
        audit = _raw_supported_audit(review_context)
        edit = {"resolved_issue_ids": [], "final_markdown": DRAFT}

        with patch.object(
            summary_review,
            "post_review_openrouter",
            side_effect=[
                _response(audit, completion_id="audit-1"),
                _response(edit, completion_id="edit-1"),
            ],
        ) as post:
            result = summary_review.generate(
                EPISODE,
                TRANSCRIPT,
                DRAFT,
                review_context,
                episode_key=EPISODE_KEY,
                source_fingerprint=SOURCE_FINGERPRINT,
            )

        self.assertEqual(post.call_count, 2)
        self.assertEqual(
            post.call_args_list[0].args[0]["response_format"]["json_schema"]["schema"],
            SUMMARY_REVIEW_AUDIT_SCHEMA,
        )
        self.assertEqual(
            post.call_args_list[1].args[0]["response_format"]["json_schema"]["schema"],
            SUMMARY_REVIEW_EDIT_SCHEMA,
        )
        self.assertEqual(result["audit_result"], _supported_audit(review_context))
        self.assertEqual(result["edit_result"], edit)
        self.assertEqual(result["accepted_final_markdown"], DRAFT)
        self.assertEqual(result["review_context"], review_context)
        self.assertEqual(result["completion_metadata"]["attempt_count"], 2)
        self.assertEqual(
            [item["phase"] for item in result["completion_metadata"]["attempts"]],
            ["audit", "edit"],
        )

    @unittest.skipUnless(MODULE_PATH.exists(), "production reviewer not implemented yet")
    def test_fail_audit_is_terminal_and_never_calls_edit(self):
        summary_review = importlib.import_module("podcast_engine.knowledge.summary_review")
        review_context = build_review_context(TRANSCRIPT, DRAFT)
        audit = {**_raw_supported_audit(review_context), "status": "fail"}

        with patch.object(
            summary_review,
            "post_review_openrouter",
            return_value=_response(audit, completion_id="audit-fail"),
        ) as post:
            result = summary_review.generate(
                EPISODE,
                TRANSCRIPT,
                DRAFT,
                review_context,
                episode_key=EPISODE_KEY,
                source_fingerprint=SOURCE_FINGERPRINT,
            )

        self.assertEqual(post.call_count, 1)
        self.assertEqual(result["audit_result"]["status"], "fail")
        self.assertIsNone(result["edit_result"])
        self.assertIsNone(result["accepted_final_markdown"])

    @unittest.skipUnless(MODULE_PATH.exists(), "production reviewer not implemented yet")
    def test_audit_repair_prompt_discloses_specific_validation_failure(self):
        summary_review = importlib.import_module("podcast_engine.knowledge.summary_review")
        review_context = build_review_context(TRANSCRIPT, DRAFT)
        invalid_audit = {"status": "pass", "risk_assessments": [], "additional_issues": []}
        valid_audit = _raw_supported_audit(review_context)
        noop_edit = {"resolved_issue_ids": [], "final_markdown": DRAFT}

        with patch.object(
            summary_review,
            "post_review_openrouter",
            side_effect=[
                _response(invalid_audit, completion_id="audit-1"),
                _response(valid_audit, completion_id="audit-2"),
                _response(noop_edit, completion_id="edit-1"),
            ],
        ) as post:
            summary_review.generate(
                EPISODE,
                TRANSCRIPT,
                DRAFT,
                review_context,
                episode_key=EPISODE_KEY,
                source_fingerprint=SOURCE_FINGERPRINT,
            )

        repair_payload = post.call_args_list[1][0][0]
        repair_instruction = repair_payload["messages"][-1]["content"]
        self.assertIn(
            "risk_assessments_must_adjudicate_every_python_owned_risk_exactly_once",
            repair_instruction,
        )
        self.assertNotIn("invalid_audit_contract", repair_instruction)

    def test_edit_repair_prompt_discloses_specific_validation_failure(self):
        summary_review = importlib.import_module("podcast_engine.knowledge.summary_review")
        review_context = build_review_context(TRANSCRIPT, DRAFT)
        audit = _raw_supported_audit(review_context)
        invalid_edit = {"resolved_issue_ids": ["R9999"], "final_markdown": DRAFT}
        repaired_edit = {"resolved_issue_ids": [], "final_markdown": DRAFT}

        with patch.object(
            summary_review,
            "post_review_openrouter",
            side_effect=[
                _response(audit, completion_id="audit-1"),
                _response(invalid_edit, completion_id="edit-1"),
                _response(repaired_edit, completion_id="edit-2"),
            ],
        ) as post:
            summary_review.generate(
                EPISODE,
                TRANSCRIPT,
                DRAFT,
                review_context,
                episode_key=EPISODE_KEY,
                source_fingerprint=SOURCE_FINGERPRINT,
            )

        repair_payload = post.call_args_list[2][0][0]
        repair_instruction = repair_payload["messages"][-1]["content"]
        self.assertIn(
            "resolved_issue_ids_must_equal_the_python_owned_obligation_set_exactly",
            repair_instruction,
        )
        self.assertNotIn("invalid_edit_contract", repair_instruction)

    def test_one_audit_repair_consumes_the_shared_repair_budget_before_edit(self):
        summary_review = importlib.import_module("podcast_engine.knowledge.summary_review")
        review_context = build_review_context(TRANSCRIPT, DRAFT)
        invalid_audit = {"status": "pass", "risk_assessments": [], "additional_issues": []}
        valid_audit = _raw_supported_audit(review_context)
        invalid_edit = {"resolved_issue_ids": [], "final_markdown": DRAFT.replace("99%", "98%", 1)}

        with patch.object(
            summary_review,
            "post_review_openrouter",
            side_effect=[
                _response(invalid_audit, completion_id="audit-1"),
                _response(valid_audit, completion_id="audit-2"),
                _response(invalid_edit, completion_id="edit-1"),
            ],
        ) as post:
            with self.assertRaises(summary_review.SummaryReviewValidationError) as raised:
                summary_review.generate(
                    EPISODE,
                    TRANSCRIPT,
                    DRAFT,
                    review_context,
                    episode_key=EPISODE_KEY,
                    source_fingerprint=SOURCE_FINGERPRINT,
                )

        self.assertEqual(post.call_count, 3)
        self.assertEqual(raised.exception.stage, "edit")
        self.assertEqual(raised.exception.completion_metadata["attempt_count"], 3)


if __name__ == "__main__":
    unittest.main()
