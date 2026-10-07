from decimal import Decimal
import json
import os
import unittest
from unittest.mock import ANY, Mock, patch

from podcast_engine.ai_pricing import ConservativeTextPricingBound, ResolvedPreset
from podcast_engine.knowledge import summary_review
from podcast_engine.knowledge.summary_review_evidence import build_review_context
from podcast_engine.preset_provenance import PresetProvenance


EPISODE_KEY = "a" * 24
SOURCE_FINGERPRINT = "sha256:" + "c" * 64


def _provenance(*, model="test/reviewer-model", max_tokens=2000) -> PresetProvenance:
    return PresetProvenance(
        status="verified",
        slug="podcast-summary-review",
        preset_id="preset-id",
        version_id="version-id-1",
        version=1,
        config={"model": model, "max_tokens": max_tokens},
        system_prompt="Verified summary review system prompt.",
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
            model_ids=("test/reviewer-model",),
            prompt_usd_per_token=Decimal("0.000001"),
            completion_usd_per_token=Decimal("0.000002"),
            context_length=200000,
            pricings=(),
        ),
    )


def _budget_plumbing(provenance):
    return (
        patch(
            "podcast_engine.knowledge.summary_review.fetch_current_designated_preset",
            return_value=provenance,
        ),
        patch(
            "podcast_engine.knowledge.summary_review.resolve_preset_model_and_pricing",
            return_value=_resolved_preset(provenance),
        ),
        patch("podcast_engine.knowledge.summary_review.reserve_budget_batch"),
        patch("podcast_engine.knowledge.summary_review.settle_budget_attempt"),
        patch("podcast_engine.knowledge.summary_review.mark_budget_attempt_uncertain"),
    )


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


def _context() -> dict:
    return build_review_context(TRANSCRIPT, DRAFT)


def _audit(review_context: dict) -> dict:
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


def _response(
    content,
    *,
    completion_id: str,
    model: str = "anthropic/claude-sonnet-4.6",
    provider: str = "Anthropic",
):
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "id": completion_id,
        "model": model,
        "openrouter_metadata": {
            "endpoints": {
                "available": [{"provider": provider, "selected": True}],
            }
        },
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "cost": 0.01,
        },
        "choices": [
            {
                "message": {
                    "content": content if isinstance(content, str) else json.dumps(content)
                }
            }
        ],
    }
    return response


class SummaryReviewGenerationV2TransportTests(unittest.TestCase):
    def test_malformed_audit_content_gets_the_one_shared_repair_turn(self):
        review_context = _context()
        audit = _audit(review_context)
        edit = {"resolved_issue_ids": [], "final_markdown": DRAFT}

        provenance = _provenance()
        p1, p2, p3, p4, p5 = _budget_plumbing(provenance)
        with patch.dict(
            os.environ, {"PODCAST_SUMMARY_REVIEW_API_KEY": "review-key"}, clear=False
        ), patch.object(
            summary_review,
            "post_review_openrouter",
            side_effect=[
                _response("not-json", completion_id="audit-1"),
                _response(audit, completion_id="audit-2"),
                _response(edit, completion_id="edit-1"),
            ],
        ) as post, p1, p2, p3, p4, p5:
            result = summary_review.generate(
                EPISODE,
                TRANSCRIPT,
                DRAFT,
                review_context,
                episode_key=EPISODE_KEY,
                source_fingerprint=SOURCE_FINGERPRINT,
            )

        self.assertEqual(post.call_count, 3)
        self.assertEqual(result["accepted_final_markdown"], DRAFT)
        self.assertEqual(result["completion_metadata"]["attempt_count"], 3)
        repair_payload = post.call_args_list[1].args[0]
        self.assertEqual(repair_payload["model"], post.call_args_list[0].args[0]["model"])
        self.assertIn(
            "expecting_value_line_1_column_1_char_0",
            repair_payload["messages"][-1]["content"],
        )
        self.assertNotIn("invalid_audit_contract", repair_payload["messages"][-1]["content"])

    def test_served_identity_must_remain_stable_across_audit_and_edit(self):
        review_context = _context()
        audit = _audit(review_context)
        edit = {"resolved_issue_ids": [], "final_markdown": DRAFT}

        provenance = _provenance()
        p1, p2, p3, p4, p5 = _budget_plumbing(provenance)
        with patch.dict(
            os.environ, {"PODCAST_SUMMARY_REVIEW_API_KEY": "review-key"}, clear=False
        ), patch.object(
            summary_review,
            "post_review_openrouter",
            side_effect=[
                _response(audit, completion_id="audit-1"),
                _response(
                    edit,
                    completion_id="edit-1",
                    provider="Different Provider",
                ),
            ],
        ), p1, p2, p3, p4, p5:
            with self.assertRaises(summary_review.SummaryReviewValidationError) as raised:
                summary_review.generate(
                    EPISODE,
                    TRANSCRIPT,
                    DRAFT,
                    review_context,
                    episode_key=EPISODE_KEY,
                    source_fingerprint=SOURCE_FINGERPRINT,
                )

        self.assertEqual(raised.exception.stage, "request")
        self.assertIn("identity", str(raised.exception).casefold())

    def test_transport_uses_only_dedicated_reviewer_key(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {}
        provenance = _provenance()
        p1, p2, p3, p4, p5 = _budget_plumbing(provenance)
        with patch.dict(
            os.environ,
            {
                "PODCAST_SUMMARY_REVIEW_API_KEY": "review-key",
                "PODCAST_KNOWLEDGE_API_KEY": "knowledge-key",
                "OPENROUTER_API_KEY": "fallback-key",
            },
            clear=True,
        ), patch.object(
            summary_review.requests, "post", return_value=response
        ) as post, p1, p2, p3, p4, p5:
            summary_review.post_review_openrouter(
                dict(provenance.config),
                episode_key=EPISODE_KEY,
                source_fingerprint=SOURCE_FINGERPRINT,
                provenance=provenance,
            )

        self.assertEqual(
            post.call_args.kwargs["headers"]["Authorization"],
            "Bearer review-key",
        )
        self.assertEqual(
            post.call_args.kwargs["json"]["provider"]["max_price"],
            {"prompt": 1, "completion": 2},
        )

        with patch.dict(
            os.environ,
            {"PODCAST_KNOWLEDGE_API_KEY": "knowledge-key"},
            clear=True,
        ):
            with self.assertRaisesRegex(RuntimeError, "PODCAST_SUMMARY_REVIEW_API_KEY"):
                summary_review.post_review_openrouter(
                    dict(provenance.config),
                    episode_key=EPISODE_KEY,
                    source_fingerprint=SOURCE_FINGERPRINT,
                    provenance=provenance,
                )

    def test_transport_rejects_conflicting_completion_ceiling_before_post(self):
        provenance = _provenance()
        provenance.config["max_completion_tokens"] = 4000
        p1, p2, p3, p4, p5 = _budget_plumbing(provenance)
        with patch.dict(os.environ, {"PODCAST_SUMMARY_REVIEW_API_KEY": "review-key"}, clear=True), patch.object(
            summary_review.requests, "post"
        ) as post, p1, p2, p3, p4, p5:
            with self.assertRaises(RuntimeError):
                summary_review.post_review_openrouter(
                    dict(provenance.config),
                    episode_key=EPISODE_KEY,
                    source_fingerprint=SOURCE_FINGERPRINT,
                    provenance=provenance,
                )
        post.assert_not_called()

    def test_transport_retries_429_once_but_not_400_and_redacts_request_text(self):
        retry_response = Mock(status_code=429, text="busy")
        retry_response.raise_for_status.side_effect = summary_review.requests.HTTPError(
            "429",
            response=retry_response,
        )
        success = Mock()
        success.raise_for_status.return_value = None
        success.json.return_value = {}
        provenance = _provenance()
        p1, p2, p3, p4, p5 = _budget_plumbing(provenance)
        with patch.dict(
            os.environ,
            {"PODCAST_SUMMARY_REVIEW_API_KEY": "review-key"},
            clear=True,
        ), patch.object(
            summary_review.requests,
            "post",
            side_effect=[retry_response, success],
        ) as post, patch.object(summary_review.time, "sleep"), p1, p2, p3, p4, p5:
            self.assertIs(
                summary_review.post_review_openrouter(
                    dict(provenance.config),
                    episode_key=EPISODE_KEY,
                    source_fingerprint=SOURCE_FINGERPRINT,
                    provenance=provenance,
                ),
                success,
            )
        self.assertEqual(post.call_count, 2)

        payload = summary_review.openrouter_audit_payload(
            EPISODE,
            TRANSCRIPT,
            DRAFT,
            _context(),
            provenance=provenance,
        )
        bad_response = Mock(status_code=400)
        bad_response.text = (
            f"Rejected {TRANSCRIPT} and {DRAFT} with Bearer review-key " + "x" * 3000
        )
        bad_response.raise_for_status.side_effect = summary_review.requests.HTTPError(
            "400",
            response=bad_response,
        )
        p1, p2, p3, p4, p5 = _budget_plumbing(provenance)
        with patch.dict(
            os.environ,
            {"PODCAST_SUMMARY_REVIEW_API_KEY": "review-key"},
            clear=True,
        ), patch.object(
            summary_review.requests,
            "post",
            return_value=bad_response,
        ) as post, p1, p2, p3, p4, p5:
            with self.assertRaises(summary_review.SummaryReviewTransportError) as raised:
                summary_review.post_review_openrouter(
                    payload,
                    episode_key=EPISODE_KEY,
                    source_fingerprint=SOURCE_FINGERPRINT,
                    provenance=provenance,
                )

        self.assertEqual(post.call_count, 1)
        message = str(raised.exception)
        self.assertNotIn("review-key", message)
        self.assertNotIn(TRANSCRIPT, message)
        self.assertNotIn(DRAFT, message)
        self.assertIn("HTTP 400", message)


if __name__ == "__main__":
    unittest.main()
