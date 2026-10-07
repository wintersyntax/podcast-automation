"""TASK-076 Task 5: downstream-reserve planning.

Proves that ``required_downstream_reserve`` -- the pure planner that keeps
optional assisted evidence (Third ASR) from starving money still needed by
mandatory downstream AI stages -- is a faithful, from-scratch recomputation
over caller-supplied stage state, never a guessed percentage of the cap,
and that it composes correctly with ``reserve_budget_batch``'s existing
``required_downstream_reserve_usd`` admission term (see test_ai_budget.py's
Task 4 CAS harness, reused here for the end-to-end admission test).
"""

from __future__ import annotations

from decimal import Decimal
import unittest

import json
import os
from unittest.mock import patch

from podcast_engine.ai_budget import (
    AI_BUDGET_STAGE_IDS,
    STAGE_MATERIALITY,
    STAGE_METADATA,
    STAGE_NOTE_WRITER,
    STAGE_RESOLVER,
    STAGE_SUMMARY,
    STAGE_SUMMARY_REVIEW,
    STAGE_THIRD_ASR,
    STAGE_TRIAGE,
    STAGE_WHISPER_TRANSCRIPTION,
    DownstreamReserveError,
    required_downstream_reserve,
    resolve_required_downstream_reserve_usd,
)
from podcast_engine.knowledge import metadata as knowledge_metadata
from podcast_engine.knowledge import summary as knowledge_summary
from podcast_engine.knowledge import summary_review as knowledge_summary_review
from podcast_engine.knowledge.models import SUMMARY_REVIEW_PRESET_ENV
from podcast_engine.preset_provenance import PresetProvenance

from tests.test_ai_budget import (
    EPISODE_KEY,
    FINGERPRINT,
    _BudgetLedgerTestCase,
)
from podcast_engine.ai_budget import reserve_budget_batch


_MANDATORY_STAGES = (
    STAGE_RESOLVER,
    STAGE_TRIAGE,
    STAGE_SUMMARY,
    STAGE_METADATA,
    STAGE_SUMMARY_REVIEW,
    STAGE_NOTE_WRITER,
)


def _all_not_pending() -> dict[str, bool]:
    return {stage: False for stage in _MANDATORY_STAGES}


class StageIdVocabularyTests(unittest.TestCase):
    def test_stage_ids_cover_every_documented_stage(self):
        self.assertEqual(
            AI_BUDGET_STAGE_IDS,
            {
                STAGE_RESOLVER,
                STAGE_TRIAGE,
                STAGE_THIRD_ASR,
                STAGE_SUMMARY,
                STAGE_METADATA,
                STAGE_SUMMARY_REVIEW,
                STAGE_NOTE_WRITER,
                STAGE_WHISPER_TRANSCRIPTION,
                STAGE_MATERIALITY,
            },
        )

    def test_materiality_is_not_a_downstream_reserve_eligible_stage(self):
        pending = _all_not_pending()
        pending[STAGE_MATERIALITY] = False
        with self.assertRaises(DownstreamReserveError):
            required_downstream_reserve(
                stage_pending=pending, stage_reservation_usd={}
            )

    def test_third_asr_is_not_a_downstream_reserve_eligible_stage(self):
        pending = _all_not_pending()
        pending[STAGE_THIRD_ASR] = False
        with self.assertRaises(DownstreamReserveError):
            required_downstream_reserve(
                stage_pending=pending, stage_reservation_usd={}
            )


class InactiveCachedActiveStageStateTests(unittest.TestCase):
    def test_inactive_stage_contributes_zero_and_needs_no_bound(self):
        pending = _all_not_pending()
        total = required_downstream_reserve(
            stage_pending=pending, stage_reservation_usd={}
        )
        self.assertEqual(total, Decimal("0"))

    def test_cached_stage_is_not_pending_and_contributes_zero(self):
        # A stage whose ledger/manifest fingerprint is already current for
        # this generation is represented the same way as a genuinely
        # inactive stage: not pending. The planner does not distinguish
        # "never applicable" from "already satisfied" -- both mean no
        # future spend is required.
        pending = _all_not_pending()
        pending[STAGE_SUMMARY] = False  # cached: fingerprint already matches
        total = required_downstream_reserve(
            stage_pending=pending,
            stage_reservation_usd={STAGE_SUMMARY: Decimal("0.05")},
        )
        self.assertEqual(total, Decimal("0"))

    def test_active_pending_stage_uses_its_own_supplied_bound(self):
        pending = _all_not_pending()
        pending[STAGE_METADATA] = True
        total = required_downstream_reserve(
            stage_pending=pending,
            stage_reservation_usd={STAGE_METADATA: Decimal("0.054")},
        )
        self.assertEqual(total, Decimal("0.054"))

    def test_sums_every_pending_stage(self):
        pending = _all_not_pending()
        pending[STAGE_SUMMARY] = True
        pending[STAGE_METADATA] = True
        pending[STAGE_SUMMARY_REVIEW] = True
        total = required_downstream_reserve(
            stage_pending=pending,
            stage_reservation_usd={
                STAGE_SUMMARY: Decimal("0.059"),
                STAGE_METADATA: Decimal("0.054"),
                STAGE_SUMMARY_REVIEW: Decimal("0.120"),
            },
        )
        self.assertEqual(total, Decimal("0.233"))


class MissingOrMalformedInputsFailClosedTests(unittest.TestCase):
    def test_missing_required_stage_key_raises(self):
        pending = _all_not_pending()
        del pending[STAGE_SUMMARY_REVIEW]
        with self.assertRaises(DownstreamReserveError):
            required_downstream_reserve(
                stage_pending=pending, stage_reservation_usd={}
            )

    def test_unknown_stage_key_raises(self):
        pending = _all_not_pending()
        pending["not_a_real_stage"] = False
        with self.assertRaises(DownstreamReserveError):
            required_downstream_reserve(
                stage_pending=pending, stage_reservation_usd={}
            )

    def test_third_asr_key_in_stage_pending_raises(self):
        pending = _all_not_pending()
        pending[STAGE_THIRD_ASR] = False
        with self.assertRaises(DownstreamReserveError):
            required_downstream_reserve(
                stage_pending=pending, stage_reservation_usd={}
            )

    def test_pending_stage_missing_its_bound_raises(self):
        pending = _all_not_pending()
        pending[STAGE_SUMMARY] = True
        with self.assertRaises(DownstreamReserveError):
            required_downstream_reserve(
                stage_pending=pending, stage_reservation_usd={}
            )

    def test_bound_must_be_a_decimal_not_a_float(self):
        # No guessed percentages: a float (e.g. 0.05 as "5% of the cap")
        # is exactly the kind of unprovable estimate this planner refuses.
        pending = _all_not_pending()
        pending[STAGE_SUMMARY] = True
        with self.assertRaises(DownstreamReserveError):
            required_downstream_reserve(
                stage_pending=pending,
                stage_reservation_usd={STAGE_SUMMARY: 0.05},
            )

    def test_bound_must_be_non_negative(self):
        pending = _all_not_pending()
        pending[STAGE_SUMMARY] = True
        with self.assertRaises(DownstreamReserveError):
            required_downstream_reserve(
                stage_pending=pending,
                stage_reservation_usd={STAGE_SUMMARY: Decimal("-0.01")},
            )

    def test_stage_pending_values_must_be_bool(self):
        pending = _all_not_pending()
        pending[STAGE_SUMMARY] = "yes"
        with self.assertRaises(DownstreamReserveError):
            required_downstream_reserve(
                stage_pending=pending, stage_reservation_usd={}
            )


class RecomputationAfterRecompileOrHumanMutationTests(unittest.TestCase):
    def test_recompile_flips_a_cached_summary_stage_back_to_pending(self):
        # Before a recompile: summary's stored input_fingerprint still
        # matches, so it is not pending.
        before = _all_not_pending()
        before_total = required_downstream_reserve(
            stage_pending=before, stage_reservation_usd={}
        )
        self.assertEqual(before_total, Decimal("0"))

        # A recompile changed the compiled transcript, so summary's
        # input_fingerprint no longer matches -- summary becomes pending
        # again. The planner has no internal cache of the earlier call; it
        # must recompute purely from the new state it is given.
        after = _all_not_pending()
        after[STAGE_SUMMARY] = True
        after_total = required_downstream_reserve(
            stage_pending=after,
            stage_reservation_usd={STAGE_SUMMARY: Decimal("0.059")},
        )
        self.assertEqual(after_total, Decimal("0.059"))

    def test_human_mutation_flips_a_settled_metadata_stage_back_to_pending(self):
        # Before a human edit: metadata's manifest entry is current.
        before = _all_not_pending()
        self.assertEqual(
            required_downstream_reserve(
                stage_pending=before, stage_reservation_usd={}
            ),
            Decimal("0"),
        )

        # A human mutation to the accepted review text changes the
        # compiled transcript, invalidating metadata's stored fingerprint.
        after = _all_not_pending()
        after[STAGE_METADATA] = True
        after_total = required_downstream_reserve(
            stage_pending=after,
            stage_reservation_usd={STAGE_METADATA: Decimal("0.054")},
        )
        self.assertEqual(after_total, Decimal("0.054"))


class OptionalEvidenceDenialTests(_BudgetLedgerTestCase):
    """End-to-end: required_downstream_reserve feeding reserve_budget_batch's
    existing required_downstream_reserve_usd admission term (Task 4)."""

    def test_third_asr_denied_when_subcap_fits_but_global_minus_downstream_does_not(self):
        pending = _all_not_pending()
        pending[STAGE_SUMMARY] = True
        pending[STAGE_METADATA] = True
        # $1.60 hard cap, $1.55 required for still-pending mandatory work
        # -> only $0.05 of true headroom, even though $0.08 comfortably
        # fits under the separate $0.10 Third-ASR subcap alone.
        downstream = required_downstream_reserve(
            stage_pending=pending,
            stage_reservation_usd={
                STAGE_SUMMARY: Decimal("1.20"),
                STAGE_METADATA: Decimal("0.35"),
            },
        )
        self.assertEqual(downstream, Decimal("1.55"))

        with self.assertRaises(Exception):
            reserve_budget_batch(
                EPISODE_KEY,
                FINGERPRINT,
                [self._reservation("t1", usd="0.08", stage=STAGE_THIRD_ASR, third_asr=True)],
                required_downstream_reserve_usd=downstream,
            )

    def test_third_asr_admitted_when_it_fits_within_true_remaining_headroom(self):
        pending = _all_not_pending()
        pending[STAGE_SUMMARY] = True
        downstream = required_downstream_reserve(
            stage_pending=pending,
            stage_reservation_usd={STAGE_SUMMARY: Decimal("0.20")},
        )
        self.assertEqual(downstream, Decimal("0.20"))

        # $1.60 - $0.20 downstream = $1.40 true headroom; $0.08 fits.
        result = reserve_budget_batch(
            EPISODE_KEY,
            FINGERPRINT,
            [self._reservation("t2", usd="0.08", stage=STAGE_THIRD_ASR, third_asr=True)],
            required_downstream_reserve_usd=downstream,
        )
        self.assertTrue(result["admitted"])

    def test_no_pending_downstream_stages_never_reduces_headroom(self):
        pending = _all_not_pending()
        downstream = required_downstream_reserve(
            stage_pending=pending, stage_reservation_usd={}
        )
        self.assertEqual(downstream, Decimal("0"))

        result = reserve_budget_batch(
            EPISODE_KEY,
            FINGERPRINT,
            [self._reservation("t3", usd="0.09", stage=STAGE_THIRD_ASR, third_asr=True)],
            required_downstream_reserve_usd=downstream,
        )
        self.assertTrue(result["admitted"])


def _fixture_preset(*, max_tokens=1000):
    return PresetProvenance(
        status="verified",
        slug="fixture-slug",
        preset_id="preset-id",
        version_id="version-id",
        version=1,
        config={"model": "prov/model", "max_tokens": max_tokens},
        system_prompt="System prompt",
        config_digest="sha256:config",
        system_prompt_digest="sha256:prompt",
    )


def _fixture_catalog(model_id="prov/model", *, prompt="0.000001", completion="0.000003", context_length=5_000_000):
    return [
        {
            "id": model_id,
            "pricing": {"prompt": prompt, "completion": completion},
            "context_length": context_length,
        }
    ]


class WorstCaseStagePayloadBuilderTests(unittest.TestCase):
    def test_summary_worst_case_payload_is_a_well_formed_request(self):
        payload = knowledge_summary.worst_case_openrouter_payload(_fixture_preset())
        self.assertIn("messages", payload)
        self.assertEqual(payload["messages"][0]["role"], "system")
        self.assertGreater(len(payload["messages"][1]["content"]), 100_000)

    def test_metadata_worst_case_payload_is_a_well_formed_request(self):
        payload = knowledge_metadata.worst_case_openrouter_payload(_fixture_preset())
        self.assertIn("messages", payload)
        self.assertEqual(payload["messages"][0]["role"], "system")
        self.assertGreater(len(payload["messages"][1]["content"]), 100_000)

    def test_summary_review_worst_case_payload_is_a_well_formed_request(self):
        payload = knowledge_summary_review.worst_case_openrouter_payload(_fixture_preset())
        self.assertIn("messages", payload)
        self.assertEqual(payload["messages"][0]["role"], "system")
        self.assertGreater(len(payload["messages"][1]["content"]), 100_000)
        self.assertEqual(knowledge_summary_review.WORST_CASE_MODEL_CALLS_PER_CASE, 3)


class ResolveRequiredDownstreamReserveUsdTests(unittest.TestCase):
    def test_resolver_or_triage_pending_is_rejected(self):
        pending = _all_not_pending()
        pending[STAGE_RESOLVER] = True
        with self.assertRaises(DownstreamReserveError):
            resolve_required_downstream_reserve_usd(
                stage_pending=pending, api_key="test-key"
            )

    def test_no_pending_stages_resolves_to_zero_with_no_network_calls(self):
        pending = _all_not_pending()
        with patch(
            "podcast_engine.ai_budget.fetch_current_designated_preset"
        ) as fetch:
            total = resolve_required_downstream_reserve_usd(
                stage_pending=pending, api_key="test-key"
            )
        self.assertEqual(total, Decimal("0"))
        fetch.assert_not_called()

    def test_pending_summary_resolves_its_own_live_bound(self):
        pending = _all_not_pending()
        pending[STAGE_SUMMARY] = True
        with patch(
            "podcast_engine.ai_budget.fetch_current_designated_preset",
            return_value=_fixture_preset(max_tokens=500),
        ):
            total = resolve_required_downstream_reserve_usd(
                stage_pending=pending,
                api_key="test-key",
                transport=lambda url, api_key: _fixture_catalog(),
            )
        self.assertGreater(total, Decimal("0"))

    def test_pending_summary_and_metadata_sum_together(self):
        pending = _all_not_pending()
        pending[STAGE_SUMMARY] = True
        pending[STAGE_METADATA] = True
        with patch(
            "podcast_engine.ai_budget.fetch_current_designated_preset",
            return_value=_fixture_preset(max_tokens=500),
        ):
            summary_only = resolve_required_downstream_reserve_usd(
                stage_pending={**_all_not_pending(), STAGE_SUMMARY: True},
                api_key="test-key",
                transport=lambda url, api_key: _fixture_catalog(),
            )
            metadata_only = resolve_required_downstream_reserve_usd(
                stage_pending={**_all_not_pending(), STAGE_METADATA: True},
                api_key="test-key",
                transport=lambda url, api_key: _fixture_catalog(),
            )
            both = resolve_required_downstream_reserve_usd(
                stage_pending=pending,
                api_key="test-key",
                transport=lambda url, api_key: _fixture_catalog(),
            )
        self.assertEqual(both, summary_only + metadata_only)

    def test_unresolvable_preset_fails_closed_rather_than_contributing_zero(self):
        pending = _all_not_pending()
        pending[STAGE_SUMMARY] = True
        with patch(
            "podcast_engine.ai_budget.fetch_current_designated_preset",
            return_value=PresetProvenance(status="invalid", reason="malformed"),
        ):
            with self.assertRaises(DownstreamReserveError):
                resolve_required_downstream_reserve_usd(
                    stage_pending=pending,
                    api_key="test-key",
                    transport=lambda url, api_key: _fixture_catalog(),
                )

    def test_inactive_summary_review_contributes_zero_with_no_preset_fetch_for_it(self):
        pending = _all_not_pending()
        pending[STAGE_SUMMARY_REVIEW] = True
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(SUMMARY_REVIEW_PRESET_ENV, None)
            with patch(
                "podcast_engine.ai_budget.fetch_current_designated_preset"
            ) as fetch:
                total = resolve_required_downstream_reserve_usd(
                    stage_pending=pending, api_key="test-key"
                )
        self.assertEqual(total, Decimal("0"))
        fetch.assert_not_called()

    def test_active_summary_review_multiplies_per_call_bound_by_call_count(self):
        from podcast_engine.ai_pricing import (
            resolve_preset_model_and_pricing,
            resolve_stage_reservation_usd,
        )

        pending = _all_not_pending()
        pending[STAGE_SUMMARY_REVIEW] = True
        with patch.dict(
            os.environ,
            {
                SUMMARY_REVIEW_PRESET_ENV: "active-reviewer-slug",
                "PODCAST_SUMMARY_REVIEW_API_KEY": "unused-in-this-test",
            },
        ):
            # The worst-case payload's "model" field is itself env-derived
            # (summary_review_preset() reads SUMMARY_REVIEW_PRESET_ENV), so
            # the expected-value computation must read it under the same
            # env override the orchestrator call below also runs under --
            # otherwise the two payloads legitimately differ by a few bytes
            # of preset-slug text, not because of a real bug.
            resolved = resolve_preset_model_and_pricing(
                _fixture_preset(max_tokens=500),
                api_key="test-key",
                transport=lambda url, api_key: _fixture_catalog(),
            )
            expected_per_call = resolve_stage_reservation_usd(
                resolved_preset=resolved,
                worst_case_request_bytes=json.dumps(
                    knowledge_summary_review.worst_case_openrouter_payload(
                        _fixture_preset(max_tokens=500)
                    ),
                    ensure_ascii=False,
                ).encode("utf-8"),
            )

            with patch(
                "podcast_engine.ai_budget.fetch_current_designated_preset",
                return_value=_fixture_preset(max_tokens=500),
            ):
                total = resolve_required_downstream_reserve_usd(
                    stage_pending=pending,
                    api_key="test-key",
                    transport=lambda url, api_key: _fixture_catalog(),
                )

        self.assertGreater(total, Decimal("0"))
        self.assertEqual(
            total,
            expected_per_call * knowledge_summary_review.WORST_CASE_MODEL_CALLS_PER_CASE,
        )


if __name__ == "__main__":
    unittest.main()
