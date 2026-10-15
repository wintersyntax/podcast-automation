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
from podcast_engine.knowledge.models import NOTE_WRITER_PRESET_ENV
from podcast_engine.knowledge.notes_v3 import writer as knowledge_note_writer
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
    STAGE_NOTE_WRITER,
)


def _all_not_pending() -> dict[str, bool]:
    return {stage: False for stage in _MANDATORY_STAGES}


class NoGenerationStatusTests(unittest.TestCase):
    def test_only_pre_generation_rejections_prove_no_charge(self):
        from podcast_engine.ai_budget import http_status_proves_no_generation

        for status in (400, 401, 403, 404, 422):
            self.assertTrue(http_status_proves_no_generation(status), status)
        for status in (None, True, "400", 402, 408, 409, 413, 429, 499, 500, 502, 503, 504):
            self.assertFalse(http_status_proves_no_generation(status), status)


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

    def test_retired_knowledge_stages_are_not_downstream_reserve_eligible(self):
        # TASK-146: summary, metadata and summary review stay valid attempt
        # stages for historical ledgers but are never protected work.
        for stage in (STAGE_SUMMARY, STAGE_METADATA, STAGE_SUMMARY_REVIEW):
            with self.subTest(stage=stage):
                pending = _all_not_pending()
                pending[stage] = False
                with self.assertRaises(DownstreamReserveError):
                    required_downstream_reserve(
                        stage_pending=pending, stage_reservation_usd={}
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
        pending[STAGE_NOTE_WRITER] = False  # cached: fingerprint already matches
        total = required_downstream_reserve(
            stage_pending=pending,
            stage_reservation_usd={STAGE_NOTE_WRITER: Decimal("0.05")},
        )
        self.assertEqual(total, Decimal("0"))

    def test_active_pending_stage_uses_its_own_supplied_bound(self):
        pending = _all_not_pending()
        pending[STAGE_TRIAGE] = True
        total = required_downstream_reserve(
            stage_pending=pending,
            stage_reservation_usd={STAGE_TRIAGE: Decimal("0.054")},
        )
        self.assertEqual(total, Decimal("0.054"))

    def test_sums_every_pending_stage(self):
        pending = _all_not_pending()
        pending[STAGE_NOTE_WRITER] = True
        pending[STAGE_TRIAGE] = True
        pending[STAGE_RESOLVER] = True
        total = required_downstream_reserve(
            stage_pending=pending,
            stage_reservation_usd={
                STAGE_NOTE_WRITER: Decimal("0.059"),
                STAGE_TRIAGE: Decimal("0.054"),
                STAGE_RESOLVER: Decimal("0.120"),
            },
        )
        self.assertEqual(total, Decimal("0.233"))


class MissingOrMalformedInputsFailClosedTests(unittest.TestCase):
    def test_missing_required_stage_key_raises(self):
        pending = _all_not_pending()
        del pending[STAGE_RESOLVER]
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
        pending[STAGE_NOTE_WRITER] = True
        with self.assertRaises(DownstreamReserveError):
            required_downstream_reserve(
                stage_pending=pending, stage_reservation_usd={}
            )

    def test_bound_must_be_a_decimal_not_a_float(self):
        # No guessed percentages: a float (e.g. 0.05 as "5% of the cap")
        # is exactly the kind of unprovable estimate this planner refuses.
        pending = _all_not_pending()
        pending[STAGE_NOTE_WRITER] = True
        with self.assertRaises(DownstreamReserveError):
            required_downstream_reserve(
                stage_pending=pending,
                stage_reservation_usd={STAGE_NOTE_WRITER: 0.05},
            )

    def test_bound_must_be_non_negative(self):
        pending = _all_not_pending()
        pending[STAGE_NOTE_WRITER] = True
        with self.assertRaises(DownstreamReserveError):
            required_downstream_reserve(
                stage_pending=pending,
                stage_reservation_usd={STAGE_NOTE_WRITER: Decimal("-0.01")},
            )

    def test_stage_pending_values_must_be_bool(self):
        pending = _all_not_pending()
        pending[STAGE_NOTE_WRITER] = "yes"
        with self.assertRaises(DownstreamReserveError):
            required_downstream_reserve(
                stage_pending=pending, stage_reservation_usd={}
            )


class RecomputationAfterRecompileOrHumanMutationTests(unittest.TestCase):
    def test_recompile_flips_a_cached_note_writer_stage_back_to_pending(self):
        # Before a recompile: the note's stored input_fingerprint still
        # matches, so the writer is not pending.
        before = _all_not_pending()
        before_total = required_downstream_reserve(
            stage_pending=before, stage_reservation_usd={}
        )
        self.assertEqual(before_total, Decimal("0"))

        # A recompile changed the compiled transcript, so the note's
        # input_fingerprint no longer matches -- the writer becomes pending
        # again. The planner has no internal cache of the earlier call; it
        # must recompute purely from the new state it is given.
        after = _all_not_pending()
        after[STAGE_NOTE_WRITER] = True
        after_total = required_downstream_reserve(
            stage_pending=after,
            stage_reservation_usd={STAGE_NOTE_WRITER: Decimal("0.059")},
        )
        self.assertEqual(after_total, Decimal("0.059"))


class OptionalEvidenceDenialTests(_BudgetLedgerTestCase):
    """End-to-end: required_downstream_reserve feeding reserve_budget_batch's
    existing required_downstream_reserve_usd admission term (Task 4)."""

    def test_third_asr_denied_when_subcap_fits_but_global_minus_downstream_does_not(self):
        pending = _all_not_pending()
        pending[STAGE_NOTE_WRITER] = True
        pending[STAGE_TRIAGE] = True
        # $1.60 hard cap, $1.55 required for still-pending mandatory work
        # -> only $0.05 of true headroom, even though $0.08 comfortably
        # fits under the separate $0.10 Third-ASR subcap alone.
        downstream = required_downstream_reserve(
            stage_pending=pending,
            stage_reservation_usd={
                STAGE_NOTE_WRITER: Decimal("1.20"),
                STAGE_TRIAGE: Decimal("0.35"),
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
        pending[STAGE_NOTE_WRITER] = True
        downstream = required_downstream_reserve(
            stage_pending=pending,
            stage_reservation_usd={STAGE_NOTE_WRITER: Decimal("0.20")},
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
        # The writer's payload builder refuses a drifted remote prompt.
        system_prompt=knowledge_note_writer.system_prompt(),
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


class ResolveRequiredDownstreamReserveUsdTests(unittest.TestCase):
    def test_resolver_or_triage_pending_is_rejected(self):
        for stage in (STAGE_RESOLVER, STAGE_TRIAGE):
            with self.subTest(stage=stage):
                pending = _all_not_pending()
                pending[stage] = True
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

    def test_pending_note_writer_resolves_its_own_live_bound(self):
        from podcast_engine.ai_pricing import (
            resolve_preset_model_and_pricing,
            resolve_stage_reservation_usd,
        )

        pending = _all_not_pending()
        pending[STAGE_NOTE_WRITER] = True
        resolved = resolve_preset_model_and_pricing(
            _fixture_preset(max_tokens=500),
            api_key="test-key",
            transport=lambda url, api_key: _fixture_catalog(),
        )
        expected = resolve_stage_reservation_usd(
            resolved_preset=resolved,
            worst_case_request_bytes=json.dumps(
                knowledge_note_writer.worst_case_openrouter_payload(
                    _fixture_preset(max_tokens=500)
                ),
                ensure_ascii=False,
            ).encode("utf-8"),
        )
        with patch.dict(os.environ, {NOTE_WRITER_PRESET_ENV: "writer-slug"}), patch(
            "podcast_engine.ai_budget.fetch_current_designated_preset",
            return_value=_fixture_preset(max_tokens=500),
        ) as fetch:
            total = resolve_required_downstream_reserve_usd(
                stage_pending=pending,
                api_key="test-key",
                transport=lambda url, api_key: _fixture_catalog(),
            )
        self.assertGreater(total, Decimal("0"))
        self.assertEqual(total, expected)
        self.assertEqual(fetch.call_args.args[0], "writer-slug")

    def test_unresolvable_preset_fails_closed_rather_than_contributing_zero(self):
        pending = _all_not_pending()
        pending[STAGE_NOTE_WRITER] = True
        with patch.dict(os.environ, {NOTE_WRITER_PRESET_ENV: "writer-slug"}), patch(
            "podcast_engine.ai_budget.fetch_current_designated_preset",
            return_value=PresetProvenance(status="invalid", reason="malformed"),
        ):
            with self.assertRaises(DownstreamReserveError):
                resolve_required_downstream_reserve_usd(
                    stage_pending=pending,
                    api_key="test-key",
                    transport=lambda url, api_key: _fixture_catalog(),
                )


if __name__ == "__main__":
    unittest.main()
