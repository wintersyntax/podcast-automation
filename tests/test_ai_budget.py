"""TASK-076: durable episode AI budget ledger with CAS admission.

Exercises podcast_engine.ai_budget against a small in-memory fake GCS blob
store that reproduces the real generation/if_generation_match/
PreconditionFailed semantics the CAS retry loop depends on, so these tests
prove the loop itself -- not just the business logic layered on top of it.
"""

from __future__ import annotations

from decimal import Decimal
import unittest
from unittest.mock import patch

from google.api_core.exceptions import NotFound, PreconditionFailed

from podcast_engine import ai_budget
from podcast_engine.ai_budget import (
    ASSISTED_THIRD_ASR_SUBCAP_USD,
    AI_BUDGET_POLICY_VERSION,
    AI_BUDGET_SCHEMA_VERSION,
    EPISODE_AI_HARD_CAP_USD,
    MAX_BUDGET_CAS_ATTEMPTS,
    BudgetAdmissionError,
    BudgetConcurrencyError,
    BudgetIntegrityError,
    BudgetLedgerIdentityError,
    budget_ledger_path,
    budget_summary,
    mark_budget_attempt_uncertain,
    release_budget_attempt_pre_send,
    reserve_budget_batch,
    settle_budget_attempt,
)


EPISODE_KEY = "a" * 24
OTHER_EPISODE_KEY = "b" * 24
FINGERPRINT = "sha256:" + "c" * 64
OTHER_FINGERPRINT = "sha256:" + "d" * 64


class _FakeBlob:
    """Reproduces the subset of google.cloud.storage.Blob this module uses."""

    def __init__(self, store: dict, path: str, *, on_upload=None):
        self._store = store
        self._path = path
        self._on_upload = on_upload
        self.generation = None

    def reload(self):
        record = self._store.get(self._path)
        if record is None:
            raise NotFound("object does not exist")
        self.generation = record["generation"]

    def exists(self) -> bool:
        return self._path in self._store

    def download_as_text(self, encoding="utf-8", if_generation_match=None):
        record = self._store.get(self._path)
        if record is None:
            raise PreconditionFailed("object does not exist")
        if if_generation_match is not None and record["generation"] != if_generation_match:
            raise PreconditionFailed("generation changed since reload")
        return record["content"]

    def upload_from_string(self, content, content_type=None, if_generation_match=None):
        # Simulates a concurrent writer's commit landing exactly between our
        # read and our own write -- the real race window for GCS's
        # server-side if_generation_match check, not between reload/download.
        if self._on_upload is not None:
            self._on_upload(self._store, self._path)
        record = self._store.get(self._path)
        if if_generation_match == 0:
            if record is not None:
                raise PreconditionFailed("object already exists")
            self._store[self._path] = {"content": content, "generation": 1}
            return
        current_generation = record["generation"] if record else None
        if record is None or current_generation != if_generation_match:
            raise PreconditionFailed("generation changed since reload")
        self._store[self._path] = {"content": content, "generation": current_generation + 1}


class _FakeBucket:
    def __init__(self, store: dict, *, on_upload=None):
        self._store = store
        self._on_upload = on_upload

    def blob(self, path: str) -> _FakeBlob:
        return _FakeBlob(self._store, path, on_upload=self._on_upload)


class _BudgetLedgerTestCase(unittest.TestCase):
    def setUp(self):
        self.store: dict = {}
        self._upload_hook = None
        patcher = patch(
            "podcast_engine.ai_budget.get_bucket",
            side_effect=lambda: _FakeBucket(self.store, on_upload=self._call_upload_hook),
        )
        self.addCleanup(patcher.stop)
        patcher.start()

    def _call_upload_hook(self, store, path):
        if self._upload_hook is not None:
            self._upload_hook(store, path)

    def _reservation(self, attempt_id, *, usd, stage="resolver", third_asr=False):
        return {
            "attempt_id": attempt_id,
            "stage": stage,
            "third_asr": third_asr,
            "reserved_usd": Decimal(usd),
        }


class BudgetLedgerPathTests(unittest.TestCase):
    def test_builds_the_canonical_path_from_the_fingerprint_digest(self):
        self.assertEqual(
            budget_ledger_path(EPISODE_KEY, FINGERPRINT),
            f"episodes/{EPISODE_KEY}/ai/budgets/{'c' * 64}.json",
        )

    def test_rejects_malformed_episode_key(self):
        with self.assertRaises(ValueError):
            budget_ledger_path("not-hex!!", FINGERPRINT)

    def test_rejects_malformed_source_fingerprint(self):
        with self.assertRaises(ValueError):
            budget_ledger_path(EPISODE_KEY, "not-a-fingerprint")
        with self.assertRaises(ValueError):
            budget_ledger_path(EPISODE_KEY, "sha256:tooshort")


class LedgerShapeAndIdentityTests(_BudgetLedgerTestCase):
    def test_wrong_schema_version_fails_closed(self):
        path = budget_ledger_path(EPISODE_KEY, FINGERPRINT)
        self.store[path] = {
            "content": '{"schema_version": 999, "policy_version": "%s", '
            '"episode_key": "%s", "source_fingerprint": "%s", "attempts": {}}'
            % (AI_BUDGET_POLICY_VERSION, EPISODE_KEY, FINGERPRINT),
            "generation": 1,
        }
        with self.assertRaises(ValueError):
            budget_summary(EPISODE_KEY, FINGERPRINT)

    def test_wrong_policy_version_fails_closed(self):
        path = budget_ledger_path(EPISODE_KEY, FINGERPRINT)
        self.store[path] = {
            "content": '{"schema_version": %d, "policy_version": "stale-v0", '
            '"episode_key": "%s", "source_fingerprint": "%s", "attempts": {}}'
            % (AI_BUDGET_SCHEMA_VERSION, EPISODE_KEY, FINGERPRINT),
            "generation": 1,
        }
        with self.assertRaises(ValueError):
            budget_summary(EPISODE_KEY, FINGERPRINT)

    def test_cross_generation_identity_mismatch_fails_closed(self):
        # Content claims a different episode_key/source_fingerprint than the
        # path it was loaded from was built for.
        path = budget_ledger_path(EPISODE_KEY, FINGERPRINT)
        self.store[path] = {
            "content": '{"schema_version": %d, "policy_version": "%s", '
            '"episode_key": "%s", "source_fingerprint": "%s", "attempts": {}}'
            % (
                AI_BUDGET_SCHEMA_VERSION,
                AI_BUDGET_POLICY_VERSION,
                OTHER_EPISODE_KEY,
                OTHER_FINGERPRINT,
            ),
            "generation": 1,
        }
        with self.assertRaises(BudgetLedgerIdentityError):
            budget_summary(EPISODE_KEY, FINGERPRINT)

    def test_malformed_attempts_field_fails_closed(self):
        path = budget_ledger_path(EPISODE_KEY, FINGERPRINT)
        self.store[path] = {
            "content": '{"schema_version": %d, "policy_version": "%s", '
            '"episode_key": "%s", "source_fingerprint": "%s", "attempts": []}'
            % (AI_BUDGET_SCHEMA_VERSION, AI_BUDGET_POLICY_VERSION, EPISODE_KEY, FINGERPRINT),
            "generation": 1,
        }
        with self.assertRaises(ValueError):
            budget_summary(EPISODE_KEY, FINGERPRINT)

    def test_malformed_attempt_state_fails_closed(self):
        path = budget_ledger_path(EPISODE_KEY, FINGERPRINT)
        self.store[path] = {
            "content": '{"schema_version": %d, "policy_version": "%s", '
            '"episode_key": "%s", "source_fingerprint": "%s", '
            '"attempts": {"x": {"state": "not_a_real_state"}}}'
            % (AI_BUDGET_SCHEMA_VERSION, AI_BUDGET_POLICY_VERSION, EPISODE_KEY, FINGERPRINT),
            "generation": 1,
        }
        with self.assertRaises(ValueError):
            budget_summary(EPISODE_KEY, FINGERPRINT)


class ReserveBudgetBatchBoundaryTests(_BudgetLedgerTestCase):
    def test_reserving_exactly_the_hard_cap_is_admitted(self):
        result = reserve_budget_batch(
            EPISODE_KEY,
            FINGERPRINT,
            [self._reservation("a1", usd=str(EPISODE_AI_HARD_CAP_USD))],
        )
        self.assertTrue(result["admitted"])
        summary = budget_summary(EPISODE_KEY, FINGERPRINT)
        self.assertEqual(summary["headroom_usd"], Decimal("0"))

    def test_reserving_one_cent_over_the_hard_cap_is_rejected(self):
        over = EPISODE_AI_HARD_CAP_USD + Decimal("0.01")
        with self.assertRaises(BudgetAdmissionError):
            reserve_budget_batch(
                EPISODE_KEY, FINGERPRINT, [self._reservation("a1", usd=str(over))]
            )
        # Nothing was admitted; the ledger stays empty.
        summary = budget_summary(EPISODE_KEY, FINGERPRINT)
        self.assertEqual(summary["headroom_usd"], EPISODE_AI_HARD_CAP_USD)

    def test_reserving_exactly_the_third_asr_subcap_is_admitted(self):
        result = reserve_budget_batch(
            EPISODE_KEY,
            FINGERPRINT,
            [
                self._reservation(
                    "asr1", usd=str(ASSISTED_THIRD_ASR_SUBCAP_USD), third_asr=True
                )
            ],
        )
        self.assertTrue(result["admitted"])
        summary = budget_summary(EPISODE_KEY, FINGERPRINT)
        self.assertEqual(summary["third_asr_headroom_usd"], Decimal("0"))

    def test_reserving_one_cent_over_the_third_asr_subcap_is_rejected_even_under_hard_cap(self):
        over = ASSISTED_THIRD_ASR_SUBCAP_USD + Decimal("0.01")
        self.assertLess(over, EPISODE_AI_HARD_CAP_USD)
        with self.assertRaises(BudgetAdmissionError):
            reserve_budget_batch(
                EPISODE_KEY,
                FINGERPRINT,
                [self._reservation("asr1", usd=str(over), third_asr=True)],
            )

    def test_third_asr_stage_cannot_bypass_subcap_when_flag_is_omitted(self):
        over = ASSISTED_THIRD_ASR_SUBCAP_USD + Decimal("0.01")
        with self.assertRaises(BudgetAdmissionError):
            reserve_budget_batch(
                EPISODE_KEY,
                FINGERPRINT,
                [{"attempt_id": "asr1", "stage": "third_asr", "reserved_usd": over}],
            )

    def test_third_asr_spend_also_counts_toward_the_shared_hard_cap(self):
        # Third-ASR sub-cap spend is not additive to the total hard cap.
        reserve_budget_batch(
            EPISODE_KEY,
            FINGERPRINT,
            [self._reservation("asr1", usd="0.10", third_asr=True)],
        )
        remaining_room = EPISODE_AI_HARD_CAP_USD - Decimal("0.10")
        reserve_budget_batch(
            EPISODE_KEY, FINGERPRINT, [self._reservation("r1", usd=str(remaining_room))]
        )
        with self.assertRaises(BudgetAdmissionError):
            reserve_budget_batch(
                EPISODE_KEY, FINGERPRINT, [self._reservation("r2", usd="0.01")]
            )

    def test_required_downstream_reserve_is_included_in_admission_math(self):
        # Even though 0.10 alone fits, a 1.51 downstream reserve pushes the
        # projected total over the 1.60 cap.
        with self.assertRaises(BudgetAdmissionError):
            reserve_budget_batch(
                EPISODE_KEY,
                FINGERPRINT,
                [self._reservation("a1", usd="0.10")],
                required_downstream_reserve_usd=Decimal("1.51"),
            )
        summary = budget_summary(EPISODE_KEY, FINGERPRINT)
        self.assertEqual(summary["headroom_usd"], EPISODE_AI_HARD_CAP_USD)


class ReserveBudgetBatchAtomicityTests(_BudgetLedgerTestCase):
    def test_whole_batch_is_admitted_together(self):
        result = reserve_budget_batch(
            EPISODE_KEY,
            FINGERPRINT,
            [
                self._reservation("a1", usd="0.10"),
                self._reservation("a2", usd="0.10"),
            ],
        )
        self.assertEqual(set(result["attempts"]), {"a1", "a2"})
        summary = budget_summary(EPISODE_KEY, FINGERPRINT)
        self.assertEqual(summary["live_reservations"], Decimal("0.20"))

    def test_whole_batch_is_rejected_together_when_any_part_would_overflow(self):
        # a1 alone fits; a1+a2 together exceed the cap. Neither is admitted.
        with self.assertRaises(BudgetAdmissionError):
            reserve_budget_batch(
                EPISODE_KEY,
                FINGERPRINT,
                [
                    self._reservation("a1", usd="0.20"),
                    self._reservation("a2", usd=str(EPISODE_AI_HARD_CAP_USD)),
                ],
            )
        summary = budget_summary(EPISODE_KEY, FINGERPRINT)
        self.assertEqual(summary["live_reservations"], Decimal("0"))
        self.assertEqual(summary["headroom_usd"], EPISODE_AI_HARD_CAP_USD)

    def test_rejects_duplicate_attempt_id_within_one_batch(self):
        with self.assertRaises(ValueError):
            reserve_budget_batch(
                EPISODE_KEY,
                FINGERPRINT,
                [self._reservation("dup", usd="0.01"), self._reservation("dup", usd="0.02")],
            )

    def test_rejects_non_positive_and_non_decimal_reserved_usd(self):
        for bad in ("0", "-0.01"):
            with self.assertRaises(ValueError):
                reserve_budget_batch(
                    EPISODE_KEY, FINGERPRINT, [self._reservation("a1", usd=bad)]
                )
        with self.assertRaises(ValueError):
            reserve_budget_batch(
                EPISODE_KEY,
                FINGERPRINT,
                [{"attempt_id": "a1", "stage": "resolver", "reserved_usd": 0.01}],
            )  # float, not Decimal


class ReserveBudgetBatchReplayTests(_BudgetLedgerTestCase):
    def test_replaying_an_already_reserved_attempt_id_is_a_noop(self):
        reserve_budget_batch(EPISODE_KEY, FINGERPRINT, [self._reservation("a1", usd="0.10")])
        path = budget_ledger_path(EPISODE_KEY, FINGERPRINT)
        generation_before = self.store[path]["generation"]

        result = reserve_budget_batch(
            EPISODE_KEY, FINGERPRINT, [self._reservation("a1", usd="0.10")]
        )
        self.assertTrue(result["admitted"])
        self.assertEqual(self.store[path]["generation"], generation_before)

    def test_a_cache_hit_batch_of_only_replayed_ids_creates_no_new_reservation(self):
        reserve_budget_batch(EPISODE_KEY, FINGERPRINT, [self._reservation("a1", usd="0.10")])
        summary_before = budget_summary(EPISODE_KEY, FINGERPRINT)

        reserve_budget_batch(EPISODE_KEY, FINGERPRINT, [self._reservation("a1", usd="0.10")])
        summary_after = budget_summary(EPISODE_KEY, FINGERPRINT)
        self.assertEqual(summary_before["live_reservations"], summary_after["live_reservations"])

    def test_replay_with_mismatched_details_is_rejected(self):
        reserve_budget_batch(EPISODE_KEY, FINGERPRINT, [self._reservation("a1", usd="0.10")])
        with self.assertRaises(BudgetAdmissionError):
            reserve_budget_batch(
                EPISODE_KEY, FINGERPRINT, [self._reservation("a1", usd="0.20")]
            )


class ReserveBudgetBatchConcurrencyTests(_BudgetLedgerTestCase):
    def test_a_single_concurrent_writer_forces_one_retry_that_still_succeeds(self):
        reserve_budget_batch(EPISODE_KEY, FINGERPRINT, [self._reservation("a1", usd="0.05")])
        ledger_path = budget_ledger_path(EPISODE_KEY, FINGERPRINT)

        calls = {"count": 0}

        def concurrent_writer(store, path):
            # Simulate exactly one other worker's commit landing between our
            # read and our own write, on the first write attempt only.
            calls["count"] += 1
            if calls["count"] != 1 or path != ledger_path:
                return
            import json

            record = store[path]
            payload = json.loads(record["content"])
            payload["attempts"]["racer"] = {
                "state": "reserved",
                "stage": "resolver",
                "third_asr": False,
                "reserved_usd": "0.01",
                "created_at": "2026-09-16T00:00:00Z",
                "updated_at": "2026-09-16T00:00:00Z",
            }
            store[path] = {
                "content": json.dumps(payload),
                "generation": record["generation"] + 1,
            }

        self._upload_hook = concurrent_writer
        result = reserve_budget_batch(
            EPISODE_KEY, FINGERPRINT, [self._reservation("a2", usd="0.05")]
        )
        self.assertTrue(result["admitted"])
        self.assertEqual(calls["count"], 2)  # first write raced, second succeeded
        summary = budget_summary(EPISODE_KEY, FINGERPRINT)
        # Both the racer's reservation and ours must have survived.
        self.assertEqual(summary["live_reservations"], Decimal("0.11"))

    def test_exhausting_cas_attempts_raises_concurrency_error(self):
        reserve_budget_batch(EPISODE_KEY, FINGERPRINT, [self._reservation("a1", usd="0.01")])
        ledger_path = budget_ledger_path(EPISODE_KEY, FINGERPRINT)

        def always_racing_writer(store, path):
            if path != ledger_path:
                return
            import json

            record = store[path]
            payload = json.loads(record["content"])
            store[path] = {
                "content": json.dumps(payload),
                "generation": record["generation"] + 1,
            }

        self._upload_hook = always_racing_writer
        with self.assertRaises(BudgetConcurrencyError):
            reserve_budget_batch(
                EPISODE_KEY, FINGERPRINT, [self._reservation("a2", usd="0.01")]
            )


class ReserveBudgetBatchIntegrityFailureTests(_BudgetLedgerTestCase):
    def test_integrity_failure_blocks_all_further_reservations(self):
        reserve_budget_batch(EPISODE_KEY, FINGERPRINT, [self._reservation("a1", usd="0.10")])
        with self.assertRaises(BudgetIntegrityError):
            settle_budget_attempt(
                EPISODE_KEY, FINGERPRINT, "a1", actual_usd=Decimal("0.15")
            )
        with self.assertRaises(BudgetIntegrityError):
            reserve_budget_batch(
                EPISODE_KEY, FINGERPRINT, [self._reservation("a2", usd="0.01")]
            )


class SettleBudgetAttemptTests(_BudgetLedgerTestCase):
    def test_settling_within_the_reservation_marks_settled_and_frees_the_unused_portion(self):
        reserve_budget_batch(EPISODE_KEY, FINGERPRINT, [self._reservation("a1", usd="0.10")])
        attempt = settle_budget_attempt(
            EPISODE_KEY, FINGERPRINT, "a1", actual_usd=Decimal("0.04")
        )
        self.assertEqual(attempt["state"], "settled")
        self.assertEqual(attempt["settled_usd"], "0.04")
        summary = budget_summary(EPISODE_KEY, FINGERPRINT)
        self.assertEqual(summary["settled_spend"], Decimal("0.04"))
        self.assertEqual(summary["live_reservations"], Decimal("0"))
        self.assertEqual(summary["headroom_usd"], EPISODE_AI_HARD_CAP_USD - Decimal("0.04"))

    def test_settling_over_the_reservation_is_a_durably_recorded_integrity_failure(self):
        reserve_budget_batch(EPISODE_KEY, FINGERPRINT, [self._reservation("a1", usd="0.10")])
        with self.assertRaises(BudgetIntegrityError):
            settle_budget_attempt(
                EPISODE_KEY, FINGERPRINT, "a1", actual_usd=Decimal("0.11")
            )
        # Durably recorded even though the call raised.
        path = budget_ledger_path(EPISODE_KEY, FINGERPRINT)
        import json

        payload = json.loads(self.store[path]["content"])
        self.assertIn("integrity_failure", payload)
        self.assertEqual(payload["integrity_failure"]["attempt_id"], "a1")

    def test_missing_actual_cost_marks_uncertain_not_released(self):
        reserve_budget_batch(EPISODE_KEY, FINGERPRINT, [self._reservation("a1", usd="0.10")])
        attempt = settle_budget_attempt(
            EPISODE_KEY, FINGERPRINT, "a1", actual_usd=None, reason="malformed usage.cost"
        )
        self.assertEqual(attempt["state"], "uncertain")
        summary = budget_summary(EPISODE_KEY, FINGERPRINT)
        # Uncertain spend still consumes the cap; it is not released optimistically.
        self.assertEqual(summary["uncertain_spend"], Decimal("0.10"))
        self.assertEqual(summary["live_reservations"], Decimal("0"))

    def test_settling_an_unknown_attempt_raises(self):
        reserve_budget_batch(EPISODE_KEY, FINGERPRINT, [self._reservation("a1", usd="0.10")])
        with self.assertRaises(ValueError):
            settle_budget_attempt(
                EPISODE_KEY, FINGERPRINT, "does-not-exist", actual_usd=Decimal("0.01")
            )

    def test_settling_an_already_released_attempt_raises(self):
        reserve_budget_batch(EPISODE_KEY, FINGERPRINT, [self._reservation("a1", usd="0.10")])
        release_budget_attempt_pre_send(EPISODE_KEY, FINGERPRINT, "a1")
        with self.assertRaises(ValueError):
            settle_budget_attempt(
                EPISODE_KEY, FINGERPRINT, "a1", actual_usd=Decimal("0.01")
            )

    def test_settling_a_previously_uncertain_attempt_is_allowed(self):
        reserve_budget_batch(EPISODE_KEY, FINGERPRINT, [self._reservation("a1", usd="0.10")])
        settle_budget_attempt(EPISODE_KEY, FINGERPRINT, "a1", actual_usd=None, reason="timeout")
        attempt = settle_budget_attempt(
            EPISODE_KEY, FINGERPRINT, "a1", actual_usd=Decimal("0.03")
        )
        self.assertEqual(attempt["state"], "settled")


class ReleaseBudgetAttemptPreSendTests(_BudgetLedgerTestCase):
    def test_releasing_a_reserved_attempt_frees_its_headroom(self):
        reserve_budget_batch(EPISODE_KEY, FINGERPRINT, [self._reservation("a1", usd="0.10")])
        attempt = release_budget_attempt_pre_send(EPISODE_KEY, FINGERPRINT, "a1")
        self.assertEqual(attempt["state"], "released")
        summary = budget_summary(EPISODE_KEY, FINGERPRINT)
        self.assertEqual(summary["live_reservations"], Decimal("0"))
        self.assertEqual(summary["headroom_usd"], EPISODE_AI_HARD_CAP_USD)

    def test_release_needs_nothing_beyond_the_ledger_itself(self):
        # No prepare-session lease (Task 10) or per-item claim (Task 9)
        # record is created anywhere in this test -- proving a reservation
        # is independently reconcilable rather than dependent on some other
        # object's fate.
        reserve_budget_batch(EPISODE_KEY, FINGERPRINT, [self._reservation("a1", usd="0.10")])
        attempt = release_budget_attempt_pre_send(EPISODE_KEY, FINGERPRINT, "a1")
        self.assertEqual(attempt["state"], "released")

    def test_cannot_pre_send_release_a_settled_attempt(self):
        reserve_budget_batch(EPISODE_KEY, FINGERPRINT, [self._reservation("a1", usd="0.10")])
        settle_budget_attempt(EPISODE_KEY, FINGERPRINT, "a1", actual_usd=Decimal("0.05"))
        with self.assertRaises(ValueError):
            release_budget_attempt_pre_send(EPISODE_KEY, FINGERPRINT, "a1")


class MarkBudgetAttemptUncertainTests(_BudgetLedgerTestCase):
    def test_marks_a_reserved_attempt_uncertain_with_a_reason(self):
        reserve_budget_batch(EPISODE_KEY, FINGERPRINT, [self._reservation("a1", usd="0.10")])
        attempt = mark_budget_attempt_uncertain(
            EPISODE_KEY, FINGERPRINT, "a1", reason="post_send_uncertain: connection reset"
        )
        self.assertEqual(attempt["state"], "uncertain")
        self.assertIn("connection reset", attempt["reason"])
        summary = budget_summary(EPISODE_KEY, FINGERPRINT)
        self.assertEqual(summary["uncertain_spend"], Decimal("0.10"))

    def test_requires_a_non_empty_reason(self):
        reserve_budget_batch(EPISODE_KEY, FINGERPRINT, [self._reservation("a1", usd="0.10")])
        with self.assertRaises(ValueError):
            mark_budget_attempt_uncertain(EPISODE_KEY, FINGERPRINT, "a1", reason="")

    def test_cannot_mark_a_settled_attempt_uncertain(self):
        reserve_budget_batch(EPISODE_KEY, FINGERPRINT, [self._reservation("a1", usd="0.10")])
        settle_budget_attempt(EPISODE_KEY, FINGERPRINT, "a1", actual_usd=Decimal("0.05"))
        with self.assertRaises(ValueError):
            mark_budget_attempt_uncertain(EPISODE_KEY, FINGERPRINT, "a1", reason="late timeout")


class BudgetSummaryTests(_BudgetLedgerTestCase):
    def test_summary_for_a_nonexistent_ledger_is_all_zero_with_full_headroom(self):
        summary = budget_summary(EPISODE_KEY, FINGERPRINT)
        self.assertEqual(summary["settled_spend"], Decimal("0"))
        self.assertEqual(summary["uncertain_spend"], Decimal("0"))
        self.assertEqual(summary["live_reservations"], Decimal("0"))
        self.assertEqual(summary["headroom_usd"], EPISODE_AI_HARD_CAP_USD)
        self.assertEqual(summary["third_asr_headroom_usd"], ASSISTED_THIRD_ASR_SUBCAP_USD)
        self.assertIsNone(summary["integrity_failure"])

    def test_summary_aggregates_across_every_state_and_third_asr_breakdown(self):
        reserve_budget_batch(
            EPISODE_KEY,
            FINGERPRINT,
            [
                self._reservation("settled", usd="0.10"),
                self._reservation("uncertain", usd="0.05"),
                self._reservation("released", usd="0.05"),
                self._reservation("still_reserved", usd="0.05"),
                self._reservation("asr", usd="0.05", third_asr=True),
            ],
        )
        settle_budget_attempt(EPISODE_KEY, FINGERPRINT, "settled", actual_usd=Decimal("0.08"))
        mark_budget_attempt_uncertain(EPISODE_KEY, FINGERPRINT, "uncertain", reason="timeout")
        release_budget_attempt_pre_send(EPISODE_KEY, FINGERPRINT, "released")

        summary = budget_summary(EPISODE_KEY, FINGERPRINT)
        self.assertEqual(summary["settled_spend"], Decimal("0.08"))
        self.assertEqual(summary["uncertain_spend"], Decimal("0.05"))
        self.assertEqual(summary["live_reservations"], Decimal("0.10"))  # still_reserved + asr
        self.assertEqual(summary["third_asr_live_reservations"], Decimal("0.05"))
        self.assertEqual(
            summary["headroom_usd"],
            EPISODE_AI_HARD_CAP_USD - Decimal("0.08") - Decimal("0.05") - Decimal("0.10"),
        )


class MultipleGenerationsAreIndependentTests(_BudgetLedgerTestCase):
    def test_two_source_fingerprints_for_the_same_episode_key_have_independent_ledgers(self):
        reserve_budget_batch(EPISODE_KEY, FINGERPRINT, [self._reservation("a1", usd="0.10")])
        reserve_budget_batch(
            EPISODE_KEY, OTHER_FINGERPRINT, [self._reservation("a1", usd="0.20")]
        )
        first = budget_summary(EPISODE_KEY, FINGERPRINT)
        second = budget_summary(EPISODE_KEY, OTHER_FINGERPRINT)
        self.assertEqual(first["live_reservations"], Decimal("0.10"))
        self.assertEqual(second["live_reservations"], Decimal("0.20"))


if __name__ == "__main__":
    unittest.main()
