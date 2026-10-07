"""TASK-125: operator reconciliation of a Whisper budget integrity failure."""

from __future__ import annotations

from decimal import Decimal
import json
import unittest
from unittest.mock import patch

from podcast_engine.ai_budget import (
    STAGE_THIRD_ASR,
    STAGE_WHISPER_TRANSCRIPTION,
    BudgetIntegrityError,
    BudgetLedgerError,
    budget_ledger_path,
    budget_summary,
    inventory_budget_integrity_failures,
    reconcile_whisper_integrity_failure,
    reserve_budget_batch,
    settle_budget_attempt,
)
from tests.test_ai_budget import EPISODE_KEY, FINGERPRINT, _FakeBlob, _FakeBucket

OVERRUN = Decimal("0.0186541665837665")
RESERVED = Decimal("0.0045375")


class _NamedBlob(_FakeBlob):
    def __init__(self, store, path, **kwargs):
        super().__init__(store, path, **kwargs)
        self.name = path


class _ListingBucket(_FakeBucket):
    def blob(self, path):
        return _NamedBlob(self._store, path)

    def list_blobs(self, prefix):
        return [_NamedBlob(self._store, path) for path in sorted(self._store) if path.startswith(prefix)]


class _ReconciliationCase(unittest.TestCase):
    def setUp(self):
        self.store: dict = {}
        patcher = patch(
            "podcast_engine.ai_budget.get_bucket",
            side_effect=lambda: _ListingBucket(self.store),
        )
        self.addCleanup(patcher.stop)
        patcher.start()

    def _overrun(self, attempt_id="w1", stage=STAGE_WHISPER_TRANSCRIPTION):
        reserve_budget_batch(
            EPISODE_KEY,
            FINGERPRINT,
            [{"attempt_id": attempt_id, "stage": stage, "third_asr": False, "reserved_usd": RESERVED}],
        )
        with self.assertRaises(BudgetIntegrityError):
            settle_budget_attempt(EPISODE_KEY, FINGERPRINT, attempt_id, actual_usd=OVERRUN)

    def _reconcile(self, **overrides):
        arguments = {
            "attempt_id": "w1",
            "actual_usd": OVERRUN,
            "reason": "OpenRouter ignored request routing; key guardrail now pins DeepInfra",
        }
        arguments.update(overrides)
        return reconcile_whisper_integrity_failure(EPISODE_KEY, FINGERPRINT, **arguments)


class WhisperIntegrityReconciliationTests(_ReconciliationCase):
    def test_inventory_lists_the_recorded_failure(self):
        self._overrun()
        rows = inventory_budget_integrity_failures(EPISODE_KEY)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["attempt_id"], "w1")
        self.assertEqual(rows[0]["stage"], STAGE_WHISPER_TRANSCRIPTION)
        self.assertEqual(rows[0]["state"], "uncertain")
        self.assertEqual(Decimal(rows[0]["actual_usd"]), OVERRUN)
        self.assertEqual(rows[0]["source_fingerprint"], FINGERPRINT)

    def test_reconciliation_settles_at_actual_cost_and_readmits_reservations(self):
        self._overrun()
        attempt = self._reconcile()
        self.assertEqual(attempt["state"], "settled")
        self.assertEqual(Decimal(attempt["settled_usd"]), OVERRUN)
        summary = budget_summary(EPISODE_KEY, FINGERPRINT)
        self.assertIsNone(summary["integrity_failure"])
        self.assertEqual(summary["settled_spend"], OVERRUN)
        self.assertEqual(summary["uncertain_spend"], Decimal("0"))
        payload = json.loads(self.store[budget_ledger_path(EPISODE_KEY, FINGERPRINT)]["content"])
        self.assertEqual(len(payload["integrity_reconciliations"]), 1)
        record = payload["integrity_reconciliations"][0]
        self.assertEqual(record["attempt_id"], "w1")
        self.assertIn("guardrail", record["reason"])
        self.assertIn("reconciled_at", record)
        self.assertEqual(inventory_budget_integrity_failures(EPISODE_KEY), [])
        reserve_budget_batch(
            EPISODE_KEY,
            FINGERPRINT,
            [{"attempt_id": "w2", "stage": STAGE_WHISPER_TRANSCRIPTION, "third_asr": False, "reserved_usd": RESERVED}],
        )

    def test_mismatched_attempt_or_cost_fails_closed(self):
        self._overrun()
        with self.assertRaises(BudgetLedgerError):
            self._reconcile(attempt_id="other")
        with self.assertRaises(BudgetLedgerError):
            self._reconcile(actual_usd=RESERVED)
        self.assertIsNotNone(budget_summary(EPISODE_KEY, FINGERPRINT)["integrity_failure"])

    def test_other_stages_and_missing_reason_are_refused(self):
        self._overrun(stage="summary")
        with self.assertRaises(BudgetLedgerError):
            self._reconcile()
        with self.assertRaises(ValueError):
            self._reconcile(reason="  ")
        with self.assertRaises(ValueError):
            self._reconcile(actual_usd=0.0186541665837665)

    def test_third_asr_whole_second_overrun_is_reconcilable(self):
        # TASK-126: a 15.6 s clip billed as 16 s exceeded its reservation.
        reserve_budget_batch(
            EPISODE_KEY,
            FINGERPRINT,
            [{"attempt_id": "t1", "stage": STAGE_THIRD_ASR, "third_asr": True, "reserved_usd": Decimal("0.00117")}],
        )
        with self.assertRaises(BudgetIntegrityError):
            settle_budget_attempt(EPISODE_KEY, FINGERPRINT, "t1", actual_usd=Decimal("0.0012"))
        attempt = reconcile_whisper_integrity_failure(
            EPISODE_KEY, FINGERPRINT, attempt_id="t1", actual_usd=Decimal("0.0012"),
            reason="fractional Third-ASR clip billed in whole seconds",
        )
        self.assertEqual(attempt["state"], "settled")
        self.assertIsNone(budget_summary(EPISODE_KEY, FINGERPRINT)["integrity_failure"])

    def test_nothing_to_reconcile_is_refused(self):
        with self.assertRaises(BudgetLedgerError):
            self._reconcile()


class ReconcileWhisperBudgetScriptTests(_ReconciliationCase):
    def _main(self, *argv):
        import contextlib
        import io

        from scripts import reconcile_whisper_budget

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = reconcile_whisper_budget.main(list(argv))
        return code, output.getvalue()

    def test_dry_run_lists_and_execute_requires_exact_restatement(self):
        self._overrun()
        code, output = self._main(EPISODE_KEY)
        self.assertEqual(code, 0)
        self.assertIn("attempt=w1", output)
        self.assertIn("DRY RUN: 1 unreconciled", output)
        code, output = self._main(EPISODE_KEY, "--execute", "--attempt", "w1")
        self.assertEqual(code, 2)
        code, output = self._main(
            EPISODE_KEY, "--execute", "--attempt", "w1", "--actual-usd", "0.0045375", "--reason", "x"
        )
        self.assertEqual(code, 1)
        self.assertIn("REFUSED", output)
        code, output = self._main(
            EPISODE_KEY, "--execute", "--attempt", "w1", "--actual-usd", str(OVERRUN), "--reason", "guardrail"
        )
        self.assertEqual(code, 0)
        self.assertIn("RECONCILED", output)
        self.assertIsNone(budget_summary(EPISODE_KEY, FINGERPRINT)["integrity_failure"])


if __name__ == "__main__":
    unittest.main()
