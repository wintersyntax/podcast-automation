"""TASK-109 fail-closed Third-ASR budget identity reconciliation."""

from __future__ import annotations

from decimal import Decimal
import json
import unittest
from unittest.mock import patch

from google.api_core.exceptions import NotFound, PreconditionFailed

from podcast_engine.ai_budget import (
    AI_BUDGET_POLICY_VERSION,
    AI_BUDGET_SCHEMA_VERSION,
    ASSISTED_THIRD_ASR_SUBCAP_USD,
    EPISODE_AI_HARD_CAP_USD,
    THIRD_ASR_BUDGET_IDENTITY_POLICY_VERSION,
    BudgetIdentityReconciliationRequired,
    BudgetIntegrityError,
    budget_ledger_path,
    budget_summary,
    ensure_fresh_third_asr_budget_identity,
    inventory_third_asr_budget_identities,
    reconcile_third_asr_budget_identity,
    require_third_asr_budget_identity_reconciled,
    reserve_budget_batch,
)


EPISODE_KEY = "a" * 24
SOURCE_FINGERPRINT = "sha256:" + "c" * 64
LEGACY_INPUT_FINGERPRINT = "sha256:" + "d" * 64
PREVIOUS_SOURCE_FINGERPRINT = "sha256:" + "e" * 64


class _FakeBlob:
    def __init__(self, store: dict, path: str):
        self._store = store
        self.name = path
        self.generation = None

    def reload(self):
        record = self._store.get(self.name)
        if record is None:
            raise NotFound("object does not exist")
        self.generation = record["generation"]

    def download_as_text(self, encoding="utf-8", if_generation_match=None):
        record = self._store.get(self.name)
        if record is None:
            raise PreconditionFailed("object does not exist")
        if if_generation_match is not None and record["generation"] != if_generation_match:
            raise PreconditionFailed("generation changed")
        return record["content"]

    def upload_from_string(self, content, content_type=None, if_generation_match=None):
        record = self._store.get(self.name)
        if if_generation_match == 0:
            if record is not None:
                raise PreconditionFailed("object already exists")
            self._store[self.name] = {"content": content, "generation": 1}
            return
        if record is None or record["generation"] != if_generation_match:
            raise PreconditionFailed("generation changed")
        self._store[self.name] = {
            "content": content,
            "generation": record["generation"] + 1,
        }


class _FakeBucket:
    def __init__(self, store: dict):
        self._store = store

    def blob(self, path: str) -> _FakeBlob:
        return _FakeBlob(self._store, path)

    def list_blobs(self, *, prefix: str):
        return [
            _FakeBlob(self._store, path)
            for path in sorted(self._store)
            if path.startswith(prefix)
        ]


class ThirdAsrBudgetIdentityTests(unittest.TestCase):
    def setUp(self):
        self.store: dict = {}
        patcher = patch(
            "podcast_engine.ai_budget.get_bucket",
            side_effect=lambda: _FakeBucket(self.store),
        )
        self.addCleanup(patcher.stop)
        patcher.start()

    def _ledger(self, fingerprint: str, attempts: dict) -> dict:
        return {
            "schema_version": AI_BUDGET_SCHEMA_VERSION,
            "policy_version": AI_BUDGET_POLICY_VERSION,
            "episode_key": EPISODE_KEY,
            "source_fingerprint": fingerprint,
            "hard_cap_usd": str(EPISODE_AI_HARD_CAP_USD),
            "third_asr_subcap_usd": str(ASSISTED_THIRD_ASR_SUBCAP_USD),
            "created_at": "2026-09-25T00:00:00+00:00",
            "attempts": attempts,
        }

    def _put_ledger(self, fingerprint: str, attempts: dict, *, generation: int = 1):
        path = budget_ledger_path(EPISODE_KEY, fingerprint)
        self.store[path] = {
            "content": json.dumps(self._ledger(fingerprint, attempts)),
            "generation": generation,
        }
        return path

    def _settled(self, amount: str) -> dict:
        return {
            "state": "settled",
            "stage": "third_asr",
            "third_asr": True,
            "reserved_usd": amount,
            "settled_usd": amount,
            "created_at": "2026-09-25T00:00:00+00:00",
            "updated_at": "2026-09-25T00:00:00+00:00",
        }

    def _uncertain(self, amount: str) -> dict:
        return {
            "state": "uncertain",
            "stage": "third_asr",
            "third_asr": True,
            "reserved_usd": amount,
            "created_at": "2026-09-25T00:00:00+00:00",
            "updated_at": "2026-09-25T00:00:00+00:00",
        }

    def _released(self, amount: str) -> dict:
        return {
            "state": "released",
            "stage": "third_asr",
            "third_asr": True,
            "reserved_usd": amount,
            "created_at": "2026-09-25T00:00:00+00:00",
            "updated_at": "2026-09-25T00:00:00+00:00",
        }

    def test_missing_marker_blocks_even_without_existing_budget_ledger(self):
        with self.assertRaises(BudgetIdentityReconciliationRequired):
            require_third_asr_budget_identity_reconciled(
                EPISODE_KEY, SOURCE_FINGERPRINT
            )

    def test_clean_generation_requires_explicit_empty_reconciliation_marker(self):
        self._put_ledger(SOURCE_FINGERPRINT, {}, generation=3)

        with self.assertRaises(BudgetIdentityReconciliationRequired):
            require_third_asr_budget_identity_reconciled(
                EPISODE_KEY, SOURCE_FINGERPRINT
            )

        reconcile_third_asr_budget_identity(
            EPISODE_KEY, SOURCE_FINGERPRINT, {}, apply=True
        )
        result = require_third_asr_budget_identity_reconciled(
            EPISODE_KEY, SOURCE_FINGERPRINT
        )
        self.assertEqual(
            result["policy_version"],
            THIRD_ASR_BUDGET_IDENTITY_POLICY_VERSION,
        )
        self.assertEqual(result["candidates"], [])

    def test_missing_marker_blocks_when_noncanonical_spend_exists(self):
        self._put_ledger(
            LEGACY_INPUT_FINGERPRINT,
            {"settled": self._settled("0.03")},
        )
        with self.assertRaises(BudgetIdentityReconciliationRequired):
            require_third_asr_budget_identity_reconciled(
                EPISODE_KEY, SOURCE_FINGERPRINT
            )

    def test_inventory_lists_only_noncanonical_consuming_third_asr_ledgers(self):
        self._put_ledger(
            LEGACY_INPUT_FINGERPRINT,
            {
                "settled": self._settled("0.03"),
                "released": self._released("0.07"),
            },
            generation=4,
        )
        self._put_ledger(
            PREVIOUS_SOURCE_FINGERPRINT,
            {"released": self._released("0.09")},
            generation=8,
        )
        self._put_ledger(
            SOURCE_FINGERPRINT,
            {"canonical": self._settled("0.01")},
            generation=2,
        )

        inventory = inventory_third_asr_budget_identities(
            EPISODE_KEY, SOURCE_FINGERPRINT
        )

        self.assertEqual(len(inventory["candidates"]), 1)
        candidate = inventory["candidates"][0]
        self.assertEqual(candidate["source_fingerprint"], LEGACY_INPUT_FINGERPRINT)
        self.assertEqual(candidate["generation"], 4)
        self.assertEqual(candidate["settled_usd"], "0.03")
        self.assertEqual(candidate["uncertain_usd"], "0")
        self.assertEqual(candidate["attempt_count"], 1)

    def test_apply_requires_explicit_classification_for_every_candidate(self):
        self._put_ledger(
            LEGACY_INPUT_FINGERPRINT,
            {"settled": self._settled("0.03")},
        )
        plan = reconcile_third_asr_budget_identity(
            EPISODE_KEY, SOURCE_FINGERPRINT, apply=False
        )
        self.assertEqual(plan["status"], "classification_required")
        self.assertEqual(plan["unclassified"], [LEGACY_INPUT_FINGERPRINT])

        with self.assertRaises(BudgetIdentityReconciliationRequired):
            reconcile_third_asr_budget_identity(
                EPISODE_KEY, SOURCE_FINGERPRINT, apply=True
            )

    def test_apply_carries_only_legacy_current_spend_and_is_idempotent(self):
        self._put_ledger(
            LEGACY_INPUT_FINGERPRINT,
            {
                "settled": self._settled("0.03"),
                "uncertain": self._uncertain("0.02"),
                "released": self._released("0.05"),
            },
            generation=5,
        )
        self._put_ledger(
            PREVIOUS_SOURCE_FINGERPRINT,
            {"settled": self._settled("0.04")},
            generation=7,
        )
        classifications = {
            LEGACY_INPUT_FINGERPRINT: "legacy_current_generation",
            PREVIOUS_SOURCE_FINGERPRINT: "previous_source_generation",
        }

        result = reconcile_third_asr_budget_identity(
            EPISODE_KEY,
            SOURCE_FINGERPRINT,
            classifications,
            apply=True,
        )
        self.assertEqual(result["status"], "reconciled")
        require_third_asr_budget_identity_reconciled(
            EPISODE_KEY, SOURCE_FINGERPRINT
        )

        summary = budget_summary(EPISODE_KEY, SOURCE_FINGERPRINT)
        self.assertEqual(summary["third_asr_settled"], Decimal("0.03"))
        self.assertEqual(summary["third_asr_uncertain"], Decimal("0.02"))
        self.assertEqual(
            summary["third_asr_headroom_usd"],
            ASSISTED_THIRD_ASR_SUBCAP_USD - Decimal("0.05"),
        )

        canonical_path = budget_ledger_path(EPISODE_KEY, SOURCE_FINGERPRINT)
        generation_before = self.store[canonical_path]["generation"]
        replay = reconcile_third_asr_budget_identity(
            EPISODE_KEY,
            SOURCE_FINGERPRINT,
            classifications,
            apply=True,
        )
        self.assertEqual(replay["status"], "already_reconciled")
        self.assertEqual(self.store[canonical_path]["generation"], generation_before)
        self.assertEqual(
            budget_summary(EPISODE_KEY, SOURCE_FINGERPRINT)["third_asr_settled"],
            Decimal("0.03"),
        )

    def test_clean_inventory_can_be_explicitly_reconciled(self):
        result = reconcile_third_asr_budget_identity(
            EPISODE_KEY, SOURCE_FINGERPRINT, {}, apply=True
        )
        self.assertEqual(result["status"], "reconciled")
        require_third_asr_budget_identity_reconciled(
            EPISODE_KEY, SOURCE_FINGERPRINT
        )

    def test_post_reconciliation_candidate_drift_closes_gate_again(self):
        path = self._put_ledger(
            LEGACY_INPUT_FINGERPRINT,
            {"settled": self._settled("0.03")},
            generation=3,
        )
        reconcile_third_asr_budget_identity(
            EPISODE_KEY,
            SOURCE_FINGERPRINT,
            {LEGACY_INPUT_FINGERPRINT: "legacy_current_generation"},
            apply=True,
        )

        payload = json.loads(self.store[path]["content"])
        payload["attempts"]["late"] = self._uncertain("0.01")
        self.store[path] = {
            "content": json.dumps(payload),
            "generation": 4,
        }

        with self.assertRaises(BudgetIdentityReconciliationRequired):
            require_third_asr_budget_identity_reconciled(
                EPISODE_KEY, SOURCE_FINGERPRINT
            )

    def test_runtime_reconciliation_check_does_not_require_bucket_listing(self):
        self._put_ledger(
            LEGACY_INPUT_FINGERPRINT,
            {"settled": self._settled("0.03")},
            generation=3,
        )
        reconcile_third_asr_budget_identity(
            EPISODE_KEY,
            SOURCE_FINGERPRINT,
            {LEGACY_INPUT_FINGERPRINT: "legacy_current_generation"},
            apply=True,
        )

        with patch.object(
            _FakeBucket,
            "list_blobs",
            side_effect=AssertionError("runtime must not list bucket objects"),
        ):
            require_third_asr_budget_identity_reconciled(
                EPISODE_KEY, SOURCE_FINGERPRINT
            )

    def test_legacy_integrity_failure_remains_fail_closed_on_canonical_ledger(self):
        path = self._put_ledger(
            LEGACY_INPUT_FINGERPRINT,
            {"uncertain": self._uncertain("0.03")},
            generation=6,
        )
        payload = json.loads(self.store[path]["content"])
        payload["integrity_failure"] = {
            "attempt_id": "uncertain",
            "reserved_usd": "0.03",
            "actual_usd": "0.04",
            "detected_at": "2026-09-25T00:00:01+00:00",
        }
        self.store[path] = {
            "content": json.dumps(payload),
            "generation": 7,
        }

        reconcile_third_asr_budget_identity(
            EPISODE_KEY,
            SOURCE_FINGERPRINT,
            {LEGACY_INPUT_FINGERPRINT: "legacy_current_generation"},
            apply=True,
        )

        summary = budget_summary(EPISODE_KEY, SOURCE_FINGERPRINT)
        self.assertEqual(
            summary["integrity_failure"]["reason"],
            "legacy_budget_identity_reconciliation",
        )
        with self.assertRaises(BudgetIntegrityError):
            reserve_budget_batch(
                EPISODE_KEY,
                SOURCE_FINGERPRINT,
                [
                    {
                        "attempt_id": "blocked",
                        "stage": "third_asr",
                        "third_asr": True,
                        "reserved_usd": Decimal("0.01"),
                    }
                ],
            )

    def test_reconciliation_classification_is_immutable_after_apply(self):
        self._put_ledger(
            LEGACY_INPUT_FINGERPRINT,
            {"settled": self._settled("0.03")},
        )
        reconcile_third_asr_budget_identity(
            EPISODE_KEY,
            SOURCE_FINGERPRINT,
            {LEGACY_INPUT_FINGERPRINT: "legacy_current_generation"},
            apply=True,
        )

        with self.assertRaises(BudgetIdentityReconciliationRequired):
            reconcile_third_asr_budget_identity(
                EPISODE_KEY,
                SOURCE_FINGERPRINT,
                {LEGACY_INPUT_FINGERPRINT: "previous_source_generation"},
                apply=True,
            )



class FreshGenerationReconciliationTests(unittest.TestCase):
    """TASK-126: runtime admission for a generation with no legacy spend."""

    setUp = ThirdAsrBudgetIdentityTests.setUp
    _ledger = ThirdAsrBudgetIdentityTests._ledger
    _put_ledger = ThirdAsrBudgetIdentityTests._put_ledger
    _settled = ThirdAsrBudgetIdentityTests._settled
    _uncertain = ThirdAsrBudgetIdentityTests._uncertain
    _released = ThirdAsrBudgetIdentityTests._released

    def _marker_path(self):
        from podcast_engine.ai_budget import third_asr_budget_reconciliation_path

        return third_asr_budget_reconciliation_path(EPISODE_KEY, SOURCE_FINGERPRINT)

    def test_absent_input_keyed_ledger_creates_a_runtime_marker(self):
        marker = ensure_fresh_third_asr_budget_identity(
            EPISODE_KEY, SOURCE_FINGERPRINT, LEGACY_INPUT_FINGERPRINT
        )
        self.assertEqual(marker["candidates"], [])
        self.assertEqual(marker["basis"], "runtime_named_read_v1")
        self.assertEqual(marker["checked_input_fingerprint"], LEGACY_INPUT_FINGERPRINT)
        self.assertIn(self._marker_path(), self.store)
        require_third_asr_budget_identity_reconciled(EPISODE_KEY, SOURCE_FINGERPRINT)

    def test_input_keyed_ledger_without_third_asr_spend_is_clean(self):
        self._put_ledger(LEGACY_INPUT_FINGERPRINT, {"r": self._released("0.01")})
        marker = ensure_fresh_third_asr_budget_identity(
            EPISODE_KEY, SOURCE_FINGERPRINT, LEGACY_INPUT_FINGERPRINT
        )
        self.assertEqual(marker["basis"], "runtime_named_read_v1")

    def test_input_keyed_third_asr_spend_still_needs_the_operator(self):
        for attempt in (self._settled("0.002"), self._uncertain("0.002")):
            with self.subTest(attempt=attempt["state"]):
                self.store.clear()
                self._put_ledger(LEGACY_INPUT_FINGERPRINT, {"a": attempt})
                with self.assertRaises(BudgetIdentityReconciliationRequired):
                    ensure_fresh_third_asr_budget_identity(
                        EPISODE_KEY, SOURCE_FINGERPRINT, LEGACY_INPUT_FINGERPRINT
                    )
                self.assertNotIn(self._marker_path(), self.store)

    def test_existing_marker_is_never_rewritten(self):
        first = ensure_fresh_third_asr_budget_identity(
            EPISODE_KEY, SOURCE_FINGERPRINT, LEGACY_INPUT_FINGERPRINT
        )
        before = dict(self.store[self._marker_path()])
        self._put_ledger(LEGACY_INPUT_FINGERPRINT, {"a": self._settled("0.002")})
        second = ensure_fresh_third_asr_budget_identity(
            EPISODE_KEY, SOURCE_FINGERPRINT, LEGACY_INPUT_FINGERPRINT
        )
        self.assertEqual(first, second)
        self.assertEqual(self.store[self._marker_path()], before)

    def test_missing_or_identical_input_fingerprint_needs_no_named_read(self):
        for input_fingerprint in (None, "not-a-fingerprint", SOURCE_FINGERPRINT):
            with self.subTest(input_fingerprint=input_fingerprint):
                self.store.clear()
                marker = ensure_fresh_third_asr_budget_identity(
                    EPISODE_KEY, SOURCE_FINGERPRINT, input_fingerprint
                )
                self.assertIsNone(marker["checked_input_fingerprint"])


if __name__ == "__main__":
    unittest.main()
