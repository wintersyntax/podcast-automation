"""TASK-076 Task 10: whole-selection prepare-session lease.

Exercises begin_assisted_preparation / reconcile_expired_prepare_session /
advance_prepare_session_item against a small in-memory fake GCS blob store
that reproduces the real generation/if_generation_match/PreconditionFailed/
NotFound semantics the CAS loop depends on -- mirroring the fixture already
proven in tests/test_third_asr_claims.py -- while the review record and
budget/pricing collaborators are mocked at the podcast_engine.human_review
module boundary, mirroring tests/test_third_asr_budget.py's _Harness
convention. reserve_budget_batch/release_budget_attempt_pre_send's own
ledger CAS/cap-enforcement behavior is already covered directly in
tests/test_ai_budget.py; this file exercises begin_assisted_preparation's
own orchestration (validation, admission-or-nothing, cache-mixture pricing,
deterministic tokens, expiry/resume, and item lifecycle), not those
collaborators' internals.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
import unittest
from unittest.mock import ANY, patch

from google.api_core.exceptions import NotFound, PreconditionFailed

from podcast_engine.ai_budget import BudgetAdmissionError
from podcast_engine.ai_pricing import AudioPricing
from podcast_engine.human_review import (
    PREPARE_SESSION_TTL_SECONDS,
    PrepareSessionConflict,
    PrepareSessionError,
    advance_prepare_session_item,
    begin_assisted_preparation,
    prepare_session_path,
    reconcile_expired_prepare_session,
)
from podcast_engine.review_audio import clip_window, third_asr_cache_key


EPISODE_KEY = "b" * 24
FINGERPRINT = "sha256:" + "d" * 64
GENERATION_FINGERPRINT = "sha256:" + "9" * 64
PRICING = AudioPricing(
    model_id="openai/gpt-transcribe",
    usd_per_second=Decimal("0.0002"),
    captured_at="2026-09-21T00:00:00Z",
    source_api="https://openrouter.ai/api/v1/models?output_modalities=transcription",
    evidence_sha256="sha256:" + "0" * 64,
)


def _iso(offset_seconds: float = 0.0) -> str:
    return (datetime(2026, 9, 21, tzinfo=timezone.utc) + timedelta(seconds=offset_seconds)).isoformat()


def _eligible_item(item_id: int, *, start: float, end: float, **overrides) -> dict:
    item = {
        "id": item_id,
        "kind": "wording_difference",
        "category": "other",
        "apple_text": "off stage and traveled back across Austria",
        "whisper_text": "offstage and traveled back from Austria",
        "source_only": False,
        "risk_reasons": [],
        "domain_terms": [],
        "citation_signal": False,
        "preservation_class": "not_source_only",
        "merge_action": "review_kept_primary",
        "anomaly": None,
        "custom_edit": None,
        "third_asr": None,
        "representation_modified": False,
        "generation_stale": False,
        "whisper_start_timestamp": start,
        "whisper_end_timestamp": end,
    }
    item.update(overrides)
    return item


def _protected_item(item_id: int, *, start: float, end: float, **overrides) -> dict:
    return _eligible_item(
        item_id,
        start=start,
        end=end,
        category="protocol_number",
        risk_reasons=["protocol_number"],
        **overrides,
    )


def _cached_evidence(item_id: int, *, start: float, end: float) -> dict:
    window = clip_window(start, end)
    cache_key = third_asr_cache_key(input_fingerprint=FINGERPRINT, item_id=item_id, window=window)
    return {"text": "already transcribed", "cache_key": cache_key, "window": window}


def _record(items: list[dict], *, generation_fingerprint: str = GENERATION_FINGERPRINT) -> dict:
    return {
        "episode_key": EPISODE_KEY,
        "input_fingerprint": FINGERPRINT,
        "human_review_generation_fingerprint": generation_fingerprint,
        "human_review": items,
    }


class _FakeBlob:
    """Reproduces the subset of google.cloud.storage.Blob this module uses."""

    def __init__(self, store: dict, path: str):
        self._store = store
        self._path = path
        self.generation = None

    def reload(self):
        record = self._store.get(self._path)
        if record is None:
            raise NotFound("object does not exist")
        self.generation = record["generation"]

    def download_as_text(self, encoding="utf-8", if_generation_match=None):
        record = self._store.get(self._path)
        if record is None:
            raise PreconditionFailed("object does not exist")
        if if_generation_match is not None and record["generation"] != if_generation_match:
            raise PreconditionFailed("generation changed since reload")
        return record["content"]

    def upload_from_string(self, content, content_type=None, if_generation_match=None):
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
    def __init__(self, store: dict):
        self._store = store

    def blob(self, path: str) -> _FakeBlob:
        return _FakeBlob(self._store, path)


class _PrepareTestCase(unittest.TestCase):
    """Bundles the standard set of begin_assisted_preparation collaborator
    patches: a real fake-GCS-backed prepare-session object, plus a mocked
    review record and mocked budget/pricing calls."""

    def setUp(self):
        self.store: dict = {}
        self.record = _record([])
        self.reserve_calls: list[list[dict]] = []
        self.release_calls: list[str] = []
        self.pricing_calls = 0
        self._reserve_side_effect = None

        self._patches = [
            patch(
                "podcast_engine.human_review.get_bucket",
                side_effect=lambda: _FakeBucket(self.store),
            ),
            patch(
                "podcast_engine.human_review.load_review_record_with_generation",
                side_effect=lambda key: (self.record, 1),
            ),
            patch(
                "podcast_engine.human_review.ensure_fresh_third_asr_budget_identity"
            ),
            patch(
                "podcast_engine.human_review.resolve_audio_model_pricing",
                side_effect=self._resolve_audio_model_pricing,
            ),
            patch(
                "podcast_engine.human_review.reserve_budget_batch",
                side_effect=self._reserve_budget_batch,
            ),
            patch(
                "podcast_engine.human_review.release_budget_attempt_pre_send",
                side_effect=self._release_budget_attempt_pre_send,
            ),
        ]
        self.mocks = {}
        for one in self._patches:
            mock = one.start()
            self.addCleanup(one.stop)
            self.mocks[one.attribute] = mock

    def _resolve_audio_model_pricing(self, *args, **kwargs):
        self.pricing_calls += 1
        return PRICING

    def _reserve_budget_batch(self, episode_key, source_fingerprint, reservations, **kwargs):
        reservations = list(reservations)
        self.reserve_calls.append(reservations)
        if self._reserve_side_effect is not None:
            self._reserve_side_effect(reservations)
        return {"admitted": True}

    def _release_budget_attempt_pre_send(self, episode_key, source_fingerprint, attempt_id):
        self.release_calls.append(attempt_id)
        return {"attempt_id": attempt_id, "state": "released"}

    def _begin(self, selected_ids, *, session_id=None, now_value=None, **kwargs):
        kwargs.setdefault("expected_review_generation_fingerprint", GENERATION_FINGERPRINT)
        now_fn = (lambda: now_value) if now_value is not None else (lambda: _iso())
        return begin_assisted_preparation(
            EPISODE_KEY,
            FINGERPRINT,
            selected_ids,
            session_id=session_id,
            api_key="third-asr-key",
            now=now_fn,
            **kwargs,
        )


class PrepareSessionPathTests(unittest.TestCase):
    def test_path_includes_fingerprint_digest_and_session_id(self):
        path = prepare_session_path(EPISODE_KEY, FINGERPRINT, "session-1")
        self.assertEqual(
            path, f"episodes/{EPISODE_KEY}/ai/prepare-sessions/{'d' * 64}/session-1.json"
        )

    def test_rejects_malformed_episode_key(self):
        with self.assertRaises(ValueError):
            prepare_session_path("not-hex", FINGERPRINT, "session-1")

    def test_rejects_malformed_fingerprint(self):
        with self.assertRaises(ValueError):
            prepare_session_path(EPISODE_KEY, "not-a-fingerprint", "session-1")

    def test_rejects_empty_session_id(self):
        with self.assertRaises(ValueError):
            prepare_session_path(EPISODE_KEY, FINGERPRINT, "")


class BeginAssistedPreparationValidationTests(_PrepareTestCase):
    """RED max-40/stale-generation/policy mismatch."""

    def test_rejects_empty_selection(self):
        with self.assertRaises(ValueError):
            self._begin([])
        self.assertEqual(self.reserve_calls, [])

    def test_rejects_more_than_max_items(self):
        with self.assertRaises(ValueError):
            self._begin(list(range(1, 42)))
        self.assertEqual(self.reserve_calls, [])

    def test_rejects_duplicate_ids(self):
        with self.assertRaises(ValueError):
            self._begin([1, 1])
        self.assertEqual(self.reserve_calls, [])

    def test_rejects_non_int_id(self):
        with self.assertRaises(ValueError):
            self._begin([1, "2"])
        self.assertEqual(self.reserve_calls, [])

    def test_rejects_bool_id(self):
        with self.assertRaises(ValueError):
            self._begin([True])
        self.assertEqual(self.reserve_calls, [])

    def test_rejects_stale_review_generation(self):
        self.record = _record(
            [_eligible_item(1, start=10.0, end=12.0)],
            generation_fingerprint="sha256:" + "1" * 64,
        )
        with self.assertRaises(ValueError):
            self._begin([1])
        self.assertEqual(self.reserve_calls, [])
        self.assertEqual(self.store, {})

    def test_rejects_non_eligible_selected_item(self):
        self.record = _record([_protected_item(1, start=10.0, end=12.0)])
        with self.assertRaises(ValueError) as ctx:
            self._begin([1])
        self.assertIn("1", str(ctx.exception))
        self.assertEqual(self.reserve_calls, [])
        self.assertEqual(self.store, {})

    def test_rejects_unknown_item_id(self):
        self.record = _record([_eligible_item(1, start=10.0, end=12.0)])
        with self.assertRaises(ValueError):
            self._begin([999])
        self.assertEqual(self.reserve_calls, [])


class BeginAssistedPreparationBudgetTests(_PrepareTestCase):
    """RED whole-selection over-budget -> zero new reservations/provider calls."""

    def test_over_budget_raises_and_writes_nothing(self):
        self.record = _record([_eligible_item(1, start=10.0, end=12.0)])

        def _deny(reservations):
            raise BudgetAdmissionError("would exceed the hard cap")

        self._reserve_side_effect = _deny

        with self.assertRaises(BudgetAdmissionError):
            self._begin([1])

        self.assertEqual(len(self.reserve_calls), 1)
        # The whole batch was attempted atomically in one call -- never a
        # partial per-item sequence -- and since it was denied, no
        # prepare-session object was ever persisted.
        self.assertEqual(self.store, {})


class BeginAssistedPreparationSuccessTests(_PrepareTestCase):
    """RED atomic success + deterministic item tokens/attempt IDs."""

    def test_creates_session_with_deterministic_attempt_ids_and_reserves_once(self):
        self.record = _record(
            [
                _eligible_item(1, start=10.0, end=12.0),
                _eligible_item(2, start=20.0, end=23.0),
            ]
        )

        session = self._begin([2, 1], session_id="fixed-session")

        self.assertEqual(session["session_id"], "fixed-session")
        self.assertEqual(session["selected_ids"], [1, 2])
        self.assertEqual(len(self.reserve_calls), 1)
        self.mocks["ensure_fresh_third_asr_budget_identity"].assert_called_once_with(
            EPISODE_KEY, FINGERPRINT, ANY
        )
        reserved_ids = {r["attempt_id"] for r in self.reserve_calls[0]}
        self.assertEqual(
            reserved_ids,
            {
                session["items"]["1"]["attempt_id"],
                session["items"]["2"]["attempt_id"],
            },
        )
        for difference_id in ("1", "2"):
            item = session["items"][difference_id]
            self.assertEqual(item["state"], "queued")
            self.assertFalse(item["cached"])
            self.assertTrue(item["attempt_id"].startswith("prepare-fixed-session-"))
            self.assertTrue(item["attempt_id"].endswith(f"-{difference_id}"))
            self.assertIsNotNone(item["reserved_usd"])
        self.assertIn(prepare_session_path(EPISODE_KEY, FINGERPRINT, "fixed-session"), self.store)

    def test_session_id_defaults_to_a_generated_id_when_not_supplied(self):
        self.record = _record([_eligible_item(1, start=10.0, end=12.0)])

        session = self._begin([1])

        self.assertTrue(session["session_id"])
        self.assertIn(
            prepare_session_path(EPISODE_KEY, FINGERPRINT, session["session_id"]), self.store
        )


class BeginAssistedPreparationCacheMixtureTests(_PrepareTestCase):
    """RED cached mixture and replay/resume without double reservation."""

    def test_cached_item_costs_nothing_and_is_marked_prepared(self):
        cached_evidence = _cached_evidence(1, start=10.0, end=12.0)
        self.record = _record(
            [
                _eligible_item(1, start=10.0, end=12.0, third_asr=cached_evidence),
                _eligible_item(2, start=20.0, end=23.0),
            ]
        )

        session = self._begin([1, 2], session_id="mix-session")

        self.assertEqual(len(self.reserve_calls), 1)
        self.assertEqual(len(self.reserve_calls[0]), 1)
        self.assertEqual(self.reserve_calls[0][0]["attempt_id"], session["items"]["2"]["attempt_id"])
        self.assertEqual(session["items"]["1"]["state"], "prepared")
        self.assertTrue(session["items"]["1"]["cached"])
        self.assertIsNone(session["items"]["1"]["attempt_id"])
        self.assertIsNone(session["items"]["1"]["reserved_usd"])
        self.assertEqual(session["items"]["2"]["state"], "queued")

    def test_all_cached_selection_skips_pricing_and_reservation_entirely(self):
        cached_evidence = _cached_evidence(1, start=10.0, end=12.0)
        self.record = _record([_eligible_item(1, start=10.0, end=12.0, third_asr=cached_evidence)])

        session = self._begin([1])

        self.assertEqual(self.reserve_calls, [])
        self.assertEqual(self.pricing_calls, 0)
        self.mocks["ensure_fresh_third_asr_budget_identity"].assert_not_called()
        self.assertEqual(session["items"]["1"]["state"], "prepared")

    def test_replay_with_same_session_id_and_selection_is_a_no_op(self):
        self.record = _record([_eligible_item(1, start=10.0, end=12.0)])

        first = self._begin([1], session_id="replay-session", now_value=_iso(0))
        second = self._begin([1], session_id="replay-session", now_value=_iso(1))

        self.assertEqual(first, second)
        self.assertEqual(len(self.reserve_calls), 1)
        self.assertEqual(self.pricing_calls, 1)

    def test_same_session_id_different_selection_raises_conflict(self):
        self.record = _record(
            [
                _eligible_item(1, start=10.0, end=12.0),
                _eligible_item(2, start=20.0, end=23.0),
            ]
        )

        self._begin([1], session_id="collide-session", now_value=_iso(0))
        with self.assertRaises(PrepareSessionError):
            self._begin([1, 2], session_id="collide-session", now_value=_iso(1))

        # Only the first call's reservation ever happened.
        self.assertEqual(len(self.reserve_calls), 1)


class BeginAssistedPreparationExpiryTests(_PrepareTestCase):
    """RED session TTL/expiry: only unconsumed pre-send reservations
    release; in-flight/uncertain remain charged."""

    def test_expired_session_is_reconciled_and_a_fresh_session_is_created(self):
        self.record = _record([_eligible_item(1, start=10.0, end=12.0)])

        first = self._begin([1], session_id="expiring-session", now_value=_iso(0))
        first_attempt_id = first["items"]["1"]["attempt_id"]

        past_expiry = _iso(PREPARE_SESSION_TTL_SECONDS + 1)
        second = self._begin([1], session_id="expiring-session", now_value=past_expiry)
        second_attempt_id = second["items"]["1"]["attempt_id"]

        # The stale attempt was released exactly once (pre-send -- it was
        # still "queued", never touched by a provider), and the fresh
        # session's attempt_id is provably different so reserve_budget_batch
        # cannot mistake it for a replay of the released (no-longer-live)
        # attempt.
        self.assertEqual(self.release_calls, [first_attempt_id])
        self.assertNotEqual(first_attempt_id, second_attempt_id)
        self.assertEqual(len(self.reserve_calls), 2)
        self.assertEqual(second["items"]["1"]["state"], "queued")
        self.assertNotEqual(first["created_at"], second["created_at"])

    def test_in_flight_item_is_not_released_on_expiry(self):
        self.record = _record([_eligible_item(1, start=10.0, end=12.0)])
        session = self._begin([1], session_id="in-flight-session", now_value=_iso(0))
        attempt_id = session["items"]["1"]["attempt_id"]

        advance_prepare_session_item(
            EPISODE_KEY,
            FINGERPRINT,
            "in-flight-session",
            1,
            state="in_flight",
            attempt_id=attempt_id,
        )

        reconciled = reconcile_expired_prepare_session(
            EPISODE_KEY, FINGERPRINT, "in-flight-session", now=lambda: _iso(PREPARE_SESSION_TTL_SECONDS + 1)
        )

        self.assertEqual(self.release_calls, [])
        self.assertEqual(reconciled["items"]["1"]["state"], "in_flight")

    def test_reconcile_is_a_noop_when_not_expired(self):
        self.record = _record([_eligible_item(1, start=10.0, end=12.0)])
        session = self._begin([1], session_id="fresh-session", now_value=_iso(0))

        reconciled = reconcile_expired_prepare_session(
            EPISODE_KEY, FINGERPRINT, "fresh-session", now=lambda: _iso(1)
        )

        self.assertEqual(reconciled, session)
        self.assertEqual(self.release_calls, [])

    def test_reconcile_is_idempotent(self):
        self.record = _record([_eligible_item(1, start=10.0, end=12.0)])
        self._begin([1], session_id="double-reconcile-session", now_value=_iso(0))

        past_expiry = lambda: _iso(PREPARE_SESSION_TTL_SECONDS + 1)
        reconcile_expired_prepare_session(EPISODE_KEY, FINGERPRINT, "double-reconcile-session", now=past_expiry)
        reconcile_expired_prepare_session(EPISODE_KEY, FINGERPRINT, "double-reconcile-session", now=past_expiry)

        self.assertEqual(len(self.release_calls), 1)

    def test_reconcile_raises_for_missing_session(self):
        with self.assertRaises(PrepareSessionError):
            reconcile_expired_prepare_session(EPISODE_KEY, FINGERPRINT, "no-such-session")


class AdvancePrepareSessionItemTests(_PrepareTestCase):
    """RED item lifecycle transitions and canonical reload/resume semantics."""

    def _session_with_one_item(self, session_id="lifecycle-session"):
        self.record = _record([_eligible_item(1, start=10.0, end=12.0)])
        session = self._begin([1], session_id=session_id, now_value=_iso(0))
        return session, session["items"]["1"]["attempt_id"]

    def test_valid_transition_queued_to_in_flight_succeeds(self):
        _, attempt_id = self._session_with_one_item()

        updated = advance_prepare_session_item(
            EPISODE_KEY, FINGERPRINT, "lifecycle-session", 1, state="in_flight", attempt_id=attempt_id
        )

        self.assertEqual(updated["items"]["1"]["state"], "in_flight")

    def test_valid_transition_chain_to_prepared_succeeds(self):
        _, attempt_id = self._session_with_one_item()

        advance_prepare_session_item(
            EPISODE_KEY, FINGERPRINT, "lifecycle-session", 1, state="in_flight", attempt_id=attempt_id
        )
        updated = advance_prepare_session_item(
            EPISODE_KEY, FINGERPRINT, "lifecycle-session", 1, state="prepared", attempt_id=attempt_id
        )

        self.assertEqual(updated["items"]["1"]["state"], "prepared")

    def test_invalid_transition_is_rejected(self):
        _, attempt_id = self._session_with_one_item()

        with self.assertRaises(PrepareSessionError):
            advance_prepare_session_item(
                EPISODE_KEY, FINGERPRINT, "lifecycle-session", 1, state="prepared", attempt_id=attempt_id
            )

    def test_unknown_state_value_is_rejected(self):
        _, attempt_id = self._session_with_one_item()

        with self.assertRaises(ValueError):
            advance_prepare_session_item(
                EPISODE_KEY, FINGERPRINT, "lifecycle-session", 1, state="bogus", attempt_id=attempt_id
            )

    def test_mismatched_token_is_rejected(self):
        self._session_with_one_item()

        with self.assertRaises(PrepareSessionError):
            advance_prepare_session_item(
                EPISODE_KEY,
                FINGERPRINT,
                "lifecycle-session",
                1,
                state="in_flight",
                attempt_id="not-the-real-token",
            )

    def test_unknown_item_is_rejected(self):
        self._session_with_one_item()

        with self.assertRaises(ValueError):
            advance_prepare_session_item(
                EPISODE_KEY, FINGERPRINT, "lifecycle-session", 999, state="in_flight", attempt_id="anything"
            )

    def test_unknown_session_is_rejected(self):
        with self.assertRaises(PrepareSessionError):
            advance_prepare_session_item(
                EPISODE_KEY, FINGERPRINT, "no-such-session", 1, state="in_flight", attempt_id="anything"
            )


if __name__ == "__main__":
    unittest.main()
