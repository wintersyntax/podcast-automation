"""TASK-076 Task 9: ensure_third_asr's claim/budget/failure-taxonomy/
generation-verified-finalize orchestration.

The generation-CAS claim primitives (acquire/release/reclaim/TTL/stale
rejection) are already exercised directly against a fake GCS blob store
in tests/test_third_asr_claims.py. This file instead exercises
ensure_third_asr itself -- the orchestration that wires claims, budget
reservation/settlement, the failure taxonomy, bounded retry, and the
generation-verified review-record finalize together -- by mocking each
collaborator at the podcast_engine.human_review module boundary, the
same convention already used for ensure_third_asr in
tests/test_human_review.py.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import requests
from google.api_core.exceptions import PreconditionFailed

from podcast_engine.ai_pricing import AudioPricing
from podcast_engine.human_review import (
    THIRD_ASR_MAX_ATTEMPTS,
    ThirdAsrClaimConflict,
    ThirdAsrInFlight,
    ensure_third_asr,
)
from podcast_engine.review_audio import clip_window, third_asr_cache_key


EPISODE_KEY = "d" * 24
FINGERPRINT = "sha256:" + "e" * 64
SOURCE_FINGERPRINT = "sha256:" + "f" * 64
DIFFERENCE_ID = 1
WINDOW = clip_window(100.0, 101.0)
CACHE_KEY = third_asr_cache_key(
    input_fingerprint=FINGERPRINT, item_id=DIFFERENCE_ID, window=WINDOW
)
PRICING = AudioPricing(
    model_id="openai/gpt-transcribe",
    usd_per_second=Decimal("0.0002"),
    captured_at="2026-09-21T00:00:00Z",
    source_api="https://openrouter.ai/api/v1/models?output_modalities=transcription",
    evidence_sha256="sha256:" + "0" * 64,
)
EXPECTED_RESERVED_USD = Decimal(str(WINDOW["duration"])) * PRICING.usd_per_second


def _item() -> dict:
    return {
        "id": DIFFERENCE_ID,
        "whisper_start_timestamp": 100.0,
        "whisper_end_timestamp": 101.0,
    }


def _record() -> dict:
    return {
        "episode_key": EPISODE_KEY,
        "input_fingerprint": FINGERPRINT,
        "source_fingerprint": SOURCE_FINGERPRINT,
        "human_review": [_item()],
    }


def _claim(*, request_id: str, budget_attempt_id: str) -> dict:
    return {
        "difference_id": DIFFERENCE_ID,
        "cache_key": CACHE_KEY,
        "request_id": request_id,
        "status": "in_flight",
        "created_at": "2026-09-21T00:00:00+00:00",
        "expires_at": "2026-09-21T00:05:00+00:00",
        "model": "openai/gpt-transcribe",
        "window": WINDOW,
        "prepare_session_id": None,
        "budget_attempt_id": budget_attempt_id,
    }


def _http_error(status_code: int) -> requests.HTTPError:
    response = Mock()
    response.status_code = status_code
    error = requests.HTTPError(f"HTTP {status_code}")
    error.response = response
    return error


class _Harness:
    """Bundles the standard set of ensure_third_asr collaborator patches."""

    def __init__(self, test: unittest.TestCase, *, record: dict | None = None):
        self.test = test
        self.record = record if record is not None else _record()
        self.reserve_calls: list[dict] = []
        self.reserve_fingerprints: list[str] = []
        self.claim_fingerprints: list[str] = []
        self.acquire_calls = 0
        self.saved_records: list[dict] = []
        self.sleep_calls: list[float] = []
        self.acquire_prepare_session_ids: list[str | None] = []

        self._patches = [
            patch(
                "podcast_engine.human_review.load_review_record_with_generation",
                side_effect=lambda key: (self.record, 7),
            ),
            patch(
                "podcast_engine.human_review.save_review_record",
                side_effect=self._save_review_record,
            ),
            patch(
                "podcast_engine.human_review.resolve_audio_model_pricing",
                return_value=PRICING,
            ),
            patch(
                "podcast_engine.human_review.ensure_fresh_third_asr_budget_identity"
            ),
            patch(
                "podcast_engine.human_review.reserve_budget_batch",
                side_effect=self._reserve_budget_batch,
            ),
            patch("podcast_engine.human_review.release_budget_attempt_pre_send"),
            patch("podcast_engine.human_review.mark_budget_attempt_uncertain"),
            patch("podcast_engine.human_review.settle_budget_attempt"),
            patch(
                "podcast_engine.human_review.acquire_third_asr_claim",
                side_effect=self._acquire_third_asr_claim,
            ),
            patch(
                "podcast_engine.human_review._refresh_third_asr_claim_for_retry",
                side_effect=self._refresh_third_asr_claim_for_retry,
            ),
            patch("podcast_engine.human_review.release_third_asr_claim"),
            patch(
                "podcast_engine.human_review.ensure_audio_clip",
                return_value=(Path("/tmp/nonexistent-task076-clip.wav"), WINDOW),
            ),
        ]
        self.mocks = {}

    def _save_review_record(self, key, record, *, if_generation_match=None):
        self.saved_records.append(record)
        return record

    def _reserve_budget_batch(self, episode_key, source_fingerprint, reservations, **kwargs):
        self.reserve_fingerprints.append(source_fingerprint)
        self.reserve_calls.append(dict(reservations[0]))
        return {"admitted": True}

    def _acquire_third_asr_claim(self, *args, **kwargs):
        self.acquire_calls += 1
        self.claim_fingerprints.append(args[1])
        self.acquire_prepare_session_ids.append(kwargs.get("prepare_session_id"))
        return _claim(
            request_id=f"request-{self.acquire_calls}",
            budget_attempt_id=kwargs["budget_attempt_id"],
        )

    def _refresh_third_asr_claim_for_retry(self, *args, **kwargs):
        self.acquire_calls += 1
        self.claim_fingerprints.append(args[1])
        return _claim(
            request_id=f"request-{self.acquire_calls}",
            budget_attempt_id=kwargs["budget_attempt_id"],
        )

    def sleep(self, seconds: float) -> None:
        self.sleep_calls.append(seconds)

    def __enter__(self):
        for one in self._patches:
            mock = one.start()
            self.mocks[one.attribute] = mock
        return self

    def __exit__(self, *exc_info):
        for one in reversed(self._patches):
            one.stop()
        return False

    def call(self, *, transcribe_side_effect, **extra_kwargs):
        episode = {"episode_key": EPISODE_KEY}
        with patch(
            "podcast_engine.human_review.transcribe_review_clip",
            side_effect=transcribe_side_effect,
        ) as transcribe:
            self.transcribe = transcribe
            return ensure_third_asr(
                episode,
                DIFFERENCE_ID,
                api_key="third-asr-key",
                sleep=self.sleep,
                random_fn=lambda: 0.0,
                **extra_kwargs,
            )


class ThirdAsrBudgetOrchestrationTests(unittest.TestCase):
    def test_happy_path_reserves_claims_transcribes_settles_and_finalizes(self):
        with _Harness(self) as harness:
            evidence = harness.call(
                transcribe_side_effect=[{"text": "third source", "usage": {"cost": 0.0021}}]
            )

        self.assertEqual(evidence["text"], "third source")
        self.assertEqual(evidence["cache_key"], CACHE_KEY)
        self.assertEqual(len(harness.reserve_calls), 1)
        self.assertEqual(harness.reserve_calls[0]["reserved_usd"], EXPECTED_RESERVED_USD)
        self.assertTrue(harness.reserve_calls[0]["third_asr"])
        self.assertEqual(harness.acquire_calls, 1)
        self.assertEqual(len(harness.saved_records), 1)
        saved_item = harness.saved_records[0]["human_review"][0]
        self.assertEqual(saved_item["third_asr"]["text"], "third source")
        self.assertEqual(harness.sleep_calls, [])

    def test_budget_identity_is_source_generation_while_evidence_identity_stays_input(self):
        with _Harness(self) as harness:
            evidence = harness.call(
                transcribe_side_effect=[{"text": "third source", "usage": {"cost": 0.0021}}]
            )

        self.assertEqual(evidence["cache_key"], CACHE_KEY)
        self.assertEqual(harness.reserve_fingerprints, [SOURCE_FINGERPRINT])
        self.assertEqual(harness.claim_fingerprints, [FINGERPRINT])
        self.assertEqual(
            harness.mocks["settle_budget_attempt"].call_args.args[1],
            SOURCE_FINGERPRINT,
        )
        self.assertEqual(
            harness.mocks["release_third_asr_claim"].call_args.args[1],
            FINGERPRINT,
        )
        self.assertEqual(
            harness.mocks["ensure_audio_clip"].call_args.kwargs["fingerprint"],
            FINGERPRINT,
        )
        harness.mocks["ensure_fresh_third_asr_budget_identity"].assert_called_once_with(
            EPISODE_KEY, SOURCE_FINGERPRINT, FINGERPRINT
        )

    def test_prepare_session_id_is_threaded_through_to_the_claim_and_finalized_evidence(self):
        # An Assisted per-item caller identifies the prepare-session that
        # triggered this spend; the claim/audit trail records it, but it
        # never influences budget admission/settlement/retry mechanics.
        with _Harness(self) as harness:
            evidence = harness.call(
                transcribe_side_effect=[{"text": "third source", "usage": {"cost": 0.0021}}],
                prepare_session_id="session-abc123",
            )

        self.assertEqual(evidence["prepare_session_id"], "session-abc123")
        self.assertEqual(harness.acquire_prepare_session_ids, ["session-abc123"])
        saved_item = harness.saved_records[0]["human_review"][0]
        self.assertEqual(saved_item["third_asr"]["prepare_session_id"], "session-abc123")

    def test_prepare_session_id_is_never_fabricated_when_not_supplied(self):
        # The existing Detailed Review "Run third ASR" caller does not
        # know about prepare-sessions at all; evidence must not gain a
        # null/placeholder field it never asked for.
        with _Harness(self) as harness:
            evidence = harness.call(
                transcribe_side_effect=[{"text": "third source", "usage": {"cost": 0.0021}}],
            )

        self.assertNotIn("prepare_session_id", evidence)
        self.assertEqual(harness.acquire_prepare_session_ids, [None])

    def test_pre_send_retryable_failure_releases_budget_and_retries_once(self):
        with _Harness(self) as harness:
            evidence = harness.call(
                transcribe_side_effect=[
                    requests.exceptions.ConnectionError("connection refused"),
                    {"text": "recovered", "usage": {"cost": 0.001}},
                ]
            )

        self.assertEqual(evidence["text"], "recovered")
        self.assertEqual(len(harness.reserve_calls), 2)
        self.assertEqual(harness.mocks["release_budget_attempt_pre_send"].call_count, 1)
        self.assertEqual(harness.mocks["mark_budget_attempt_uncertain"].call_count, 0)
        self.assertEqual(harness.mocks["settle_budget_attempt"].call_count, 1)
        self.assertEqual(len(harness.sleep_calls), 1)
        self.assertGreaterEqual(harness.sleep_calls[0], 1.0)
        self.assertLessEqual(harness.sleep_calls[0], 1.5)
        # The claim stays exclusively owned by this call across the retry
        # (refreshed, never released mid-flight): acquire once, refresh once.
        self.assertEqual(harness.acquire_calls, 2)
        harness.mocks["release_third_asr_claim"].assert_called_once()

    def test_post_send_uncertain_failure_keeps_reservation_and_retries_once(self):
        with _Harness(self) as harness:
            evidence = harness.call(
                transcribe_side_effect=[
                    requests.exceptions.ReadTimeout("read timed out"),
                    {"text": "recovered", "usage": {"cost": 0.001}},
                ]
            )

        self.assertEqual(evidence["text"], "recovered")
        self.assertEqual(len(harness.reserve_calls), 2)
        self.assertEqual(harness.mocks["release_budget_attempt_pre_send"].call_count, 0)
        self.assertEqual(harness.mocks["mark_budget_attempt_uncertain"].call_count, 1)

    def test_provider_retryable_5xx_retries_with_fresh_reservation(self):
        with _Harness(self) as harness:
            evidence = harness.call(
                transcribe_side_effect=[
                    _http_error(503),
                    {"text": "recovered", "usage": {"cost": 0.001}},
                ]
            )

        self.assertEqual(evidence["text"], "recovered")
        self.assertEqual(len(harness.reserve_calls), 2)
        self.assertEqual(harness.mocks["mark_budget_attempt_uncertain"].call_count, 1)

    def test_permanent_failure_does_not_retry_and_releases_claim(self):
        error = _http_error(400)
        with _Harness(self) as harness:
            with self.assertRaises(requests.HTTPError):
                harness.call(transcribe_side_effect=[error])

        self.assertEqual(len(harness.reserve_calls), 1)
        self.assertEqual(harness.mocks["mark_budget_attempt_uncertain"].call_count, 1)
        self.assertEqual(harness.mocks["release_budget_attempt_pre_send"].call_count, 0)
        harness.mocks["release_third_asr_claim"].assert_called_once()
        self.assertEqual(harness.sleep_calls, [])
        self.assertEqual(harness.saved_records, [])

    def test_exhausted_retries_raises_original_error_bounded_to_two_attempts(self):
        with _Harness(self) as harness:
            with self.assertRaises(requests.exceptions.ReadTimeout):
                harness.call(
                    transcribe_side_effect=[
                        requests.exceptions.ReadTimeout("first timeout"),
                        requests.exceptions.ReadTimeout("second timeout"),
                    ]
                )

        self.assertEqual(len(harness.reserve_calls), THIRD_ASR_MAX_ATTEMPTS)
        self.assertEqual(harness.mocks["mark_budget_attempt_uncertain"].call_count, 2)
        harness.mocks["release_third_asr_claim"].assert_called_once()
        self.assertEqual(len(harness.sleep_calls), 1)

    def test_pre_send_failure_before_clip_exists_is_classified_pre_send_retryable(self):
        with _Harness(self) as harness, patch(
            "podcast_engine.human_review.ensure_audio_clip",
            side_effect=[
                RuntimeError("FFmpeg review clip extraction failed: boom"),
                (Path("/tmp/nonexistent-task076-clip.wav"), WINDOW),
            ],
        ):
            evidence = harness.call(
                transcribe_side_effect=[{"text": "recovered", "usage": {"cost": 0.001}}]
            )

        self.assertEqual(evidence["text"], "recovered")
        self.assertEqual(harness.mocks["release_budget_attempt_pre_send"].call_count, 1)
        self.assertEqual(len(harness.reserve_calls), 2)

    def test_in_flight_claim_releases_the_pre_send_reservation_and_propagates(self):
        with _Harness(self) as harness:
            with patch(
                "podcast_engine.human_review.acquire_third_asr_claim",
                side_effect=ThirdAsrInFlight(retry_after_seconds=12.0, cache_key=CACHE_KEY),
            ):
                with self.assertRaises(ThirdAsrInFlight):
                    harness.call(transcribe_side_effect=[])

        self.assertEqual(len(harness.reserve_calls), 1)
        harness.mocks["release_budget_attempt_pre_send"].assert_called_once()
        harness.mocks["release_third_asr_claim"].assert_not_called()

    def test_finalize_skips_stale_identity_without_raising(self):
        record = _record()
        with _Harness(self, record=record) as harness:
            # Mutate the record's input_fingerprint in place right before
            # ensure_third_asr's finalize reload sees it, simulating a
            # recompile that changed episode input identity mid-flight.
            def transcribe(_clip):
                record["input_fingerprint"] = "sha256:" + "f" * 64
                return {"text": "now stale", "usage": {"cost": 0.001}}

            evidence = harness.call(transcribe_side_effect=transcribe)

        self.assertEqual(evidence["text"], "now stale")
        self.assertEqual(harness.saved_records, [])

    def test_finalize_retries_on_generation_conflict_bounded_to_three_attempts(self):
        with _Harness(self) as harness:
            save_mock = harness.mocks["save_review_record"]
            save_mock.side_effect = [
                PreconditionFailed("changed"),
                PreconditionFailed("changed"),
                None,
            ]
            evidence = harness.call(
                transcribe_side_effect=[{"text": "third source", "usage": {"cost": 0.001}}]
            )

        self.assertEqual(evidence["text"], "third source")
        self.assertEqual(save_mock.call_count, 3)

    def test_finalize_conflict_exhausted_raises(self):
        with _Harness(self) as harness:
            harness.mocks["save_review_record"].side_effect = PreconditionFailed("changed")
            with self.assertRaises(ThirdAsrClaimConflict):
                harness.call(
                    transcribe_side_effect=[{"text": "third source", "usage": {"cost": 0.001}}]
                )


if __name__ == "__main__":
    unittest.main()


class ClassifyThirdAsrFailureTests(unittest.TestCase):
    """TASK-076 Task 9: the standalone failure-taxonomy classifier."""

    def test_connect_timeout_is_pre_send_retryable(self):
        from podcast_engine.review_audio import (
            THIRD_ASR_FAILURE_PRE_SEND_RETRYABLE,
            classify_third_asr_failure,
        )

        self.assertEqual(
            classify_third_asr_failure(requests.exceptions.ConnectTimeout("boom")),
            THIRD_ASR_FAILURE_PRE_SEND_RETRYABLE,
        )

    def test_connection_error_is_pre_send_retryable(self):
        from podcast_engine.review_audio import (
            THIRD_ASR_FAILURE_PRE_SEND_RETRYABLE,
            classify_third_asr_failure,
        )

        self.assertEqual(
            classify_third_asr_failure(requests.exceptions.ConnectionError("refused")),
            THIRD_ASR_FAILURE_PRE_SEND_RETRYABLE,
        )

    def test_read_timeout_is_post_send_uncertain(self):
        from podcast_engine.review_audio import (
            THIRD_ASR_FAILURE_POST_SEND_UNCERTAIN,
            classify_third_asr_failure,
        )

        self.assertEqual(
            classify_third_asr_failure(requests.exceptions.ReadTimeout("slow")),
            THIRD_ASR_FAILURE_POST_SEND_UNCERTAIN,
        )

    def test_429_is_provider_retryable(self):
        from podcast_engine.review_audio import (
            THIRD_ASR_FAILURE_PROVIDER_RETRYABLE,
            classify_third_asr_failure,
        )

        self.assertEqual(
            classify_third_asr_failure(_http_error(429)),
            THIRD_ASR_FAILURE_PROVIDER_RETRYABLE,
        )

    def test_503_is_provider_retryable(self):
        from podcast_engine.review_audio import (
            THIRD_ASR_FAILURE_PROVIDER_RETRYABLE,
            classify_third_asr_failure,
        )

        self.assertEqual(
            classify_third_asr_failure(_http_error(503)),
            THIRD_ASR_FAILURE_PROVIDER_RETRYABLE,
        )

    def test_400_is_permanent(self):
        from podcast_engine.review_audio import (
            THIRD_ASR_FAILURE_PERMANENT,
            classify_third_asr_failure,
        )

        self.assertEqual(classify_third_asr_failure(_http_error(400)), THIRD_ASR_FAILURE_PERMANENT)

    def test_401_is_permanent(self):
        from podcast_engine.review_audio import (
            THIRD_ASR_FAILURE_PERMANENT,
            classify_third_asr_failure,
        )

        self.assertEqual(classify_third_asr_failure(_http_error(401)), THIRD_ASR_FAILURE_PERMANENT)

    def test_missing_credential_runtime_error_is_permanent(self):
        from podcast_engine.review_audio import (
            THIRD_ASR_FAILURE_PERMANENT,
            classify_third_asr_failure,
        )

        self.assertEqual(
            classify_third_asr_failure(RuntimeError("Missing PODCAST_REVIEW_ASR_API_KEY")),
            THIRD_ASR_FAILURE_PERMANENT,
        )

    def test_malformed_response_value_error_is_post_send_uncertain(self):
        from podcast_engine.review_audio import (
            THIRD_ASR_FAILURE_POST_SEND_UNCERTAIN,
            classify_third_asr_failure,
        )

        self.assertEqual(
            classify_third_asr_failure(ValueError("no transcript text")),
            THIRD_ASR_FAILURE_POST_SEND_UNCERTAIN,
        )

    def test_unexpected_error_defaults_to_post_send_uncertain(self):
        from podcast_engine.review_audio import (
            THIRD_ASR_FAILURE_POST_SEND_UNCERTAIN,
            classify_third_asr_failure,
        )

        self.assertEqual(
            classify_third_asr_failure(KeyError("surprise")),
            THIRD_ASR_FAILURE_POST_SEND_UNCERTAIN,
        )
