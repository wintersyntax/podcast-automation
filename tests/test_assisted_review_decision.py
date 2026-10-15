"""TASK-076 Task 11: atomic Assisted human decisions and audit.

record_assisted_human_decision_batch() is a separate atomic decision path
from the strict low-risk record_human_decision_batch (tests/
test_human_review_batch_decision.py): it accepts any admitted
(assisted_routing-eligible) card regardless of machine-supported state,
never requires the human's choice to agree with the derived machine
recommendation, and persists a bounded decision_support audit bundle
alongside the ordinary routing_provenance snapshot every decision channel
now carries. See docs/superpowers/specs/2026-09-16-human-review-evidence-
assisted-adjudication-design.md, "Atomic assisted persistence" and
"Decision and routing audit".
"""

from __future__ import annotations

import copy
import hashlib
import unittest
from unittest.mock import Mock, patch

from google.api_core.exceptions import PreconditionFailed

from compiler.review_policy import ASSISTED_REVIEW_POLICY_VERSION
from podcast_engine import human_review


FIXED_TIME = "2026-09-22T09:00:00+00:00"


def _assisted_item(item_id, apple_text, whisper_text, *, audio=None, **overrides):
    item = {
        "id": item_id,
        "kind": "wording_difference",
        "category": "other",
        "apple_text": apple_text,
        "whisper_text": whisper_text,
        "source_only": False,
        "risk_reasons": [],
        "domain_terms": [],
        "citation_signal": False,
        "preservation_class": "not_source_only",
        "merge_action": "review_kept_primary",
        "anomaly": None,
        "custom_edit": None,
        "third_asr": (
            None
            if audio is None
            else {
                "text": audio,
                "cache_key": f"sha256:cache-{item_id}",
                "model": "test-third-asr-model",
                "window": {"start": 0.0, "end": 5.0},
            }
        ),
        "representation_modified": False,
        "generation_stale": False,
        "whisper_start_timestamp": 10.0,
        "whisper_end_timestamp": 12.0,
    }
    item.update(overrides)
    return item


def _record(items):
    cards = list(items)
    input_fingerprint = "sha256:input"
    return {
        "episode_key": "episode-1",
        "input_fingerprint": input_fingerprint,
        "human_decisions": [],
        "human_review": cards,
        "human_review_queue_fingerprint": human_review.review_queue_fingerprint(cards),
        "human_review_generation_fingerprint": human_review.review_generation_fingerprint(
            input_fingerprint, cards
        ),
    }


# Known-calibrated machine-supported fixtures (short/exclusive-exact rule),
# reused from tests/test_assisted_review_state.py's own coverage.
def _machine_supported_apple_item(item_id=10):
    return _assisted_item(item_id, "back", "home", audio="well back then")


def _machine_supported_whisper_item(item_id=11):
    return _assisted_item(item_id, "go back", "go home", audio="well go home then")


def _unresolved_pending_item(item_id=12):
    # Eligible, but no Third-ASR evidence yet -> assisted state "audio_pending".
    return _assisted_item(item_id, "we will proceed", "we will continue")


def _noncandidate_item(item_id=13, **overrides):
    return _assisted_item(
        item_id,
        "the protocol used",
        "the protocol utilized",
        category="protocol_number",
        risk_reasons=["protocol_number"],
        **overrides,
    )


class AssistedReviewDecisionTests(unittest.TestCase):
    def test_mixed_apple_whisper_success_equivalence(self):
        apple_item = _machine_supported_apple_item(10)
        whisper_item = _machine_supported_whisper_item(11)
        record = _record([apple_item, whisper_item])
        generation = record["human_review_generation_fingerprint"]

        saved = []

        def save(_episode_key, value, *, if_generation_match=None):
            saved.append((if_generation_match, copy.deepcopy(value)))
            return value

        with (
            patch(
                "podcast_engine.human_review.load_review_record_with_generation",
                return_value=(copy.deepcopy(record), 41),
            ),
            patch("podcast_engine.human_review.save_review_record", side_effect=save),
            patch("podcast_engine.human_review.now_iso", return_value=FIXED_TIME),
        ):
            batch_result = human_review.record_assisted_human_decision_batch(
                "episode-1",
                [{"id": 10, "source": "apple"}, {"id": 11, "source": "whisper"}],
                expected_generation_fingerprint=generation,
                assisted_policy_version=ASSISTED_REVIEW_POLICY_VERSION,
            )

        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0][0], 41)
        self.assertEqual(batch_result["human_review"], [])
        self.assertEqual(
            [decision["id"] for decision in batch_result["human_decisions"]],
            [10, 11],
        )

        # Equivalence: deciding item 10 alone (fresh single-item record)
        # produces the exact same decision as within the mixed batch.
        solo_record = _record([apple_item])
        with (
            patch(
                "podcast_engine.human_review.load_review_record_with_generation",
                return_value=(copy.deepcopy(solo_record), 99),
            ),
            patch("podcast_engine.human_review.save_review_record", side_effect=save),
            patch("podcast_engine.human_review.now_iso", return_value=FIXED_TIME),
        ):
            solo_result = human_review.record_assisted_human_decision_batch(
                "episode-1",
                [{"id": 10, "source": "apple"}],
                expected_generation_fingerprint=solo_record["human_review_generation_fingerprint"],
                assisted_policy_version=ASSISTED_REVIEW_POLICY_VERSION,
            )

        batch_decision_10 = next(
            decision for decision in batch_result["human_decisions"] if decision["id"] == 10
        )
        solo_decision_10 = solo_result["human_decisions"][0]
        self.assertEqual(batch_decision_10, solo_decision_10)
        self.assertEqual(batch_decision_10["chosen_source"], "apple")
        self.assertEqual(batch_decision_10["chosen_text"], "back")
        self.assertEqual(batch_decision_10["decision_support"]["relation"], "confirmed_machine")
        self.assertEqual(batch_decision_10["decision_support"]["machine_state"], "machine_supported_apple")

        decision_11 = next(
            decision for decision in batch_result["human_decisions"] if decision["id"] == 11
        )
        self.assertEqual(decision_11["chosen_source"], "whisper")
        self.assertEqual(decision_11["decision_support"]["relation"], "confirmed_machine")
        self.assertEqual(decision_11["decision_support"]["machine_state"], "machine_supported_whisper")

    def test_prepared_per_card_choice_matches_assisted_batch_support(self):
        item = _machine_supported_apple_item(70)
        record = _record([item])
        generation = record["human_review_generation_fingerprint"]
        with (
            patch("podcast_engine.human_review.load_review_record", return_value=copy.deepcopy(record)),
            patch("podcast_engine.human_review.save_review_record") as save,
            patch("podcast_engine.human_review.now_iso", return_value=FIXED_TIME),
        ):
            manual = human_review.record_human_decision("episode-1", 70, source="whisper")
        save.assert_called_once()
        with (
            patch("podcast_engine.human_review.load_review_record_with_generation", return_value=(copy.deepcopy(record), 9)),
            patch("podcast_engine.human_review.save_review_record", side_effect=lambda key, value, **kwargs: value),
            patch("podcast_engine.human_review.now_iso", return_value=FIXED_TIME),
        ):
            batch = human_review.record_assisted_human_decision_batch(
                "episode-1", [{"id": 70, "source": "whisper"}],
                expected_generation_fingerprint=generation,
                assisted_policy_version=ASSISTED_REVIEW_POLICY_VERSION,
            )["human_decisions"][0]
        self.assertIn('decision_support', manual)
        self.assertEqual(manual['decision_support'], batch['decision_support'])
        self.assertEqual(manual['decision_support']['relation'], 'human_override')
        self.assertEqual(manual['chosen_source'], 'whisper')

    def test_human_override_and_manual_unresolved_audit(self):
        override_item = _machine_supported_apple_item(20)  # machine recommends apple
        unresolved_item = _unresolved_pending_item(21)  # no machine recommendation yet
        record = _record([override_item, unresolved_item])
        generation = record["human_review_generation_fingerprint"]

        with (
            patch(
                "podcast_engine.human_review.load_review_record_with_generation",
                return_value=(copy.deepcopy(record), 51),
            ),
            patch("podcast_engine.human_review.save_review_record", side_effect=lambda *a, **k: a[1]),
            patch("podcast_engine.human_review.now_iso", return_value=FIXED_TIME),
        ):
            result = human_review.record_assisted_human_decision_batch(
                "episode-1",
                [{"id": 20, "source": "whisper"}, {"id": 21, "source": "apple"}],
                expected_generation_fingerprint=generation,
                assisted_policy_version=ASSISTED_REVIEW_POLICY_VERSION,
            )

        override_decision = next(d for d in result["human_decisions"] if d["id"] == 20)
        self.assertEqual(override_decision["chosen_source"], "whisper")
        self.assertEqual(override_decision["decision_support"]["relation"], "human_override")
        self.assertEqual(override_decision["decision_support"]["machine_state"], "machine_supported_apple")
        self.assertEqual(override_decision["decision_support"]["recommended_source"], "apple")

        manual_decision = next(d for d in result["human_decisions"] if d["id"] == 21)
        self.assertEqual(manual_decision["chosen_source"], "apple")
        self.assertEqual(manual_decision["decision_support"]["relation"], "manual_from_unresolved")
        self.assertEqual(manual_decision["decision_support"]["machine_state"], "audio_pending")
        self.assertIsNone(manual_decision["decision_support"]["recommended_source"])

    def test_defer_semantics_omitted_rows_remain_pending(self):
        decided = _machine_supported_apple_item(30)
        deferred_a = _unresolved_pending_item(31)
        deferred_b = _machine_supported_whisper_item(32)
        record = _record([decided, deferred_a, deferred_b])
        generation = record["human_review_generation_fingerprint"]

        with (
            patch(
                "podcast_engine.human_review.load_review_record_with_generation",
                return_value=(copy.deepcopy(record), 61),
            ),
            patch("podcast_engine.human_review.save_review_record", side_effect=lambda *a, **k: a[1]),
            patch("podcast_engine.human_review.now_iso", return_value=FIXED_TIME),
        ):
            result = human_review.record_assisted_human_decision_batch(
                "episode-1",
                [{"id": 30, "source": "apple"}],
                expected_generation_fingerprint=generation,
                assisted_policy_version=ASSISTED_REVIEW_POLICY_VERSION,
            )

        self.assertEqual([d["id"] for d in result["human_decisions"]], [30])
        self.assertEqual(
            sorted(item["id"] for item in result["human_review"]),
            [31, 32],
        )

    def test_decision_support_is_bounded_and_never_stores_raw_provider_response(self):
        item = _assisted_item(
            40,
            "we traveled back",
            "we traveled home",
            audio="well we traveled back anyway",
        )
        # Simulate a realistic persisted evidence object carrying provider
        # billing/usage fields that must never leak into decision_support.
        item["third_asr"]["usage"] = {"seconds": 4.2}
        item["third_asr"]["cost"] = 0.0031
        item["third_asr"]["audio_format"] = "wav"
        item["third_asr"]["generated_at"] = "2026-09-22T08:00:00+00:00"
        item["triage"] = {
            "status": "advisory",
            "recommendation": "recommend_apple",
            "confidence": "high",
            "source": "apple",
            "text": "we traveled back",
        }
        item["suggestion"] = {
            "source": "apple",
            "reason": "compiler_default",
            "automatic_resolution": False,
        }
        record = _record([item])
        generation = record["human_review_generation_fingerprint"]

        with (
            patch(
                "podcast_engine.human_review.load_review_record_with_generation",
                return_value=(copy.deepcopy(record), 71),
            ),
            patch("podcast_engine.human_review.save_review_record", side_effect=lambda *a, **k: a[1]),
            patch("podcast_engine.human_review.now_iso", return_value=FIXED_TIME),
        ):
            result = human_review.record_assisted_human_decision_batch(
                "episode-1",
                [{"id": 40, "source": "apple"}],
                expected_generation_fingerprint=generation,
                assisted_policy_version=ASSISTED_REVIEW_POLICY_VERSION,
            )

        support = result["human_decisions"][0]["decision_support"]

        third_asr_support = support["third_asr"]
        self.assertEqual(
            set(third_asr_support),
            {"cache_key", "model", "window", "text_hash"},
        )
        self.assertEqual(
            third_asr_support["text_hash"],
            hashlib.sha256("well we traveled back anyway".encode("utf-8")).hexdigest(),
        )
        self.assertNotIn("text", third_asr_support)
        self.assertNotIn("usage", third_asr_support)
        self.assertNotIn("cost", third_asr_support)

        self.assertEqual(
            support["triage"],
            {"status": "advisory", "recommendation": "recommend_apple", "confidence": "high"},
        )
        self.assertNotIn("source", support["triage"])
        self.assertNotIn("text", support["triage"])

        self.assertEqual(support["compiler_suggestion_source"], "apple")
        self.assertIn("match_evidence", support)
        self.assertIn("margin", support)
        self.assertEqual(support["routing"], {"policy_version": ASSISTED_REVIEW_POLICY_VERSION, "reason_codes": []})

    def test_atomic_rejection_for_noncandidate_protected_stale_wrong_policy_tampered_duplicate_already_decided(self):
        base_item = _machine_supported_apple_item(50)
        protected_item = _noncandidate_item(51)
        stale_flag_item = _assisted_item(52, "we will go", "we will proceed", generation_stale=True)
        record = _record([base_item, protected_item, stale_flag_item])
        generation = record["human_review_generation_fingerprint"]

        cases = {
            "noncandidate": (
                [{"id": 51, "source": "apple"}],
                generation,
                ASSISTED_REVIEW_POLICY_VERSION,
                "candidate",
            ),
            "protected_reason_code": (
                [{"id": 51, "source": "whisper"}],
                generation,
                ASSISTED_REVIEW_POLICY_VERSION,
                "candidate",
            ),
            "stale_item_generation_flag": (
                [{"id": 52, "source": "apple"}],
                generation,
                ASSISTED_REVIEW_POLICY_VERSION,
                "candidate",
            ),
            "stale_record_generation": (
                [{"id": 50, "source": "apple"}],
                "sha256:stale-does-not-match",
                ASSISTED_REVIEW_POLICY_VERSION,
                "generation",
            ),
            "wrong_policy_version": (
                [{"id": 50, "source": "apple"}],
                generation,
                "human-review-assisted-v0-stale",
                "policy",
            ),
            "tampered_extra_field": (
                [{"id": 50, "source": "apple", "note": "forged"}],
                generation,
                ASSISTED_REVIEW_POLICY_VERSION,
                "id and source",
            ),
            "duplicate_id": (
                [{"id": 50, "source": "apple"}, {"id": 50, "source": "whisper"}],
                generation,
                ASSISTED_REVIEW_POLICY_VERSION,
                "duplicate",
            ),
            "already_decided": (
                [{"id": 999, "source": "apple"}],
                generation,
                ASSISTED_REVIEW_POLICY_VERSION,
                "not pending",
            ),
        }

        for label, (decisions, gen, policy_version, message_fragment) in cases.items():
            with self.subTest(label=label):
                save = Mock()
                with (
                    patch(
                        "podcast_engine.human_review.load_review_record_with_generation",
                        return_value=(copy.deepcopy(record), 81),
                    ),
                    patch("podcast_engine.human_review.save_review_record", save),
                ):
                    with self.assertRaisesRegex(ValueError, message_fragment):
                        human_review.record_assisted_human_decision_batch(
                            "episode-1",
                            decisions,
                            expected_generation_fingerprint=gen,
                            assisted_policy_version=policy_version,
                        )
                save.assert_not_called()

        # No case above ever produced a write; the original record is untouched.
        self.assertEqual(record["human_decisions"], [])
        self.assertEqual(
            sorted(item["id"] for item in record["human_review"]),
            [50, 51, 52],
        )

    def test_precondition_race_reloads_fresh_record_and_never_duplicates(self):
        item = _machine_supported_apple_item(60)
        record = _record([item])
        generation = record["human_review_generation_fingerprint"]

        load = Mock(
            side_effect=[
                (copy.deepcopy(record), 91),
                (copy.deepcopy(record), 92),
            ]
        )
        saved = []

        def save(_episode_key, value, *, if_generation_match=None):
            saved.append((if_generation_match, copy.deepcopy(value)))
            if len(saved) == 1:
                raise PreconditionFailed("concurrent review write")
            return value

        with (
            patch("podcast_engine.human_review.load_review_record_with_generation", load),
            patch("podcast_engine.human_review.save_review_record", side_effect=save),
            patch("podcast_engine.human_review.now_iso", return_value=FIXED_TIME),
        ):
            result = human_review.record_assisted_human_decision_batch(
                "episode-1",
                [{"id": 60, "source": "apple"}],
                expected_generation_fingerprint=generation,
                assisted_policy_version=ASSISTED_REVIEW_POLICY_VERSION,
            )

        self.assertEqual(load.call_count, 2)
        self.assertEqual([entry[0] for entry in saved], [91, 92])
        self.assertEqual([d["id"] for d in result["human_decisions"]], [60])

    def test_per_card_third_support_and_unprepared_or_excluded_guards(self):
        prepared = _machine_supported_apple_item(71)
        # TASK-123: "Use Third" needs context anchors so that only the clip
        # words covering this card are inserted.
        prepared['apple_context'] = 'so it was well back then we trained'
        prepared['whisper_context'] = 'so it was well home then we trained'
        prepared['third_asr']['text'] = 'so it was well back then we trained hard'
        excluded = _noncandidate_item(72, third_asr=copy.deepcopy(prepared['third_asr']))
        pending = _assisted_item(73, 'back', 'home', third_asr={'status': 'pending'})
        unprepared = _unresolved_pending_item(74)
        results = []
        for item, source in ((prepared, 'third'), (excluded, 'apple'), (pending, 'apple'), (unprepared, 'apple')):
            record = _record([item])
            with (
                patch('podcast_engine.human_review.load_review_record', return_value=record),
                patch('podcast_engine.human_review.save_review_record', side_effect=lambda key, value: value),
            ):
                results.append(human_review.record_human_decision('episode-1', item['id'], source=source))
        manual = results[0]
        self.assertIn('decision_support', manual)
        support = manual['decision_support']
        self.assertEqual(support['relation'], 'human_override')
        self.assertEqual(support['recommended_source'], 'apple')
        self.assertEqual(support['third_asr']['text_hash'], hashlib.sha256(prepared['third_asr']['text'].encode('utf-8')).hexdigest())
        self.assertNotIn('text', support['third_asr'])
        for decision in results[1:]:
            self.assertNotIn('decision_support', decision)


if __name__ == "__main__":
    unittest.main()
