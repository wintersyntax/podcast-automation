"""TASK-141: legacy-safe, versioned Human Review queue identity."""

import copy
import hashlib
import json
import unittest
from unittest.mock import Mock, patch

from podcast_engine import compilation, human_review


def card(card_id=210, *, filled_focus=False):
    source = "lat pull down"
    context = "Do a lat pull down now."
    return {
        "id": card_id,
        "reason": "compiler_requires_human_review",
        "kind": "replace",
        "severity": "high",
        "category": "training_term",
        "apple_text": source,
        "whisper_text": "lat pulldown",
        "focus": {
            "scope": "full",
            "apple_text": source if filled_focus else None,
            "whisper_text": "lat pulldown" if filled_focus else None,
        },
        "merge_preview": {
            "base_source": "apple",
            "context_text": context,
            "span_start": context.index(source),
            "span_end": context.index(source) + len(source),
            "span_text": source,
        },
    }


class ReviewQueueIdentityTests(unittest.TestCase):
    def test_legacy_hash_is_byte_identical_and_not_silently_migrated(self):
        source = card()
        payload = [{
            "id": 210, "reason": "compiler_requires_human_review",
            "kind": "replace", "severity": "high",
            "category": "training_term", "apple_text": "lat pull down",
            "whisper_text": "lat pulldown",
            "focus": {"scope": "full", "apple_text": None, "whisper_text": None},
        }]
        canonical = lambda value: json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
        expected_queue = "sha256:" + hashlib.sha256(canonical(payload)).hexdigest()
        expected_generation = "sha256:" + hashlib.sha256(canonical({
            "input_fingerprint": "sha256:episode", "queue_fingerprint": expected_queue,
        })).hexdigest()
        self.assertEqual(human_review.review_queue_fingerprint([source], version=1), expected_queue)
        self.assertEqual(
            human_review.review_generation_fingerprint("sha256:episode", [source], version=1),
            expected_generation,
        )
        self.assertNotEqual(
            human_review.review_generation_fingerprint("sha256:episode", [source]),
            expected_generation,
        )

    def test_v2_ignores_full_focus_only_but_not_real_changes(self):
        old, fixed = card(), card(filled_focus=True)
        self.assertNotEqual(
            human_review.review_queue_fingerprint([old], version=1),
            human_review.review_queue_fingerprint([fixed], version=1),
        )
        self.assertEqual(
            human_review.review_queue_fingerprint([old]),
            human_review.review_queue_fingerprint([fixed]),
        )
        self.assertEqual(
            human_review.review_generation_fingerprint("sha256:input", [old]),
            human_review.review_generation_fingerprint("sha256:input", [fixed]),
        )
        for key in ("apple_text", "whisper_text", "merge_preview"):
            changed = copy.deepcopy(old)
            if key == "merge_preview":
                changed[key]["span_start"] += 1
            else:
                changed[key] += " different"
            self.assertNotEqual(
                human_review.review_queue_fingerprint([old]),
                human_review.review_queue_fingerprint([changed]),
            )
        self.assertEqual([x["id"] for x in [old]], [x["id"] for x in [fixed]])

    def test_omitted_full_focus_is_the_same_v2_card(self):
        missing = card()
        del missing["focus"]
        filled = card(filled_focus=True)
        self.assertEqual(
            human_review.review_queue_fingerprint([missing]),
            human_review.review_queue_fingerprint([filled]),
        )
        self.assertNotEqual(
            human_review.review_queue_fingerprint([missing], version=1),
            human_review.review_queue_fingerprint([filled], version=1),
        )

    def test_partial_scope_keeps_focus_sensitive_identity(self):
        old = card()
        old["focus"] = {
            "scope": "partial", "apple_text": "pull down", "whisper_text": "pulldown",
        }
        changed = copy.deepcopy(old)
        changed["focus"]["apple_text"] = "pull"
        self.assertNotEqual(
            human_review.review_queue_fingerprint([old]),
            human_review.review_queue_fingerprint([changed]),
        )
        self.assertEqual(old["id"], changed["id"])

    def test_legacy_record_normal_decision_preserves_generation_and_other_cards(self):
        existing = [card(210), card(141)]
        old_generation = human_review.review_generation_fingerprint(
            "sha256:original", existing, version=1,
        )
        record = {
            "episode_key": "episode-1", "human_review": copy.deepcopy(existing),
            "human_decisions": [],
            "human_review_queue_fingerprint": human_review.review_queue_fingerprint(
                existing, version=1,
            ),
            "human_review_generation_fingerprint": old_generation,
        }
        with (
            patch("podcast_engine.human_review.load_review_record", return_value=record),
            patch("podcast_engine.human_review.save_review_record") as save,
            patch("podcast_engine.human_review._routing_snapshot", return_value={}),
            patch("podcast_engine.human_review.review_tier_snapshot", return_value={}),
        ):
            decision = human_review.record_human_decision(
                "episode-1", 210, source="apple",
            )
        save.assert_called_once()
        self.assertEqual(decision["id"], 210)
        self.assertEqual([x["id"] for x in record["human_review"]], [141])
        self.assertEqual([x["id"] for x in record["human_decisions"]], [210])
        self.assertEqual(record["human_review_generation_fingerprint"], old_generation)
        self.assertNotIn("human_review_queue_fingerprint_version", record)
        self.assertEqual(
            record["human_review_queue_fingerprint"],
            human_review.review_queue_fingerprint(record["human_review"], version=1),
        )

    def test_new_records_are_explicitly_v2_without_extra_cards(self):
        cards = [card(210), card(141)]
        original = copy.deepcopy(cards)
        provenance = Mock()
        provenance.record.return_value = {"status": "not_required"}
        result = compilation._build_review_record(
            episode_key="episode-1",
            input_fingerprint="sha256:input",
            source_fingerprint="sha256:source",
            resolution={"accepted": [], "review": [], "outcomes": []},
            human_review=cards,
            resolver_batch={"batch": {"diff_items": []}},
            source_paths={
                side: {"text": f"episodes/e/sources/{side}/transcript.txt", "metadata": None}
                for side in ("apple", "whisper")
            },
            preset_provenance=provenance,
        )
        self.assertEqual(result["human_review_queue_fingerprint_version"], 2)
        self.assertEqual(
            result["human_review_queue_fingerprint"],
            human_review.review_queue_fingerprint(cards, version=2),
        )
        self.assertEqual(
            result["human_review_generation_fingerprint"],
            human_review.review_generation_fingerprint("sha256:input", cards, version=2),
        )
        self.assertEqual(result["human_review"], original)
        self.assertEqual(result["human_decisions"], [])

    def test_unknown_versions_fail_closed(self):
        for version in (0, 3, True, "2"):
            with self.subTest(version=version):
                with self.assertRaisesRegex(ValueError, "fingerprint version"):
                    human_review.review_queue_fingerprint([card()], version=version)
                with self.assertRaisesRegex(ValueError, "fingerprint version"):
                    human_review.review_queue_fingerprint_version(
                        {"human_review_queue_fingerprint_version": version},
                    )


if __name__ == "__main__":
    unittest.main()
