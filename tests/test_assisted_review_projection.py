"""TASK-076 Task 10: project routing/assisted analysis onto pending cards.

project_assisted_review_items() is a thin, pure wiring layer over
pending_review_items() (podcast_engine.human_review) and
derive_assisted_state() (compiler.assisted_review, TASK-076 Task 8). The
admission/state derivation logic itself is already covered by
tests/test_assisted_review_admission.py and tests/test_assisted_review_state.py;
this file only exercises the projection wiring: stable ordering, that
every pending card is covered, that non-admitted cards carry routing
provenance only, and that the projection never mutates the source record.
"""

from __future__ import annotations

import unittest

from podcast_engine.human_review import project_assisted_review_items


def _eligible_item(item_id: int = 1, **overrides) -> dict:
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
        "whisper_start_timestamp": 10.0,
        "whisper_end_timestamp": 12.0,
    }
    item.update(overrides)
    return item


def _protected_item(item_id: int = 2, **overrides) -> dict:
    return _eligible_item(
        item_id,
        category="protocol_number",
        risk_reasons=["protocol_number"],
        **overrides,
    )


class ProjectAssistedReviewItemsTests(unittest.TestCase):
    def test_protected_card_projects_routing_only(self):
        record = {"human_review": [_protected_item()]}

        projected = project_assisted_review_items(record)

        self.assertEqual(len(projected), 1)
        assisted = projected[0]["assisted_review"]
        self.assertFalse(assisted["routing"]["eligible"])
        self.assertIn("protected_protocol_number", assisted["routing"]["reason_codes"])
        self.assertNotIn("state", assisted)
        self.assertNotIn("state", assisted["routing"])

    def test_eligible_card_without_evidence_projects_audio_pending(self):
        record = {"human_review": [_eligible_item()]}

        projected = project_assisted_review_items(record)

        assisted = projected[0]["assisted_review"]
        self.assertTrue(assisted["routing"]["eligible"])
        self.assertEqual(assisted["state"], "audio_pending")
        self.assertIn("audio_not_prepared", assisted["reason_codes"])

    def test_eligible_card_with_exact_evidence_projects_machine_supported(self):
        item = _eligible_item(
            apple_text="the quick brown fox",
            whisper_text="the quick brown socks",
            third_asr={"text": "the quick brown fox", "cache_key": "sha256:" + "a" * 64},
        )
        record = {"human_review": [item]}

        projected = project_assisted_review_items(record)

        assisted = projected[0]["assisted_review"]
        self.assertEqual(assisted["state"], "machine_supported_apple")
        self.assertEqual(assisted["recommendation"], "apple")
        # Third-ASR evidence (including any cache_key/window/model metadata
        # already persisted on the card) is carried through for display.
        self.assertEqual(assisted["third_asr"]["text"], "the quick brown fox")

    def test_projection_preserves_stable_id_order_regardless_of_input_order(self):
        record = {
            "human_review": [
                _eligible_item(item_id=5),
                _eligible_item(item_id=1),
                _protected_item(item_id=3),
            ]
        }

        projected = project_assisted_review_items(record)

        self.assertEqual([item["id"] for item in projected], [1, 3, 5])

    def test_covers_every_pending_card_none_dropped(self):
        record = {
            "human_review": [
                _eligible_item(item_id=1),
                _protected_item(item_id=2),
                _eligible_item(item_id=3, kind="unknown_kind_value"),
            ]
        }

        projected = project_assisted_review_items(record)

        self.assertEqual({item["id"] for item in projected}, {1, 2, 3})
        # An unknown/malformed kind still gets a well-formed routing-only
        # projection rather than being silently dropped.
        third = next(item for item in projected if item["id"] == 3)
        self.assertFalse(third["assisted_review"]["routing"]["eligible"])
        self.assertIn("unknown_kind", third["assisted_review"]["routing"]["reason_codes"])

    def test_projection_does_not_mutate_the_source_record(self):
        item = _eligible_item()
        record = {"human_review": [item]}

        project_assisted_review_items(record)

        self.assertNotIn("assisted_review", item)
        self.assertNotIn("assisted_review", record["human_review"][0])

    def test_empty_pending_queue_projects_nothing(self):
        self.assertEqual(project_assisted_review_items({"human_review": []}), [])


if __name__ == "__main__":
    unittest.main()
