"""TASK-160: record past episodes' terminology candidates at most once."""

from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import unittest

from google.api_core.exceptions import PreconditionFailed

from podcast_engine.terminology_candidates import (
    MAX_EPISODE_KEYS,
    CandidateObservation,
    CandidateRegistryStore,
    candidate_id_for,
    initial_candidate_registry,
    merge_backfill_observation,
    merge_candidate_observation,
)

ROOT = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location("terminology_candidates_cli", ROOT / "scripts" / "terminology_candidates.py")
cli = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cli)

PATH = "knowledge/terminology/candidates-v1.json"


class _Blob:
    def __init__(self, bucket, name):
        self.bucket, self.name = bucket, name
        self.generation = None

    def exists(self):
        return self.name in self.bucket.objects

    def reload(self):
        self.generation = self.bucket.objects[self.name][1]

    def download_as_text(self, *, encoding, if_generation_match):
        content, generation = self.bucket.objects[self.name]
        if generation != if_generation_match:
            raise PreconditionFailed("moved")
        return content

    def upload_from_string(self, content, *, content_type, if_generation_match):
        current = self.bucket.objects.get(self.name, (None, 0))[1]
        if current != if_generation_match:
            raise PreconditionFailed("moved")
        self.bucket.writes += 1
        self.bucket.objects[self.name] = (content, current + 1)


class _Bucket:
    def __init__(self):
        self.objects = {}
        self.writes = 0

    def blob(self, name):
        return _Blob(self, name)


def _observation(text="lengthened partials", episode="a" * 24, provenance="corpus-confirmed"):
    return CandidateObservation(
        text=text,
        suggested_category="training_term",
        provenance=provenance,
        episode_key=episode,
        observed_form=text,
        evidence={"difference_id": 4},
    )


def _report(*texts):
    return {
        "episode_local_memory": {
            "mode": "shadow",
            "evidence_contract": "aligned_equal_same_episode_only",
            "signals": [
                {
                    "difference_id": index,
                    "category": "training_term",
                    "signal": "one_side_confirmed",
                    "forms": [{"source": "whisper", "text": text, "episode_confirmed": True}],
                }
                for index, text in enumerate(texts)
            ],
        }
    }


class MergeBackfillObservationTests(unittest.TestCase):
    def test_one_observation_per_candidate_and_episode(self):
        registry = initial_candidate_registry(updated_at="2026-10-30T00:00:00Z")
        self.assertEqual(merge_backfill_observation(registry, _observation())["status"], "recorded")
        self.assertEqual(merge_backfill_observation(registry, _observation())["status"], "already_recorded")
        self.assertEqual(
            merge_backfill_observation(registry, _observation(provenance="human-reviewed"))["status"],
            "already_recorded",
        )
        self.assertEqual(merge_backfill_observation(registry, _observation(episode="b" * 24))["status"], "recorded")
        entry = registry["candidates"][candidate_id_for("lengthened partials")]
        self.assertEqual(entry["observation_count"], 2)
        self.assertEqual(entry["episode_keys"], ["a" * 24, "b" * 24])

    def test_episode_recorded_by_a_natural_compile_is_not_counted_again(self):
        registry = initial_candidate_registry(updated_at="2026-10-30T00:00:00Z")
        merge_candidate_observation(registry, _observation())
        self.assertEqual(merge_backfill_observation(registry, _observation())["status"], "already_recorded")
        self.assertEqual(registry["candidates"][candidate_id_for("lengthened partials")]["observation_count"], 1)

    def test_full_episode_list_is_skipped_rather_than_double_counted(self):
        registry = initial_candidate_registry(updated_at="2026-10-30T00:00:00Z")
        for index in range(MAX_EPISODE_KEYS):
            merge_candidate_observation(registry, _observation(episode=f"{index:024x}"))
        before = deepcopy(registry)
        self.assertEqual(merge_backfill_observation(registry, _observation(episode="f" * 24))["status"],
                         "episode_list_full")
        self.assertEqual(registry, before)

    def test_episode_key_is_required(self):
        registry = initial_candidate_registry(updated_at="2026-10-30T00:00:00Z")
        with self.assertRaises(ValueError):
            merge_backfill_observation(registry, _observation(episode=None))


class BackfillCommandTests(unittest.TestCase):
    def setUp(self):
        self.bucket = _Bucket()
        self.store = CandidateRegistryStore(bucket=self.bucket, path=PATH)
        self.reports = {"a" * 24: _report("lengthened partials", "myofibrillar hypertrophy"), "b" * 24: _report("lengthened partials")}
        self.reviews = {
            "a" * 24: {"human_decisions": [{
                "reviewed_by": "human",
                "chosen_text": "mechanistic target of rapamycin",
                "review_item": {"id": 11, "category": "scientific_medical_term"},
            }]},
        }

    def _run(self, keys, *, execute):
        return cli.run_backfill(
            self.store, keys, execute=execute,
            read_report=self.reports.get, read_review=self.reviews.get,
        )

    def test_dry_run_writes_nothing_and_predicts_the_execute_counts(self):
        keys = ["a" * 24, "b" * 24, "c" * 24]
        dry = self._run(keys, execute=False)
        self.assertEqual(self.bucket.writes, 0)
        self.assertEqual(self.bucket.objects, {})
        executed = self._run(keys, execute=True)
        self.assertEqual(dry["totals"], executed["totals"])
        self.assertEqual(executed["totals"]["recorded"], 4)
        self.assertEqual(executed["episodes"][2], {"episode_key": "c" * 24, "status": "no_compiler_report"})

    def test_repeated_execute_is_a_no_op(self):
        keys = ["a" * 24, "b" * 24]
        self._run(keys, execute=True)
        writes = self.bucket.writes
        snapshot = self.bucket.objects[PATH]
        again = self._run(keys, execute=True)
        self.assertEqual(self.bucket.writes, writes)
        self.assertEqual(self.bucket.objects[PATH], snapshot)
        self.assertEqual(again["totals"]["recorded"], 0)
        self.assertEqual(again["totals"]["already_recorded"], 4)
        registry = json.loads(snapshot[0])
        entry = registry["candidates"][candidate_id_for("lengthened partials")]
        self.assertEqual(entry["observation_count"], 2)


if __name__ == "__main__":
    unittest.main()
