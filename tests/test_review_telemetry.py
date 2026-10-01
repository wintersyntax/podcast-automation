import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "review_telemetry.py"
SPEC = importlib.util.spec_from_file_location("review_telemetry", SCRIPT_PATH)
review_telemetry = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(review_telemetry)


class _Blob:
    def __init__(self, bucket, name):
        self.bucket = bucket
        self.name = name

    def exists(self):
        self.bucket.calls.append(("exists", self.name))
        return self.name in self.bucket.objects

    def download_as_bytes(self):
        self.bucket.calls.append(("download_as_bytes", self.name))
        return self.bucket.objects[self.name].encode("utf-8")

    def __getattr__(self, name):
        if name.startswith(("upload", "delete", "rewrite", "patch", "compose")):
            raise AssertionError(f"read-only telemetry attempted GCS mutation: {name}")
        raise AttributeError(name)


class _Bucket:
    def __init__(self, objects):
        self.objects = dict(objects)
        self.calls = []

    def blob(self, name):
        self.calls.append(("blob", name))
        return _Blob(self, name)


def _decision(source, *, kind="replace", category="wording", severity="high"):
    return {
        "id": 1,
        "chosen_source": source,
        "chosen_text": "sensitive raw text is intentionally not reported",
        "scope": "full",
        "reviewed_by": "human",
        "reviewed_at": "2026-08-29T10:00:00+00:00",
        "note": "private note",
        "review_item": {"id": 1, "kind": kind, "category": category, "severity": severity, "apple_text": "Apple evidence", "whisper_text": "Whisper evidence"},
    }


def _report(*, modern=True):
    value = {
        "differences": [
            {"severity": "high", "kind": "replace", "resolver_category": "wording", "preservation_class": "preserved"},
            {"severity": "low", "kind": "insert", "resolver_category": "other", "preservation_class": "discarded_confirmed_filler"},
        ],
        "review_required": 1,
    }
    if modern:
        value["source_only_summary"] = {"apple_only": 1, "whisper_only": 1, "preserved": 1}
    return value


class ReviewTelemetryTests(unittest.TestCase):
    def setUp(self):
        self.episodes = [
            {"episode_key": "ep-v1", "podcast": "Legacy Podcast", "title": "Legacy episode"},
            {"episode_key": "ep-v2", "podcast": "Modern Podcast", "title": "Modern episode"},
        ]
        self.objects = {
            "episodes.json": json.dumps(self.episodes),
            "episodes/ep-v1/review/resolver.json": json.dumps({
                "schema_version": 1, "policy_version": "transcript-reviewer-v1", "reviewed_at": "2026-08-20T00:00:00+00:00",
                "resolver_item_ids": [1, 2], "accepted": [{"id": 1}], "resolver_review": [{"id": 2}],
                "human_decisions": [_decision("apple"), _decision("third", kind="insert", category="other", severity="low")],
            }),
            "episodes/ep-v1/compiled/report.json": json.dumps(_report(modern=False)),
            "episodes/ep-v2/review/resolver.json": json.dumps({
                "schema_version": 2, "policy_version": "resolver-policy-v3", "reviewed_at": "2026-08-29T00:00:00+00:00",
                "resolver_item_ids": [3, 4, 5], "accepted": [{"id": 3}], "resolver_review": [{"id": 4}],
                "resolver_bypassed": [{"id": 6}], "resolver_deferred": [{"id": 7}],
                "resolver_outcomes": [
                    {"id": 3, "status": "accepted"}, {"id": 4, "status": "abstained"}, {"id": 5, "status": "rejected_by_python"},
                ],
                "human_decisions": [_decision("whisper", category="number"), _decision("custom", kind="delete", severity="medium")],
            }),
            "episodes/ep-v2/compiled/report.json": json.dumps(_report(modern=True)),
        }

    def test_aggregates_v1_and_v2_without_claiming_legacy_lifecycle_metrics(self):
        bucket = _Bucket(self.objects)
        report = review_telemetry.build_telemetry(bucket, bucket_name="test-bucket")

        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["human_decisions"]["reviewed_episodes"], 2)
        self.assertEqual(report["human_decisions"]["total"], 4)
        for source in ("apple", "whisper", "third", "custom"):
            self.assertEqual(report["human_decisions"]["source_choice"][source]["count"], 1)
        self.assertEqual(report["human_decisions"]["source_choice"]["apple"]["percentage"], 25.0)
        self.assertEqual(report["resolver"]["metrics"]["structurally_bypassed"]["availability"], "mixed")
        self.assertEqual(report["resolver"]["metrics"]["structurally_bypassed"]["available_episodes"], 1)
        self.assertEqual(report["resolver"]["metrics"]["structurally_bypassed"]["count"], 1)
        self.assertEqual(report["resolver"]["metrics"]["resolver_outcomes"]["counts"], {"abstained": 1, "accepted": 1, "rejected_by_python": 1})
        self.assertEqual(report["compiler"]["total_differences"], 4)
        self.assertEqual(report["compiler"]["source_only_summary"]["availability"], "mixed")
        self.assertEqual(report["episodes"][0]["resolver"]["metrics"]["deferred"]["availability"], "unavailable")
        self.assertNotIn("chosen_text", json.dumps(report))
        self.assertNotIn("Apple evidence", json.dumps(report))
        self.assertTrue(all(call[0] in {"blob", "exists", "download_as_bytes"} for call in bucket.calls))

    def test_output_is_deterministic_and_markdown_warns_for_small_sample(self):
        first = review_telemetry.build_telemetry(_Bucket(self.objects))
        second = review_telemetry.build_telemetry(_Bucket(self.objects))
        self.assertEqual(first, second)
        markdown = review_telemetry.render_markdown(first)
        self.assertIn("Telemetry is descriptive evidence, not a compiler source prior.", markdown)
        self.assertIn("Small-sample warning", markdown)
        with tempfile.TemporaryDirectory() as directory:
            json_path, markdown_path = review_telemetry.write_report(first, directory)
            self.assertEqual(json.loads(json_path.read_text(encoding="utf-8")), first)
            self.assertEqual(markdown_path.read_text(encoding="utf-8"), markdown)

    def test_missing_optional_fields_use_unknown_and_single_episode_filter(self):
        objects = {
            "episodes.json": json.dumps([{"episode_key": "only", "podcast": None, "title": None}]),
            "episodes/only/review/resolver.json": json.dumps({"schema_version": 1, "human_decisions": [{"chosen_source": "unexpected", "review_item": {}}]}),
            "episodes/only/compiled/report.json": json.dumps({"differences": [{}]}),
        }
        report = review_telemetry.build_telemetry(_Bucket(objects), episode_key="only")
        self.assertEqual(report["episodes"][0]["podcast"], "<unknown>")
        self.assertEqual(report["human_decisions"]["source_choice"]["<unknown>"]["count"], 1)
        self.assertEqual(report["human_decisions"]["by_kind"]["<unknown>"]["<unknown>"]["count"], 1)
        self.assertEqual(report["compiler"]["preservation_class"], {"<unknown>": 1})


def _eligible_item(item_id, **overrides):
    item = {
        "id": item_id,
        "kind": "wording_difference",
        "category": "other",
        "apple_text": "the quick brown fox",
        "whisper_text": "the quick brown socks",
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


def _protected_item(item_id, **overrides):
    return _eligible_item(item_id, category="protocol_number", risk_reasons=["protocol_number"], **overrides)


class AssistedReviewTelemetryTests(unittest.TestCase):
    """TASK-076 Task 10, part 3: assisted-review and budget/session telemetry."""

    def _build(self, pending_items, *, ledger=None):
        objects = {
            "episodes.json": json.dumps([{"episode_key": "ep-1", "podcast": "Pod", "title": "Ep"}]),
            "episodes/ep-1/review/resolver.json": json.dumps({
                "schema_version": 2,
                "input_fingerprint": "sha256:" + "a" * 64,
                "source_fingerprint": "sha256:" + "a" * 64,
                "human_decisions": [],
                "human_review": pending_items,
            }),
        }
        if ledger is not None:
            # Keyed by source_fingerprint, matching the real canonical path
            # (podcast_engine/ai_budget.py::budget_ledger_path) -- this helper
            # aliases input_fingerprint/source_fingerprint to the same fake
            # value for convenience; test_ledger_lookup_uses_source_fingerprint_
            # not_input_fingerprint below covers the case where they differ.
            objects["episodes/ep-1/ai/budgets/" + "a" * 64 + ".json"] = json.dumps(ledger)
        bucket = _Bucket(objects)
        report = review_telemetry.build_telemetry(bucket, bucket_name="test-bucket")
        return report, bucket

    def test_assisted_review_counts_and_distributions(self):
        pending_items = [
            _protected_item(1),
            _eligible_item(2),
            _eligible_item(3, third_asr={"text": "the quick brown fox"}),
        ]
        report, bucket = self._build(pending_items)

        episode = report["episodes"][0]["assisted_review"]
        self.assertEqual(episode["candidate"], 2)
        self.assertEqual(episode["prepared"], 1)
        self.assertEqual(episode["audio_pending"], 1)
        self.assertEqual(episode["machine_supported"], 1)
        self.assertEqual(episode["conflict"], 0)
        self.assertEqual(episode["unresolved"], 0)
        self.assertEqual(episode["malformed"], 0)
        self.assertEqual(episode["state_distribution"]["audio_pending"], 1)
        self.assertEqual(episode["state_distribution"]["machine_supported_apple"], 1)
        self.assertIn("protected_protocol_number", episode["routing_reason_distribution"])

        aggregate = report["assisted_review"]
        self.assertEqual(aggregate["candidate"], 2)
        self.assertEqual(aggregate["machine_supported"], 1)
        self.assertTrue(all(call[0] in {"blob", "exists", "download_as_bytes"} for call in bucket.calls))

    def test_non_dict_pending_item_counts_as_malformed(self):
        report, _ = self._build(["not-a-dict"])
        episode = report["episodes"][0]["assisted_review"]
        self.assertEqual(episode["malformed"], 1)
        self.assertEqual(episode["candidate"], 0)

    def test_assisted_review_never_folds_into_human_decisions(self):
        pending_items = [_eligible_item(2, third_asr={"text": "the quick brown fox"})]
        report, _ = self._build(pending_items)
        self.assertEqual(report["human_decisions"]["total"], 0)
        self.assertEqual(report["assisted_review"]["candidate"], 1)

    def test_ledger_summary_reads_attempts_by_state_and_committed_usd(self):
        ledger = {
            "hard_cap_usd": "0.70",
            "third_asr_subcap_usd": "0.10",
            "attempts": {
                "prepare-s1-1-1": {"state": "reserved", "reserved_usd": "0.02", "third_asr": True},
                "prepare-s1-1-2": {"state": "settled", "reserved_usd": "0.02", "settled_usd": "0.015", "third_asr": True},
                "prepare-s1-1-3": {"state": "released", "reserved_usd": "0.02", "third_asr": True},
                "resolver-attempt-1": {"state": "uncertain", "reserved_usd": "0.05", "third_asr": False},
            },
        }
        report, _ = self._build([], ledger=ledger)

        budget = report["episodes"][0]["budget"]
        self.assertEqual(budget["attempts_by_state"], {"released": 1, "reserved": 1, "settled": 1, "uncertain": 1})
        self.assertEqual(budget["third_asr_attempts_by_state"], {"released": 1, "reserved": 1, "settled": 1})
        # committed = reserved(0.02) + settled(0.015, the settled amount not the reservation) + uncertain(0.05); released excluded.
        self.assertEqual(budget["committed_usd"], "0.085")
        self.assertEqual(budget["third_asr_committed_usd"], "0.035")
        self.assertFalse(budget["integrity_failure"])

        aggregate = report["budget"]
        self.assertEqual(aggregate["ledgers_available"], 1)
        self.assertEqual(aggregate["committed_usd"], "0.085")

    def test_missing_fingerprint_or_ledger_is_unavailable_not_an_error(self):
        report, _ = self._build([])
        self.assertIsNone(report["episodes"][0]["budget"])
        self.assertEqual(report["budget"]["ledgers_available"], 0)

    def test_ledger_lookup_uses_source_fingerprint_not_input_fingerprint(self):
        """TASK-076 Task 14 production proof: input_fingerprint and source_fingerprint
        are distinct identities (podcast_engine/compilation.py). The AI budget ledger's
        canonical GCS path is keyed by source_fingerprint
        (podcast_engine/ai_budget.py::budget_ledger_path), never input_fingerprint. A
        fixture where the two differ must still find the ledger.
        """
        objects = {
            "episodes.json": json.dumps([{"episode_key": "ep-1", "podcast": "Pod", "title": "Ep"}]),
            "episodes/ep-1/review/resolver.json": json.dumps({
                "schema_version": 2,
                "input_fingerprint": "sha256:" + "a" * 64,
                "source_fingerprint": "sha256:" + "b" * 64,
                "human_decisions": [],
                "human_review": [],
            }),
            "episodes/ep-1/ai/budgets/" + "b" * 64 + ".json": json.dumps({
                "hard_cap_usd": "0.70",
                "third_asr_subcap_usd": "0.10",
                "attempts": {
                    "resolver-attempt-1": {"state": "settled", "reserved_usd": "0.05", "settled_usd": "0.01", "third_asr": False},
                },
            }),
        }
        bucket = _Bucket(objects)
        report = review_telemetry.build_telemetry(bucket, bucket_name="test-bucket")

        budget = report["episodes"][0]["budget"]
        self.assertIsNotNone(budget, "ledger must be found via source_fingerprint, not input_fingerprint")
        self.assertEqual(budget["committed_usd"], "0.01")

    def test_markdown_includes_assisted_and_budget_sections(self):
        pending_items = [_eligible_item(2, third_asr={"text": "the quick brown fox"})]
        ledger = {
            "hard_cap_usd": "0.70",
            "third_asr_subcap_usd": "0.10",
            "attempts": {"prepare-s1-1-2": {"state": "settled", "reserved_usd": "0.02", "settled_usd": "0.015", "third_asr": True}},
        }
        report, _ = self._build(pending_items, ledger=ledger)
        markdown = review_telemetry.render_markdown(report)
        self.assertIn("## Assisted review (advisory machine support, never a human decision)", markdown)
        self.assertIn("## Budget and prepare-session summary", markdown)
        self.assertIn("Machine-supported (clear recommendation): **1**", markdown)


if __name__ == "__main__":
    unittest.main()
