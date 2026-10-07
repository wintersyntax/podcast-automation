"""TASK-118 step 2: the single-pass writer stage and its worker wiring."""

from decimal import Decimal
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from podcast_engine import ai_budget
from podcast_engine.knowledge import orchestration
from podcast_engine.knowledge.models import (
    NOTE_WRITER_POLICY_VERSION,
    NOTE_WRITER_TIMEOUT_SECONDS,
)
from podcast_engine.knowledge.notes_v3 import render, schema, writer
from podcast_engine.preset_provenance import PresetProvenance
from tests.test_knowledge import _FakeBucket
from tests.test_knowledge_notes_v3 import TRANSCRIPT, _note

ROOT = Path(__file__).resolve().parents[1]
EPISODE = {
    "id": "episode-1",
    "episode_key": "episode-1",
    "podcast": "Example Strength Podcast",
    "podcast_id": "example-strength",
    "category": "exercise_strength",
    "prompt": "strength",
    "title": "Ep 386 - Training Principles",
    "published": "2026-08-12T14:00:00Z",
    "link": "https://example.test/episode",
    "status": {"compiler": {"state": "completed"}},
    "files": {
        "sources": {
            "apple": {"text": "episodes/episode-1/sources/apple.txt"},
            "whisper": {"text": "episodes/episode-1/sources/whisper.txt"},
        }
    },
}
WRITER_ENV = {
    "PODCAST_KNOWLEDGE_WRITER_PRESET": "podcast-knowledge-writer",
    "PODCAST_KNOWLEDGE_API_KEY": "test-key",
}


def _provenance(*, prompt=None, config=None):
    return PresetProvenance(
        status="verified",
        slug="podcast-knowledge-writer",
        preset_id="preset-id",
        version_id="version-1",
        version=1,
        config=config if config is not None else {
            "models": ["anthropic/claude-sonnet-5.5"],
            "max_tokens": 48000,
        },
        system_prompt=writer.system_prompt() if prompt is None else prompt,
        config_digest="sha256:config",
        system_prompt_digest="sha256:prompt",
        verified_at="2026-09-29T00:00:00Z",
    )


def _response(note=None, *, finish_reason="stop", content=None):
    response = Mock()
    response.json.return_value = {
        "model": "anthropic/claude-sonnet-5.5",
        "provider": "Anthropic",
        "choices": [{
            "finish_reason": finish_reason,
            "message": {"content": content if content is not None else json.dumps(note or _note())},
        }],
        "usage": {"prompt_tokens": 20000, "completion_tokens": 30000, "cost": 0.34},
    }
    return response


def _ledger(episode_key="a" * 24, source_fingerprint="sha256:" + "d" * 64):
    return {
        "episode_key": episode_key,
        "source_fingerprint": source_fingerprint,
        "hard_cap_usd": "1.60",
        "attempts": {
            "r1": {"state": "settled", "stage": "resolver", "reserved_usd": "0.10", "settled_usd": "0.03"},
            "t1": {"state": "uncertain", "stage": "triage", "reserved_usd": "0.05"},
            "w1": {"state": "settled", "stage": "note_writer", "reserved_usd": "0.70", "settled_usd": "0.34"},
            "w0": {"state": "released", "stage": "note_writer", "reserved_usd": "0.70"},
        },
    }


# The fixture note has 7 anchored units, 2 of which are dropped (ad + one
# stub); the production 20 % threshold is meant for ~60-unit notes.
_LENIENT_CHECKS = patch("podcast_engine.knowledge.notes_v3.checks.MAX_DROPPED_FRACTION", 0.5)


class WriterPayloadTests(unittest.TestCase):
    def test_payload_uses_the_preset_snapshot_strict_schema_and_whole_transcript(self):
        payload = writer.openrouter_payload(EPISODE, TRANSCRIPT, {"rpe": []}, _provenance())
        self.assertEqual(payload["max_tokens"], 48000)
        self.assertEqual(payload["models"], ["anthropic/claude-sonnet-5.5"])
        self.assertEqual(payload["response_format"], schema.RESPONSE_FORMAT)
        self.assertEqual(payload["usage"], {"include": True})
        self.assertEqual(payload["provider"], {"require_parameters": True})
        with_provider = writer.openrouter_payload(EPISODE, TRANSCRIPT, {}, _provenance(config={
            "model": "anthropic/claude-sonnet-5.5", "max_tokens": 48000,
            "provider": {"allow_fallbacks": False, "data_collection": "deny"},
        }))
        self.assertEqual(with_provider["provider"], {
            "allow_fallbacks": False, "data_collection": "deny", "require_parameters": True,
        })
        self.assertEqual(payload["messages"][0], {"role": "system", "content": writer.system_prompt()})
        user = json.loads(payload["messages"][1]["content"])
        self.assertEqual(user["transcript"], TRANSCRIPT)
        self.assertEqual(user["tag_vocabulary"], {"rpe": []})
        self.assertEqual(user["episode_context"]["title"], EPISODE["title"])

    def test_show_notes_are_sent_as_bounded_episode_context(self):
        payload = writer.openrouter_payload(EPISODE, TRANSCRIPT, {}, _provenance(), description="  Chapters: creatine  ")
        user = json.loads(payload["messages"][1]["content"])
        self.assertEqual(user["episode_context"]["description"], "Chapters: creatine")
        long = writer.openrouter_payload(EPISODE, TRANSCRIPT, {}, _provenance(), description="x" * 20000)
        context = json.loads(long["messages"][1]["content"])["episode_context"]
        self.assertEqual(len(context["description"]), writer.MAX_DESCRIPTION_CHARS)
        without = json.loads(writer.openrouter_payload(EPISODE, TRANSCRIPT, {}, _provenance())["messages"][1]["content"])
        self.assertNotIn("description", without["episode_context"])

    def test_a_drifted_remote_prompt_fails_closed(self):
        with self.assertRaises(writer.NoteWriterError) as caught:
            writer.openrouter_payload(EPISODE, TRANSCRIPT, {}, _provenance(prompt="Summarize it."))
        self.assertEqual(caught.exception.code, "prompt_drift")

    def test_trailing_whitespace_in_the_remote_prompt_is_not_drift(self):
        writer.openrouter_payload(EPISODE, TRANSCRIPT, {}, _provenance(prompt=writer.system_prompt() + "\n\n"))

    def test_preset_must_bound_completion_tokens(self):
        with self.assertRaises(writer.NoteWriterError) as caught:
            writer.openrouter_payload(EPISODE, TRANSCRIPT, {}, _provenance(config={"models": ["x"]}))
        self.assertEqual(caught.exception.code, "preset_config")

    def test_worst_case_payload_is_well_formed_and_large(self):
        payload = writer.worst_case_openrouter_payload(_provenance())
        self.assertEqual(payload["messages"][0]["role"], "system")
        self.assertGreater(len(payload["messages"][1]["content"]), 100_000)

    def test_fingerprint_tracks_transcript_preset_and_prompt(self):
        base = writer.input_fingerprint(EPISODE, b"a", "p")
        self.assertEqual(base, writer.input_fingerprint(EPISODE, b"a", "p"))
        self.assertNotEqual(base, writer.input_fingerprint(EPISODE, b"b", "p"))
        self.assertNotEqual(base, writer.input_fingerprint(EPISODE, b"a", "q"))
        with patch.object(writer, "prompt_sha256", return_value="sha256:other"):
            self.assertNotEqual(base, writer.input_fingerprint(EPISODE, b"a", "p"))

    def test_prompt_ships_in_the_worker_image(self):
        self.assertTrue(writer.PROMPT_PATH.is_file())
        self.assertIn("!prompts/knowledge/*.md", (ROOT / ".dockerignore").read_text().splitlines())


class ParseResponseTests(unittest.TestCase):
    def test_truncation_is_reported_as_such(self):
        with self.assertRaises(writer.NoteWriterError) as caught:
            writer.parse_response(_response(finish_reason="length").json())
        self.assertEqual(caught.exception.code, "truncated")

    def test_invalid_json_and_invalid_structure_are_rejected(self):
        for content in ("{not json", json.dumps({"tldr": []})):
            with self.subTest(content=content), self.assertRaises(writer.NoteWriterError) as caught:
                writer.parse_response(_response(content=content).json())
            self.assertEqual(caught.exception.code, "invalid_structure")

    def test_valid_response_returns_note_and_served_identity(self):
        note, served = writer.parse_response(_response().json())
        self.assertEqual(note, _note())
        self.assertEqual(served["model"], "anthropic/claude-sonnet-5.5")
        self.assertEqual(served["usage"]["cost"], 0.34)


class GenerateTests(unittest.TestCase):
    def _generate(self, responses):
        post = Mock(side_effect=responses)
        with patch.dict(os.environ, WRITER_ENV), patch.object(
            writer, "fetch_current_designated_preset", return_value=_provenance()
        ), patch.object(writer, "post_openrouter", post), _LENIENT_CHECKS:
            try:
                result = writer.generate(
                    EPISODE, TRANSCRIPT, {}, episode_key="episode-1", source_fingerprint="sha256:" + "0" * 64
                )
            except writer.NoteWriterError as error:
                return error, post
        return result, post

    def test_one_call_produces_a_checked_rendered_note_under_the_writer_stage(self):
        result, post = self._generate([_response()])
        self.assertEqual(post.call_count, 1)
        kwargs = post.call_args.kwargs
        self.assertEqual(kwargs["stage"], ai_budget.STAGE_NOTE_WRITER)
        self.assertEqual(kwargs["timeout_seconds"], NOTE_WRITER_TIMEOUT_SECONDS)
        self.assertTrue(result.body.startswith("## TL;DR\n"))
        self.assertNotIn("MRR10", result.body)
        reasons = sorted(d.reason for d in result.check_report.dropped)
        self.assertEqual(reasons, ["ad_or_sponsor", "label_only_bullet"])
        record = writer.audit_record(result)
        self.assertEqual(record["served_provider"], "Anthropic")
        self.assertEqual(record["preset_provenance"]["version_id"], "version-1")
        self.assertNotIn("You write a permanent knowledge note", json.dumps(record))

    def test_invalid_structure_gets_exactly_one_more_request(self):
        result, post = self._generate([_response(content="{}"), _response()])
        self.assertEqual(post.call_count, 2)
        self.assertEqual(result.response_attempts, 2)
        error, post = self._generate([_response(content="{}"), _response(content="{}")])
        self.assertEqual(post.call_count, 2)
        self.assertEqual(error.code, "invalid_structure")

    def test_truncation_is_not_retried(self):
        error, post = self._generate([_response(finish_reason="length"), _response()])
        self.assertEqual(post.call_count, 1)
        self.assertEqual(error.code, "truncated")


class OrchestrationTests(unittest.TestCase):
    def _run(self, objects, *, generate):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            compiled = root / "compiled.txt"
            compiled.write_text(TRANSCRIPT, encoding="utf-8")
            bucket = _FakeBucket(objects)
            with patch.dict(os.environ, WRITER_ENV), patch.object(
                orchestration, "RUNTIME_DIR", root / "knowledge"
            ), patch.object(
                orchestration, "download_file_from_gcs", return_value=str(compiled)
            ), patch.object(
                orchestration, "get_bucket", return_value=bucket
            ), patch(
                "podcast_engine.episode_generation.download_gcs_bytes",
                return_value=b"fixture canonical source bytes",
            ), patch.object(
                orchestration.note_writer, "generate", generate
            ), patch.object(
                orchestration.summary_review, "active_summary_review_preset", return_value=None
            ), patch(
                "podcast_engine.ai_budget.episode_spend_summary",
                side_effect=lambda key, fingerprint: ai_budget.spend_breakdown(_ledger(key, fingerprint)),
            ):
                path = orchestration.build_knowledge_note(EPISODE)
                return Path(path).read_text(encoding="utf-8"), bucket

    def _writer_result(self):
        with patch.dict(os.environ, WRITER_ENV), patch.object(
            writer, "fetch_current_designated_preset", return_value=_provenance()
        ), patch.object(writer, "post_openrouter", return_value=_response()), _LENIENT_CHECKS:
            return writer.generate(
                EPISODE, TRANSCRIPT, {}, episode_key="episode-1", source_fingerprint="sha256:" + "0" * 64
            )

    def test_writer_path_publishes_body_note_manifest_and_final_note(self):
        result = self._writer_result()
        generate = Mock(return_value=result)
        final, bucket = self._run({}, generate=generate)

        generate.assert_called_once()
        self.assertEqual(generate.call_args.kwargs["episode_key"], "episode-1")
        self.assertTrue(final.startswith("---\n"))
        self.assertIn("topics:\n- creatine\n- sleep\n", final)
        self.assertIn("## TL;DR", final)
        self.assertEqual(bucket.objects["episodes/episode-1/summary/body.md"], result.body)
        self.assertNotIn("episodes/episode-1/summary/note.json", bucket.objects)

        manifest = json.loads(bucket.objects["episodes/episode-1/summary/metadata.json"])
        self.assertEqual(manifest["summary"]["policy_version"], NOTE_WRITER_POLICY_VERSION)
        self.assertEqual(manifest["summary"]["preset"], "podcast-knowledge-writer")
        self.assertNotIn("summary_review", manifest)
        self.assertEqual(manifest["metadata"]["people"], ["Eric Helms"])
        self.assertEqual(manifest["note_writer"]["note"], result.note)
        spend = manifest["note_writer"]["episode_spend"]
        self.assertEqual(spend["stages"]["note_writer"]["settled_usd"], "0.34")
        self.assertEqual(spend["hard_cap_usd"], "1.60")
        self.assertEqual(manifest["note_writer"]["checks"]["anchored_dropped"], 1)
        with patch.object(orchestration, "get_bucket", return_value=bucket):
            self.assertEqual(
                orchestration.canonical_summary_generated_at("episode-1"),
                manifest["summary"]["generated_at"],
            )

    def test_show_notes_are_fetched_passed_and_recorded(self):
        generate = Mock(return_value=self._writer_result())
        episode = {**EPISODE, "feed_url": "https://feed.test/rss", "guid": "g-1"}
        with patch.object(orchestration, "fetch_episode_description", return_value="Show notes") as fetch:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                compiled = root / "compiled.txt"
                compiled.write_text(TRANSCRIPT, encoding="utf-8")
                bucket = _FakeBucket({})
                with patch.dict(os.environ, WRITER_ENV), patch.object(orchestration, "RUNTIME_DIR", root / "k"), \
                        patch.object(orchestration, "download_file_from_gcs", return_value=str(compiled)), \
                        patch.object(orchestration, "get_bucket", return_value=bucket), \
                        patch("podcast_engine.episode_generation.download_gcs_bytes", return_value=b"src"), \
                        patch.object(orchestration.note_writer, "generate", generate), \
                        patch("podcast_engine.ai_budget.episode_spend_summary", return_value={}):
                    orchestration.build_knowledge_note(episode)
        fetch.assert_called_once_with("https://feed.test/rss", "g-1")
        self.assertEqual(generate.call_args.kwargs["description"], "Show notes")
        manifest = json.loads(bucket.objects["episodes/episode-1/summary/metadata.json"])
        self.assertTrue(manifest["note_writer"]["description_sha256"].startswith("sha256:"))

    def test_unchanged_inputs_hit_the_cache_without_a_model_call(self):
        _, bucket = self._run({}, generate=Mock(return_value=self._writer_result()))
        again = Mock()
        final, second = self._run(dict(bucket.objects), generate=again)
        again.assert_not_called()
        self.assertEqual(final, bucket.objects["episodes/episode-1/summary/summary.md"])

    def test_an_older_render_is_re_rendered_from_the_stored_note_without_a_model_call(self):
        _, bucket = self._run({}, generate=Mock(return_value=self._writer_result()))
        manifest = json.loads(bucket.objects["episodes/episode-1/summary/metadata.json"])
        self.assertEqual(manifest["summary"]["render_version"], render.RENDER_VERSION)
        del manifest["summary"]["render_version"]  # a note written by renderer v1
        objects = dict(bucket.objects)
        objects["episodes/episode-1/summary/metadata.json"] = json.dumps(manifest)
        objects["episodes/episode-1/summary/body.md"] = "stale v1 body"
        again = Mock()
        final, second = self._run(objects, generate=again)
        again.assert_not_called()
        rerendered = json.loads(second.objects["episodes/episode-1/summary/metadata.json"])
        self.assertEqual(rerendered["summary"]["render_version"], render.RENDER_VERSION)
        self.assertEqual(second.objects["episodes/episode-1/summary/body.md"], self._writer_result().body)
        self.assertIn("## TL;DR", final)

    def test_a_legacy_manifest_is_regenerated_by_the_writer(self):
        legacy = {"schema_version": 1, "summary": {"policy_version": "summary-v6", "generated_at": "x"}}
        generate = Mock(return_value=self._writer_result())
        self._run(
            {
                "episodes/episode-1/summary/metadata.json": json.dumps(legacy),
                "episodes/episode-1/summary/body.md": "old body",
            },
            generate=generate,
        )
        generate.assert_called_once()

    def test_without_the_writer_preset_the_legacy_gate_still_applies(self):
        with patch.dict(os.environ, {"PODCAST_KNOWLEDGE_WRITER_PRESET": "  "}), patch.object(
            orchestration.summary_review, "active_summary_review_preset", return_value=None
        ), patch.object(orchestration.note_writer, "generate") as generate:
            self.assertIsNone(orchestration.build_knowledge_note(EPISODE))
        generate.assert_not_called()


class SpendTrackingTests(unittest.TestCase):
    def test_spend_breakdown_splits_by_stage_and_state(self):
        spend = ai_budget.spend_breakdown(_ledger())
        self.assertEqual(spend["settled_usd"], "0.37")
        self.assertEqual(spend["uncertain_usd"], "0.05")
        self.assertEqual(spend["committed_usd"], "0.42")
        self.assertEqual(spend["stages"]["note_writer"], {
            "attempts": 2, "settled_usd": "0.34", "uncertain_usd": "0", "reserved_usd": "0",
        })
        json.dumps(spend)

    def test_unreadable_ledger_never_blocks_the_note(self):
        with patch("podcast_engine.ai_budget.episode_spend_summary", side_effect=RuntimeError("boom")):
            self.assertEqual(orchestration._episode_spend("a" * 24, "sha256:" + "d" * 64), {"unavailable": "RuntimeError"})

    def test_report_joins_episodes_and_summarises_spend(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location("ai_spend_report", ROOT / "scripts" / "ai_spend_report.py")
        report_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(report_module)
        second = _ledger("b" * 24)
        second["attempts"]["w1"]["settled_usd"] = "0.50"
        report = report_module.build_report(
            [ai_budget.spend_breakdown(_ledger()), ai_budget.spend_breakdown(second)],
            {"a" * 24: {"podcast": "Example Strength Podcast", "title": "Ep 1", "published": "2026-09-01"}},
        )
        self.assertEqual(report["episodes"], 2)
        self.assertEqual(report["total_committed_usd"], "1.000000")
        self.assertEqual(report["note_writer_settled"]["max_usd"], "0.500000")
        self.assertEqual(report["rows"][0]["title"], "Ep 1")
        with tempfile.TemporaryDirectory() as directory:
            json_path, csv_path = report_module.write_outputs(report, Path(directory))
            header, first = csv_path.read_text().splitlines()[:2]
        self.assertIn("note_writer_usd", header)
        self.assertIn("Example Strength Podcast", first)


class BudgetWiringTests(unittest.TestCase):
    def test_cap_and_stage_vocabulary(self):
        self.assertEqual(ai_budget.EPISODE_AI_HARD_CAP_USD, Decimal("1.60"))
        self.assertIn(ai_budget.STAGE_NOTE_WRITER, ai_budget.AI_BUDGET_STAGE_IDS)

    def test_inactive_writer_contributes_no_downstream_reserve(self):
        pending = {stage: False for stage in ai_budget._DOWNSTREAM_RESERVE_ELIGIBLE_STAGES}
        pending[ai_budget.STAGE_NOTE_WRITER] = True
        with patch.dict(os.environ, {"PODCAST_KNOWLEDGE_WRITER_PRESET": ""}), patch(
            "podcast_engine.ai_budget.fetch_current_designated_preset"
        ) as fetch:
            total = ai_budget.resolve_required_downstream_reserve_usd(
                stage_pending=pending, api_key="test-key"
            )
        self.assertEqual(total, Decimal("0"))
        fetch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
