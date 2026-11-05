"""TASK-119: the production coverage-fill stage and its worker wiring."""

import copy
import json
import os
import unittest
from unittest.mock import Mock, patch

from podcast_engine import ai_budget
from podcast_engine.knowledge import orchestration
from podcast_engine.knowledge.models import NOTE_FILL_TIMEOUT_SECONDS
from podcast_engine.knowledge.notes_v3 import coverage, fill, fill_stage, render
from podcast_engine.preset_provenance import PresetProvenance
from tests.test_knowledge_coverage_fill import TRANSCRIPT, bullet, note
from tests.test_knowledge_note_writer import EPISODE, WRITER_ENV, OrchestrationTests

FILL_ENV = {"PODCAST_KNOWLEDGE_FILL_PRESET": "podcast-knowledge-fill", "PODCAST_KNOWLEDGE_API_KEY": "test-key"}
FINGERPRINT = "sha256:" + "0" * 64


def _provenance(*, prompt=None, config=None, verified=True):
    return PresetProvenance(
        status="verified" if verified else "unavailable",
        slug="podcast-knowledge-fill",
        preset_id="fill-id",
        version_id="fill-version-1",
        version=1,
        config=config if config is not None else {
            "model": "anthropic/claude-sonnet-5.5",
            "max_tokens": 14000,
            "provider": {"only": ["anthropic"], "allow_fallbacks": False},
        },
        system_prompt=fill.system_prompt() if prompt is None else prompt,
        config_digest="sha256:config",
        system_prompt_digest="sha256:prompt",
        verified_at="2026-10-15T00:00:00Z",
    )


def _decisions(report, *, add_for=None):
    empty = bullet("", "")
    decisions = []
    for candidate in report.candidates:
        decision = {"candidate_id": candidate.id, "action": "skip", "section_title": "", "old_bullet_id": "",
                    "bullet": empty, "skip_reason": "not_practical"}
        if candidate.type == add_for:
            decision.update(action="add", section_title="Leg press", skip_reason="not_applicable", bullet=bullet(
                "On the leg press he keeps his feet high and uses a full range of motion with a slow eccentric.",
                "On the leg press I keep my feet high on the platform and use a full range of motion",
                "personal_experience",
            ))
        decisions.append(decision)
    return decisions


def _response(decisions=None, *, finish_reason="stop", content=None):
    response = Mock()
    response.json.return_value = {
        "model": "anthropic/claude-sonnet-5.5",
        "provider": "Anthropic",
        "choices": [{"finish_reason": finish_reason,
                     "message": {"content": content if content is not None else json.dumps({"decisions": decisions or []})}}],
        "usage": {"prompt_tokens": 30000, "completion_tokens": 1800, "cost": 0.08},
    }
    return response


class PayloadTests(unittest.TestCase):
    def setUp(self):
        self.report = coverage.find_candidates(note(), TRANSCRIPT)

    def test_payload_uses_the_preset_snapshot_strict_schema_and_whole_input(self):
        payload = fill_stage.openrouter_payload(note(), TRANSCRIPT, self.report, _provenance(), title="Ep 1")
        self.assertEqual(payload["model"], "anthropic/claude-sonnet-5.5")
        self.assertEqual(payload["max_tokens"], 14000)
        self.assertTrue(payload["provider"]["require_parameters"])
        self.assertEqual(payload["response_format"], fill.RESPONSE_FORMAT)
        self.assertEqual(payload["messages"][0]["content"], fill.system_prompt())
        content = json.loads(payload["messages"][1]["content"])
        self.assertEqual(content["transcript"], TRANSCRIPT)
        self.assertEqual(len(content["all_bullets"]), 2)
        self.assertEqual(len(content["candidates"]), len(self.report.candidates))

    def test_drifted_prompt_unverified_preset_and_unbounded_tokens_fail(self):
        for provenance, code in (
            (_provenance(prompt="something else"), "prompt_drift"),
            (_provenance(verified=False), "preset_unverified"),
            (_provenance(config={"model": "x"}), "preset_config"),
        ):
            with self.subTest(code), self.assertRaises(fill_stage.NoteFillError) as caught:
                fill_stage.openrouter_payload(note(), TRANSCRIPT, self.report, provenance)
            self.assertEqual(caught.exception.code, code)


class RunFillTests(unittest.TestCase):
    def _run(self, responses=None, *, env=FILL_ENV, provenance=None, the_note=None, transcript=TRANSCRIPT):
        post = Mock(side_effect=responses or [])
        with patch.dict(os.environ, env), patch.object(
            fill_stage, "fetch_current_designated_preset", return_value=provenance or _provenance()
        ), patch.object(fill_stage, "post_openrouter", post):
            outcome = fill_stage.run_fill(
                the_note or note(), transcript, title="Ep 1", episode_key="episode-1", source_fingerprint=FINGERPRINT
            )
        return outcome, post

    def test_a_triggered_fill_is_applied_under_the_fill_stage_and_recorded(self):
        report = coverage.find_candidates(note(), TRANSCRIPT)
        outcome, post = self._run([_response(_decisions(report, add_for="exercise_advice"))])
        self.assertEqual(post.call_count, 1)
        self.assertEqual(post.call_args.kwargs["stage"], ai_budget.STAGE_NOTE_FILL)
        self.assertEqual(post.call_args.kwargs["timeout_seconds"], NOTE_FILL_TIMEOUT_SECONDS)
        self.assertEqual(outcome.status, "applied")
        record = outcome.record
        self.assertEqual(record["schema_version"], fill.FILL_SCHEMA_VERSION)
        self.assertEqual(record["prompt_sha256"], fill.prompt_sha256())
        self.assertEqual(record["usage"]["cost"], 0.08)
        self.assertEqual(record["result"]["applied"]["add"], 1)
        self.assertEqual(record["preset_provenance"]["version_id"], "fill-version-1")
        self.assertEqual([s["title"] for s in outcome.note["sections"]][-1], "Leg press")
        self.assertIs(record["note"], outcome.note)
        self.assertNotIn("You maintain a permanent knowledge note", json.dumps(record))

    def test_nothing_uncovered_means_no_call(self):
        report = coverage.CoverageReport([], 0, False, {}, "sha256:x")
        with patch.object(fill_stage.coverage, "find_candidates", return_value=report):
            outcome, post = self._run()
        post.assert_not_called()
        self.assertEqual(outcome.status, "not_triggered")
        self.assertEqual(outcome.note, note())

    def test_without_the_fill_preset_the_fill_is_disabled(self):
        outcome, post = self._run(env={"PODCAST_KNOWLEDGE_FILL_PRESET": " "})
        post.assert_not_called()
        self.assertEqual(outcome.status, "disabled")

    def test_failures_publish_the_writer_note_and_are_recorded(self):
        cases = {
            "prompt_drift": dict(provenance=_provenance(prompt="drifted")),
            "provider_error": dict(responses=[RuntimeError("HTTP 503")]),
            "truncated": dict(responses=[_response(finish_reason="length"), _response()]),
            "invalid_twice": dict(responses=[_response(content="{}"), _response(content="nope")]),
        }
        for name, kwargs in cases.items():
            with self.subTest(name):
                outcome, post = self._run(**kwargs)
                self.assertEqual(outcome.status, "failed")
                self.assertEqual(outcome.note, note())
                self.assertIn("type", outcome.record["error"])
                if name == "truncated":
                    self.assertEqual(post.call_count, 1)
                if name == "invalid_twice":
                    self.assertEqual(post.call_count, 2)

    def test_an_invalid_structure_gets_exactly_one_more_request(self):
        report = coverage.find_candidates(note(), TRANSCRIPT)
        outcome, post = self._run([_response(content="{}"), _response(_decisions(report))])
        self.assertEqual(post.call_count, 2)
        self.assertEqual(outcome.status, "applied")
        self.assertEqual(outcome.record["response_attempts"], 2)


class PendingTests(unittest.TestCase):
    def current(self, status="applied"):
        return {"status": status, "schema_version": fill.FILL_SCHEMA_VERSION,
                "prompt_sha256": fill.prompt_sha256(), "coverage_policy": coverage.COVERAGE_POLICY_VERSION}

    def test_pending_matrix(self):
        self.assertFalse(fill_stage.fill_pending(None, enabled=False))
        self.assertTrue(fill_stage.fill_pending(None, enabled=True))
        self.assertFalse(fill_stage.fill_pending(self.current(), enabled=True))
        self.assertFalse(fill_stage.fill_pending(self.current("not_triggered"), enabled=True))
        self.assertTrue(fill_stage.fill_pending(self.current("failed"), enabled=True))
        self.assertTrue(fill_stage.fill_pending(self.current("disabled"), enabled=True))
        self.assertTrue(fill_stage.fill_pending({**self.current(), "prompt_sha256": "sha256:old"}, enabled=True))

    def test_published_note_prefers_the_applied_fill(self):
        filled = {"filled": True}
        manifest = {"note_writer": {"note": {"writer": True}}, "note_fill": {"status": "applied", "note": filled}}
        self.assertIs(fill_stage.published_note(manifest), filled)
        manifest["note_fill"]["status"] = "failed"
        self.assertEqual(fill_stage.published_note(manifest), {"writer": True})
        self.assertEqual(fill_stage.published_note({"note_writer": {"note": {"writer": True}}}), {"writer": True})


class WiringTests(unittest.TestCase):
    """The writer orchestration harness with the fill enabled."""

    ENV = {**WRITER_ENV, "PODCAST_KNOWLEDGE_FILL_PRESET": "podcast-knowledge-fill"}
    _run = OrchestrationTests._run
    _writer_result = OrchestrationTests._writer_result

    def _filled(self, the_note):
        filled = copy.deepcopy(the_note)
        filled["sections"][0]["bullets"].append(dict(filled["sections"][0]["bullets"][0], text="A filled detail."))
        return filled

    def _fill_mock(self, status="applied"):
        def run(the_note, transcript, **kwargs):
            note_out = self._filled(the_note) if status == "applied" else the_note
            record = {"status": status, "schema_version": fill.FILL_SCHEMA_VERSION,
                      "prompt_sha256": fill.prompt_sha256(), "coverage_policy": coverage.COVERAGE_POLICY_VERSION}
            if status == "applied":
                record["note"] = note_out
            return fill_stage.FillOutcome(note_out, record)
        return Mock(side_effect=run)

    def _run_fill(self, objects, *, generate, run_fill):
        with patch.object(orchestration.note_fill, "run_fill", run_fill):
            return self._run(objects, generate=generate)

    def test_a_new_note_is_filled_and_rendered_from_the_filled_note(self):
        result = self._writer_result()
        run_fill = self._fill_mock()
        final, bucket = self._run_fill({}, generate=Mock(return_value=result), run_fill=run_fill)
        run_fill.assert_called_once()
        self.assertEqual(run_fill.call_args.kwargs["title"], EPISODE["title"])
        manifest = json.loads(bucket.objects["episodes/episode-1/summary/metadata.json"])
        self.assertEqual(manifest["note_writer"]["note"], result.note)
        self.assertEqual(manifest["note_fill"]["status"], "applied")
        self.assertIn("A filled detail.", bucket.objects["episodes/episode-1/summary/body.md"])
        self.assertIn("A filled detail.", final)

    def test_a_failed_fill_still_publishes_the_writer_note(self):
        result = self._writer_result()
        final, bucket = self._run_fill({}, generate=Mock(return_value=result), run_fill=self._fill_mock("failed"))
        self.assertEqual(bucket.objects["episodes/episode-1/summary/body.md"], result.body)
        manifest = json.loads(bucket.objects["episodes/episode-1/summary/metadata.json"])
        self.assertEqual(manifest["note_fill"]["status"], "failed")

    def test_a_stored_note_without_a_fill_is_filled_once_without_the_writer(self):
        _, bucket = self._run_fill(
            {}, generate=Mock(return_value=self._writer_result()), run_fill=self._fill_mock("disabled")
        )
        before = json.loads(bucket.objects["episodes/episode-1/summary/metadata.json"])
        self.assertEqual(before["note_fill"]["status"], "disabled")

        writer_again, run_fill = Mock(), self._fill_mock()
        final, second = self._run_fill(dict(bucket.objects), generate=writer_again, run_fill=run_fill)
        writer_again.assert_not_called()
        run_fill.assert_called_once()
        manifest = json.loads(second.objects["episodes/episode-1/summary/metadata.json"])
        self.assertEqual(manifest["note_fill"]["status"], "applied")
        self.assertEqual(manifest["summary"]["render_version"], render.RENDER_VERSION)
        self.assertIn("A filled detail.", second.objects["episodes/episode-1/summary/body.md"])

        third_fill = self._fill_mock()
        final_again, third = self._run_fill(dict(second.objects), generate=Mock(), run_fill=third_fill)
        third_fill.assert_not_called()
        self.assertEqual(final_again, final)

    def test_a_re_render_keeps_the_filled_note(self):
        _, bucket = self._run_fill({}, generate=Mock(return_value=self._writer_result()), run_fill=self._fill_mock())
        manifest = json.loads(bucket.objects["episodes/episode-1/summary/metadata.json"])
        del manifest["summary"]["render_version"]
        objects = dict(bucket.objects)
        objects["episodes/episode-1/summary/metadata.json"] = json.dumps(manifest)
        run_fill = self._fill_mock()
        _, second = self._run_fill(objects, generate=Mock(), run_fill=run_fill)
        run_fill.assert_not_called()
        self.assertIn("A filled detail.", second.objects["episodes/episode-1/summary/body.md"])


class BudgetStageTests(unittest.TestCase):
    def test_fill_is_a_budget_stage_but_not_a_protected_downstream_stage(self):
        self.assertIn(ai_budget.STAGE_NOTE_FILL, ai_budget.AI_BUDGET_STAGE_IDS)
        self.assertNotIn(ai_budget.STAGE_NOTE_FILL, ai_budget._DOWNSTREAM_RESERVE_ELIGIBLE_STAGES)


if __name__ == "__main__":
    unittest.main()
