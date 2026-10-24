"""TASK-131: recompile an episode waiting in Human Review without re-transcribing."""

from __future__ import annotations

import contextlib
import io
import unittest
from unittest.mock import patch

from google.api_core.exceptions import PreconditionFailed

from podcast_engine import storage
from scripts import recompile_episode


def _episode(key, compiler_state):
    return {
        "episode_key": key,
        "status": {
            "whisper": {"state": "ready"},
            "compiler": {"state": compiler_state},
        },
    }


class RequestRecompilationTests(unittest.TestCase):
    def test_review_required_episode_becomes_ready_and_sources_stay(self):
        episodes = [_episode("a" * 24, "review_required"), _episode("b" * 24, "review_required")]
        saved = []
        with (
            patch.object(storage, "load_episodes_with_generation", return_value=(episodes, 7)),
            patch.object(storage, "save_episodes", side_effect=lambda value, if_generation_match: saved.append((value, if_generation_match))),
        ):
            storage.request_recompilation("a" * 24)
        value, generation = saved[0]
        self.assertEqual(generation, 7)
        self.assertEqual(value[0]["status"]["compiler"]["state"], "ready")
        self.assertEqual(value[0]["status"]["whisper"], {"state": "ready"})
        self.assertEqual(value[1]["status"]["compiler"]["state"], "review_required")

    def test_other_states_and_unknown_keys_are_refused(self):
        for state in ("completed", "pending", "ready", "blocked"):
            with self.subTest(state=state), patch.object(
                storage, "load_episodes_with_generation", return_value=([_episode("a" * 24, state)], 1)
            ), patch.object(storage, "save_episodes") as save:
                with self.assertRaises(ValueError):
                    storage.request_recompilation("a" * 24)
                save.assert_not_called()
        with patch.object(storage, "load_episodes_with_generation", return_value=([], 1)):
            with self.assertRaises(ValueError):
                storage.request_recompilation("a" * 24)

    def test_concurrent_write_retries_from_fresh_state(self):
        loads = [([_episode("a" * 24, "review_required")], 1), ([_episode("a" * 24, "review_required")], 2)]
        calls = []

        def save(value, if_generation_match):
            calls.append(if_generation_match)
            if if_generation_match == 1:
                raise PreconditionFailed("changed")

        with (
            patch.object(storage, "load_episodes_with_generation", side_effect=loads),
            patch.object(storage, "save_episodes", side_effect=save),
        ):
            storage.request_recompilation("a" * 24)
        self.assertEqual(calls, [1, 2])


class RecompileScriptTests(unittest.TestCase):
    rows = [
        {"episode_key": "a" * 24, "podcast": "P", "title": "T", "pending_cards": 57, "human_decisions": 0},
        {"episode_key": "c" * 24, "podcast": "P", "title": "T", "pending_cards": 3, "human_decisions": 2},
    ]

    def _main(self, *argv):
        output = io.StringIO()
        with (
            patch.object(recompile_episode, "candidates", return_value=self.rows),
            patch.object(recompile_episode, "request_recompilation") as request,
            contextlib.redirect_stdout(output),
        ):
            code = recompile_episode.main(list(argv))
        return code, request, output.getvalue()

    def test_dry_run_requests_nothing(self):
        code, request, output = self._main()
        self.assertEqual(code, 0)
        request.assert_not_called()
        self.assertIn("DRY RUN: 2 episode(s)", output)

    def test_execute_requests_listed_episodes(self):
        code, request, _ = self._main("--execute", "a" * 24)
        self.assertEqual(code, 0)
        request.assert_called_once_with("a" * 24)

    def test_recorded_decisions_and_unknown_keys_are_refused(self):
        code, request, _ = self._main("--execute", "a" * 24, "c" * 24)
        self.assertEqual(code, 1)
        request.assert_not_called()
        code, request, _ = self._main("--execute", "c" * 24, "--discard-decisions")
        self.assertEqual(code, 0)
        request.assert_called_once_with("c" * 24)
        code, request, _ = self._main("--execute", "f" * 24)
        self.assertEqual(code, 1)
        request.assert_not_called()


if __name__ == "__main__":
    unittest.main()
