import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from compiler.transcript import build_resolver_batch, compile_transcripts
from podcast_engine.compilation import (
    REVIEW_POLICY_VERSION,
    REVIEW_RECORD_SCHEMA_VERSION,
    _build_review_record,
    compile_episode_sources,
    _human_review_items,
    _resolve_with_cache,
    _review_input_fingerprint,
    _review_source_fingerprint,
    _triage_from_record,
    _unavailable_triage,
    review_record_is_current,
)
from podcast_engine.human_review import review_queue_fingerprint, validated_human_resolutions
from podcast_engine.episode_generation import source_generation_fingerprint
from podcast_engine.preset_provenance import PresetProvenance
from podcast_engine.review import REVIEW_PRESET, TriageResponseParseError


class ReviewPersistenceTests(
    unittest.TestCase
):
    def _provenance(self, *, version=4):
        return PresetProvenance(
            status="verified",
            slug=REVIEW_PRESET,
            preset_id="preset-id",
            version_id=f"version-id-{version}",
            version=version,
            config={"model": "test/model", "temperature": 0},
            system_prompt="Prompt",
            config_digest="sha256:config",
            system_prompt_digest="sha256:prompt",
            verified_at="2026-08-30T00:00:00+00:00",
        )

    def _record(
        self,
        fingerprint="sha256:same",
        source_fingerprint="sha256:sources",
    ):
        return {
            "schema_version": (
                REVIEW_RECORD_SCHEMA_VERSION
            ),
            "policy_version": (
                REVIEW_POLICY_VERSION
            ),
            "episode_key": (
                "episode-1"
            ),
            "input_fingerprint": (
                fingerprint
            ),
            "source_fingerprint": source_fingerprint,
            "reviewer_preset": (
                REVIEW_PRESET
            ),
            "preset_provenance": self._provenance().record(),
            "reviewed_at": (
                "2026-08-25T23:00:00+00:00"
            ),
            "accepted": [
                {
                    "id": 103,
                    "source": "apple",
                    "text": "RPE",
                    "scope": "partial",
                    "focus_apple_text": "RPE",
                    "focus_whisper_text": "RP",
                }
            ],
            "resolver_review": [
                {
                    "id": 246,
                    "reason": (
                        "needs_human_review"
                    ),
                }
            ],
            "human_review": [
                {
                    "id": 246,
                    "reason": (
                        "needs_human_review"
                    ),
                }
            ],
        }


    def test_matching_record_reuses_decisions_without_ai_call(
        self,
    ):
        record = self._record()

        batch = {
            "batch": {
                "diff_items": [
                    {"id": 103},
                    {"id": 246},
                ],
            },
        }

        with patch(
            "podcast_engine.compilation._load_review_record",
            return_value=record,
        ), patch(
            "podcast_engine.compilation.resolve_compiler_batch",
        ) as resolver:

            (
                resolution,
                returned_record,
                cache_hit,
            ) = _resolve_with_cache(
                resolver_batch=batch,
                input_fingerprint=(
                    "sha256:same"
                ),
                episode_key="episode-1",
                source_fingerprint="sha256:sources",
                resolver_record_path=(
                    "episodes/episode-1/review/resolver.json"
                ),
                preset_provenance=self._provenance(),
            )

        resolver.assert_not_called()

        self.assertTrue(
            cache_hit
        )

        self.assertIs(
            returned_record,
            record,
        )

        self.assertEqual(
            resolution[
                "accepted"
            ],
            record[
                "accepted"
            ],
        )

        self.assertEqual(
            resolution[
                "review"
            ],
            record[
                "resolver_review"
            ],
        )


    def test_changed_fingerprint_invalidates_cache_and_calls_ai(
        self,
    ):
        record = self._record(
            "sha256:old"
        )

        batch = {
            "batch": {
                "diff_items": [
                    {"id": 103},
                ],
            },
        }

        fresh = {
            "accepted": [
                {
                    "id": 103,
                    "source": "apple",
                    "text": "RPE",
                }
            ],
            "review": [],
        }

        with patch(
            "podcast_engine.compilation._load_review_record",
            return_value=record,
        ), patch(
            "podcast_engine.compilation.resolve_compiler_batch",
            return_value=fresh,
        ) as resolver:

            (
                resolution,
                returned_record,
                cache_hit,
            ) = _resolve_with_cache(
                resolver_batch=batch,
                input_fingerprint=(
                    "sha256:new"
                ),
                episode_key="episode-1",
                source_fingerprint="sha256:sources",
                resolver_record_path=(
                    "episodes/episode-1/review/resolver.json"
                ),
                preset_provenance=self._provenance(),
            )

        resolver.assert_called_once_with(
            batch,
            self._provenance(),
            episode_key="episode-1",
            source_fingerprint="sha256:sources",
        )

        self.assertFalse(
            cache_hit
        )

        self.assertIs(returned_record, record)

        self.assertEqual(
            resolution,
            fresh,
        )


    def test_previous_policy_record_never_reuses_same_fingerprint(
        self,
    ):
        record = self._record("sha256:stable")
        record["policy_version"] = "resolver-policy-v2"
        batch = {"batch": {"diff_items": []}}
        fresh = {"accepted": [], "review": [], "outcomes": []}

        with patch(
            "podcast_engine.compilation._load_review_record",
            return_value=record,
        ), patch(
            "podcast_engine.compilation.resolve_compiler_batch",
            return_value=fresh,
        ) as resolver:
            resolution, returned_record, cache_hit = _resolve_with_cache(
                resolver_batch=batch,
                input_fingerprint="sha256:stable",
                episode_key="episode-1",
                source_fingerprint="sha256:sources",
                resolver_record_path="episodes/episode-1/review/resolver.json",
                preset_provenance=self._provenance(),
            )

        resolver.assert_called_once_with(
            batch,
            self._provenance(),
            episode_key="episode-1",
            source_fingerprint="sha256:sources",
        )
        self.assertFalse(cache_hit)
        self.assertIs(returned_record, record)
        self.assertEqual(resolution, fresh)

    def test_unavailable_provenance_skips_ai_and_existing_ai_cache(self):
        record = self._record()
        unavailable = PresetProvenance(
            status="unavailable",
            reason="network_or_timeout",
            slug=REVIEW_PRESET,
        )
        batch = {"batch": {"diff_items": [{"id": 103}]}}

        with patch("podcast_engine.compilation._load_review_record", return_value=record), patch(
            "podcast_engine.compilation.resolve_compiler_batch"
        ) as resolver:
            resolution, returned_record, cache_hit = _resolve_with_cache(
                resolver_batch=batch,
                input_fingerprint="sha256:same",
                episode_key="episode-1",
                source_fingerprint="sha256:sources",
                resolver_record_path="episodes/episode-1/review/resolver.json",
                preset_provenance=unavailable,
            )

        resolver.assert_not_called()
        self.assertFalse(cache_hit)
        self.assertIs(returned_record, record)
        self.assertEqual(resolution["accepted"], [])
        self.assertEqual(resolution["review"], [{"id": 103, "reason": "preset_provenance_unavailable"}])

    def test_all_non_verified_provenance_states_skip_ai_and_route_human_review(self):
        batch = {"batch": {"diff_items": [{"id": 103}]}}
        for status in ("drift", "unavailable", "invalid"):
            with self.subTest(status=status), patch(
                "podcast_engine.compilation._load_review_record", return_value=None
            ), patch("podcast_engine.compilation.resolve_compiler_batch") as resolver:
                resolution, _record, cache_hit = _resolve_with_cache(
                    resolver_batch=batch,
                    input_fingerprint="sha256:same",
                    episode_key="episode-1",
                    source_fingerprint="sha256:sources",
                    resolver_record_path="episodes/episode-1/review/resolver.json",
                    preset_provenance=PresetProvenance(status=status, reason="safe_reason", slug=REVIEW_PRESET),
                )

            resolver.assert_not_called()
            self.assertFalse(cache_hit)
            self.assertEqual(resolution["accepted"], [])
            self.assertEqual(resolution["review"], [{"id": 103, "reason": f"preset_provenance_{status}"}])

    def test_exact_version_config_and_prompt_identity_each_change_fingerprint(self):
        batch = {"batch": {"episode_id": "episode-1", "diff_items": []}, "response_schema": {"type": "object"}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            apple = root / "apple.txt"
            whisper = root / "whisper.txt"
            apple.write_text("Stable transcript.", encoding="utf-8")
            whisper.write_text("Stable transcript.", encoding="utf-8")

            def fingerprint(provenance):
                return _review_input_fingerprint(
                    apple_text_path=apple,
                    apple_metadata_path=None,
                    whisper_text_path=whisper,
                    whisper_metadata_path=None,
                    resolver_batch=batch,
                    preset_provenance=provenance,
                )

            baseline = fingerprint(self._provenance())
            changed_version = fingerprint(self._provenance(version=5))
            changed_config = fingerprint(PresetProvenance(
                **{**self._provenance().__dict__, "config_digest": "sha256:other-config"}
            ))
            changed_prompt = fingerprint(PresetProvenance(
                **{**self._provenance().__dict__, "system_prompt_digest": "sha256:other-prompt"}
            ))

        self.assertNotEqual(baseline, changed_version)
        self.assertNotEqual(baseline, changed_config)
        self.assertNotEqual(baseline, changed_prompt)

    def test_schema_v3_record_cannot_satisfy_current_reuse_contract(self):
        record = self._record("sha256:stable")
        record["schema_version"] = 3
        batch = {"batch": {"diff_items": [{"id": 103}]}}
        fresh = {"accepted": [], "review": [], "outcomes": []}

        with patch(
            "podcast_engine.compilation._load_review_record",
            return_value=record,
        ), patch(
            "podcast_engine.compilation.resolve_compiler_batch",
            return_value=fresh,
        ) as resolver:
            resolution, returned_record, cache_hit = _resolve_with_cache(
                resolver_batch=batch,
                input_fingerprint="sha256:stable",
                episode_key="episode-1",
                source_fingerprint="sha256:sources",
                resolver_record_path="episodes/episode-1/review/resolver.json",
                preset_provenance=self._provenance(),
            )

        resolver.assert_called_once_with(
            batch,
            self._provenance(),
            episode_key="episode-1",
            source_fingerprint="sha256:sources",
        )
        self.assertFalse(cache_hit)
        self.assertIs(returned_record, record)
        self.assertEqual(resolution, fresh)

    def test_cached_triage_replays_only_safe_per_chunk_failure_metadata(self):
        raw_content = '{"triage": ['
        record = {
            "triage_recommendations": [],
            "triage_outcomes": [
                {
                    "id": 11,
                    "status": "unavailable",
                    "recommendation": "needs_audio",
                    "reason": "triage_unavailable",
                }
            ],
            "triage_completion_metadata": {
                "chunks": [],
            },
            "triage_failure_metadata": [
                {
                    "item_ids": [11],
                    "error_type": "TriageResponseParseError",
                    "finish_reason": "length",
                    "completion_id": "completion-bad",
                    "served_model": "google/gemini-3.6-flash",
                    "served_provider": "Google AI Studio",
                    "content_chars": len(raw_content),
                    "content_sha256": "d" * 64,
                    "raw_content": raw_content,
                    "unexpected": "must-not-replay",
                }
            ],
        }

        triage = _triage_from_record(record)

        self.assertEqual(
            triage["failure_metadata"],
            [
                {
                    "item_ids": [11],
                    "error_type": "TriageResponseParseError",
                    "finish_reason": "length",
                    "completion_id": "completion-bad",
                    "served_model": "google/gemini-3.6-flash",
                    "served_provider": "Google AI Studio",
                    "content_chars": len(raw_content),
                    "content_sha256": "d" * 64,
                }
            ],
        )
        serialized = json.dumps(triage, sort_keys=True)
        self.assertNotIn("raw_content", serialized)
        self.assertNotIn(raw_content, serialized)
        self.assertNotIn("must-not-replay", serialized)

    def test_previous_schema_ai_cache_cannot_satisfy_current_reuse_contract(self):
        record = self._record()
        record["schema_version"] = REVIEW_RECORD_SCHEMA_VERSION - 1
        batch = {"batch": {"diff_items": [{"id": 103}]}}
        fresh = {"accepted": [], "review": [], "outcomes": []}

        with patch("podcast_engine.compilation._load_review_record", return_value=record), patch(
            "podcast_engine.compilation.resolve_compiler_batch", return_value=fresh
        ) as resolver:
            resolution, returned_record, cache_hit = _resolve_with_cache(
                resolver_batch=batch,
                input_fingerprint="sha256:same",
                episode_key="episode-1",
                source_fingerprint="sha256:sources",
                resolver_record_path="episodes/episode-1/review/resolver.json",
                preset_provenance=self._provenance(),
            )

        resolver.assert_called_once_with(
            batch,
            self._provenance(),
            episode_key="episode-1",
            source_fingerprint="sha256:sources",
        )
        self.assertFalse(cache_hit)
        self.assertIs(returned_record, record)
        self.assertEqual(resolution, fresh)

    def test_valid_human_truth_remains_usable_when_provenance_is_unavailable(self):
        item = {
            "id": 103,
            "apple_text": "Apple text",
            "whisper_text": "Whisper text",
            "focus": {"scope": "full", "apple_text": None, "whisper_text": None},
        }
        record = {
            "human_decisions": [{
                "id": 103,
                "chosen_source": "apple",
                "chosen_text": "Apple text",
                "scope": "full",
                "reviewed_by": "human",
                "review_item": dict(item),
            }],
        }
        unavailable = PresetProvenance(status="unavailable", reason="network_or_timeout", slug=REVIEW_PRESET)

        with patch("podcast_engine.compilation._load_review_record", return_value=record), patch(
            "podcast_engine.compilation.resolve_compiler_batch"
        ) as resolver:
            _resolution, cached_record, cache_hit = _resolve_with_cache(
                resolver_batch={"batch": {"diff_items": [{"id": 103}]}},
                input_fingerprint="sha256:same",
                episode_key="episode-1",
                source_fingerprint="sha256:sources",
                resolver_record_path="episodes/episode-1/review/resolver.json",
                preset_provenance=unavailable,
            )

        resolver.assert_not_called()
        self.assertFalse(cache_hit)
        self.assertIs(cached_record, record)
        self.assertEqual(
            validated_human_resolutions(cached_record, [item], require_current_item_evidence=True),
            [{"id": 103, "source": "apple", "text": "Apple text", "scope": "full", "reviewed_by": "human"}],
        )

    def test_unavailable_provenance_keeps_compiler_running_and_creates_human_review(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            apple = root / "apple.txt"
            whisper = root / "whisper.txt"
            apple.write_text("Start use 5 grams now. The protocol uses 5 grams.", encoding="utf-8")
            whisper.write_text("Start use 6 grams now. The protocol uses 5 grams.", encoding="utf-8")
            episode = {
                "episode_key": "episode-1",
                "files": {"sources": {"apple": {"text": "apple.txt", "metadata": None}, "whisper": {"text": "whisper.txt", "metadata": None}}},
            }
            saved = []
            unavailable = PresetProvenance(status="unavailable", reason="network_or_timeout", slug=REVIEW_PRESET)
            with (
                patch("podcast_engine.compilation.RUNTIME_DIR", root / "runtime"),
                patch("podcast_engine.compilation._local_copy", side_effect=[apple, whisper]),
                patch("podcast_engine.compilation._load_review_record", return_value=None),
                patch("podcast_engine.compilation.verify_transcript_reviewer", return_value=unavailable),
                patch("podcast_engine.compilation.resolve_compiler_batch") as resolver,
                patch("podcast_engine.compilation._save_review_record", side_effect=lambda _path, record: saved.append(record) or _path),
                patch("podcast_engine.compilation.upload_path_to_gcs", side_effect=lambda _path, gcs_path: gcs_path),
            ):
                outcome = compile_episode_sources(episode)

        resolver.assert_not_called()
        self.assertGreater(outcome["review_required"], 0)
        self.assertEqual(saved[-1]["preset_provenance"]["status"], "unavailable")

    def test_deterministic_compilation_without_ai_work_never_verifies_preset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            apple = root / "apple.txt"
            whisper = root / "whisper.txt"
            apple.write_text("Identical deterministic transcript.", encoding="utf-8")
            whisper.write_text("Identical deterministic transcript.", encoding="utf-8")
            episode = {
                "episode_key": "episode-no-ai",
                "files": {"sources": {"apple": {"text": "apple.txt", "metadata": None}, "whisper": {"text": "whisper.txt", "metadata": None}}},
            }
            saved = []
            with (
                patch("podcast_engine.compilation.RUNTIME_DIR", root / "runtime"),
                patch("podcast_engine.compilation._local_copy", side_effect=[apple, whisper]),
                patch("podcast_engine.compilation._load_review_record", return_value=None),
                patch("podcast_engine.compilation.verify_transcript_reviewer") as verify,
                patch("podcast_engine.compilation.resolve_compiler_batch") as resolver,
                patch("podcast_engine.compilation._save_review_record", side_effect=lambda _path, record: saved.append(record) or _path),
                patch("podcast_engine.compilation.upload_path_to_gcs", side_effect=lambda _path, gcs_path: gcs_path),
            ):
                outcome = compile_episode_sources(episode)

        verify.assert_not_called()
        resolver.assert_not_called()
        self.assertEqual(outcome["review_required"], 0)
        self.assertEqual(saved[-1]["preset_provenance"]["status"], "not_required")

    def test_recompile_rebuild_links_result_generation_before_terminal_completion(self):
        apple_text = "Start use 5 grams now. The protocol uses 5 grams."
        whisper_text = "Start use 6 grams now. The protocol uses 5 grams."
        result = compile_transcripts(apple_text, whisper_text)
        batch = build_resolver_batch(result, "episode-human-only")
        review_item = _human_review_items(
            result,
            batch,
            {"accepted": [], "review": [], "outcomes": []},
        )[0]
        record = {
            "human_review_generation_fingerprint": "sha256:original-review",
            "human_decisions": [{
                "id": review_item["id"],
                "chosen_source": "apple",
                "chosen_text": review_item["apple_text"],
                "scope": review_item["focus"]["scope"],
                "reviewed_by": "human",
                "review_item": review_item,
            }],
            "recompile_requests": [{
                "request_id": "request-123",
                "review_generation_fingerprint": "sha256:original-review",
                "status": "started",
            }],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            apple = root / "apple.txt"
            whisper = root / "whisper.txt"
            apple.write_text(apple_text, encoding="utf-8")
            whisper.write_text(whisper_text, encoding="utf-8")
            episode = {
                "episode_key": "episode-human-only",
                "files": {"sources": {"apple": {"text": "apple.txt", "metadata": None}, "whisper": {"text": "whisper.txt", "metadata": None}}},
            }
            saved = []
            with (
                patch("podcast_engine.compilation.RUNTIME_DIR", root / "runtime"),
                patch("podcast_engine.compilation._local_copy", side_effect=[apple, whisper]),
                patch("podcast_engine.compilation._load_review_record", return_value=record),
                patch("podcast_engine.compilation.verify_transcript_reviewer") as verify,
                patch("podcast_engine.compilation.resolve_compiler_batch") as resolver,
                patch(
                    "podcast_engine.compilation._save_review_record",
                    side_effect=lambda _path, saved_record: saved.append(saved_record) or _path,
                ),
                patch("podcast_engine.compilation.upload_path_to_gcs", side_effect=lambda _path, gcs_path: gcs_path),
                patch.dict(
                    os.environ,
                    {
                        "PODCAST_RECOMPILE_REQUEST_ID": "request-123",
                        "PODCAST_REVIEW_GENERATION": "sha256:original-review",
                        "PODCAST_RECOMPILE_EPISODE_KEY": "episode-human-only",
                    },
                    clear=False,
                ),
            ):
                outcome = compile_episode_sources(episode)

        verify.assert_not_called()
        resolver.assert_not_called()
        self.assertEqual(outcome["review_required"], 0)
        request = saved[-1]["recompile_requests"][0]
        self.assertEqual(request["status"], "started")
        self.assertNotEqual(
            saved[-1]["human_review_generation_fingerprint"],
            "sha256:original-review",
        )
        self.assertEqual(
            request["result_review_generation_fingerprint"],
            saved[-1]["human_review_generation_fingerprint"],
        )

    def test_review_gate_requires_current_policy_and_source_bytes(self):
        episode = {
            "files": {
                "sources": {
                    "apple": {"text": "apple.txt", "metadata": None},
                    "whisper": {"text": "whisper.txt", "metadata": None},
                }
            }
        }
        source_fingerprint = _review_source_fingerprint(
            apple_text=b"Apple current",
            apple_metadata=None,
            whisper_text=b"Whisper current",
            whisper_metadata=None,
        )
        record = self._record(source_fingerprint=source_fingerprint)

        with patch(
            "podcast_engine.compilation.download_gcs_bytes",
            side_effect=[b"Apple current", b"Whisper current"],
        ):
            self.assertTrue(review_record_is_current(episode, record))

        old_policy = dict(record)
        old_policy["policy_version"] = "resolver-policy-v2"
        with patch(
            "podcast_engine.compilation.download_gcs_bytes",
            side_effect=[b"Apple current", b"Whisper current"],
        ):
            self.assertFalse(review_record_is_current(episode, old_policy))

        with patch(
            "podcast_engine.compilation.download_gcs_bytes",
            side_effect=[b"Apple current", b"Whisper changed"],
        ):
            self.assertFalse(review_record_is_current(episode, record))

    def test_stale_zero_decision_record_replaces_obsolete_review_cards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            apple = root / "apple.txt"
            whisper = root / "whisper.txt"
            apple.write_text("Start 0.2 grams end", encoding="utf-8")
            whisper.write_text("Start two grams end", encoding="utf-8")
            episode = {
                "episode_key": "episode-1",
                "files": {
                    "sources": {
                        "apple": {"text": "apple.txt", "metadata": None},
                        "whisper": {"text": "whisper.txt", "metadata": None},
                    }
                },
            }
            stale_record = self._record()
            stale_record["policy_version"] = "resolver-policy-v2"
            stale_record["human_review"] = [{"id": 999, "reason": "obsolete"}]
            stale_record["human_decisions"] = []
            saved = []

            with (
                patch("podcast_engine.compilation.RUNTIME_DIR", root / "runtime"),
                patch(
                    "podcast_engine.compilation._local_copy",
                    side_effect=[apple, whisper],
                ),
                patch("podcast_engine.compilation._load_review_record", return_value=stale_record),
                patch("podcast_engine.compilation.verify_transcript_reviewer", return_value=self._provenance()),
                patch(
                    "podcast_engine.compilation.resolve_compiler_batch",
                    return_value={"accepted": [], "review": [], "outcomes": []},
                ),
                patch(
                    "podcast_engine.compilation._save_review_record",
                    side_effect=lambda _path, record: saved.append(record) or _path,
                ),
                patch(
                    "podcast_engine.compilation.upload_path_to_gcs",
                    side_effect=lambda _path, gcs_path: gcs_path,
                ),
            ):
                outcome = compile_episode_sources(episode)

        self.assertFalse(outcome["review_cache_hit"])
        self.assertGreater(outcome["review_required"], 0)
        self.assertEqual(saved[-1]["human_review"], outcome["review"])
        self.assertNotIn({"id": 999, "reason": "obsolete"}, saved[-1]["human_review"])
        self.assertEqual(saved[-1]["human_decisions"], [])


    def test_source_byte_change_changes_fingerprint(
        self,
    ):
        batch = {
            "batch": {
                "episode_id": (
                    "episode-1"
                ),
                "diff_items": [
                    {"id": 103},
                ],
            },
            "response_schema": {
                "type": "object",
            },
        }

        with tempfile.TemporaryDirectory() as directory:

            root = Path(
                directory
            )

            apple = (
                root
                / "apple.txt"
            )

            whisper = (
                root
                / "whisper.txt"
            )

            apple.write_text(
                "Creatine is useful.",
                encoding="utf-8",
            )

            whisper.write_text(
                "Creatine is useful.",
                encoding="utf-8",
            )

            first = (
                _review_input_fingerprint(
                    apple_text_path=(
                        apple
                    ),
                    apple_metadata_path=(
                        None
                    ),
                    whisper_text_path=(
                        whisper
                    ),
                    whisper_metadata_path=(
                        None
                    ),
                    resolver_batch=(
                        batch
                    ),
                    preset_provenance=self._provenance(),
                )
            )

            whisper.write_text(
                "Creatine was useful.",
                encoding="utf-8",
            )

            second = (
                _review_input_fingerprint(
                    apple_text_path=(
                        apple
                    ),
                    apple_metadata_path=(
                        None
                    ),
                    whisper_text_path=(
                        whisper
                    ),
                    whisper_metadata_path=(
                        None
                    ),
                    resolver_batch=(
                        batch
                    ),
                    preset_provenance=self._provenance(),
                )
            )

        self.assertNotEqual(
            first,
            second,
        )

    def test_remote_preset_policy_version_change_invalidates_fingerprint(self):
        batch = {
            "batch": {"episode_id": "episode-1", "diff_items": []},
            "response_schema": {"type": "object"},
        }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            apple = root / "apple.txt"
            whisper = root / "whisper.txt"
            apple.write_text("Stable transcript.", encoding="utf-8")
            whisper.write_text("Stable transcript.", encoding="utf-8")

            before = _review_input_fingerprint(
                apple_text_path=apple,
                apple_metadata_path=None,
                whisper_text_path=whisper,
                whisper_metadata_path=None,
                resolver_batch=batch,
                preset_provenance=self._provenance(),
            )
            with patch(
                "podcast_engine.compilation.REVIEW_POLICY_VERSION",
                REVIEW_POLICY_VERSION + "-changed",
            ):
                after = _review_input_fingerprint(
                    apple_text_path=apple,
                    apple_metadata_path=None,
                    whisper_text_path=whisper,
                    whisper_metadata_path=None,
                    resolver_batch=batch,
                    preset_provenance=self._provenance(),
                )

        self.assertNotEqual(before, after)


    def test_high_risk_source_only_review_path_is_symmetric_and_resolvable(self):
        whisper_metadata = {
            "segments": [{
                "start": 10.0,
                "end": 12.0,
                "avg_logprob": -0.2,
                "no_speech_prob": 0.01,
                "words": [
                    {"word": "Start", "start": 10.0, "end": 10.2, "probability": 0.99},
                    {"word": "5", "start": 10.3, "end": 10.4, "probability": 0.99},
                    {"word": "grams", "start": 10.4, "end": 10.7, "probability": 0.99},
                    {"word": "end", "start": 10.8, "end": 11.0, "probability": 0.99},
                ],
            }],
        }

        cases = (
            {
                "name": "whisper_only",
                "apple": "Start end",
                "whisper": "Start 5 grams end",
                "metadata": whisper_metadata,
                "evidence_source": "whisper",
                "timestamp_field": "whisper_start_timestamp",
            },
            {
                "name": "apple_only",
                "apple": "0:10\nStart 5 grams end",
                "whisper": "Start end",
                "metadata": None,
                "evidence_source": "apple",
                "timestamp_field": "apple_timestamp",
            },
        )

        for case in cases:
            with self.subTest(case=case["name"]):
                compile_kwargs = {"merge_gap": 0}
                if case["metadata"] is not None:
                    compile_kwargs["whisper_metadata"] = case["metadata"]

                result = compile_transcripts(
                    case["apple"],
                    case["whisper"],
                    **compile_kwargs,
                )
                difference = next(
                    item for item in result.differences if item.source_only
                )
                batch = build_resolver_batch(result, f"review-{case['name']}")

                # High-risk one-sided evidence has no independent textual
                # corroboration here, so it bypasses Gemini and becomes a
                # direct Human Review item.
                self.assertEqual(batch["batch"]["diff_items"], [])
                review = _human_review_items(
                    result,
                    batch,
                    {"accepted": [], "review": []},
                )
                self.assertEqual(len(review), 1)
                self.assertEqual(review[0]["id"], difference.id)
                self.assertEqual(review[0]["category"], "unit")
                self.assertIsNotNone(review[0][case["timestamp_field"]])

                evidence_text = (
                    difference.apple_text
                    if case["evidence_source"] == "apple"
                    else difference.whisper_text
                )
                accepted = compile_transcripts(
                    case["apple"],
                    case["whisper"],
                    resolver_resolutions=[{
                        "id": difference.id,
                        "source": case["evidence_source"],
                        "text": evidence_text,
                        "reviewed_by": "human",
                    }],
                    **compile_kwargs,
                )
                self.assertIn("5 grams", accepted.compiled_transcript)
                self.assertEqual(accepted.review_required, 0)

                other_source = (
                    "whisper" if case["evidence_source"] == "apple" else "apple"
                )
                other_text = (
                    difference.whisper_text
                    if other_source == "whisper"
                    else difference.apple_text
                )
                rejected = compile_transcripts(
                    case["apple"],
                    case["whisper"],
                    resolver_resolutions=[{
                        "id": difference.id,
                        "source": other_source,
                        "text": other_text,
                        "reviewed_by": "human",
                    }],
                    **compile_kwargs,
                )
                self.assertNotIn("5 grams", rejected.compiled_transcript)
                self.assertEqual(rejected.review_required, 0)

    def test_audit_record_keeps_decisions_and_identity(
        self,
    ):
        resolution = {
            "accepted": [
                {
                    "id": 223,
                    "source": "apple",
                    "text": "creatine",
                }
            ],
            "review": [
                {
                    "id": 246,
                    "reason": (
                        "needs_human_review"
                    ),
                }
            ],
            "completion_metadata": {
                "completion_id": "chatcmpl-1",
                "served_model": "google/gemini-test",
                "served_provider": "Test Provider",
            },
        }

        human_review = [
            {
                "id": 35,
                "reason": (
                    "compiler_requires_human_review"
                ),
            },
            {
                "id": 246,
                "reason": (
                    "needs_human_review"
                ),
            },
        ]

        batch = {
            "batch": {
                "diff_items": [
                    {"id": 223},
                    {"id": 246},
                ],
            },
        }

        sources = {
            "apple": {
                "text": (
                    "episodes/key/sources/apple/transcript.txt"
                ),
                "metadata": (
                    "episodes/key/sources/apple/transcript.json"
                ),
            },
            "whisper": {
                "text": (
                    "episodes/key/sources/whisper/transcript.txt"
                ),
                "metadata": (
                    "episodes/key/sources/whisper/transcript.json"
                ),
            },
        }

        record = (
            _build_review_record(
                episode_key="key",
                input_fingerprint=(
                    "sha256:abc"
                ),
                source_fingerprint="sha256:sources",
                resolution=(
                    resolution
                ),
                human_review=(
                    human_review
                ),
                resolver_batch=(
                    batch
                ),
                source_paths=(
                    sources
                ),
                preset_provenance=self._provenance(),
            )
        )

        self.assertEqual(
            record[
                "schema_version"
            ],
            REVIEW_RECORD_SCHEMA_VERSION,
        )

        self.assertEqual(
            record[
                "policy_version"
            ],
            REVIEW_POLICY_VERSION,
        )

        self.assertEqual(
            record[
                "reviewer_preset"
            ],
            REVIEW_PRESET,
        )

        self.assertEqual(
            record[
                "input_fingerprint"
            ],
            "sha256:abc",
        )

        self.assertEqual(record["source_fingerprint"], "sha256:sources")
        self.assertEqual(
            record["human_review_queue_fingerprint"],
            review_queue_fingerprint(human_review),
        )

        self.assertEqual(
            record[
                "accepted"
            ],
            resolution[
                "accepted"
            ],
        )

        self.assertEqual(
            record[
                "resolver_review"
            ],
            resolution[
                "review"
            ],
        )
        self.assertEqual(
            record["preset_provenance"]["completion_id"],
            "chatcmpl-1",
        )
        self.assertEqual(
            record["preset_provenance"]["served_model"],
            "google/gemini-test",
        )
        self.assertEqual(
            record["preset_provenance"]["served_provider"],
            "Test Provider",
        )
        provenance = record["preset_provenance"]
        for field in (
            "status",
            "preset_slug",
            "preset_id",
            "version_id",
            "version",
            "config_sha256",
            "system_prompt_sha256",
            "verified_at",
            "completion_id",
            "served_model",
            "served_provider",
        ):
            self.assertIn(field, provenance)
        self.assertNotIn("system_prompt", provenance)
        self.assertNotIn("request_id", provenance)
        self.assertNotIn("Prompt", json.dumps(record))

        self.assertEqual(
            record[
                "human_review"
            ],
            human_review,
        )

        self.assertEqual(
            record[
                "resolver_item_ids"
            ],
            [
                223,
                246,
            ],
        )

        self.assertIn(
            "reviewed_at",
            record,
        )


class ReviewTriagePersistenceTests(unittest.TestCase):
    def _provenance(self):
        return PresetProvenance(
            status="verified",
            slug=REVIEW_PRESET,
            preset_id="preset-id",
            version_id="version-id-4",
            version=4,
            config={"model": "test/model", "temperature": 0},
            system_prompt="Prompt",
            config_digest="sha256:config",
            system_prompt_digest="sha256:prompt",
            verified_at="2026-08-30T00:00:00+00:00",
        )

    def _triage_batch(self, *, text="attention"):
        return {
            "response_schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "triage": {
                        "type": "array",
                    },
                },
                "required": ["triage"],
            },
            "deferred_ids": [],
            "batch": {
                "diff_items": [
                    {
                        "id": 246,
                        "kind": "transcription_difference",
                        "severity": "medium",
                        "category": "other",
                        "source_only": False,
                        "apple_text": "tension",
                        "whisper_text": text,
                        "apple_context": "before tension after",
                        "whisper_context": f"before {text} after",
                    }
                ],
            },
        }

    def _triage_result(self):
        return {
            "accepted": [],
            "recommendations": [
                {
                    "id": 246,
                    "recommendation": "recommend_whisper",
                    "source": "whisper",
                    "text": "attention",
                    "confidence": "high",
                    "reason": "Whisper better matches the supplied local evidence",
                }
            ],
            "outcomes": [
                {
                    "id": 246,
                    "status": "advisory",
                    "recommendation": "recommend_whisper",
                    "confidence": "high",
                    "reason": "Whisper better matches the supplied local evidence",
                }
            ],
            "completion_metadata": {
                "completion_id": "triage-completion-1",
                "served_model": "test/model",
                "served_provider": "Test Provider",
            },
        }

    def test_compile_episode_sources_persists_triage_parse_failure_diagnostics(self):
        raw_content = '{"triage": ['
        diagnostics = {
            "finish_reason": "length",
            "completion_id": "completion-truncated",
            "served_model": "google/gemini-3.6-flash",
            "served_provider": "Google AI Studio",
            "content_chars": len(raw_content),
            "content_sha256": "b" * 64,
            "raw_content": raw_content,
        }
        expected = {
            "finish_reason": "length",
            "completion_id": "completion-truncated",
            "served_model": "google/gemini-3.6-flash",
            "served_provider": "Google AI Studio",
            "content_chars": len(raw_content),
            "content_sha256": "b" * 64,
        }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            apple = root / "apple.txt"
            whisper = root / "whisper.txt"
            apple.write_text("Stable transcript.", encoding="utf-8")
            whisper.write_text("Stable transcript.", encoding="utf-8")
            episode = {
                "episode_key": "episode-triage-parse-failure",
                "files": {
                    "sources": {
                        "apple": {"text": "apple.txt", "metadata": None},
                        "whisper": {"text": "whisper.txt", "metadata": None},
                    }
                },
            }
            saved = []

            with (
                patch("podcast_engine.compilation.RUNTIME_DIR", root / "runtime"),
                patch(
                    "podcast_engine.compilation._local_copy",
                    side_effect=[apple, whisper],
                ),
                patch(
                    "podcast_engine.compilation._load_review_record",
                    return_value=None,
                ),
                patch(
                    "podcast_engine.compilation.build_triage_batch",
                    return_value=self._triage_batch(),
                ),
                patch(
                    "podcast_engine.compilation.verify_transcript_reviewer",
                    return_value=self._provenance(),
                ),
                patch(
                    "podcast_engine.compilation.resolve_compiler_batch",
                    return_value={
                        "accepted": [],
                        "review": [],
                        "outcomes": [],
                    },
                ),
                patch(
                    "podcast_engine.compilation.triage_compiler_batch",
                    side_effect=TriageResponseParseError(diagnostics),
                ),
                patch(
                    "podcast_engine.compilation._save_review_record",
                    side_effect=lambda _path, record: saved.append(record) or _path,
                ),
                patch(
                    "podcast_engine.compilation.upload_path_to_gcs",
                    side_effect=lambda _path, gcs_path: gcs_path,
                ),
            ):
                compile_episode_sources(episode)

        self.assertTrue(saved)
        self.assertEqual(
            saved[-1]["triage_failure_metadata"],
            expected,
        )
        serialized = json.dumps(saved[-1], sort_keys=True)
        self.assertNotIn("raw_content", serialized)
        self.assertNotIn(raw_content, serialized)

    def test_triage_contract_changes_review_input_fingerprint(self):
        resolver_batch = {
            "batch": {
                "episode_id": "episode-1",
                "diff_items": [],
            },
            "response_schema": {
                "type": "object",
            },
        }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            apple = root / "apple.txt"
            whisper = root / "whisper.txt"
            apple.write_text("Stable transcript.", encoding="utf-8")
            whisper.write_text("Stable transcript.", encoding="utf-8")

            baseline = _review_input_fingerprint(
                apple_text_path=apple,
                apple_metadata_path=None,
                whisper_text_path=whisper,
                whisper_metadata_path=None,
                resolver_batch=resolver_batch,
                triage_batch=self._triage_batch(),
                preset_provenance=self._provenance(),
            )
            changed = _review_input_fingerprint(
                apple_text_path=apple,
                apple_metadata_path=None,
                whisper_text_path=whisper,
                whisper_metadata_path=None,
                resolver_batch=resolver_batch,
                triage_batch=self._triage_batch(text="different"),
                preset_provenance=self._provenance(),
            )

        self.assertNotEqual(baseline, changed)

    def test_unavailable_triage_persists_only_safe_failure_metadata(self):
        raw_content = '{"triage": ['
        supplied = {
            "finish_reason": "length",
            "completion_id": "completion-truncated",
            "served_model": "google/gemini-3.6-flash",
            "served_provider": "Google AI Studio",
            "content_chars": len(raw_content),
            "content_sha256": "a" * 64,
            "raw_content": raw_content,
        }
        expected = {
            "finish_reason": "length",
            "completion_id": "completion-truncated",
            "served_model": "google/gemini-3.6-flash",
            "served_provider": "Google AI Studio",
            "content_chars": len(raw_content),
            "content_sha256": "a" * 64,
        }

        triage = _unavailable_triage(
            self._triage_batch(),
            failure_metadata=supplied,
        )

        sources = {
            "apple": {
                "text": "episodes/key/sources/apple/transcript.txt",
                "metadata": None,
            },
            "whisper": {
                "text": "episodes/key/sources/whisper/transcript.txt",
                "metadata": None,
            },
        }

        record = _build_review_record(
            episode_key="key",
            input_fingerprint="sha256:abc",
            source_fingerprint="sha256:sources",
            resolution={
                "accepted": [],
                "review": [],
                "outcomes": [],
            },
            triage=triage,
            human_review=[
                {
                    "id": 246,
                    "reason": "compiler_requires_human_review",
                }
            ],
            resolver_batch={"batch": {"diff_items": []}},
            triage_batch=self._triage_batch(),
            source_paths=sources,
            preset_provenance=self._provenance(),
        )

        self.assertEqual(
            record["triage_failure_metadata"],
            expected,
        )

        serialized = json.dumps(record, sort_keys=True)
        self.assertNotIn("raw_content", serialized)
        self.assertNotIn(raw_content, serialized)

    def test_durable_record_persists_safe_per_chunk_failure_metadata(self):
        raw_content = '{"triage": ['
        triage = {
            "accepted": [],
            "recommendations": [],
            "outcomes": [
                {
                    "id": 246,
                    "status": "unavailable",
                    "recommendation": "needs_audio",
                    "reason": "triage_unavailable",
                }
            ],
            "failure_metadata": [
                {
                    "item_ids": [246],
                    "error_type": "TriageResponseParseError",
                    "finish_reason": "length",
                    "completion_id": "completion-truncated",
                    "served_model": "google/gemini-3.6-flash",
                    "served_provider": "Google AI Studio",
                    "content_chars": len(raw_content),
                    "content_sha256": "c" * 64,
                    "raw_content": raw_content,
                    "unexpected": "must-not-persist",
                }
            ],
        }
        expected = [
            {
                "item_ids": [246],
                "error_type": "TriageResponseParseError",
                "finish_reason": "length",
                "completion_id": "completion-truncated",
                "served_model": "google/gemini-3.6-flash",
                "served_provider": "Google AI Studio",
                "content_chars": len(raw_content),
                "content_sha256": "c" * 64,
            }
        ]
        sources = {
            "apple": {
                "text": "episodes/key/sources/apple/transcript.txt",
                "metadata": None,
            },
            "whisper": {
                "text": "episodes/key/sources/whisper/transcript.txt",
                "metadata": None,
            },
        }

        record = _build_review_record(
            episode_key="key",
            input_fingerprint="sha256:abc",
            source_fingerprint="sha256:sources",
            resolution={
                "accepted": [],
                "review": [],
                "outcomes": [],
            },
            triage=triage,
            human_review=[
                {
                    "id": 246,
                    "reason": "compiler_requires_human_review",
                }
            ],
            resolver_batch={"batch": {"diff_items": []}},
            triage_batch=self._triage_batch(),
            source_paths=sources,
            preset_provenance=self._provenance(),
        )

        self.assertEqual(
            record["triage_failure_metadata"],
            expected,
        )

        serialized = json.dumps(record, sort_keys=True)
        self.assertNotIn("raw_content", serialized)
        self.assertNotIn(raw_content, serialized)
        self.assertNotIn("must-not-persist", serialized)

    def test_durable_record_persists_triage_evidence_separately(self):
        resolution = {
            "accepted": [],
            "review": [],
            "outcomes": [],
        }
        triage = self._triage_result()
        human_review = [
            {
                "id": 246,
                "reason": "compiler_requires_human_review",
            }
        ]
        sources = {
            "apple": {
                "text": "episodes/key/sources/apple/transcript.txt",
                "metadata": None,
            },
            "whisper": {
                "text": "episodes/key/sources/whisper/transcript.txt",
                "metadata": None,
            },
        }

        record = _build_review_record(
            episode_key="key",
            input_fingerprint="sha256:abc",
            source_fingerprint="sha256:sources",
            resolution=resolution,
            triage=triage,
            human_review=human_review,
            resolver_batch={"batch": {"diff_items": []}},
            triage_batch=self._triage_batch(),
            source_paths=sources,
            preset_provenance=self._provenance(),
        )

        self.assertEqual(record["accepted"], [])
        self.assertEqual(record["triage_item_ids"], [246])
        self.assertEqual(
            record["triage_recommendations"],
            triage["recommendations"],
        )
        self.assertEqual(
            record["triage_outcomes"],
            triage["outcomes"],
        )
        self.assertEqual(
            record["triage_completion_metadata"],
            triage["completion_metadata"],
        )

    def test_human_review_projection_exposes_triage_separately_from_suggestion(self):
        result = compile_transcripts(
            "Start use 5 grams now. The protocol uses 5 grams.",
            "Start use 6 grams now. The protocol uses 5 grams.",
        )
        difference = next(
            item
            for item in result.differences
            if item.review_required
        )
        batch = build_resolver_batch(
            result,
            "triage-projection",
        )
        triage = {
            "accepted": [],
            "recommendations": [
                {
                    "id": difference.id,
                    "recommendation": "recommend_whisper",
                    "source": "whisper",
                    "text": difference.whisper_text,
                    "confidence": "high",
                    "reason": "Whisper better matches the supplied local evidence",
                }
            ],
            "outcomes": [
                {
                    "id": difference.id,
                    "status": "advisory",
                    "recommendation": "recommend_whisper",
                    "confidence": "high",
                    "reason": "Whisper better matches the supplied local evidence",
                }
            ],
        }

        review = _human_review_items(
            result,
            batch,
            {
                "accepted": [],
                "review": [],
                "outcomes": [],
            },
            triage=triage,
        )

        self.assertEqual(len(review), 1)
        self.assertEqual(
            review[0]["triage"],
            {
                "status": "advisory",
                "recommendation": "recommend_whisper",
                "source": "whisper",
                "text": difference.whisper_text,
                "confidence": "high",
                "reason": "Whisper better matches the supplied local evidence",
            },
        )
        self.assertEqual(
            review[0]["suggestion"]["source"],
            difference.selected_source,
        )
        self.assertFalse(
            review[0]["suggestion"]["automatic_resolution"],
        )

    def test_triage_transport_failure_preserves_human_review(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            apple = root / "apple.txt"
            whisper = root / "whisper.txt"
            apple.write_text(
                "Start use 5 grams now. The protocol uses 5 grams.",
                encoding="utf-8",
            )
            whisper.write_text(
                "Start use 6 grams now. The protocol uses 5 grams.",
                encoding="utf-8",
            )
            episode = {
                "episode_key": "episode-triage-failure",
                "files": {
                    "sources": {
                        "apple": {
                            "text": "apple.txt",
                            "metadata": None,
                        },
                        "whisper": {
                            "text": "whisper.txt",
                            "metadata": None,
                        },
                    }
                },
            }
            saved = []

            with (
                patch(
                    "podcast_engine.compilation.RUNTIME_DIR",
                    root / "runtime",
                ),
                patch(
                    "podcast_engine.compilation._local_copy",
                    side_effect=[apple, whisper],
                ),
                patch(
                    "podcast_engine.compilation._load_review_record",
                    return_value=None,
                ),
                patch(
                    "podcast_engine.compilation.verify_transcript_reviewer",
                    return_value=self._provenance(),
                ),
                patch(
                    "podcast_engine.compilation.resolve_compiler_batch",
                    return_value={
                        "accepted": [],
                        "review": [],
                        "outcomes": [],
                    },
                ),
                patch(
                    "podcast_engine.compilation.triage_compiler_batch",
                    side_effect=RuntimeError("triage transport failed"),
                    create=True,
                ) as triage_call,
                patch(
                    "podcast_engine.compilation._save_review_record",
                    side_effect=lambda _path, record: saved.append(record) or _path,
                ),
                patch(
                    "podcast_engine.compilation.upload_path_to_gcs",
                    side_effect=lambda _path, gcs_path: gcs_path,
                ),
            ):
                outcome = compile_episode_sources(episode)

        triage_call.assert_called_once()
        self.assertGreater(outcome["review_required"], 0)
        self.assertEqual(saved[-1]["accepted"], [])
        self.assertTrue(saved[-1]["triage_outcomes"])
        self.assertTrue(
            all(
                item["status"] == "unavailable"
                and item["recommendation"] == "needs_audio"
                for item in saved[-1]["triage_outcomes"]
            )
        )
        self.assertTrue(
            all(
                item.get("triage", {}).get("status") == "unavailable"
                for item in saved[-1]["human_review"]
            )
        )


class ReviewSourceFingerprintDelegatesToSharedGenerationIdentityTests(
    unittest.TestCase
):
    """TASK-076 Task 2: the resolver's source_fingerprint contract now
    delegates to the shared episode source-generation identity, so both
    must keep producing byte-for-byte identical output for identical
    source bytes.
    """

    def test_review_source_fingerprint_matches_shared_generation_identity(self):
        self.assertEqual(
            _review_source_fingerprint(
                apple_text=b"Apple current",
                apple_metadata=b"Apple metadata",
                whisper_text=b"Whisper current",
                whisper_metadata=None,
            ),
            source_generation_fingerprint(
                apple_text=b"Apple current",
                apple_metadata=b"Apple metadata",
                whisper_text=b"Whisper current",
                whisper_metadata=None,
            ),
        )


if __name__ == "__main__":
    unittest.main()
