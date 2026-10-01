import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from compiler.review_policy import ASSISTED_REVIEW_POLICY_VERSION
from compiler.transcript import compile_transcripts
from podcast_engine.human_review import (
    ThirdAsrInFlight,
    ensure_audio_clip,
    ensure_third_asr,
    record_human_decision,
    validated_human_resolutions,
)
from podcast_engine.review_alignment import align_transcripts
from podcast_engine.review_audio import (
    _audio_request_payload,
    clip_window,
    parse_third_asr_response,
    third_asr_cache_key,
    transcribe_review_clip,
)
from podcast_engine.review_web import create_review_app


def review_item(difference_id=1, *, partial=False):
    item = {
        "id": difference_id,
        "category": "protocol_number",
        "severity": "high",
        "reason": "needs_human_review",
        "apple_text": "0.2 grams",
        "whisper_text": "two grams",
        "apple_context": "Apple context",
        "whisper_context": "Whisper context",
        "whisper_start_timestamp": 100.0,
        "whisper_end_timestamp": 101.0,
    }
    if partial:
        item["apple_text"] = "gonna have messed up RPE"
        item["whisper_text"] = "gonna have messed up RP"
        item["focus"] = {"scope": "partial", "apple_text": "RPE", "whisper_text": "RP"}
    return item


def _anchorable_third_item():
    item = review_item()
    item["apple_context"] = "for most lifters about 0.2 grams per pound of body weight each day"
    item["whisper_context"] = "for most lifters about two grams per pound of body weight each day"
    item["third_asr"] = {"text": "So for most lifters, about 0.25 grams per pound of body weight each day is plenty."}
    return item


def _assisted_shape_item(difference_id=200, **overrides):
    """A card shaped for compiler.review_policy.assisted_routing().

    TASK-076 Task 11: used to prove Detailed Review decisions snapshot
    routing provenance without the routing computation itself ever
    influencing which transcript text a decision chooses.
    """

    item = {
        "id": difference_id,
        "kind": "wording_difference",
        "category": "other",
        "apple_text": "the study included Dr Alvarez",
        "whisper_text": "the study included doctor Alvarez",
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


class HumanReviewTests(unittest.TestCase):
    def test_anchor_sliding_alignment_is_monotonic_and_locates_changed_span(self):
        apple = "one two anchor words stay aligned here then RPE after another anchor words stay aligned again"
        whisper = "one two anchor words stay aligned here then RP after another anchor words stay aligned again"
        alignment = align_transcripts(apple, whisper)

        mapped = list(alignment.apple_to_whisper.items())
        self.assertTrue(all(left[0] < right[0] and left[1] < right[1] for left, right in zip(mapped, mapped[1:])))
        located = alignment.locate_apple_span(8, 9)
        self.assertIn(located["method"], {"interpolated_anchors", "left_anchor", "right_anchor", "matched_tokens"})
        self.assertGreaterEqual(located["whisper_start_word"], 1)

    def test_audio_clip_window_is_centered_and_bounded(self):
        window = clip_window(100.0, 101.0, duration=105.0)
        self.assertEqual(window, {"start": 93.0, "end": 105.0, "duration": 12.0})
        long = clip_window(10.0, 40.0)
        self.assertEqual(long["duration"], 15.0)
        self.assertEqual(long["start"], 17.5)

    def test_audio_clip_uses_rss_enclosure_and_ephemeral_storage(self):
        episode = {"episode_key": "episode-1", "audio_url": "https://cdn.example.test/episode.mp3"}
        with patch("podcast_engine.human_review.extract_audio_clip") as extract:
            clip, _ = ensure_audio_clip(episode, review_item(), fingerprint="sha256:stable")
        try:
            self.assertEqual(extract.call_args.args[0], "https://cdn.example.test/episode.mp3")
            self.assertTrue(str(clip).endswith(".wav"))
            self.assertNotIn(Path.cwd(), clip.parents)
        finally:
            clip.unlink(missing_ok=True)

    def test_third_asr_request_and_response_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "review.wav"
            audio.write_bytes(b"RIFFfake")
            payload = _audio_request_payload(audio)
        self.assertEqual(payload["model"], "openai/gpt-transcribe")
        self.assertEqual(payload["input_audio"]["format"], "wav")
        self.assertTrue(payload["input_audio"]["data"])
        self.assertEqual(parse_third_asr_response({"text": "  confirmed words ", "usage": {"cost": 0.001}})["text"], "confirmed words")
        with self.assertRaises(ValueError):
            parse_third_asr_response({"usage": {}})

    def test_third_asr_uses_only_dedicated_third_asr_key(self):
        response = Mock()
        response.json.return_value = {"text": "audio evidence", "usage": {"seconds": 12}}
        response.raise_for_status.return_value = None
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"PODCAST_REVIEW_ASR_API_KEY": "third-asr-key", "PODCAST_TRANSCRIPT_REVIEW_API_KEY": "review-key", "PODCAST_KNOWLEDGE_API_KEY": "summary-key"}, clear=False), patch("podcast_engine.review_audio.requests.post", return_value=response) as post:
            audio = Path(directory) / "review.wav"
            audio.write_bytes(b"RIFFfake")
            evidence = transcribe_review_clip(audio)
        self.assertEqual(evidence["text"], "audio evidence")
        self.assertEqual(post.call_args.kwargs["headers"]["Authorization"], "Bearer third-asr-key")

    def test_human_decision_persists_separately_and_partial_hybrid_applies(self):
        record = {"episode_key": "episode-1", "human_decisions": [], "human_review": [review_item(partial=True)]}
        saved = []
        with patch("podcast_engine.human_review.load_review_record", return_value=record), patch("podcast_engine.human_review.save_review_record", side_effect=lambda key, value: saved.append(value)):
            decision = record_human_decision("episode-1", 1, source="apple")

        self.assertEqual(decision["chosen_text"], "RPE")
        self.assertEqual(record["human_review"], [])
        self.assertEqual(record["human_decisions"][0]["reviewed_by"], "human")
        resolutions = validated_human_resolutions(record, [review_item(partial=True)])
        result = compile_transcripts(
            "gonna have messed up RPE",
            "gonna have messed up RP",
            primary="whisper",
            resolver_resolutions=resolutions,
        )
        self.assertEqual(result.compiled_transcript, "gonna have messed up RPE")
        self.assertTrue(saved)

    def test_detailed_review_decision_snapshots_routing_provenance_without_changing_outcome(self):
        # TASK-076 Task 11: every Detailed Review decision snapshots the
        # current routing-policy version and exclusion reason codes for
        # later corpus analysis, but routing is provenance only -- it must
        # never change which source/text a decision actually records.
        cases = {
            "eligible": ({}, []),
            "protected_proper_name": ({"category": "proper_name"}, ["protected_proper_name"]),
            "anomaly": ({"anomaly": {"kind": "duplicate"}}, ["anomaly"]),
            "span_over_limit": (
                {
                    "apple_text": " ".join(f"word{i}" for i in range(12)),
                    "whisper_text": " ".join(f"term{i}" for i in range(12)),
                },
                ["span_over_limit"],
            ),
        }
        for label, (overrides, expected_reason_codes) in cases.items():
            with self.subTest(label=label):
                item = _assisted_shape_item(200, **overrides)
                record = {"episode_key": "episode-1", "human_decisions": [], "human_review": [item]}
                saved = []
                with (
                    patch("podcast_engine.human_review.load_review_record", return_value=record),
                    patch(
                        "podcast_engine.human_review.save_review_record",
                        side_effect=lambda key, value: saved.append(value),
                    ),
                ):
                    decision = record_human_decision("episode-1", 200, source="apple")

                self.assertEqual(decision["chosen_source"], "apple")
                self.assertEqual(decision["chosen_text"], item["apple_text"])
                self.assertEqual(
                    decision["routing_provenance"],
                    {
                        "policy_version": ASSISTED_REVIEW_POLICY_VERSION,
                        "reason_codes": expected_reason_codes,
                    },
                )

    def test_cached_third_asr_does_not_rerun_for_unchanged_card(self):
        # TASK-076 Task 9: ensure_third_asr now requires canonical
        # episode_key/source_fingerprint identity (for its claim/budget
        # collaborators) even on the cache-hit path, so this fixture uses
        # valid-format identifiers rather than the old placeholder strings.
        # The cache hit itself must short-circuit before touching budget,
        # pricing, or claim machinery at all -- covered separately (with
        # the full first-run flow) in tests/test_third_asr_budget.py.
        episode_key = "a" * 24
        fingerprint = "sha256:" + "b" * 64
        item = review_item()
        window = clip_window(item["whisper_start_timestamp"], item["whisper_end_timestamp"])
        cache_key = third_asr_cache_key(input_fingerprint=fingerprint, item_id=1, window=window)
        item["third_asr"] = {"text": "third source", "cache_key": cache_key}
        record = {
            "episode_key": episode_key,
            "input_fingerprint": fingerprint,
            "source_fingerprint": "sha256:" + "c" * 64,
            "human_review": [item],
        }

        with patch(
            "podcast_engine.human_review.load_review_record_with_generation",
            return_value=(record, 7),
        ), patch("podcast_engine.human_review.save_review_record") as save, patch(
            "podcast_engine.human_review.transcribe_review_clip"
        ) as third, patch("podcast_engine.human_review.reserve_budget_batch") as reserve, patch(
            "podcast_engine.human_review.acquire_third_asr_claim"
        ) as acquire:
            episode = {"episode_key": episode_key}
            first = ensure_third_asr(episode, 1)
            second = ensure_third_asr(episode, 1)

        self.assertEqual(first, second)
        self.assertEqual(first["text"], "third source")
        third.assert_not_called()
        reserve.assert_not_called()
        acquire.assert_not_called()
        save.assert_not_called()

    def test_third_asr_in_flight_is_a_non_blocking_web_response(self):
        episode = {"episode_key": "episode-1", "title": "Episode One", "status": {"compiler": {"state": "review_required"}}}
        in_flight = ThirdAsrInFlight(retry_after_seconds=42.5, cache_key="sha256:" + "c" * 64)
        with patch("podcast_engine.review_web.load_episodes", return_value=[episode]), patch(
            "podcast_engine.review_web.ensure_third_asr", side_effect=in_flight
        ):
            response = create_review_app().test_client().post(
                "/api/review/episodes/episode-1/items/1/third-asr"
            )

        self.assertEqual(response.status_code, 202)
        body = response.get_json()
        self.assertEqual(body["status"], "in_flight")
        self.assertAlmostEqual(body["retry_after_seconds"], 42.5)

    def test_human_resolution_removes_the_pipeline_gate(self):
        record = {"human_decisions": [{"id": 1, "chosen_source": "custom", "chosen_text": "0.2 grams", "scope": "full", "reviewed_by": "human"}]}
        current = [review_item()]
        resolutions = validated_human_resolutions(record, current)
        unresolved = [item for item in current if item["id"] not in {choice["id"] for choice in resolutions}]
        self.assertEqual(unresolved, [])
        self.assertEqual(resolutions[0]["source"], "human")

    def test_stale_lifecycle_reuses_only_decisions_with_unchanged_review_text(self):
        original = review_item()
        decision = {
            "id": 1,
            "chosen_source": "custom",
            "chosen_text": "0.2 grams",
            "scope": "full",
            "reviewed_by": "human",
            "review_item": original,
        }
        record = {"human_decisions": [decision]}

        self.assertEqual(
            validated_human_resolutions(
                record,
                [review_item()],
                require_current_item_evidence=True,
            ),
            [{"id": 1, "source": "human", "text": "0.2 grams", "scope": "full", "reviewed_by": "human"}],
        )

        changed = review_item()
        changed["whisper_text"] = "two point zero grams"
        self.assertEqual(
            validated_human_resolutions(
                record,
                [changed],
                require_current_item_evidence=True,
            ),
            [],
        )

    def test_empty_source_choice_is_a_valid_human_omission_resolution(self):
        omission = review_item(35)
        omission["apple_text"] = "Yeah"
        omission["whisper_text"] = ""
        record = {"human_decisions": [{"id": 35, "chosen_source": "whisper", "chosen_text": "", "scope": "full", "reviewed_by": "human"}]}

        resolutions = validated_human_resolutions(record, [omission])

        self.assertEqual(resolutions, [{"id": 35, "source": "whisper", "text": "", "scope": "full", "reviewed_by": "human"}])

    def test_full_scope_source_choices_persist_the_exact_source_text(self):
        record = {"episode_key": "episode-1", "human_decisions": [], "human_review": [review_item()]}
        with patch("podcast_engine.human_review.load_review_record", return_value=record), patch("podcast_engine.human_review.save_review_record"):
            decision = record_human_decision("episode-1", 1, source="apple")

        self.assertEqual(decision["chosen_source"], "apple")
        self.assertEqual(decision["chosen_text"], "0.2 grams")

    def test_empty_omission_source_can_be_selected_and_persisted(self):
        omission = review_item()
        omission["apple_text"] = "Yeah"
        omission["whisper_text"] = ""
        record = {"episode_key": "episode-1", "human_decisions": [], "human_review": [omission]}
        with patch("podcast_engine.human_review.load_review_record", return_value=record), patch("podcast_engine.human_review.save_review_record"):
            decision = record_human_decision("episode-1", 1, source="whisper")

        self.assertEqual(decision["chosen_text"], "")

    def test_third_choice_requires_persisted_nonempty_evidence(self):
        record = {"episode_key": "episode-1", "human_decisions": [], "human_review": [review_item()]}
        with patch("podcast_engine.human_review.load_review_record", return_value=record), patch("podcast_engine.human_review.save_review_record") as save:
            with self.assertRaisesRegex(ValueError, "Run third ASR"):
                record_human_decision("episode-1", 1, source="third")

        save.assert_not_called()

    def test_third_choice_persists_only_the_anchored_clip_words(self):
        # TASK-123: the clip transcript covers the whole audio window; only the
        # words between this card's own context anchors may replace its span.
        item = _anchorable_third_item()
        record = {"episode_key": "episode-1", "human_decisions": [], "human_review": [item]}
        with patch("podcast_engine.human_review.load_review_record", return_value=record), patch("podcast_engine.human_review.save_review_record"):
            decision = record_human_decision("episode-1", 1, source="third")

        self.assertEqual(decision["chosen_text"], "0.25 grams")
        self.assertEqual(decision["chosen_source"], "third")

    def test_whole_clip_text_is_not_accepted_as_third(self):
        item = _anchorable_third_item()
        record = {"episode_key": "episode-1", "human_decisions": [], "human_review": [item]}
        with patch("podcast_engine.human_review.load_review_record", return_value=record), patch("podcast_engine.human_review.save_review_record") as save:
            with self.assertRaisesRegex(ValueError, "persisted third-ASR"):
                record_human_decision("episode-1", 1, source="third", text=item["third_asr"]["text"])

        save.assert_not_called()

    def test_unaligned_third_evidence_cannot_be_chosen(self):
        item = review_item()
        item["third_asr"] = {"text": "Independent evidence"}
        record = {"episode_key": "episode-1", "human_decisions": [], "human_review": [item]}
        with patch("podcast_engine.human_review.load_review_record", return_value=record), patch("podcast_engine.human_review.save_review_record") as save:
            with self.assertRaisesRegex(ValueError, "could not be aligned"):
                record_human_decision("episode-1", 1, source="third")

        save.assert_not_called()

    def test_edited_third_evidence_is_not_accepted_as_third(self):
        item = _anchorable_third_item()
        record = {"episode_key": "episode-1", "human_decisions": [], "human_review": [item]}
        with patch("podcast_engine.human_review.load_review_record", return_value=record), patch("podcast_engine.human_review.save_review_record") as save:
            with self.assertRaisesRegex(ValueError, "persisted third-ASR"):
                record_human_decision("episode-1", 1, source="third", text="Edited evidence")

        save.assert_not_called()

    def test_custom_edit_requires_nonempty_text(self):
        record = {"episode_key": "episode-1", "human_decisions": [], "human_review": [review_item()]}
        with patch("podcast_engine.human_review.load_review_record", return_value=record), patch("podcast_engine.human_review.save_review_record") as save:
            with self.assertRaisesRegex(ValueError, "Custom/Edit requires"):
                record_human_decision("episode-1", 1, source="custom", text="  ")

        save.assert_not_called()

    def test_local_web_endpoints_expose_the_review_data_contract(self):
        record = {"episode_key": "episode-1", "human_review": [review_item()]}
        episode = {"episode_key": "episode-1", "title": "Episode One", "status": {"compiler": {"state": "review_required"}}}
        with patch("podcast_engine.review_web.load_episodes", return_value=[episode]), patch("podcast_engine.review_web.load_review_record", return_value=record):
            client = create_review_app().test_client()
            listing = client.get("/api/review/episodes")
            detail = client.get("/api/review/episodes/episode-1")
            page = client.get("/")
        self.assertEqual(listing.status_code, 200)
        self.assertEqual(listing.get_json()["episodes"][0]["pending_count"], 1)
        self.assertEqual(detail.get_json()["cards"][0]["id"], 1)
        self.assertIn(b"Use Apple", page.data)
        self.assertIn(b"data-source=whisper", page.data)
        self.assertIn(b"Run third ASR", page.data)
        self.assertIn(b"Selected replacement", page.data)
        self.assertIn(b"Confirm selection", page.data)
        self.assertIn(b"Optional audit note", page.data)
        self.assertIn(b"Evidence cached", page.data)
        self.assertNotIn(b"Run third ASR again", page.data)

    def test_generic_error_handler_returns_safe_responses_without_reraising(self):
        app = create_review_app()

        @app.get("/unexpected")
        def unexpected_page():
            raise RuntimeError("page failure")

        @app.get("/api/unexpected")
        def unexpected_api():
            raise RuntimeError("api failure")

        client = app.test_client()
        page = client.get("/unexpected")
        api = client.get("/api/unexpected")

        self.assertEqual(page.status_code, 500)
        self.assertEqual(page.get_data(as_text=True), "Internal Server Error")
        self.assertEqual(api.status_code, 500)
        self.assertEqual(api.get_json(), {"error": "Request failed (500): Internal Server Error"})

    def test_review_presentation_labels_do_not_change_persisted_enums(self):
        item = review_item()
        item.update({"category": "other", "severity": "medium", "reason": "compiler_requires_human_review"})
        record = {"episode_key": "episode-1", "human_review": [item]}
        with patch("podcast_engine.review_web.load_review_record", return_value=record):
            response = create_review_app().test_client().get("/api/review/episodes/episode-1")

        body = response.get_json()
        self.assertEqual(body["record"]["human_review"][0]["category"], "other")
        self.assertEqual(body["cards"][0]["display"], {
            "category": "General transcript difference",
            "severity": "Needs review",
            "reason": "Compiler requires human review",
        })
        self.assertEqual(body["progress"], {"total": 1, "reviewed": 0, "remaining": 1, "assisted_unprepared": 0, "triage_unavailable": 0})

    def test_review_progress_counts_cards_with_unavailable_triage(self):
        # TASK-123: a silent whole-run triage outage must be visible in the UI.
        unavailable = review_item(1)
        unavailable["triage"] = {"status": "unavailable", "recommendation": "needs_audio", "reason": "triage_unavailable"}
        advisory = review_item(2)
        advisory["triage"] = {"status": "advisory", "recommendation": "recommend_apple", "confidence": "high", "reason": "clear"}
        record = {"episode_key": "episode-1", "human_review": [unavailable, advisory]}
        with patch("podcast_engine.review_web.load_review_record", return_value=record):
            body = create_review_app().test_client().get("/api/review/episodes/episode-1").get_json()

        self.assertEqual(body["progress"]["triage_unavailable"], 1)

    def test_review_card_presents_only_the_anchored_third_window(self):
        aligned = _anchorable_third_item()
        unaligned = review_item(2)
        unaligned["third_asr"] = {"text": "Independent evidence"}
        record = {"episode_key": "episode-1", "human_review": [aligned, unaligned]}
        with patch("podcast_engine.review_web.load_review_record", return_value=record):
            cards = create_review_app().test_client().get("/api/review/episodes/episode-1").get_json()["cards"]

        self.assertEqual(cards[0]["third_window"], "0.25 grams")
        self.assertTrue(cards[0]["third_available"])
        self.assertIsNone(cards[1]["third_window"])
        self.assertFalse(cards[1]["third_available"])
        # The full clip transcript stays visible as evidence.
        self.assertEqual(cards[0]["third_asr"]["text"], aligned["third_asr"]["text"])

    def test_third_asr_expected_failure_returns_sanitized_json(self):
        episode = {"episode_key": "episode-1"}
        with patch("podcast_engine.review_web.load_episodes", return_value=[episode]), patch("podcast_engine.review_web.ensure_third_asr", side_effect=RuntimeError("Missing PODCAST_REVIEW_ASR_API_KEY for third-asr review")):
            response = create_review_app().test_client().post("/api/review/episodes/episode-1/items/1/third-asr")

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.get_json(), {"error": "Third ASR is not configured."})
        self.assertNotIn(b"PODCAST_REVIEW_ASR_API_KEY", response.data)

    def test_audio_route_returns_wav_and_failure_stays_structured(self):
        record = {"episode_key": "episode-1", "input_fingerprint": "sha256:stable", "source_fingerprint": "sha256:stable", "human_review": [review_item()]}
        episode = {"episode_key": "episode-1"}
        with tempfile.TemporaryDirectory() as directory:
            clip = Path(directory) / "clip.wav"
            clip.write_bytes(b"RIFFtest")
            with patch("podcast_engine.review_web.load_episodes", return_value=[episode]), patch("podcast_engine.review_web.load_review_record", return_value=record), patch("podcast_engine.review_web.ensure_audio_clip", return_value=(clip, {"start": 94.0, "end": 106.0, "duration": 12.0})):
                response = create_review_app().test_client().get("/api/review/episodes/episode-1/items/1/audio")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.mimetype, "audio/wav")
            self.assertEqual(response.data, b"RIFFtest")
            response.close()

        with patch("podcast_engine.review_web.load_episodes", return_value=[episode]), patch("podcast_engine.review_web.load_review_record", return_value=record), patch("podcast_engine.review_web.ensure_audio_clip", side_effect=RuntimeError("FFmpeg review clip extraction failed: details that should not reach UI")):
            failed = create_review_app().test_client().get("/api/review/episodes/episode-1/items/1/audio")
        self.assertEqual(failed.status_code, 422)
        self.assertEqual(failed.get_json(), {"error": "Review audio is unavailable: FFmpeg review clip extraction failed."})

    def test_api_404_is_json_not_flask_html(self):
        response = create_review_app().test_client().get("/api/review/not-a-route")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.get_json(), {"error": "Request failed (404): Not Found"})

    def test_web_decision_endpoint_accepts_the_whisper_button_source(self):
        record = {"episode_key": "episode-1", "human_review": [review_item()]}
        with patch("podcast_engine.review_web.record_human_decision", return_value={"chosen_source": "whisper"}) as decide, patch("podcast_engine.review_web.load_review_record", return_value=record):
            response = create_review_app().test_client().post("/api/review/episodes/episode-1/items/1/decision", json={"source": "whisper"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(decide.call_args.kwargs["source"], "whisper")

    def test_web_recompile_endpoint_requires_no_pending_cards_and_starts_worker(self):
        record = {"episode_key": "episode-1", "human_review": []}
        episode = {"episode_key": "episode-1", "status": {"compiler": {"state": "review_required"}}}
        with patch("podcast_engine.review_web.load_episodes", return_value=[episode]), patch("podcast_engine.review_web.request_worker_recompile", return_value={"operation": "operations/worker-1"}) as start:
            response = create_review_app().test_client().post("/api/review/episodes/episode-1/recompile")
        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.get_json()["recompile"]["operation"], "operations/worker-1")
        self.assertEqual(start.call_args.kwargs["requested_by"], "local-human-review")

    def test_completed_review_page_keeps_recompile_disabled_after_reload(self):
        record = {
            "episode_key": "episode-1",
            "human_review": [],
            "human_review_generation_fingerprint": "sha256:rebuilt",
            "recompile_requests": [{
                "request_id": "request-123",
                "review_generation_fingerprint": "sha256:original",
                "status": "completed",
                "result_review_generation_fingerprint": "sha256:rebuilt",
            }],
        }
        episode = {"episode_key": "episode-1", "status": {"compiler": {"state": "review_required"}}}
        with patch("podcast_engine.review_web.load_review_record", return_value=record), patch(
            "podcast_engine.review_web.load_episodes", return_value=[episode]
        ):
            page = create_review_app().test_client().get("/")
            response = create_review_app().test_client().get("/api/review/episodes/episode-1")
        self.assertEqual(page.status_code, 200)
        self.assertEqual(response.get_json()["recompile_status"], "completed")
        self.assertIn(b"function recompileStatus()", page.data)
        self.assertIn(b"Recompile completed", page.data)
        self.assertIn(b"record?.recompile_status", page.data)
        self.assertIn(b"${active?'disabled':''}", page.data)

    def test_independent_future_review_generation_is_not_blocked_by_old_completion(self):
        record = {
            "episode_key": "episode-1",
            "human_review": [],
            "human_review_generation_fingerprint": "sha256:future",
            "recompile_requests": [{
                "request_id": "request-123",
                "review_generation_fingerprint": "sha256:original",
                "result_review_generation_fingerprint": "sha256:rebuilt",
                "status": "completed",
            }],
        }
        with patch("podcast_engine.review_web.load_review_record", return_value=record):
            response = create_review_app().test_client().get("/api/review/episodes/episode-1")

        self.assertIsNone(response.get_json()["recompile_status"])

    def test_google_login_gates_the_cloud_review_ui_to_allowed_email(self):
        record = {"episode_key": "episode-1", "human_review": [review_item()]}
        episode = {"episode_key": "episode-1", "title": "Episode One", "status": {"compiler": {"state": "review_required"}}}
        environment = {
            "REVIEW_REQUIRE_AUTH": "true",
            "GOOGLE_OAUTH_CLIENT_ID": "client-id.apps.googleusercontent.com",
            "REVIEW_SESSION_SECRET": "session-secret",
            "REVIEW_ALLOWED_EMAIL": "reviewer@example.com",
        }
        with patch.dict(os.environ, environment, clear=False), patch("podcast_engine.review_web.load_episodes", return_value=[episode]), patch("podcast_engine.review_web.load_review_record", return_value=record), patch("podcast_engine.review_web.id_token.verify_oauth2_token", return_value={"email": "reviewer@example.com", "email_verified": True}):
            client = create_review_app().test_client()
            self.assertEqual(client.get("/").status_code, 302)
            self.assertIn(b"client-id.apps.googleusercontent.com", client.get("/login").data)
            logged_in = client.post("/auth/google", json={"credential": "google-id-token"})
            detail = client.get("/api/review/episodes/episode-1", base_url="https://localhost")
        self.assertEqual(logged_in.status_code, 200)
        self.assertEqual(detail.status_code, 200)

    def test_tag_vocabulary_routes_are_in_the_authenticated_review_surface(self):
        environment = {
            "REVIEW_REQUIRE_AUTH": "true",
            "GOOGLE_OAUTH_CLIENT_ID": "client-id.apps.googleusercontent.com",
            "REVIEW_SESSION_SECRET": "session-secret",
        }
        with patch.dict(os.environ, environment, clear=False), patch(
            "podcast_engine.review_web.TagRegistry"
        ) as registry:
            client = create_review_app().test_client()
            response = client.post(
                "/api/tag-vocabulary/candidates/lengthened-partials/decision",
                json={"action": "promote"},
                base_url="https://localhost",
            )
            page = client.get("/tags", base_url="https://localhost")
        self.assertEqual(response.status_code, 401)
        self.assertEqual(page.status_code, 302)
        registry.assert_not_called()


if __name__ == "__main__":
    unittest.main()
