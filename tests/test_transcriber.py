import contextlib
from contextlib import contextmanager
from decimal import Decimal
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from podcast_engine import transcriber
from podcast_engine.ai_pricing import AudioPricing
from podcast_engine.episode_contract import paths_for
from podcast_engine.transcriber import AudioChunk


class TranscriberTests(unittest.TestCase):
    def test_chunk_ranges_include_five_second_overlap(self):
        # TASK-125: 600 s chunks keep each request within OpenRouter's
        # upload and processing limits.
        ranges = transcriber._chunk_ranges(4036.519256)

        self.assertEqual(len(ranges), 7)
        self.assertEqual(ranges[0], (0.0, 605.0))
        self.assertEqual(ranges[1], (600.0, 605.0))
        self.assertEqual(ranges[5], (3000.0, 605.0))
        self.assertAlmostEqual(ranges[6][0], 3600.0)
        self.assertAlmostEqual(ranges[6][1], 436.519256)

    def test_midpoint_stitch_handles_conflicting_overlap_words(self):
        previous = [
            {
                "text": (
                    "totally mess up people's sleep "
                    "for eight weeks in a way"
                ),
                "start": 1199.72,
                "end": 1203.14,
                "words": [
                    _word(
                        " totally",
                        1199.72,
                        1200.10,
                    ),
                    _word(
                        " mess",
                        1200.10,
                        1200.44,
                    ),
                    _word(
                        " up",
                        1200.44,
                        1200.70,
                    ),
                    _word(
                        " people's",
                        1200.70,
                        1201.18,
                    ),
                    _word(
                        " sleep",
                        1201.18,
                        1201.42,
                    ),
                    _word(
                        " for",
                        1201.42,
                        1201.66,
                    ),
                    _word(
                        " eight",
                        1201.66,
                        1201.94,
                    ),
                    _word(
                        " weeks",
                        1201.94,
                        1202.28,
                    ),
                    _word(
                        " in",
                        1202.28,
                        1202.88,
                    ),
                    _word(
                        " a",
                        1202.88,
                        1203.00,
                    ),
                    _word(
                        " way",
                        1203.00,
                        1203.14,
                    ),
                ],
            }
        ]

        current = [
            {
                "text": (
                    "really mess up people's sleep "
                    "for eight weeks in a way that "
                    "seems plausible"
                ),
                "start": 1200.0,
                "end": 1204.54,
                "words": [
                    _word(
                        " really",
                        1200.00,
                        1200.20,
                    ),
                    _word(
                        " mess",
                        1200.20,
                        1200.46,
                    ),
                    _word(
                        " up",
                        1200.46,
                        1200.72,
                    ),
                    _word(
                        " people's",
                        1200.72,
                        1201.24,
                    ),
                    _word(
                        " sleep",
                        1201.24,
                        1201.40,
                    ),
                    _word(
                        " for",
                        1201.40,
                        1201.66,
                    ),
                    _word(
                        " eight",
                        1201.66,
                        1201.96,
                    ),
                    _word(
                        " weeks",
                        1201.96,
                        1202.32,
                    ),
                    _word(
                        " in",
                        1202.32,
                        1202.88,
                    ),
                    _word(
                        " a",
                        1202.88,
                        1203.00,
                    ),
                    _word(
                        " way",
                        1203.00,
                        1203.14,
                    ),
                    _word(
                        " that",
                        1203.14,
                        1203.34,
                    ),
                    _word(
                        " seems",
                        1203.34,
                        1203.92,
                    ),
                    _word(
                        " plausible.",
                        1203.92,
                        1204.54,
                    ),
                ],
            }
        ]

        stitched, info = (
            transcriber._stitch_chunk_overlap(
                previous,
                current,
                chunk_start=1200.0,
            )
        )

        words = [
            word["word"].strip()
            for segment in stitched
            for word in segment["words"]
        ]

        self.assertEqual(
            info["cut_time"],
            1202.5,
        )

        self.assertEqual(
            words,
            [
                "totally",
                "mess",
                "up",
                "people's",
                "sleep",
                "for",
                "eight",
                "weeks",
                "in",
                "a",
                "way",
                "that",
                "seems",
                "plausible.",
            ],
        )

        transcriber._validate_monotonic_timestamps(
            stitched
        )

    def test_midpoint_stitch_produces_monotonic_word_timeline(self):
        previous = [
            {
                "text": "hard to overcome sleep deficit",
                "start": 3599.82,
                "end": 3602.42,
                "words": [
                    _word(
                        " hard",
                        3599.82,
                        3600.16,
                    ),
                    _word(
                        " to",
                        3600.16,
                        3600.38,
                    ),
                    _word(
                        " overcome",
                        3600.38,
                        3600.80,
                    ),
                    _word(
                        " sleep",
                        3601.20,
                        3601.46,
                    ),
                    _word(
                        " deficit",
                        3601.46,
                        3601.98,
                    ),
                    _word(
                        " is",
                        3601.98,
                        3602.42,
                    ),
                ],
            }
        ]

        current = [
            {
                "text": (
                    "or to overcome sleep deficit is "
                    "impacting my performance"
                ),
                "start": 3600.0,
                "end": 3603.42,
                "words": [
                    _word(
                        " or",
                        3600.00,
                        3600.18,
                    ),
                    _word(
                        " to",
                        3600.18,
                        3600.36,
                    ),
                    _word(
                        " overcome",
                        3600.36,
                        3600.82,
                    ),
                    _word(
                        " sleep",
                        3601.22,
                        3601.46,
                    ),
                    _word(
                        " deficit",
                        3601.46,
                        3602.02,
                    ),
                    _word(
                        " is",
                        3602.02,
                        3602.42,
                    ),
                    _word(
                        " impacting",
                        3602.42,
                        3602.76,
                    ),
                    _word(
                        " my",
                        3602.76,
                        3603.02,
                    ),
                    _word(
                        " performance",
                        3603.02,
                        3603.42,
                    ),
                ],
            }
        ]

        stitched, _ = (
            transcriber._stitch_chunk_overlap(
                previous,
                current,
                chunk_start=3600.0,
            )
        )

        transcriber._validate_monotonic_timestamps(
            stitched
        )

        starts = [
            word["start"]
            for segment in stitched
            for word in segment["words"]
        ]

        self.assertEqual(
            starts,
            sorted(starts),
        )

    def test_overlap_preserves_untimestamped_word_with_segment_fallback(self):
        previous = [
            {
                "text": "before",
                "start": 10.0,
                "end": 12.0,
                "words": [_word(" before", 10.0, 10.4)],
            }
        ]
        current = [
            {
                "text": "evidence after",
                "start": 12.0,
                "end": 13.0,
                "words": [
                    {"word": " evidence", "start": None, "end": None},
                    _word(" after", 12.7, 13.0),
                ],
            }
        ]

        stitched, _ = transcriber._stitch_chunk_overlap(
            previous, current, chunk_start=10.0
        )

        self.assertEqual(
            [segment["text"] for segment in stitched], ["before", "evidence after"]
        )
        self.assertEqual(
            [word["word"].strip() for word in stitched[-1]["words"]],
            ["evidence", "after"],
        )

    def test_overlap_retains_unique_untimestamped_evidence_on_wrong_side_of_cut(self):
        previous = [{"text": "before", "start": 10.0, "end": 10.4,
                     "words": [_word(" before", 10.0, 10.4)]}]
        current = [{"text": "evidence after", "start": 10.0, "end": 13.0,
                    "words": [
                        {"word": " evidence", "start": None, "end": None},
                        _word(" after", 12.7, 13.0),
                    ]}]
        stitched, _ = transcriber._stitch_chunk_overlap(previous, current, chunk_start=10.0)
        self.assertEqual(
            [word["word"].strip() for segment in stitched for word in segment["words"]],
            ["before", "evidence", "after"],
        )

    def test_overlap_deduplicates_equivalent_untimestamped_evidence(self):
        previous = [{"text": "evidence", "start": 11.0, "end": 13.0,
                     "words": [{"word": " evidence", "start": None, "end": None}]}]
        current = [{"text": "evidence after", "start": 10.0, "end": 13.0,
                    "words": [
                        {"word": " evidence", "start": None, "end": None},
                        _word(" after", 12.7, 13.0),
                    ]}]
        stitched, _ = transcriber._stitch_chunk_overlap(previous, current, chunk_start=10.0)
        words = [word["word"].strip() for segment in stitched for word in segment["words"]]
        self.assertEqual(words, ["evidence", "after"])
        self.assertEqual(words.count("evidence"), 1)

    def test_overlap_preserves_order_and_available_timestamps_for_mixed_words(self):
        previous = [{"text": "before evidence", "start": 10.0, "end": 13.0,
                     "words": [
                         _word(" before", 10.0, 10.4),
                         {"word": " evidence", "start": None, "end": None},
                     ]}]
        current = [{"text": "after", "start": 12.7, "end": 13.0,
                    "words": [_word(" after", 12.7, 13.0)]}]
        stitched, _ = transcriber._stitch_chunk_overlap(previous, current, chunk_start=10.0)
        words = [word for segment in stitched for word in segment["words"]]
        self.assertEqual([word["word"].strip() for word in words], ["before", "evidence", "after"])
        self.assertEqual(words[0]["start"], 10.0)
        self.assertIsNone(words[1]["start"])
        self.assertEqual(words[2]["start"], 12.7)
        transcriber._validate_monotonic_timestamps(stitched)

    def test_overlap_does_not_collapse_same_common_word_in_distinct_contexts(self):
        previous = [{
            "text": "the result",
            "start": 10.5,
            "end": 12.0,
            "words": [
                {"word": " the", "start": None, "end": None},
                _word(" result", 11.4, 12.0),
            ],
        }]
        current = [{
            "text": "later the participants",
            "start": 12.6,
            "end": 14.5,
            "words": [
                _word(" later", 12.6, 12.9),
                {"word": " the", "start": None, "end": None},
                _word(" participants", 13.8, 14.5),
            ],
        }]

        stitched, _ = transcriber._stitch_chunk_overlap(
            previous, current, chunk_start=10.0
        )
        words = [
            word["word"].strip()
            for segment in stitched
            for word in segment["words"]
        ]

        self.assertEqual(words.count("the"), 2)
        self.assertEqual(
            words,
            ["the", "result", "later", "the", "participants"],
        )

    def test_overlap_preserves_multiplicity_for_repeated_untimestamped_words(self):
        previous = [{
            "text": "very very",
            "start": 11.0,
            "end": 13.0,
            "words": [
                {"word": " very", "start": None, "end": None},
                {"word": " very", "start": None, "end": None},
            ],
        }]
        current = [{
            "text": "very very after",
            "start": 11.0,
            "end": 13.5,
            "words": [
                {"word": " very", "start": None, "end": None},
                {"word": " very", "start": None, "end": None},
                _word(" after", 13.0, 13.5),
            ],
        }]

        stitched, _ = transcriber._stitch_chunk_overlap(
            previous, current, chunk_start=10.0
        )
        words = [
            word["word"].strip()
            for segment in stitched
            for word in segment["words"]
        ]

        self.assertEqual(words, ["very", "very", "after"])
        self.assertEqual(words.count("very"), 2)

    def test_substantial_empty_chunks_are_suspicious_at_every_position(self):
        chunk = AudioChunk(path=Path("chunk.wav"), start=1200.0, duration=120.0)
        for chunk_number, chunk_count in ((1, 1), (1, 2), (2, 2), (2, 3), (3, 3)):
            with self.subTest(chunk_number=chunk_number, chunk_count=chunk_count):
                self.assertTrue(transcriber._suspicious_empty_chunk(
                    chunk, chunk_number=chunk_number, chunk_count=chunk_count, segments=[]
                ))
        self.assertFalse(
            transcriber._suspicious_empty_chunk(
                AudioChunk(path=Path("short.wav"), start=1200.0, duration=5.0),
                chunk_number=3,
                chunk_count=3,
                segments=[],
            )
        )

    def test_empty_substantial_middle_chunk_stops_before_upload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "episode.mp3"
            audio.write_bytes(b"fake audio")
            chunk_dir = root / "chunks"
            chunk_dir.mkdir()
            chunks = []
            for index in range(3):
                path = chunk_dir / f"chunk_{index:03d}.mp3"
                path.write_bytes(b"chunk")
                chunks.append(AudioChunk(path=path, start=index * 120.0, duration=120.0))

            def respond(chunk, **_):
                if chunk.path.name == "chunk_001.mp3":
                    return _stt_response([], language="en")
                return _stt_response([(1.0, 2.0, " speech")], language="en")

            with (
                _openrouter_environment(),
                patch.object(transcriber, "_transcribe_chunk", side_effect=respond),
                patch.object(
                    transcriber,
                    "_prepare_audio_chunks",
                    return_value=(chunk_dir, chunks, 360.0),
                ),
                patch.object(
                    transcriber,
                    "upload_path_to_gcs",
                    side_effect=AssertionError("incomplete transcript must not upload"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "substantial chunk"):
                    transcriber.transcribe_audio(str(audio), "episode-key")

    def test_validator_rejects_backwards_word_timestamps(self):
        segments = [
            {
                "start": 10.0,
                "end": 12.0,
                "words": [
                    _word(
                        " first",
                        10.0,
                        10.4,
                    ),
                    _word(
                        " second",
                        9.9,
                        10.2,
                    ),
                ],
            }
        ]

        with self.assertRaises(
            RuntimeError
        ):
            transcriber._validate_monotonic_timestamps(
                segments
            )

    def test_chunked_transcription_preserves_global_timestamps(self):
        episode_key = "0123456789abcdef01234567"

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "episode.mp3"
            audio.write_bytes(b"fake audio")
            chunk_dir = root / "chunks"
            chunk_dir.mkdir()
            chunk_0 = chunk_dir / "chunk_000.mp3"
            chunk_1 = chunk_dir / "chunk_001.mp3"
            chunk_0.write_bytes(b"chunk 0")
            chunk_1.write_bytes(b"chunk 1")

            uploaded = {}
            upload_calls = []
            requests_sent = []

            def upload(local, destination):
                upload_calls.append(destination)
                uploaded[destination] = Path(local).read_text(encoding="utf-8")
                return destination

            def post(url, *, headers, json, timeout):
                requests_sent.append((url, headers, json, timeout))
                if len(requests_sent) == 1:
                    # Qwen3-ASR names the language ("english"); the hint is the ISO code.
                    return _FakeResponse(_stt_response([(1.0, 2.0, " First chunk.")], language="english"))
                return _FakeResponse(_stt_response([(3.0, 4.0, " Second chunk.")], language="en"))

            with (
                _openrouter_environment() as budget,
                patch.object(transcriber.requests, "post", side_effect=post),
                patch.object(
                    transcriber,
                    "_prepare_audio_chunks",
                    return_value=(
                        chunk_dir,
                        [
                            AudioChunk(path=chunk_0, start=0.0, duration=15.0),
                            AudioChunk(path=chunk_1, start=10.0, duration=5.0),
                        ],
                        15.0,
                    ),
                ),
                patch.object(transcriber, "WHISPER_METADATA_SCHEMA_VERSION", 9, create=True),
                patch.object(transcriber, "upload_path_to_gcs", side_effect=upload),
            ):
                result = transcriber.transcribe_audio(str(audio), episode_key)

            # Every request names the manifest model. Provider routing is not
            # sent: OpenRouter ignores it for transcription, and the key's
            # guardrail pins the provider instead.
            self.assertEqual(len(requests_sent), 2)
            for url, headers, body, timeout in requests_sent:
                self.assertEqual(url, transcriber.STT_ENDPOINT)
                self.assertEqual(headers["Authorization"], "Bearer test-whisper-key")
                self.assertEqual(body["model"], "qwen/qwen3-asr-1.7b")
                self.assertNotIn("provider", body)
                self.assertEqual(body["response_format"], "verbose_json")
                self.assertEqual(body["timestamp_granularities"], ["word", "segment"])
                self.assertEqual(body["input_audio"]["format"], "mp3")
                self.assertEqual(timeout, transcriber.REQUEST_TIMEOUT_SECONDS)
            self.assertNotIn("language", requests_sent[0][2])
            self.assertEqual(requests_sent[1][2]["language"], "en")

            # Each physical request is admitted, then settled at its reported cost.
            reserve, settle = budget["reserve"], budget["settle"]
            self.assertEqual(reserve.call_count, 2)
            self.assertEqual(settle.call_count, 2)
            request = reserve.call_args_list[0].args[2][0]
            self.assertEqual(request["stage"], "whisper_transcription")
            self.assertEqual(request["reserved_usd"], Decimal("15") * Decimal("0.0000075"))
            ledger_key = reserve.call_args_list[0].args[1]
            self.assertRegex(ledger_key, r"^sha256:[0-9a-f]{64}$")
            self.assertEqual(settle.call_args_list[0].kwargs["actual_usd"], Decimal("0.0001"))

            metadata = json.loads(uploaded[paths_for(episode_key)["whisper_metadata"]])
            self.assertEqual(metadata["schema_version"], 9)
            self.assertEqual(metadata["source"], "openrouter-whisper")
            self.assertEqual(metadata["producer"], transcriber.current_whisper_producer())
            self.assertEqual(metadata["audio_fingerprint"], ledger_key)
            self.assertEqual(metadata["billed_usd"], "0.0002")
            self.assertEqual(uploaded[paths_for(episode_key)["whisper_text"]], "First chunk. Second chunk.")
            self.assertEqual(metadata["chunk_count"], 2)
            self.assertEqual(metadata["chunk_overlap_seconds"], 5)
            self.assertEqual(metadata["deduplication"], "timestamp_midpoint_overlap_v4")
            self.assertEqual(metadata["duration"], 15.0)
            self.assertEqual(metadata["language"], "en")
            self.assertEqual(metadata["segments"][0]["start"], 1.0)
            # Chunk 2 begins globally at 10 seconds, so local 3.0 becomes 13.0.
            self.assertEqual(metadata["segments"][1]["start"], 13.0)
            self.assertEqual(metadata["segments"][1]["words"][0]["start"], 13.0)
            self.assertEqual(metadata["segments"][1]["avg_logprob"], -0.2)
            self.assertEqual(metadata["segments"][1]["no_speech_prob"], 0.01)
            self.assertEqual(metadata["stitch_boundaries"][0]["cut_time"], 12.5)

            paths = paths_for(episode_key)
            self.assertEqual(result["text"], paths["whisper_text"])
            self.assertEqual(result["metadata"], paths["whisper_metadata"])
            self.assertEqual(upload_calls, [paths["whisper_text"], paths["whisper_metadata"]])

    def test_missing_credential_fails_before_any_request(self):
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "episode.mp3"
            audio.write_bytes(b"fake audio")
            with (
                patch.dict("os.environ", {"PODCAST_WHISPER_API_KEY": ""}),
                patch.object(transcriber.requests, "post", side_effect=AssertionError("no request")),
            ):
                with self.assertRaisesRegex(RuntimeError, "PODCAST_WHISPER_API_KEY"):
                    transcriber.transcribe_audio(str(audio), "episode-key")

    def test_rejected_request_is_marked_uncertain_and_not_retried(self):
        chunk = AudioChunk(path=Path(__file__), start=0.0, duration=12.0)
        with (
            _openrouter_environment() as budget,
            patch.object(transcriber.requests, "post", return_value=_FakeResponse({}, status=400)),
            patch.object(transcriber.time, "sleep") as sleep,
        ):
            with self.assertRaisesRegex(RuntimeError, "HTTP 400"):
                transcriber._transcribe_chunk(
                    chunk,
                    api_key="k",
                    language=None,
                    episode_key="0123456789abcdef01234567",
                    budget_fingerprint="sha256:" + "a" * 64,
                    usd_per_second=Decimal("0.0000075"),
                )
        self.assertEqual(budget["reserve"].call_count, 1)
        self.assertEqual(budget["uncertain"].call_count, 1)
        budget["settle"].assert_not_called()
        sleep.assert_not_called()

    def test_transient_failure_retries_once_with_a_fresh_reservation(self):
        chunk = AudioChunk(path=Path(__file__), start=0.0, duration=12.0)
        responses = [_FakeResponse({}, status=503), _FakeResponse(_stt_response([(0.0, 1.0, " ok")]))]
        with (
            _openrouter_environment() as budget,
            patch.object(transcriber.requests, "post", side_effect=responses),
            patch.object(transcriber.time, "sleep"),
        ):
            payload = transcriber._transcribe_chunk(
                chunk,
                api_key="k",
                language="en",
                episode_key="0123456789abcdef01234567",
                budget_fingerprint="sha256:" + "a" * 64,
                usd_per_second=Decimal("0.0000075"),
            )
        self.assertEqual(payload["text"], " ok")
        self.assertEqual(budget["reserve"].call_count, 2)
        first, second = (call.args[2][0]["attempt_id"] for call in budget["reserve"].call_args_list)
        self.assertNotEqual(first, second)
        self.assertEqual(budget["uncertain"].call_count, 1)
        self.assertEqual(budget["settle"].call_count, 1)

    def _chunk_with_responses(self, responses):
        chunk = AudioChunk(path=Path(__file__), start=0.0, duration=12.0)
        with (
            _openrouter_environment() as budget,
            patch.object(transcriber.requests, "post", side_effect=responses),
            patch.object(transcriber.time, "sleep") as sleep,
        ):
            try:
                payload = transcriber._transcribe_chunk(
                    chunk,
                    api_key="k",
                    language="en",
                    episode_key="0123456789abcdef01234567",
                    budget_fingerprint="sha256:" + "a" * 64,
                    usd_per_second=Decimal("0.0000075"),
                )
            except RuntimeError as error:
                payload = error
        return payload, budget, [call.args[0] for call in sleep.call_args_list]

    def test_rate_limit_waits_with_backoff_and_a_fresh_reservation(self):
        ok = _FakeResponse(_stt_response([(0.0, 1.0, " ok")]))
        payload, budget, sleeps = self._chunk_with_responses(
            [_FakeResponse({}, status=429), _FakeResponse({}, status=429), ok]
        )
        self.assertEqual(payload["text"], " ok")
        self.assertEqual(sleeps, [15.0, 30.0])
        self.assertEqual(budget["reserve"].call_count, 3)
        self.assertEqual(budget["uncertain"].call_count, 2)
        self.assertEqual(budget["settle"].call_count, 1)

    def test_rate_limit_honors_retry_after_within_the_cap(self):
        ok = _FakeResponse(_stt_response([(0.0, 1.0, " ok")]))
        payload, _budget, sleeps = self._chunk_with_responses(
            [
                _FakeResponse({}, status=429, headers={"Retry-After": "7"}),
                _FakeResponse({}, status=429, headers={"Retry-After": "999"}),
                _FakeResponse({}, status=429, headers={"Retry-After": "Wed, 30 Sep 2026 19:00:00 GMT"}),
                ok,
            ]
        )
        self.assertEqual(payload["text"], " ok")
        self.assertEqual(sleeps, [7.0, 120.0, 60.0])

    def test_rate_limit_log_names_the_reason_without_the_key(self):
        limited = _FakeResponse(
            {"error": {"message": "Rate limit exceeded:\n upstream   provider busy", "code": 429}},
            status=429,
            headers={"Retry-After": "3"},
        )
        ok = _FakeResponse(_stt_response([(0.0, 1.0, " ok")]))
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            payload, _budget, sleeps = self._chunk_with_responses([limited, ok])
        self.assertEqual(payload["text"], " ok")
        self.assertEqual(sleeps, [3.0])
        self.assertIn("[Rate limit exceeded: upstream provider busy; Retry-After]", output.getvalue())
        self.assertNotIn("Bearer", output.getvalue())

    def test_persistent_rate_limit_fails_after_the_bounded_attempts(self):
        responses = [_FakeResponse({}, status=429) for _ in range(transcriber.RATE_LIMIT_ATTEMPTS)]
        error, budget, sleeps = self._chunk_with_responses(responses)
        self.assertIsInstance(error, RuntimeError)
        self.assertIn("HTTP 429", str(error))
        self.assertEqual(sleeps, [15.0, 30.0, 60.0, 120.0, 120.0, 120.0, 120.0])
        self.assertLessEqual(sum(sleeps), 600.0)
        self.assertEqual(budget["reserve"].call_count, transcriber.RATE_LIMIT_ATTEMPTS)
        self.assertEqual(budget["uncertain"].call_count, transcriber.RATE_LIMIT_ATTEMPTS)
        budget["settle"].assert_not_called()

    def test_server_errors_still_retry_only_once(self):
        error, budget, sleeps = self._chunk_with_responses(
            [_FakeResponse({}, status=503), _FakeResponse({}, status=502)]
        )
        self.assertIsInstance(error, RuntimeError)
        self.assertIn("HTTP 502", str(error))
        self.assertEqual(sleeps, [transcriber.RETRY_DELAY_SECONDS])
        self.assertEqual(budget["reserve"].call_count, 2)

    def test_words_attach_to_the_segment_containing_their_midpoint(self):
        payload = {
            "segments": [
                {"start": 0.0, "end": 2.0, "text": " one two", "avg_logprob": -0.1, "no_speech_prob": 0.0},
                {"start": 2.0, "end": 4.0, "text": " three", "avg_logprob": -0.3, "no_speech_prob": 0.1},
            ],
            "words": [
                {"word": " one", "start": 0.0, "end": 0.9},
                {"word": " two", "start": 1.5, "end": 2.2},
                {"word": " three", "start": 2.1, "end": 3.0},
            ],
        }
        segments = transcriber._openrouter_segment_metadata(payload, offset_seconds=100.0)
        self.assertEqual([w["word"] for w in segments[0]["words"]], [" one", " two"])
        self.assertEqual([w["word"] for w in segments[1]["words"]], [" three"])
        self.assertEqual(segments[1]["words"][0]["start"], 102.1)
        self.assertNotIn("probability", segments[0]["words"][0])

    def test_small_backward_word_start_at_segment_join_is_clamped(self):
        # Qwen3-ASR restarts a segment's first word ~0.02 s before the
        # previous word's start; the parser clamps steps within tolerance.
        payload = {
            "segments": [
                {"start": 0.0, "end": 2.0, "text": " one two", "avg_logprob": -0.1, "no_speech_prob": 0.0},
                {"start": 2.0, "end": 4.0, "text": " three", "avg_logprob": -0.1, "no_speech_prob": 0.0},
            ],
            "words": [
                {"word": " one", "start": 0.0, "end": 0.9},
                {"word": " two", "start": 1.98, "end": 1.98},
                {"word": " three", "start": 1.96, "end": 2.5},
            ],
        }
        segments = transcriber._openrouter_segment_metadata(payload, offset_seconds=10.0)
        words = [word for segment in segments for word in segment["words"]]
        self.assertEqual([word["word"] for word in words], [" one", " two", " three"])
        self.assertAlmostEqual(words[2]["start"], 11.98)
        self.assertAlmostEqual(words[2]["end"], 12.5)
        transcriber._validate_monotonic_timestamps(segments)

    def test_large_backward_word_start_still_fails_closed(self):
        payload = {
            "segments": [{"start": 0.0, "end": 4.0, "text": " one two", "avg_logprob": -0.1, "no_speech_prob": 0.0}],
            "words": [
                {"word": " one", "start": 2.0, "end": 2.5},
                {"word": " two", "start": 1.4, "end": 2.6},
            ],
        }
        segments = transcriber._openrouter_segment_metadata(payload)
        self.assertEqual(segments[0]["words"][1]["start"], 1.4)
        with self.assertRaises(RuntimeError):
            transcriber._validate_monotonic_timestamps(segments)

    def test_segment_join_step_of_a_tenth_of_a_second_is_clamped(self):
        # Synthetic segment-join case: one segment ends at 27.6 s while the
        # next segment begins at 27.5 s.
        payload = {
            "segments": [
                {"start": 20.0, "end": 27.5, "text": "the final stage example.", "avg_logprob": -0.1, "no_speech_prob": 0.0},
                {"start": 27.5, "end": 30.0, "text": "Next words", "avg_logprob": -0.1, "no_speech_prob": 0.0},
            ],
            "words": [
                {"word": "stage", "start": 26.88, "end": 27.2},
                {"word": "example.", "start": 27.6, "end": 27.6},
                {"word": "Next", "start": 27.5, "end": 28.3},
                {"word": "words", "start": 28.3, "end": 28.46},
            ],
        }
        segments = transcriber._openrouter_segment_metadata(payload)
        transcriber._validate_monotonic_timestamps(segments)
        words = [word for segment in segments for word in segment["words"]]
        self.assertAlmostEqual(words[2]["start"], 27.6)

    def test_words_without_leading_space_keep_word_boundaries_after_stitching(self):
        # Qwen3-ASR words carry no leading space; segment text rebuilt from
        # words during stitching must still separate words.
        payload = {
            "segments": [{"start": 0.0, "end": 3.0, "text": "What's up, everybody?", "avg_logprob": -0.1, "no_speech_prob": 0.0}],
            "words": [
                {"word": "What's", "start": 0.0, "end": 0.4},
                {"word": "up,", "start": 0.4, "end": 0.8},
                {"word": "everybody?", "start": 0.8, "end": 1.5},
            ],
        }
        segments = transcriber._openrouter_segment_metadata(payload)
        self.assertEqual([word["word"] for word in segments[0]["words"]], [" What's", " up,", " everybody?"])
        rebuilt = transcriber._rebuild_segment_from_words(segments[0], segments[0]["words"])
        self.assertEqual(rebuilt["text"], "What's up, everybody?")

    def test_language_hint_uses_iso_codes(self):
        self.assertEqual(transcriber._language_hint("english"), "en")
        self.assertEqual(transcriber._language_hint(" English "), "en")
        self.assertEqual(transcriber._language_hint("en"), "en")
        self.assertIsNone(transcriber._language_hint("klingon"))
        self.assertIsNone(transcriber._language_hint(None))

    def test_malformed_response_is_rejected(self):
        for payload in ({}, {"segments": [], "words": [{"word": " x", "start": 0, "end": 1}]},
                        {"segments": [{"start": 2, "end": 1}], "words": []}):
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    transcriber._openrouter_segment_metadata(payload)


class _FakeResponse:
    def __init__(self, payload, status=200, headers=None):
        self._payload = payload
        self.status_code = status
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise transcriber.requests.HTTPError(f"{self.status_code}", response=self)

    def json(self):
        return self._payload


def _stt_response(segments, language="en"):
    return {
        "text": "".join(text for _, _, text in segments),
        "language": language,
        "usage": {"seconds": 12, "cost": 0.0001},
        "segments": [
            {"id": index, "start": start, "end": end, "text": text, "avg_logprob": -0.2, "no_speech_prob": 0.01}
            for index, (start, end, text) in enumerate(segments)
        ],
        "words": [
            {"word": text, "start": start, "end": end}
            for start, end, text in segments
        ],
    }


@contextmanager
def _openrouter_environment():
    with (
        patch.dict("os.environ", {"PODCAST_WHISPER_API_KEY": "test-whisper-key"}),
        patch.object(
            transcriber,
            "resolve_audio_endpoint_pricing",
            return_value=AudioPricing("openai/whisper-large-v3", Decimal("0.0000075"), "now", "url", "sha256:x"),
        ),
        patch.object(transcriber, "reserve_budget_batch") as reserve,
        patch.object(transcriber, "settle_budget_attempt") as settle,
        patch.object(transcriber, "mark_budget_attempt_uncertain") as uncertain,
    ):
        yield {"reserve": reserve, "settle": settle, "uncertain": uncertain}


def _word(
    word,
    start,
    end,
):
    return {
        "word": word,
        "start": start,
        "end": end,
        "probability": 0.99,
    }


if __name__ == "__main__":
    unittest.main()
