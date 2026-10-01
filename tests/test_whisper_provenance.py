"""Tests for the canonical current Whisper provenance contract."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from podcast_engine import whisper_provenance as provenance


class WhisperProvenanceTests(unittest.TestCase):
    def _metadata(self, producer: dict | None = None) -> dict:
        return {
            "schema_version": provenance.WHISPER_METADATA_SCHEMA_VERSION,
            "source": provenance.WHISPER_METADATA_SOURCE,
            "model": provenance.MODEL_ID,
            "producer": (
                provenance.current_whisper_producer()
                if producer is None
                else producer
            ),
            "segments": [],
        }

    def test_model_manifest_schema_constant_drives_validation(self):
        future = 9
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "whisper-model.json"
            manifest.write_text(
                json.dumps({"schema_version": future}),
                encoding="utf-8",
            )

            with patch.object(
                provenance,
                "WHISPER_MODEL_MANIFEST_SCHEMA_VERSION",
                future,
                create=True,
            ), patch.object(
                provenance,
                "MANIFEST_PATH",
                manifest,
            ):
                try:
                    loaded = provenance._load_manifest()
                except RuntimeError as error:
                    self.fail(
                        f"manifest schema constant was ignored: {error}"
                    )

        self.assertEqual(loaded["schema_version"], future)

    def test_producer_schema_constant_drives_identity(self):
        future = 9
        with patch.object(
            provenance,
            "WHISPER_PRODUCER_SCHEMA_VERSION",
            future,
            create=True,
        ):
            payload = provenance.producer_payload()

        self.assertEqual(
            payload["producer_schema_version"],
            future,
        )

    def test_current_metadata_schema_constant_drives_reuse_contract(self):
        future = 9
        payload = provenance.producer_payload()
        producer = {
            **payload,
            "fingerprint": provenance.producer_fingerprint(payload),
        }
        metadata = {
            "schema_version": future,
            "source": provenance.WHISPER_METADATA_SOURCE,
            "model": provenance.MODEL_ID,
            "producer": producer,
            "segments": [],
        }

        with patch.object(
            provenance,
            "WHISPER_METADATA_SCHEMA_VERSION",
            future,
            create=True,
        ), patch.object(
            provenance,
            "current_whisper_producer",
            return_value=producer,
        ):
            self.assertTrue(
                provenance.whisper_metadata_matches_current_producer(
                    metadata
                )
            )

    def test_current_producer_has_expected_identity(self):
        producer = provenance.current_whisper_producer()

        self.assertEqual(
            producer["model"],
            {"id": "openai/whisper-large-v3", "provider": "deepinfra"},
        )
        self.assertEqual(
            producer["engine"],
            {
                "name": "openrouter-stt",
                "endpoint": "https://openrouter.ai/api/v1/audio/transcriptions",
                "response_format": "verbose_json",
                "timestamp_granularities": ["word", "segment"],
            },
        )
        self.assertEqual(
            producer["audio"],
            {"format": "mp3", "sample_rate": 16000, "channels": 1, "bitrate_kbps": 48},
        )
        self.assertEqual(
            producer["chunking"],
            {
                "chunk_seconds": 600,
                "overlap_seconds": 5,
                "stitch_method": "timestamp_midpoint_overlap_v3",
            },
        )
        self.assertEqual(
            producer["fingerprint"],
            provenance.producer_fingerprint(provenance.producer_payload()),
        )

    def test_local_faster_whisper_metadata_is_not_reusable(self):
        # TASK-125: every artifact of the earlier bundled Faster-Whisper
        # producer must be re-transcribed, never reused.
        metadata = self._metadata()
        metadata["schema_version"] = 2
        metadata["source"] = "faster-whisper"
        self.assertFalse(provenance.whisper_metadata_matches_current_producer(metadata))

    def test_current_metadata_is_reusable(self):
        self.assertTrue(
            provenance.whisper_metadata_matches_current_producer(
                self._metadata()
            )
        )

    def test_legacy_v1_metadata_is_not_reusable(self):
        metadata = {
            "schema_version": 1,
            "source": "faster-whisper",
            "model": "small",
            "segments": [],
        }

        self.assertFalse(
            provenance.whisper_metadata_matches_current_producer(
                metadata
            )
        )

    def test_tampered_producer_is_not_reusable(self):
        producer = provenance.current_whisper_producer()
        producer["audio"]["bitrate_kbps"] = 128

        # Fingerprint intentionally remains the original one.
        self.assertFalse(
            provenance.whisper_metadata_matches_current_producer(
                self._metadata(producer)
            )
        )

    def test_stale_but_self_consistent_producer_is_not_reusable(self):
        producer = deepcopy(
            provenance.current_whisper_producer()
        )

        producer["model"]["provider"] = "historical-provider"

        payload = {
            key: value
            for key, value in producer.items()
            if key != "fingerprint"
        }
        producer["fingerprint"] = (
            provenance.producer_fingerprint(payload)
        )

        # Internal fingerprint is valid, but the producer is no longer
        # the current canonical producer.
        self.assertFalse(
            provenance.whisper_metadata_matches_current_producer(
                self._metadata(producer)
            )
        )


if __name__ == "__main__":
    unittest.main()
