"""RED/GREEN contract tests for the shared TASK-076 episode source-generation identity.

podcast_engine.episode_generation.source_generation_fingerprint(...) must stay
byte-for-byte compatible with the resolver's existing source_fingerprint
contract (podcast_engine.compilation._review_source_fingerprint), since
already-persisted Human Review records key off it. This module also proves
the from-episode loader (used to key the future budget ledger off the same
identity) fails closed on missing or malformed canonical source shapes.
"""

import hashlib
import json
import unittest
from unittest.mock import patch

from podcast_engine.episode_generation import (
    EpisodeSourceGenerationError,
    source_generation_fingerprint,
    source_generation_fingerprint_from_episode,
)


def _expected_fingerprint(apple_text, apple_metadata, whisper_text, whisper_metadata):
    def digest(value):
        return (
            "sha256:" + hashlib.sha256(value).hexdigest()
            if value is not None
            else None
        )

    payload = {
        "apple_text": digest(apple_text),
        "apple_metadata": digest(apple_metadata),
        "whisper_text": digest(whisper_text),
        "whisper_metadata": digest(whisper_metadata),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


class SourceGenerationFingerprintCompatibilityTests(unittest.TestCase):
    def test_complete_sources_match_the_resolver_contract_payload(self):
        fingerprint = source_generation_fingerprint(
            apple_text=b"apple transcript bytes",
            apple_metadata=b"apple metadata bytes",
            whisper_text=b"whisper transcript bytes",
            whisper_metadata=b"whisper metadata bytes",
        )
        self.assertEqual(
            fingerprint,
            _expected_fingerprint(
                b"apple transcript bytes",
                b"apple metadata bytes",
                b"whisper transcript bytes",
                b"whisper metadata bytes",
            ),
        )
        self.assertTrue(fingerprint.startswith("sha256:"))

    def test_missing_metadata_matches_the_resolver_contract_payload(self):
        fingerprint = source_generation_fingerprint(
            apple_text=b"apple transcript bytes",
            apple_metadata=None,
            whisper_text=b"whisper transcript bytes",
            whisper_metadata=None,
        )
        self.assertEqual(
            fingerprint,
            _expected_fingerprint(
                b"apple transcript bytes",
                None,
                b"whisper transcript bytes",
                None,
            ),
        )

    def test_changed_bytes_produce_a_different_fingerprint(self):
        base = source_generation_fingerprint(
            apple_text=b"a",
            apple_metadata=None,
            whisper_text=b"w",
            whisper_metadata=None,
        )
        changed = source_generation_fingerprint(
            apple_text=b"a-changed",
            apple_metadata=None,
            whisper_text=b"w",
            whisper_metadata=None,
        )
        self.assertNotEqual(base, changed)


def _episode(**source_overrides):
    sources = {
        "apple": {
            "text": "episodes/e1/sources/apple/transcript.txt",
            "metadata": "episodes/e1/sources/apple/metadata.json",
        },
        "whisper": {
            "text": "episodes/e1/sources/whisper/transcript.txt",
            "metadata": "episodes/e1/sources/whisper/metadata.json",
        },
    }
    sources.update(source_overrides)
    return {"files": {"sources": sources}}


class SourceGenerationFingerprintFromEpisodeTests(unittest.TestCase):
    def test_loads_and_fingerprints_canonical_sources(self):
        episode = _episode()
        payloads = {
            "episodes/e1/sources/apple/transcript.txt": b"apple text",
            "episodes/e1/sources/apple/metadata.json": b"apple meta",
            "episodes/e1/sources/whisper/transcript.txt": b"whisper text",
            "episodes/e1/sources/whisper/metadata.json": b"whisper meta",
        }
        with patch(
            "podcast_engine.episode_generation.download_gcs_bytes",
            side_effect=lambda path: payloads[path],
        ):
            fingerprint = source_generation_fingerprint_from_episode(episode)

        self.assertEqual(
            fingerprint,
            source_generation_fingerprint(
                apple_text=b"apple text",
                apple_metadata=b"apple meta",
                whisper_text=b"whisper text",
                whisper_metadata=b"whisper meta",
            ),
        )

    def test_missing_metadata_paths_are_treated_as_absent(self):
        episode = _episode(
            apple={"text": "episodes/e1/sources/apple/transcript.txt"},
            whisper={"text": "episodes/e1/sources/whisper/transcript.txt"},
        )
        payloads = {
            "episodes/e1/sources/apple/transcript.txt": b"apple text",
            "episodes/e1/sources/whisper/transcript.txt": b"whisper text",
        }
        with patch(
            "podcast_engine.episode_generation.download_gcs_bytes",
            side_effect=lambda path: payloads[path],
        ):
            fingerprint = source_generation_fingerprint_from_episode(episode)

        self.assertEqual(
            fingerprint,
            source_generation_fingerprint(
                apple_text=b"apple text",
                apple_metadata=None,
                whisper_text=b"whisper text",
                whisper_metadata=None,
            ),
        )

    def test_non_dict_episode_is_rejected(self):
        with self.assertRaises(EpisodeSourceGenerationError):
            source_generation_fingerprint_from_episode("not a dict")

    def test_missing_files_is_rejected(self):
        with self.assertRaises(EpisodeSourceGenerationError):
            source_generation_fingerprint_from_episode({})

    def test_missing_apple_text_path_is_rejected(self):
        episode = _episode(
            apple={"metadata": "episodes/e1/sources/apple/metadata.json"}
        )
        with self.assertRaises(EpisodeSourceGenerationError):
            source_generation_fingerprint_from_episode(episode)

    def test_missing_whisper_text_path_is_rejected(self):
        episode = _episode(whisper={})
        with self.assertRaises(EpisodeSourceGenerationError):
            source_generation_fingerprint_from_episode(episode)

    def test_malformed_sources_shape_is_rejected(self):
        episode = {"files": {"sources": "not a dict"}}
        with self.assertRaises(EpisodeSourceGenerationError):
            source_generation_fingerprint_from_episode(episode)

    def test_malformed_apple_source_shape_is_rejected(self):
        episode = _episode(apple="not a dict")
        with self.assertRaises(EpisodeSourceGenerationError):
            source_generation_fingerprint_from_episode(episode)


if __name__ == "__main__":
    unittest.main()
