"""Versioned contracts for the podcast knowledge-note pipeline."""

from __future__ import annotations

KNOWLEDGE_SCHEMA_VERSION = 1
# summary-v6 is publishable only through the independent summary-review-v2 evidence gate.
SUMMARY_POLICY_VERSION = "summary-v6"
SUMMARY_REVIEW_POLICY_VERSION = "summary-review-v2"
SUMMARY_REVIEW_EVIDENCE_SCHEMA_VERSION = 2
METADATA_POLICY_VERSION = "metadata-v4"
FRONTMATTER_SCHEMA_VERSION = 1

SUMMARY_PRESET_ENV = "PODCAST_SUMMARY_PRESET"
SUMMARY_REVIEW_PRESET_ENV = "PODCAST_SUMMARY_REVIEW_PRESET"
METADATA_PRESET_ENV = "PODCAST_METADATA_PRESET"
DEFAULT_SUMMARY_PRESET = "podcast-summary"
DEFAULT_SUMMARY_REVIEW_PRESET = "podcast-summary-review"
DEFAULT_METADATA_PRESET = "podcast-metadata"

# TASK-118: single-pass knowledge-note writer (knowledge-note-v3). The
# writer path is active only when this preset env var is nonblank; it then
# replaces the summary -> summary-review -> metadata chain for new notes.
# There is deliberately no default slug: activation is an explicit
# deployment decision, never an accident of an unset variable.
NOTE_WRITER_PRESET_ENV = "PODCAST_KNOWLEDGE_WRITER_PRESET"
# Bump when the prompt, note schema, checks or rendering change what a
# cached note would contain.
NOTE_WRITER_POLICY_VERSION = "note-writer-v1"
# Sonnet 5.5 writer calls took 60-220 s in the pilots; the shared
# KNOWLEDGE_TIMEOUT_SECONDS (120 s) would cut the longest ones off.
NOTE_WRITER_TIMEOUT_SECONDS = 420


METADATA_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["topics", "people", "existing_tags", "new_tag_candidates"],
    "properties": {
        "topics": {
            "type": "array",
            "maxItems": 8,
            "items": {"type": "string"},
        },
        "people": {
            "type": "array",
            "maxItems": 12,
            "items": {"type": "string"},
        },
        "existing_tags": {
            "type": "array",
            "maxItems": 8,
            "items": {"type": "string"},
        },
        "new_tag_candidates": {
            "type": "array",
            "maxItems": 2,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["tag", "category"],
                "properties": {
                    "tag": {"type": "string"},
                    "category": {
                        "type": "string",
                        "enum": [
                            "domain", "training", "nutrition", "supplements",
                            "recovery", "research", "other",
                        ],
                    },
                },
            },
        },
    },
}
