"""Versioned contracts for the podcast knowledge-note pipeline."""

from __future__ import annotations

KNOWLEDGE_SCHEMA_VERSION = 1
FRONTMATTER_SCHEMA_VERSION = 1

# TASK-118: single-pass knowledge-note writer (knowledge-note-v3), the only
# note path since TASK-146 retired the summary -> summary-review -> metadata
# chain. There is deliberately no default slug: the Worker refuses to build
# a note while this variable is unset or blank.
NOTE_WRITER_PRESET_ENV = "PODCAST_KNOWLEDGE_WRITER_PRESET"
# Bump when the prompt, note schema, checks or rendering change what a
# cached note would contain.
NOTE_WRITER_POLICY_VERSION = "note-writer-v1"
# Sonnet 5.5 writer calls took 60-220 s in the pilots; the shared
# KNOWLEDGE_TIMEOUT_SECONDS (120 s) would cut the longest ones off.
NOTE_WRITER_TIMEOUT_SECONDS = 420
# TASK-119: the forced-decision coverage fill, its own OpenRouter preset
# governed like the writer's (system prompt must equal
# prompts/knowledge/note-fill-v1.md). Unset or blank disables the fill; the
# Worker deploy gate requires it in production.
NOTE_FILL_PRESET_ENV = "PODCAST_KNOWLEDGE_FILL_PRESET"
# Fill calls took 6-23 s in the 2026-10-15 comparison.
NOTE_FILL_TIMEOUT_SECONDS = 180


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
