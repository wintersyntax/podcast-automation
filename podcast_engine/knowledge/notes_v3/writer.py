"""Production writer stage for single-pass knowledge notes (TASK-118).

This is the only ``notes_v3`` module that calls a model. One
``anthropic/claude-sonnet-5.5`` request reads the whole canonical compiled
transcript and returns a ``knowledge-note-v3`` JSON note; Python then
validates its structure, applies the deterministic checks and renders the
Markdown body.

Contracts shared with the other knowledge stages:

* the request is built from one verified current OpenRouter preset snapshot
  (``PODCAST_KNOWLEDGE_WRITER_PRESET``), never a mutable ``@preset`` alias;
* the preset's system prompt must equal the committed
  ``prompts/knowledge/note-writer-v1.md`` -- the prompt and the response
  schema change together in the repository, so a drifted remote prompt
  fails closed instead of silently producing a different contract;
* every physical attempt reserves episode AI budget through
  ``client.post_openrouter`` under stage ``note_writer``.

Failure handling: an invalid structure gets one more request; truncation
(``finish_reason = length``) does not, because the same input would be cut
off again. Either way the episode is held and nothing is published.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path

from ...episode_contract import now_iso
from ...preset_provenance import PresetProvenance, fetch_current_designated_preset
from ..client import post_openrouter
from ..models import (
    NOTE_WRITER_POLICY_VERSION,
    NOTE_WRITER_PRESET_ENV,
    NOTE_WRITER_TIMEOUT_SECONDS,
)
from ..summary import _WORST_CASE_EPISODE, episode_context, worst_case_transcript_text
from . import checks, render, schema
from .units import UnitReport

PROMPT_PATH = (
    Path(__file__).resolve().parents[3] / "prompts" / "knowledge" / "note-writer-v1.md"
)
MAX_RESPONSE_ATTEMPTS = 2
# Show notes are context, not content: bounded so a pathological feed entry
# cannot inflate the request or its budget reservation.
MAX_DESCRIPTION_CHARS = 8000
_ERROR_DETAIL_LIMIT = 5


class NoteWriterError(RuntimeError):
    """The writer response could not become a note; the episode is held."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


@dataclass
class WriterResult:
    """A checked note, its rendered body, and everything worth auditing."""

    note: dict
    body: str
    check_report: checks.CheckReport
    unit_report: UnitReport
    provenance: PresetProvenance
    served_model: str | None = None
    served_provider: str | None = None
    usage: dict = field(default_factory=dict)
    response_attempts: int = 1


def note_writer_preset() -> str | None:
    """The configured writer preset slug, or ``None`` when inactive."""

    value = os.getenv(NOTE_WRITER_PRESET_ENV, "").strip()
    return value or None


def system_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def prompt_sha256() -> str:
    return "sha256:" + hashlib.sha256(system_prompt().encode("utf-8")).hexdigest()


def input_fingerprint(episode: dict, transcript_bytes: bytes, preset: str) -> str:
    """Fingerprint every input that can change the note body.

    The RSS show notes are excluded too: they are best-effort context, and
    a transient feed miss must not make an already-paid note look stale.
    The tag vocabulary is deliberately excluded: it only steers tag
    suggestions, changes whenever any tag is promoted, and tag transitions
    on existing notes are handled by the tag-registry backfill -- including
    it would silently re-buy the whole note after every tag decision.
    """

    payload = {
        "policy_version": NOTE_WRITER_POLICY_VERSION,
        "note_schema_version": schema.NOTE_SCHEMA_VERSION,
        "preset": preset,
        "prompt_sha256": prompt_sha256(),
        "transcript_sha256": hashlib.sha256(transcript_bytes).hexdigest(),
        "episode": episode_context(episode),
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def writer_episode_context(episode: dict, description: str | None = None) -> dict:
    """Episode context sent to the writer: the shared episode fields plus
    the RSS show notes when available (as in the accepted pilot)."""

    context = episode_context(episode)
    if isinstance(description, str) and description.strip():
        context["description"] = description.strip()[:MAX_DESCRIPTION_CHARS]
    return context


def openrouter_payload(
    episode: dict,
    transcript: str,
    tag_vocabulary: dict | None,
    provenance: PresetProvenance,
    description: str | None = None,
) -> dict:
    """Build the physical request from one verified preset snapshot."""

    if (
        not provenance.verified
        or not isinstance(provenance.config, dict)
        or not isinstance(provenance.system_prompt, str)
    ):
        raise RuntimeError("Verified OpenRouter preset provenance is required")
    if provenance.system_prompt.strip() != system_prompt().strip():
        raise NoteWriterError(
            "prompt_drift",
            f"the system prompt of OpenRouter preset {provenance.slug!r} "
            "differs from prompts/knowledge/note-writer-v1.md; update the "
            "preset (or the repository prompt and its policy version) first",
        )
    max_tokens = provenance.config.get("max_tokens")
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens <= 0:
        raise NoteWriterError(
            "preset_config",
            "the writer preset must set a positive integer max_tokens",
        )

    payload = copy.deepcopy(provenance.config)
    payload.update(
        {
            "response_format": copy.deepcopy(schema.RESPONSE_FORMAT),
            "messages": [
                {"role": "system", "content": provenance.system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "episode_context": writer_episode_context(episode, description),
                            "tag_vocabulary": tag_vocabulary or {},
                            "transcript": transcript,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            "usage": {"include": True},
        }
    )
    # The note contract depends on strict structured output, so never let
    # OpenRouter route to a provider that would silently drop
    # ``response_format`` -- whatever the mutable preset says.
    provider = payload.get("provider")
    payload["provider"] = {**(provider if isinstance(provider, dict) else {}), "require_parameters": True}
    return payload


def worst_case_openrouter_payload(provenance: PresetProvenance) -> dict:
    """The same-shaped request at the documented maximum transcript size and
    a generous tag vocabulary, for downstream-reserve planning only. Never
    sent to a provider."""

    # Imported locally: metadata imports podcast_engine.ai_budget, which
    # pulls in google-cloud-storage, and this module must stay importable
    # without it.
    from ..metadata import _worst_case_tag_vocabulary

    return openrouter_payload(
        _WORST_CASE_EPISODE,
        worst_case_transcript_text(),
        _worst_case_tag_vocabulary(),
        provenance,
        description="w" * MAX_DESCRIPTION_CHARS,
    )


def parse_response(response_payload: object) -> tuple[dict, dict]:
    """Return ``(note, served)`` from a chat-completions response or raise."""

    try:
        choice = response_payload["choices"][0]
        finish_reason = choice.get("finish_reason")
        content = choice["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise NoteWriterError("invalid_structure", "response has no message content") from error
    if finish_reason == "length":
        raise NoteWriterError(
            "truncated",
            "the writer hit max_tokens; raise the preset's max_tokens rather than retrying",
        )
    if not isinstance(content, str):
        raise NoteWriterError("invalid_structure", "message content is not text")
    try:
        note = json.loads(content)
    except json.JSONDecodeError as error:
        raise NoteWriterError("invalid_structure", f"content is not JSON ({error.msg})") from error
    errors = schema.validate_note(note)
    if errors:
        raise NoteWriterError(
            "invalid_structure", "; ".join(errors[:_ERROR_DETAIL_LIMIT])
        )
    usage = response_payload.get("usage") if isinstance(response_payload, dict) else None
    served = {
        "model": response_payload.get("model"),
        "provider": response_payload.get("provider"),
        "finish_reason": finish_reason,
        "usage": {
            key: usage[key]
            for key in ("prompt_tokens", "completion_tokens", "total_tokens", "cost")
            if isinstance(usage, dict) and key in usage
        },
    }
    return note, served


def generate(
    episode: dict,
    transcript: str,
    tag_vocabulary: dict | None,
    *,
    episode_key: str,
    source_fingerprint: str,
    description: str | None = None,
    pricing_transport=None,
    pricing_now=None,
) -> WriterResult:
    """Request, validate, check and render one knowledge note."""

    # Imported locally for the same import-boundary reason as above.
    from ...ai_budget import STAGE_NOTE_WRITER

    api_key = os.getenv("PODCAST_KNOWLEDGE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing PODCAST_KNOWLEDGE_API_KEY for knowledge requests")
    slug = note_writer_preset()
    if slug is None:
        raise RuntimeError(f"{NOTE_WRITER_PRESET_ENV} is not configured")

    provenance = fetch_current_designated_preset(slug, api_key=api_key, verified_at=now_iso())
    if not provenance.verified:
        raise RuntimeError(
            f"Note writer preset could not be resolved: {provenance.status} "
            f"({provenance.reason or 'unknown'})"
        )
    payload = openrouter_payload(episode, transcript, tag_vocabulary, provenance, description)

    last_error: NoteWriterError | None = None
    for attempt in range(1, MAX_RESPONSE_ATTEMPTS + 1):
        response = post_openrouter(
            payload,
            episode_key=episode_key,
            source_fingerprint=source_fingerprint,
            stage=STAGE_NOTE_WRITER,
            provenance=provenance,
            pricing_transport=pricing_transport,
            pricing_now=pricing_now,
            timeout_seconds=NOTE_WRITER_TIMEOUT_SECONDS,
        )
        try:
            response_payload = response.json()
        except ValueError:
            response_payload = None
        try:
            raw_note, served = parse_response(response_payload)
        except NoteWriterError as error:
            if error.code != "invalid_structure":
                raise
            last_error = error
            continue
        break
    else:
        raise last_error  # type: ignore[misc]

    checked, check_report = checks.apply_checks(raw_note, transcript)
    unit_report = UnitReport()
    body = render.render_body(checked, unit_report)
    return WriterResult(
        note=checked,
        body=body,
        check_report=check_report,
        unit_report=unit_report,
        provenance=provenance,
        served_model=served["model"],
        served_provider=served["provider"],
        usage=served["usage"],
        response_attempts=attempt,
    )


def audit_record(result: WriterResult) -> dict:
    """JSON-safe record of how one note was produced (no prompt text)."""

    report = result.check_report
    return {
        "note_schema_version": schema.NOTE_SCHEMA_VERSION,
        "prompt_sha256": prompt_sha256(),
        "preset_provenance": result.provenance.record(),
        "served_model": result.served_model,
        "served_provider": result.served_provider,
        "usage": result.usage,
        "response_attempts": result.response_attempts,
        "checks": {
            "anchored_units": report.anchored_units,
            "anchored_dropped": report.anchored_dropped,
            "fuzzy_anchors": report.fuzzy_anchors,
            "dropped": [
                {"path": unit.path, "reason": unit.reason, "text": unit.text}
                for unit in report.dropped
            ],
        },
        "units": {
            "unbraced_imperial": list(result.unit_report.unbraced_imperial),
            "unknown_units": list(result.unit_report.unknown_units),
        },
    }


__all__ = [
    "MAX_RESPONSE_ATTEMPTS",
    "NoteWriterError",
    "PROMPT_PATH",
    "WriterResult",
    "audit_record",
    "generate",
    "input_fingerprint",
    "note_writer_preset",
    "openrouter_payload",
    "writer_episode_context",
    "parse_response",
    "prompt_sha256",
    "system_prompt",
    "worst_case_openrouter_payload",
]
