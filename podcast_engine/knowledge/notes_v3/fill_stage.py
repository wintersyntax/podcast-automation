"""Production coverage-fill stage for single-pass knowledge notes (TASK-119).

Runs after the writer on the checked note. The deterministic detectors
(``coverage``) decide whether a fill is needed; when they trigger, one
request built from the verified current ``PODCAST_KNOWLEDGE_FILL_PRESET``
snapshot sends the whole transcript, every note bullet and the candidates
(``fill.fill_input``), and Python applies the returned decisions
(``fill.apply_decisions``).

Contracts shared with the writer stage:

* the preset's system prompt must equal the committed
  ``prompts/knowledge/note-fill-v1.md``; a drifted prompt fails the fill;
* every physical attempt reserves episode AI budget through
  ``client.post_openrouter`` under stage ``note_fill``.

The fill only adds to a note that has already passed the writer checks, so
a failed fill never holds the episode: the writer note is published and the
failure is recorded in the manifest (``note_fill.status = "failed"``), where
``fill_pending`` makes the one-time regeneration tool retry it.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import json
import os

from ...episode_contract import now_iso
from ...preset_provenance import PresetProvenance, fetch_current_designated_preset
from ..client import post_openrouter
from ..models import NOTE_FILL_PRESET_ENV, NOTE_FILL_TIMEOUT_SECONDS
from . import coverage, fill

MAX_RESPONSE_ATTEMPTS = 2
_ERROR_LIMIT = 300
FILL_STATUSES = ("not_triggered", "applied", "disabled", "failed")


class NoteFillError(RuntimeError):
    """The fill could not run or its response could not be applied."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


@dataclass
class FillOutcome:
    """The note to publish and the manifest record of how it was filled."""

    note: dict
    record: dict

    @property
    def status(self) -> str:
        return self.record["status"]


def note_fill_preset() -> str | None:
    """The configured fill preset slug, or ``None`` when the fill is disabled."""

    value = os.getenv(NOTE_FILL_PRESET_ENV, "").strip()
    return value or None


def fill_pending(record: object, *, enabled: bool) -> bool:
    """True when an enabled fill has not produced a current result yet.

    A missing, disabled or failed record, or one made by another fill schema,
    prompt or coverage policy, is pending. A lexicon change alone is not:
    lexicon terms are added through the coverage review (TASK-120).
    """

    if not enabled:
        return False
    if not isinstance(record, dict):
        return True
    return (
        record.get("status") not in ("applied", "not_triggered")
        or record.get("schema_version") != fill.FILL_SCHEMA_VERSION
        or record.get("prompt_sha256") != fill.prompt_sha256()
        or record.get("coverage_policy") != coverage.COVERAGE_POLICY_VERSION
    )


def published_note(manifest: dict) -> dict:
    """The note the body is rendered from: the filled note when there is one."""

    record = manifest.get("note_fill")
    if isinstance(record, dict) and record.get("status") == "applied" and isinstance(record.get("note"), dict):
        return record["note"]
    return manifest["note_writer"]["note"]


def openrouter_payload(
    note: dict,
    transcript: str,
    report: coverage.CoverageReport,
    provenance: PresetProvenance,
    *,
    title: str = "",
) -> dict:
    """Build the physical request from one verified preset snapshot."""

    if (
        not provenance.verified
        or not isinstance(provenance.config, dict)
        or not isinstance(provenance.system_prompt, str)
    ):
        raise NoteFillError("preset_unverified", "verified OpenRouter preset provenance is required")
    if provenance.system_prompt.strip() != fill.system_prompt().strip():
        raise NoteFillError(
            "prompt_drift",
            f"the system prompt of OpenRouter preset {provenance.slug!r} differs from "
            "prompts/knowledge/note-fill-v1.md",
        )
    max_tokens = provenance.config.get("max_tokens")
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens <= 0:
        raise NoteFillError("preset_config", "the fill preset must set a positive integer max_tokens")

    payload = copy.deepcopy(provenance.config)
    payload.update(
        {
            "response_format": copy.deepcopy(fill.RESPONSE_FORMAT),
            "messages": [
                {"role": "system", "content": provenance.system_prompt},
                fill.user_message(fill.fill_input(note, transcript, report, title=title)),
            ],
            "usage": {"include": True},
        }
    )
    provider = payload.get("provider")
    payload["provider"] = {**(provider if isinstance(provider, dict) else {}), "require_parameters": True}
    return payload


def _base_record(report: coverage.CoverageReport | None, status: str) -> dict:
    return {
        "schema_version": fill.FILL_SCHEMA_VERSION,
        "coverage_policy": coverage.COVERAGE_POLICY_VERSION,
        "prompt_sha256": fill.prompt_sha256(),
        "status": status,
        "filled_at": now_iso(),
        "coverage": report.as_record() if report is not None else None,
    }


def run_fill(
    note: dict,
    transcript: str,
    *,
    title: str,
    episode_key: str,
    source_fingerprint: str,
    pricing_transport=None,
    pricing_now=None,
) -> FillOutcome:
    """Detect, and when triggered fill, one checked note. Never raises for a
    fill failure: the writer note is returned with a ``failed`` record."""

    from ...ai_budget import STAGE_NOTE_FILL

    report = coverage.find_candidates(note, transcript)
    if not report.triggered:
        return FillOutcome(note, _base_record(report, "not_triggered"))
    slug = note_fill_preset()
    if slug is None:
        return FillOutcome(note, _base_record(report, "disabled"))

    record = _base_record(report, "failed")
    record["preset"] = slug
    try:
        api_key = os.getenv("PODCAST_KNOWLEDGE_API_KEY")
        if not api_key:
            raise NoteFillError("missing_api_key", "PODCAST_KNOWLEDGE_API_KEY is not set")
        provenance = fetch_current_designated_preset(slug, api_key=api_key, verified_at=now_iso())
        if not provenance.verified:
            raise NoteFillError(
                "preset_unverified", f"{provenance.status} ({provenance.reason or 'unknown'})"
            )
        record["preset_provenance"] = provenance.record()
        payload = openrouter_payload(note, transcript, report, provenance, title=title)
        for attempt in range(1, MAX_RESPONSE_ATTEMPTS + 1):
            response = post_openrouter(
                payload,
                episode_key=episode_key,
                source_fingerprint=source_fingerprint,
                stage=STAGE_NOTE_FILL,
                provenance=provenance,
                pricing_transport=pricing_transport,
                pricing_now=pricing_now,
                timeout_seconds=NOTE_FILL_TIMEOUT_SECONDS,
            )
            try:
                response_payload = response.json()
            except ValueError:
                response_payload = None
            try:
                decisions, served = fill.parse_response(response_payload)
                break
            except fill.FillTruncatedError:
                raise
            except fill.FillResponseError:
                # An invalid structure gets one more request, as for the writer.
                if attempt >= MAX_RESPONSE_ATTEMPTS:
                    raise
        record["response_attempts"] = attempt
    except Exception as error:  # noqa: BLE001 -- a failed fill never holds the note
        record["error"] = {"type": type(error).__name__, "message": str(error)[:_ERROR_LIMIT]}
        print(
            json.dumps(
                {
                    "severity": "ERROR",
                    "event": "knowledge_note_fill_failed",
                    "message": "Coverage fill failed; publishing the writer note",
                    "episode_key": episode_key,
                    "error_type": type(error).__name__,
                    "error_code": getattr(error, "code", None),
                }
            ),
            flush=True,
        )
        return FillOutcome(note, record)

    updated, fill_report = fill.apply_decisions(note, transcript, report.candidates, decisions)
    record.update(
        {
            "status": "applied",
            "served_model": served["model"],
            "served_provider": served["provider"],
            "usage": served["usage"],
            "result": fill_report.as_record(),
            "note": updated,
        }
    )
    return FillOutcome(updated, record)


__all__ = [
    "FILL_STATUSES",
    "FillOutcome",
    "MAX_RESPONSE_ATTEMPTS",
    "NoteFillError",
    "fill_pending",
    "note_fill_preset",
    "openrouter_payload",
    "published_note",
    "run_fill",
]
