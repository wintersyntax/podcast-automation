"""Forced-decision coverage fill for single-pass notes (TASK-119).

Design: ``docs/KNOWLEDGE_GENERATION.md``
section 7. After the deterministic detectors (``coverage``) find at least one
uncovered exercise-advice spot or unit-bearing number, one model call receives
the whole transcript, every note bullet (with a ``bullet_id``), the section
titles and the candidate excerpts, and must return exactly one decision per
candidate: ``extend`` a listed bullet, ``add`` a bullet, or ``skip`` with an enum
reason. Python validates every decision -- anchor in the transcript, no ad
wording, no label-only stub, a known bullet for ``extend`` -- applies the valid
ones and records every decision for the coverage review (TASK-120).

Full input was chosen over a reduced input (nearby bullets and short excerpts
only) by the 2026-10-15 paid comparison: the reduced variants were cheaper
but skipped exercise technique the full input added.

This module builds the request content, parses the response and applies it; it
makes no network call itself (``fill_stage`` does).
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from . import schema
from .checks import AD_RE, TranscriptIndex, _is_stub
from .coverage import Candidate, CoverageReport

FILL_SCHEMA_VERSION = "note-fill-v1"
PROMPT_PATH = Path(__file__).resolve().parents[3] / "prompts" / "knowledge" / "note-fill-v1.md"
MAX_EXCERPT_CHARS = 1500
ACTIONS = ("add", "extend", "skip")
SKIP_REASONS = ("ad_or_plug", "off_scope_or_small_talk", "already_covered", "not_practical", "not_applicable")

_S = {"type": "string"}
DECISION_SCHEMA = schema._obj({
    "candidate_id": _S,
    "action": {"type": "string", "enum": list(ACTIONS)},
    "section_title": _S,
    "old_bullet_id": _S,
    "bullet": schema.BULLET_SCHEMA,
    "skip_reason": {"type": "string", "enum": list(SKIP_REASONS)},
})
FILL_RESPONSE_SCHEMA = schema._obj({"decisions": schema._arr(DECISION_SCHEMA)})
RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "podcast_knowledge_note_fill_v1", "strict": True, "schema": FILL_RESPONSE_SCHEMA},
}


class FillResponseError(ValueError):
    """The fill response is not a valid decision list."""


class FillTruncatedError(FillResponseError):
    """The fill hit max_tokens; the same input would be cut off again."""


def system_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def prompt_sha256() -> str:
    return "sha256:" + hashlib.sha256(system_prompt().encode("utf-8")).hexdigest()


def _bullet_ids(note: dict) -> dict[str, tuple[int, int]]:
    ids: dict[str, tuple[int, int]] = {}
    number = 0
    for s_index, section in enumerate(note["sections"]):
        for b_index, _ in enumerate(section["bullets"]):
            number += 1
            ids[f"b{number:03d}"] = (s_index, b_index)
    return ids


def fill_input(note: dict, transcript: str, report: CoverageReport, *, title: str = "") -> dict:
    """The fill request content: the whole transcript and note plus the candidates."""

    return {
        "episode_title": title,
        "section_titles": [section["title"] for section in note["sections"]],
        "all_bullets": [
            {
                "bullet_id": bullet_id,
                "section_title": note["sections"][s_index]["title"],
                "text": note["sections"][s_index]["bullets"][b_index]["text"],
            }
            for bullet_id, (s_index, b_index) in _bullet_ids(note).items()
        ],
        "transcript": transcript,
        "candidates": [
            {
                "candidate_id": c.id,
                "type": c.type,
                "detector_detail": c.detail[:300],
                "transcript_excerpt": c.excerpt[:MAX_EXCERPT_CHARS],
            }
            for c in report.candidates
        ],
    }


def user_message(payload: dict) -> dict:
    return {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)}


def parse_decisions(content: object) -> list[dict]:
    """Validate the model content (JSON text or object) against the schema."""

    if isinstance(content, str):
        try:
            content = json.loads(content)
        except ValueError as exc:
            raise FillResponseError("fill response is not JSON") from exc
    errors: list[str] = []
    schema._check(content, FILL_RESPONSE_SCHEMA, "fill", errors)
    if errors:
        raise FillResponseError("; ".join(errors[:5]))
    return content["decisions"]


def parse_response(response_payload: object) -> tuple[list[dict], dict]:
    """Return ``(decisions, served)`` from a chat-completions response or raise."""

    try:
        choice = response_payload["choices"][0]
        finish_reason = choice.get("finish_reason")
        content = choice["message"]["content"]
    except (KeyError, IndexError, TypeError, AttributeError) as error:
        raise FillResponseError("fill response has no message content") from error
    if finish_reason == "length":
        raise FillTruncatedError("the fill hit max_tokens; raise the preset's max_tokens")
    decisions = parse_decisions(content)
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
    return decisions, served


@dataclass
class FillReport:
    candidates: int = 0
    decisions: list[dict] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)
    applied: dict[str, int] = field(default_factory=lambda: {"add": 0, "extend": 0, "skip": 0})
    rejected: dict[str, int] = field(default_factory=dict)

    def as_record(self) -> dict:
        return {
            "schema_version": FILL_SCHEMA_VERSION,
            "candidates": self.candidates,
            "applied": dict(self.applied),
            "rejected": dict(self.rejected),
            "missing": list(self.missing),
            "unknown": list(self.unknown),
            "decisions": list(self.decisions),
        }


def _rejection(bullet: dict, index: TranscriptIndex) -> str | None:
    if _is_stub(bullet["text"]):
        return "label_only_bullet"
    if AD_RE.search(bullet["text"]):
        return "ad_or_sponsor"
    if index.locate(bullet["anchor"])[0] is None:
        return "anchor_not_in_transcript"
    return None


def apply_decisions(note: dict, transcript: str, candidates: list[Candidate], decisions: list[dict]) -> tuple[dict, FillReport]:
    """Apply valid decisions to a copy of ``note``; record every decision."""

    updated = copy.deepcopy(note)
    index = TranscriptIndex(transcript)
    bullet_ids = _bullet_ids(note)
    by_id = {c.id: c for c in candidates}
    report = FillReport(candidates=len(candidates))
    seen: set[str] = set()
    extended: set[str] = set()
    for decision in decisions:
        cid = decision["candidate_id"]
        if cid not in by_id:
            report.unknown.append(cid)
            continue
        if cid in seen:
            continue
        seen.add(cid)
        candidate = by_id[cid]
        record = {
            "candidate_id": cid,
            "type": candidate.type,
            "detail": candidate.detail,
            "excerpt_sha256": candidate.excerpt_sha256,
            "action": decision["action"],
            "skip_reason": decision["skip_reason"],
            "section_title": decision["section_title"],
            "old_bullet_id": decision["old_bullet_id"],
            "bullet": decision["bullet"],
            "applied": False,
            "rejected": None,
        }
        if decision["action"] == "skip":
            record["applied"] = True
            report.applied["skip"] += 1
            report.decisions.append(record)
            continue
        reason = _rejection(decision["bullet"], index)
        target = bullet_ids.get(decision["old_bullet_id"]) if decision["action"] == "extend" else None
        if reason is None and decision["action"] == "extend" and target is None:
            reason = "unknown_bullet_id"
        if reason is None and decision["action"] == "extend" and decision["old_bullet_id"] in extended:
            reason = "bullet_already_extended"
        if reason is None and decision["action"] == "add" and not decision["section_title"].strip():
            reason = "missing_section_title"
        if reason is not None:
            record["rejected"] = reason
            report.rejected[reason] = report.rejected.get(reason, 0) + 1
            report.decisions.append(record)
            continue
        if decision["action"] == "extend":
            s_index, b_index = target
            record["old_text"] = note["sections"][s_index]["bullets"][b_index]["text"]
            updated["sections"][s_index]["bullets"][b_index] = dict(decision["bullet"])
            extended.add(decision["old_bullet_id"])
        else:
            title = decision["section_title"].strip()
            section = next((s for s in updated["sections"] if s["title"].strip().lower() == title.lower()), None)
            if section is None:
                section = {"title": title, "bottom_line": "", "bullets": [], "protocols": []}
                updated["sections"].append(section)
            section["bullets"].append(dict(decision["bullet"]))
        record["applied"] = True
        report.applied[decision["action"]] += 1
        report.decisions.append(record)
    report.missing = [c.id for c in candidates if c.id not in seen]
    return updated, report
