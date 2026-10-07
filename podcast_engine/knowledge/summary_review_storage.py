"""Immutable evidence storage for transcript-grounded summary reviews."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import math
from typing import Any

from google.api_core.exceptions import NotFound, PreconditionFailed

from podcast_engine.episode_contract import summary_review_artifact_paths

from .models import SUMMARY_REVIEW_EVIDENCE_SCHEMA_VERSION
from .summary_review_contract import (
    accepted_final_markdown,
    validate_audit_result,
    validate_edit_result,
)
from .summary_review_evidence import validate_review_context


_FAILURE_FIELDS = frozenset({"stage", "code"})
_COMPLETION_TEXT_FIELDS = ("completion_id", "served_model", "served_provider")
_COMPLETION_NUMBER_FIELDS = (
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "cost",
)
_MAX_MODEL_CALLS_PER_CASE = 3
_SUMMARY_REVIEW_HISTORY_SCHEMA_VERSION = 1


def _sha256(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("Summary-review source artifacts must be strings")
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _stable_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"


def _bounded_failure(value: dict | None) -> dict | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != _FAILURE_FIELDS:
        raise ValueError("Summary-review failure must contain only stage and code")
    bounded = {}
    for field in ("stage", "code"):
        item = value.get(field)
        if not isinstance(item, str) or not item or len(item) > 80:
            raise ValueError(f"Summary-review failure {field} is invalid")
        bounded[field] = item
    return bounded


_SUMMARY_REVIEW_FAILURE_EXPLANATIONS = {
    ("request", "request_failed"):
        "Reviewer request failed before a valid response was available.",
    ("request", "independent_or_stable_served_identity_required"):
        "Reviewer model/provider identity was missing, invalid, or changed during the review.",
    ("audit", "invalid_response_json"):
        "Reviewer returned invalid JSON during the audit stage.",
    ("audit", "missing_review_content"):
        "Reviewer response was missing required audit content.",
    ("audit", "invalid_audit_contract"):
        "Reviewer audit evidence did not satisfy the required contract.",
    ("audit", "review_call_budget_exhausted"):
        "Reviewer exhausted the allowed call budget during the audit stage.",
    ("audit", "review_rejected"):
        "Reviewer rejected the draft during the audit stage.",
    ("edit", "invalid_response_json"):
        "Reviewer returned invalid JSON during the edit stage.",
    ("edit", "missing_review_content"):
        "Reviewer response was missing required edit content.",
    ("edit", "invalid_edit_contract"):
        "Reviewer edit evidence did not satisfy the required contract.",
    ("edit", "review_call_budget_exhausted"):
        "Reviewer exhausted the allowed call budget during the edit stage.",
}

_SUMMARY_REVIEW_FAILURE_STAGE_FALLBACKS = {
    "request":
        "Summary review stopped during the request stage for an unrecognized bounded failure.",
    "audit":
        "Summary review stopped during the audit stage for an unrecognized bounded failure.",
    "edit":
        "Summary review stopped during the edit stage for an unrecognized bounded failure.",
}


def explain_summary_review_failure(failure: dict) -> str:
    """Return safe operator prose for one canonical bounded failure."""

    bounded = _bounded_failure(failure)
    if bounded is None:
        raise ValueError("Summary-review failure is required")

    explanation = _SUMMARY_REVIEW_FAILURE_EXPLANATIONS.get(
        (bounded["stage"], bounded["code"])
    )
    if explanation is not None:
        return explanation

    return _SUMMARY_REVIEW_FAILURE_STAGE_FALLBACKS.get(
        bounded["stage"],
        "Summary review stopped for an unrecognized bounded failure.",
    )


def derive_summary_review_stalled_state(
    review_records: list[dict],
    *,
    episode_key: str,
    transcript_sha256: str,
    draft_sha256: str,
    review_policy_version: str,
    review_preset: str,
) -> dict:
    """Derive the trailing terminal-failure streak for one exact review identity."""

    if not isinstance(review_records, list):
        raise ValueError("Summary-review records must be a list")

    expected_identity = {
        "episode_key": episode_key,
        "transcript_sha256": transcript_sha256,
        "draft_sha256": draft_sha256,
        "review_policy_version": review_policy_version,
        "review_preset": review_preset,
    }
    for field, value in expected_identity.items():
        if not isinstance(value, str) or not value:
            raise ValueError(f"Summary-review stalled identity {field} is invalid")

    events = []
    for record in review_records:
        if not isinstance(record, dict):
            raise ValueError("Summary-review evidence record must be an object")
        if record.get("schema_version") != SUMMARY_REVIEW_EVIDENCE_SCHEMA_VERSION:
            raise ValueError("Summary-review evidence schema version is invalid")

        identity = {}
        for field in expected_identity:
            value = record.get(field)
            if not isinstance(value, str) or not value:
                raise ValueError(f"Summary-review evidence {field} is invalid")
            identity[field] = value

        review_id = record.get("review_id")
        if not isinstance(review_id, str) or not review_id:
            raise ValueError("Summary-review evidence review_id is invalid")

        reviewed_at = record.get("reviewed_at")
        if not isinstance(reviewed_at, str) or not reviewed_at:
            raise ValueError("Summary-review evidence reviewed_at is invalid")
        normalized_reviewed_at = (
            reviewed_at[:-1] + "+00:00" if reviewed_at.endswith("Z") else reviewed_at
        )
        try:
            reviewed_at_value = datetime.fromisoformat(normalized_reviewed_at)
        except ValueError as error:
            raise ValueError("Summary-review evidence reviewed_at is invalid") from error
        if reviewed_at_value.tzinfo is None:
            raise ValueError("Summary-review evidence reviewed_at must include timezone")

        outcome = record.get("outcome")
        if outcome not in {"accepted", "failed"}:
            raise ValueError("Summary-review evidence outcome is invalid")

        matching_identity = identity == expected_identity
        failure = None
        if matching_identity:
            failure = _bounded_failure(record.get("failure"))
            if outcome == "failed" and failure is None:
                raise ValueError(
                    "Failed summary-review evidence for stalled identity requires failure"
                )
            if outcome == "accepted" and failure is not None:
                raise ValueError(
                    "Accepted summary-review evidence for stalled identity cannot contain failure"
                )

        events.append(
            {
                "identity": identity,
                "review_id": review_id,
                "reviewed_at": reviewed_at,
                "reviewed_at_value": reviewed_at_value,
                "outcome": outcome,
                "failure": failure,
            }
        )

    events.sort(
        key=lambda item: (item["reviewed_at_value"], item["review_id"]),
        reverse=True,
    )

    failures = []
    for event in events:
        if event["identity"] != expected_identity:
            break
        if event["outcome"] == "accepted":
            break
        failures.append(
            {
                "review_id": event["review_id"],
                "reviewed_at": event["reviewed_at"],
                "failure": event["failure"],
            }
        )

    latest = failures[0] if failures else None
    earliest = failures[-1] if failures else None
    return {
        "stalled": len(failures) >= 2,
        "failure_count": len(failures),
        "first_failure_at": earliest["reviewed_at"] if earliest else None,
        "last_failure_at": latest["reviewed_at"] if latest else None,
        "latest_failure": dict(latest["failure"]) if latest else None,
        "latest_review_id": latest["review_id"] if latest else None,
    }


def build_summary_review_stalled_diagnostics(
    review_records: list[dict],
    *,
    episode_key: str,
    transcript_sha256: str,
    draft_sha256: str,
    review_policy_version: str,
    review_preset: str,
    canonical_note_preserved: bool,
) -> dict:
    """Build one read-only operator diagnostic object from immutable review evidence."""

    if not isinstance(canonical_note_preserved, bool):
        raise ValueError("Summary-review canonical-note preservation must be boolean")

    stalled_state = derive_summary_review_stalled_state(
        review_records,
        episode_key=episode_key,
        transcript_sha256=transcript_sha256,
        draft_sha256=draft_sha256,
        review_policy_version=review_policy_version,
        review_preset=review_preset,
    )

    latest_failure = stalled_state["latest_failure"]
    latest_review_id = stalled_state["latest_review_id"]
    technical = {}
    if latest_review_id is not None:
        for record in review_records:
            if (
                isinstance(record, dict)
                and record.get("review_id") == latest_review_id
            ):
                technical = _safe_completion_metadata(
                    record.get("completion_metadata")
                )
                break

    failure_diagnostic = None
    if latest_failure is not None:
        failure_diagnostic = {
            "stage": latest_failure["stage"],
            "code": latest_failure["code"],
            "explanation": explain_summary_review_failure(latest_failure),
        }

    return {
        "state": "stalled" if stalled_state["stalled"] else "retry_pending",
        "stalled": stalled_state["stalled"],
        "failure_count": stalled_state["failure_count"],
        "first_failure_at": stalled_state["first_failure_at"],
        "last_failure_at": stalled_state["last_failure_at"],
        "latest_review_id": latest_review_id,
        "latest_failure": failure_diagnostic,
        "retry_state": "resumable",
        "summary_completion": "pending",
        "canonical_note_preserved": canonical_note_preserved,
        "technical": technical,
    }


def _clear_summary_review_diagnostics() -> dict:
    return {
        "state": "clear",
        "stalled": False,
        "failure_count": 0,
        "first_failure_at": None,
        "last_failure_at": None,
        "latest_review_id": None,
        "latest_failure": None,
        "retry_state": None,
        "summary_completion": None,
        "canonical_note_preserved": True,
        "technical": {},
    }


def _summary_review_history_index_path(episode_key: str) -> str:
    return f"episodes/{episode_key}/summary/reviews/index.json"


def _review_ids_from_history_manifest(value: object) -> list[str]:
    if not isinstance(value, dict) or set(value) != {"schema_version", "review_ids"}:
        raise ValueError("Summary-review history manifest shape is invalid")
    if value.get("schema_version") != _SUMMARY_REVIEW_HISTORY_SCHEMA_VERSION:
        raise ValueError("Summary-review history manifest schema version is invalid")

    review_ids = value.get("review_ids")
    if not isinstance(review_ids, list):
        raise ValueError("Summary-review history manifest review_ids is invalid")

    normalized = []
    seen = set()
    for review_id in review_ids:
        if (
            not isinstance(review_id, str)
            or not review_id
            or "/" in review_id
            or not review_id.startswith("sr-")
        ):
            raise ValueError("Summary-review history manifest review_id is invalid")
        if review_id in seen:
            raise ValueError("Summary-review history manifest contains duplicate review_id")
        seen.add(review_id)
        normalized.append(review_id)
    return normalized


def _load_review_records_from_history(*, bucket, episode_key: str) -> list[dict]:
    index_blob = bucket.blob(_summary_review_history_index_path(episode_key))
    if not index_blob.exists():
        return []

    try:
        manifest = json.loads(index_blob.download_as_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("Summary-review history manifest is malformed") from error

    review_records = []
    for review_id in _review_ids_from_history_manifest(manifest):
        path = summary_review_artifact_paths(episode_key, review_id)["review"]
        blob = bucket.blob(path)
        if not blob.exists():
            raise ValueError("Summary-review history references missing review.json")
        try:
            envelope = json.loads(blob.download_as_text(encoding="utf-8"))
        except (UnicodeError, json.JSONDecodeError) as error:
            raise ValueError("Summary-review review.json is malformed") from error

        if not isinstance(envelope, dict):
            raise ValueError("Summary-review review.json must contain an object")
        if envelope.get("review_id") != review_id:
            raise ValueError(
                "Summary-review review.json review_id does not match its path"
            )
        if envelope.get("episode_key") != episode_key:
            raise ValueError(
                "Summary-review review.json episode_key does not match its path"
            )
        review_records.append(envelope)
    return review_records


def load_summary_review_stalled_diagnostics(*, bucket, episode_key: str) -> dict:
    """Read immutable review history and derive current operator diagnostics."""

    if not isinstance(episode_key, str) or not episode_key or "/" in episode_key:
        raise ValueError("Summary-review diagnostics episode_key is invalid")

    review_records = _load_review_records_from_history(
        bucket=bucket,
        episode_key=episode_key,
    )
    if not review_records:
        return _clear_summary_review_diagnostics()

    def sort_key(record: dict):
        review_id = record.get("review_id")
        reviewed_at = record.get("reviewed_at")
        if not isinstance(review_id, str) or not review_id:
            raise ValueError("Summary-review evidence review_id is invalid")
        if not isinstance(reviewed_at, str) or not reviewed_at:
            raise ValueError("Summary-review evidence reviewed_at is invalid")
        normalized = (
            reviewed_at[:-1] + "+00:00"
            if reviewed_at.endswith("Z")
            else reviewed_at
        )
        try:
            reviewed_at_value = datetime.fromisoformat(normalized)
        except ValueError as error:
            raise ValueError(
                "Summary-review evidence reviewed_at is invalid"
            ) from error
        if reviewed_at_value.tzinfo is None:
            raise ValueError(
                "Summary-review evidence reviewed_at must include timezone"
            )
        return reviewed_at_value, review_id

    latest_record = max(review_records, key=sort_key)

    identity_fields = (
        "transcript_sha256",
        "draft_sha256",
        "review_policy_version",
        "review_preset",
    )
    identity = {}
    for field in identity_fields:
        value = latest_record.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"Summary-review evidence {field} is invalid")
        identity[field] = value

    return build_summary_review_stalled_diagnostics(
        review_records,
        episode_key=episode_key,
        transcript_sha256=identity["transcript_sha256"],
        draft_sha256=identity["draft_sha256"],
        review_policy_version=identity["review_policy_version"],
        review_preset=identity["review_preset"],
        canonical_note_preserved=True,
    )


def _safe_attempt(value: object) -> dict:
    if not isinstance(value, dict):
        raise ValueError("Summary-review completion attempt must be an object")
    allowed = {
        "phase",
        "phase_attempt",
        *_COMPLETION_TEXT_FIELDS,
        *_COMPLETION_NUMBER_FIELDS,
    }
    if not set(value).issubset(allowed):
        raise ValueError("Summary-review completion attempt contains unsupported fields")

    phase = value.get("phase")
    if phase not in {"audit", "edit"}:
        raise ValueError("Summary-review completion attempt phase is invalid")
    phase_attempt = value.get("phase_attempt")
    if (
        not isinstance(phase_attempt, int)
        or isinstance(phase_attempt, bool)
        or phase_attempt < 1
        or phase_attempt > 2
    ):
        raise ValueError("Summary-review completion phase_attempt is invalid")

    result: dict[str, Any] = {"phase": phase, "phase_attempt": phase_attempt}
    for field in _COMPLETION_TEXT_FIELDS:
        item = value.get(field)
        if item is None:
            continue
        if not isinstance(item, str) or not item or len(item) > 500:
            raise ValueError(f"Summary-review completion attempt {field} is invalid")
        result[field] = item
    for field in _COMPLETION_NUMBER_FIELDS:
        item = value.get(field)
        if item is None:
            continue
        if (
            not isinstance(item, (int, float))
            or isinstance(item, bool)
            or not math.isfinite(item)
            or item < 0
        ):
            raise ValueError(f"Summary-review completion attempt {field} is invalid")
        result[field] = item
    return result


def _safe_completion_metadata(value: dict | None) -> dict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("Summary-review completion metadata must be an object")

    allowed = {
        *_COMPLETION_TEXT_FIELDS,
        *_COMPLETION_NUMBER_FIELDS,
        "attempt_count",
        "attempts",
    }
    if not set(value).issubset(allowed):
        raise ValueError("Summary-review completion metadata contains unsupported fields")

    result: dict[str, Any] = {}
    for field in _COMPLETION_TEXT_FIELDS:
        item = value.get(field)
        if item is None:
            continue
        if not isinstance(item, str) or not item or len(item) > 500:
            raise ValueError(f"Summary-review completion metadata {field} is invalid")
        result[field] = item
    for field in _COMPLETION_NUMBER_FIELDS:
        item = value.get(field)
        if item is None:
            continue
        if (
            not isinstance(item, (int, float))
            or isinstance(item, bool)
            or not math.isfinite(item)
            or item < 0
        ):
            raise ValueError(f"Summary-review completion metadata {field} is invalid")
        result[field] = item

    attempt_count = value.get("attempt_count")
    if attempt_count is not None:
        if (
            not isinstance(attempt_count, int)
            or isinstance(attempt_count, bool)
            or attempt_count < 1
            or attempt_count > _MAX_MODEL_CALLS_PER_CASE
        ):
            raise ValueError("Summary-review completion metadata attempt_count is invalid")
        result["attempt_count"] = attempt_count

    attempts = value.get("attempts")
    if attempts is not None:
        if not isinstance(attempts, list) or len(attempts) > _MAX_MODEL_CALLS_PER_CASE:
            raise ValueError("Summary-review completion metadata attempts is invalid")
        result["attempts"] = [_safe_attempt(item) for item in attempts]
        if attempt_count is not None and len(result["attempts"]) > attempt_count:
            raise ValueError("Summary-review completion attempts exceed attempt_count")

    return result


def _review_id(identity: dict) -> str:
    digest = _canonical_json_sha256(identity).split(":", 1)[1]
    return "sr-" + digest[:24]


def _write_immutable(blob, content: str, *, content_type: str) -> str:
    expected = content.encode("utf-8")
    if blob.exists():
        if blob.download_as_bytes() != expected:
            raise RuntimeError(f"immutable review artifact conflict: {blob.name}")
        return "unchanged"

    try:
        blob.upload_from_string(
            content,
            content_type=content_type,
            if_generation_match=0,
        )
    except PreconditionFailed:
        if not blob.exists() or blob.download_as_bytes() != expected:
            raise RuntimeError(f"immutable review artifact conflict: {blob.name}")
        return "unchanged"
    return "created"


def _verify_immutable(blob, content: str) -> None:
    if not blob.exists() or blob.download_as_bytes() != content.encode("utf-8"):
        raise RuntimeError(f"immutable review artifact verification failed: {blob.name}")


def _register_review_in_history(
    *,
    bucket,
    episode_key: str,
    review_id: str,
) -> str:
    """Register one verified immutable review through a generation-safe manifest."""

    path = _summary_review_history_index_path(episode_key)

    for _ in range(5):
        blob = bucket.blob(path)

        try:
            blob.reload()
        except NotFound:
            review_ids = []
            generation = 0
        else:
            generation = int(blob.generation)
            try:
                raw = blob.download_as_text(
                    encoding="utf-8",
                    if_generation_match=generation,
                )
            except (NotFound, PreconditionFailed):
                continue

            try:
                manifest = json.loads(raw)
            except (UnicodeError, json.JSONDecodeError) as error:
                raise ValueError(
                    "Summary-review history manifest is malformed"
                ) from error

            review_ids = _review_ids_from_history_manifest(manifest)
            if review_id in review_ids:
                return "unchanged"

        manifest_text = _stable_json(
            {
                "schema_version": _SUMMARY_REVIEW_HISTORY_SCHEMA_VERSION,
                "review_ids": [*review_ids, review_id],
            }
        )

        try:
            blob.upload_from_string(
                manifest_text,
                content_type="application/json",
                if_generation_match=generation,
            )
        except PreconditionFailed:
            continue

        return "created" if generation == 0 else "updated"

    raise RuntimeError(
        "Summary-review history manifest changed repeatedly during registration"
    )


def _evidence_context(review_context: dict) -> dict:
    return {
        "transcript_span_algorithm": review_context["transcript_span_index"]["algorithm"],
        "transcript_span_index_sha256": review_context["transcript_span_index"]["index_sha256"],
        "draft_block_algorithm": review_context["draft_block_index"]["algorithm"],
        "draft_block_index_sha256": review_context["draft_block_index"]["index_sha256"],
        "risk_inventory_algorithm": review_context["risk_inventory"]["algorithm"],
        "risk_inventory_sha256": review_context["risk_inventory"]["index_sha256"],
    }


def persist_review_attempt(
    *,
    bucket,
    episode_key: str,
    transcript: str,
    draft: str,
    review_context: dict,
    audit_result: dict | None,
    edit_result: dict | None,
    accepted_final: str | None,
    failure: dict | None,
    summary_policy_version: str,
    summary_preset: str,
    review_policy_version: str,
    review_preset: str,
    completion_metadata: dict | None,
    reviewed_at: str,
) -> dict:
    """Persist one V2 review attempt before any canonical summary publication."""

    for field_name, value in (
        ("episode_key", episode_key),
        ("summary_policy_version", summary_policy_version),
        ("summary_preset", summary_preset),
        ("review_policy_version", review_policy_version),
        ("review_preset", review_preset),
        ("reviewed_at", reviewed_at),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(f"Summary-review {field_name} is required")
    if not isinstance(transcript, str) or not isinstance(draft, str):
        raise ValueError("Summary-review transcript and draft must be strings")

    validate_review_context(review_context, transcript, draft)
    safe_failure = _bounded_failure(failure)
    safe_completion = _safe_completion_metadata(completion_metadata)
    transcript_sha256 = _sha256(transcript)
    draft_sha256 = _sha256(draft)
    context_binding = _evidence_context(review_context)

    validated_audit = None
    validated_edit = None
    if audit_result is not None:
        validated_audit = validate_audit_result(audit_result, transcript, review_context)

    if validated_audit is not None and validated_audit["status"] in {"pass", "revised"}:
        if edit_result is not None:
            validated_edit = validate_edit_result(
                edit_result,
                validated_audit,
                draft,
                transcript,
                review_context,
            )

    accepted = (
        validated_audit is not None
        and validated_audit["status"] in {"pass", "revised"}
        and validated_edit is not None
    )

    if accepted:
        if safe_failure is not None:
            raise ValueError("Accepted summary review cannot contain a failure")
        if not isinstance(accepted_final, str) or not accepted_final:
            raise ValueError("Accepted summary review requires final Markdown")
        expected_final = accepted_final_markdown(validated_audit, validated_edit, draft)
        if accepted_final != expected_final:
            raise ValueError("Accepted summary review final does not match validated edit")
    else:
        if accepted_final is not None:
            raise ValueError("Failed summary review cannot contain final Markdown")
        if validated_audit is None and safe_failure is None:
            raise ValueError("Failed summary review requires an audit result or failure")
        if validated_audit is not None and validated_audit["status"] in {"pass", "revised"}:
            if safe_failure is None:
                raise ValueError("Incomplete actionable summary review requires a failure")
        if validated_audit is not None and validated_audit["status"] == "fail" and edit_result is not None:
            raise ValueError("Fail audit cannot contain an edit result")

    final_sha256 = _sha256(accepted_final) if accepted else None
    identity = {
        "episode_key": episode_key,
        "outcome": "accepted" if accepted else "failed",
        "transcript_sha256": transcript_sha256,
        "draft_sha256": draft_sha256,
        "final_sha256": final_sha256,
        "evidence_context": context_binding,
        "audit_result": validated_audit,
        "edit_result": validated_edit,
        "failure": safe_failure,
        "summary_policy_version": summary_policy_version,
        "summary_preset": summary_preset,
        "review_policy_version": review_policy_version,
        "review_preset": review_preset,
        "completion_metadata": safe_completion,
        "reviewed_at": reviewed_at,
    }
    review_id = _review_id(identity)
    paths = summary_review_artifact_paths(episode_key, review_id)

    envelope = {
        "schema_version": SUMMARY_REVIEW_EVIDENCE_SCHEMA_VERSION,
        "review_id": review_id,
        **identity,
        "review_model": safe_completion.get("served_model"),
        "review_provider": safe_completion.get("served_provider"),
    }
    review_json = _stable_json(envelope)
    transcript_index_json = _stable_json(review_context["transcript_span_index"])
    draft_index_json = _stable_json(review_context["draft_block_index"])
    risk_inventory_json = _stable_json(review_context["risk_inventory"])

    writes = {
        "draft": _write_immutable(
            bucket.blob(paths["draft"]),
            draft,
            content_type="text/markdown; charset=utf-8",
        ),
        "transcript_index": _write_immutable(
            bucket.blob(paths["transcript_index"]),
            transcript_index_json,
            content_type="application/json",
        ),
        "draft_index": _write_immutable(
            bucket.blob(paths["draft_index"]),
            draft_index_json,
            content_type="application/json",
        ),
        "risk_inventory": _write_immutable(
            bucket.blob(paths["risk_inventory"]),
            risk_inventory_json,
            content_type="application/json",
        ),
    }
    if accepted:
        writes["final"] = _write_immutable(
            bucket.blob(paths["final"]),
            accepted_final,
            content_type="text/markdown; charset=utf-8",
        )
    writes["review"] = _write_immutable(
        bucket.blob(paths["review"]),
        review_json,
        content_type="application/json",
    )

    required = {
        paths["draft"]: draft,
        paths["transcript_index"]: transcript_index_json,
        paths["draft_index"]: draft_index_json,
        paths["risk_inventory"]: risk_inventory_json,
        paths["review"]: review_json,
    }
    if accepted:
        required[paths["final"]] = accepted_final
    for path, content in required.items():
        _verify_immutable(bucket.blob(path), content)

    writes["history_index"] = _register_review_in_history(
        bucket=bucket,
        episode_key=episode_key,
        review_id=review_id,
    )

    return {
        "review_id": review_id,
        "artifact_root": paths["root"],
        "status": validated_audit["status"] if accepted else "failed",
        "final_sha256": final_sha256,
        "writes": writes,
    }