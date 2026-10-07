"""Pure structured contract for grounded podcast-summary review."""

from __future__ import annotations

import json
import re

from .summary_guard import normalize_and_validate_summary_body

REVIEW_CONTRACT_ID = "summary-review-v1"
MAX_EVIDENCE_CHARS = 1200
MAX_ISSUES = 40
MAX_RESOLUTION_CHARS = 1000
MAX_CONTRACT_ATTEMPTS = 2
ISSUE_TYPES = (
    "coverage_omission",
    "unsupported_claim",
    "unsupported_claim_absent",
    "unsupported_precision",
    "unsupported_term_normalization",
    "epistemic_drift",
    "causal_overstatement",
    "recommendation_drift",
    "redundancy",
    "structure_violation",
    "other",
)
ISSUE_SEVERITIES = ("minor", "moderate", "material")
SEMANTIC_TYPES = {
    "unsupported_claim",
    "unsupported_precision",
    "unsupported_term_normalization",
    "epistemic_drift",
    "causal_overstatement",
    "recommendation_drift",
}
STRUCTURAL_TYPES = {"redundancy", "structure_violation"}
ALLOWED_H2 = ("TL;DR", "Key Ideas", "Research & Evidence", "Follow Up")

ISSUE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "type",
        "severity",
        "draft_excerpt",
        "transcript_evidence",
        "resolution",
    ],
    "properties": {
        "type": {"type": "string", "enum": list(ISSUE_TYPES)},
        "severity": {"type": "string", "enum": list(ISSUE_SEVERITIES)},
        "draft_excerpt": {
            "type": ["string", "null"],
            "maxLength": MAX_EVIDENCE_CHARS,
        },
        "transcript_evidence": {
            "type": ["string", "null"],
            "maxLength": MAX_EVIDENCE_CHARS,
        },
        "resolution": {
            "type": "string",
            "minLength": 1,
            "maxLength": MAX_RESOLUTION_CHARS,
        },
    },
}
SUMMARY_REVIEW_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["status", "issues_found", "final_markdown"],
    "properties": {
        "status": {"type": "string", "enum": ["pass", "revised", "fail"]},
        "issues_found": {
            "type": "array",
            "maxItems": MAX_ISSUES,
            "items": ISSUE_SCHEMA,
        },
        "final_markdown": {"type": ["string", "null"]},
    },
}


def build_contract_repair_payload(
    payload: dict,
    previous_content: str | None,
    failure_code: str,
) -> dict:
    """Build one same-request-model repair turn without weakening validation."""

    if not isinstance(payload, dict):
        raise TypeError("Summary review repair payload must be an object")
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("Summary review repair payload requires messages")
    if not isinstance(failure_code, str) or not re.fullmatch(
        r"[a-z0-9_]{1,80}", failure_code
    ):
        raise ValueError("Summary review repair failure code is invalid")

    repaired = dict(payload)
    repaired_messages = [dict(message) for message in messages]
    if isinstance(previous_content, str) and previous_content.strip():
        repaired_messages.append(
            {"role": "assistant", "content": previous_content}
        )
    repaired_messages.append(
        {
            "role": "user",
            "content": (
                "Your previous response failed the strict review contract with code "
                f"{failure_code}. Return the complete structured response again. "
                "For every non-null draft_excerpt or transcript_evidence, copy one "
                "short contiguous substring character-for-character from its supplied "
                "source. Do not add or remove punctuation, capitalization, whitespace, "
                "or Markdown markers; do not use ellipses or combine separate spans. "
                "Use null only where the issue-type contract permits it. Recheck the "
                "status/issues/final_markdown relationship before answering."
            ),
        }
    )
    repaired["messages"] = repaired_messages
    return repaired


def aggregate_completion_metadata(
    attempts: list[dict],
    *,
    attempt_count: int,
) -> dict:
    """Keep final identity while summing known usage across contract attempts."""

    if not isinstance(attempt_count, int) or isinstance(attempt_count, bool) or attempt_count < 1:
        raise ValueError("Summary review attempt_count must be a positive integer")
    if not isinstance(attempts, list) or any(not isinstance(item, dict) for item in attempts):
        raise ValueError("Summary review completion attempts must be objects")

    result: dict = {"attempt_count": attempt_count}
    if attempts:
        final = attempts[-1]
        for field in ("completion_id", "served_model", "served_provider"):
            value = final.get(field)
            if isinstance(value, str) and value:
                result[field] = value

    for field in ("prompt_tokens", "completion_tokens", "total_tokens", "cost"):
        values = [item[field] for item in attempts if field in item]
        if values:
            result[field] = sum(values)
    return result


def normalize_newlines(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("Review text must be a string")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def parse_review_content(content: str) -> dict:
    if not isinstance(content, str):
        raise ValueError("Summary review response must contain JSON text")
    parsed = json.loads(content.strip())
    if not isinstance(parsed, dict):
        raise ValueError("Summary review response must be a JSON object")
    return parsed


def validate_review_shape(payload: dict) -> dict:
    """Validate only the strict structured-response shape.

    This deliberately does not enforce cross-field/source semantics such as
    pass requiring a null model-authored final or evidence excerpts matching
    sources. Those remain the responsibility of validate_review_result().
    """

    if not isinstance(payload, dict):
        raise ValueError("Summary review result must be an object")
    if set(payload) != {"status", "issues_found", "final_markdown"}:
        raise ValueError("Summary review result fields do not match the contract")

    status = payload.get("status")
    if status not in {"pass", "revised", "fail"}:
        raise ValueError("Unknown summary review status")

    issues = payload.get("issues_found")
    if not isinstance(issues, list):
        raise ValueError("issues_found must be an array")
    if len(issues) > MAX_ISSUES:
        raise ValueError(f"issues_found exceeds {MAX_ISSUES} items")

    for review_issue in issues:
        if not isinstance(review_issue, dict):
            raise ValueError("Each issues_found item must be an object")
        if set(review_issue) != {
            "type",
            "severity",
            "draft_excerpt",
            "transcript_evidence",
            "resolution",
        }:
            raise ValueError("Summary review issue fields do not match the contract")

        if review_issue.get("type") not in ISSUE_TYPES:
            raise ValueError("Unknown summary review issue type")
        if review_issue.get("severity") not in ISSUE_SEVERITIES:
            raise ValueError("Unknown summary review issue severity")

        resolution = review_issue.get("resolution")
        if not isinstance(resolution, str) or not resolution.strip():
            raise ValueError("Summary review issue resolution is required")
        if len(resolution) > MAX_RESOLUTION_CHARS:
            raise ValueError(
                f"Summary review issue resolution exceeds {MAX_RESOLUTION_CHARS} characters"
            )

        for field_name in ("draft_excerpt", "transcript_evidence"):
            value = review_issue.get(field_name)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{field_name} must be a string or null")
            if isinstance(value, str) and len(value) > MAX_EVIDENCE_CHARS:
                raise ValueError(
                    f"{field_name} exceeds {MAX_EVIDENCE_CHARS} characters"
                )

    final_markdown = payload.get("final_markdown")
    if final_markdown is not None and not isinstance(final_markdown, str):
        raise ValueError("final_markdown must be a string or null")

    return payload


def _require_excerpt(value: str | None, source: str, field_name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} is required")
    if len(value) > MAX_EVIDENCE_CHARS:
        raise ValueError(f"{field_name} exceeds {MAX_EVIDENCE_CHARS} characters")
    if value not in source:
        raise ValueError(f"{field_name} is not an exact source substring")


def _validate_optional_excerpt(
    value: str | None,
    source: str,
    field_name: str,
) -> None:
    if value is None:
        return
    _require_excerpt(value, source, field_name)


def _validate_issue(issue: dict, draft: str, transcript: str) -> None:
    if not isinstance(issue, dict):
        raise ValueError("Each issues_found item must be an object")
    if set(issue) != {
        "type",
        "severity",
        "draft_excerpt",
        "transcript_evidence",
        "resolution",
    }:
        raise ValueError("Summary review issue fields do not match the contract")

    issue_type = issue.get("type")
    if issue_type not in ISSUE_TYPES:
        raise ValueError("Unknown summary review issue type")
    if issue.get("severity") not in ISSUE_SEVERITIES:
        raise ValueError("Unknown summary review issue severity")

    resolution = issue.get("resolution")
    if not isinstance(resolution, str) or not resolution.strip():
        raise ValueError("Summary review issue resolution is required")
    if len(resolution) > MAX_RESOLUTION_CHARS:
        raise ValueError(
            f"Summary review issue resolution exceeds {MAX_RESOLUTION_CHARS} characters"
        )

    draft_excerpt = issue.get("draft_excerpt")
    transcript_evidence = issue.get("transcript_evidence")
    normalized_draft_excerpt = (
        normalize_newlines(draft_excerpt)
        if isinstance(draft_excerpt, str)
        else draft_excerpt
    )
    normalized_transcript_evidence = (
        normalize_newlines(transcript_evidence)
        if isinstance(transcript_evidence, str)
        else transcript_evidence
    )

    if issue_type == "coverage_omission":
        _validate_optional_excerpt(
            normalized_draft_excerpt,
            draft,
            "draft_excerpt",
        )
        _require_excerpt(
            normalized_transcript_evidence,
            transcript,
            "transcript_evidence",
        )
    elif issue_type == "unsupported_claim_absent":
        _require_excerpt(normalized_draft_excerpt, draft, "draft_excerpt")
        if normalized_transcript_evidence is not None:
            raise ValueError(
                "transcript_evidence must be null when a claim is absent from the transcript"
            )
    elif issue_type in SEMANTIC_TYPES:
        _require_excerpt(normalized_draft_excerpt, draft, "draft_excerpt")
        _require_excerpt(
            normalized_transcript_evidence,
            transcript,
            "transcript_evidence",
        )
    elif issue_type in STRUCTURAL_TYPES:
        _require_excerpt(normalized_draft_excerpt, draft, "draft_excerpt")
        _validate_optional_excerpt(
            normalized_transcript_evidence,
            transcript,
            "transcript_evidence",
        )
    else:
        if normalized_draft_excerpt is None and normalized_transcript_evidence is None:
            raise ValueError("other issue requires at least one evidence excerpt")
        _validate_optional_excerpt(
            normalized_draft_excerpt,
            draft,
            "draft_excerpt",
        )
        _validate_optional_excerpt(
            normalized_transcript_evidence,
            transcript,
            "transcript_evidence",
        )


def validate_final_markdown(markdown: str) -> None:
    if not isinstance(markdown, str) or not markdown.strip():
        raise ValueError("Summary review final Markdown must be a non-empty string")

    normalize_and_validate_summary_body(markdown)
    text = normalize_newlines(markdown)
    headings = re.findall(r"^##\s+(.+?)\s*$", text, flags=re.MULTILINE)
    if headings[:2] != ["TL;DR", "Key Ideas"]:
        raise ValueError("Summary review final Markdown must start with TL;DR then Key Ideas")
    if any(name not in ALLOWED_H2 for name in headings):
        raise ValueError("Summary review final Markdown contains an unsupported section")
    if len(headings) != len(set(headings)):
        raise ValueError("Summary review final Markdown contains duplicate H2 sections")
    if headings != [name for name in ALLOWED_H2 if name in headings]:
        raise ValueError("Summary review final Markdown sections are out of order")


def validate_review_result(payload: dict, draft: str, transcript: str) -> dict:
    """Validate a grounded reviewer result against the supplied sources."""

    validate_review_shape(payload)

    if not isinstance(payload, dict):
        raise ValueError("Summary review result must be an object")
    if set(payload) != {"status", "issues_found", "final_markdown"}:
        raise ValueError("Summary review result fields do not match the contract")

    status = payload.get("status")
    if status not in {"pass", "revised", "fail"}:
        raise ValueError("Unknown summary review status")

    issues = payload.get("issues_found")
    if not isinstance(issues, list):
        raise ValueError("issues_found must be an array")
    if len(issues) > MAX_ISSUES:
        raise ValueError(f"issues_found exceeds {MAX_ISSUES} items")

    normalized_draft = normalize_newlines(draft)
    normalized_transcript = normalize_newlines(transcript)
    for review_issue in issues:
        _validate_issue(review_issue, normalized_draft, normalized_transcript)

    final = payload.get("final_markdown")
    if status == "pass":
        if issues:
            raise ValueError("pass status requires zero issues")
        if final is not None:
            raise ValueError("pass status requires null final Markdown")
        validate_final_markdown(normalized_draft)
    elif status == "revised":
        if not issues:
            raise ValueError("revised status requires at least one issue")
        if not isinstance(final, str) or not final.strip():
            raise ValueError("revised status requires non-empty final Markdown")
        if normalize_newlines(final) == normalized_draft:
            raise ValueError("revised status requires changed final Markdown")
        validate_final_markdown(final)
    else:
        if final is not None:
            raise ValueError("fail status requires null final Markdown")

    return payload


def accepted_final_markdown(payload: dict, draft: str) -> str | None:
    """Resolve the accepted user-facing Markdown after validation.

    A pass deliberately carries no model-authored Markdown. Python materializes the
    original draft so a no-change review cannot restyle or re-emit it. Revised
    output uses the reviewer-authored final; fail has no accepted final.
    """

    status = payload.get("status")
    if status == "pass":
        return normalize_newlines(draft)
    if status == "revised":
        final = payload.get("final_markdown")
        if not isinstance(final, str):
            raise ValueError("revised status has no accepted final Markdown")
        return normalize_newlines(final)
    if status == "fail":
        return None
    raise ValueError("Unknown summary review status")
