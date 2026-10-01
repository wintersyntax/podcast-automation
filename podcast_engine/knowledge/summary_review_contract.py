"""Pure structured contract for grounded podcast-summary review."""

from __future__ import annotations

import json
import re

from .summary_review_evidence import (
    build_draft_block_index,
    build_risk_inventory,
    draft_blocks_by_id,
    normalize_newlines,
    risks_by_id,
    transcript_spans_by_id,
)
from .summary_guard import normalize_and_validate_summary_body

REVIEW_CONTRACT_ID = "summary-review-v2"
MAX_ISSUES = 40
MAX_RESOLUTION_CHARS = 1000
MAX_REPAIR_TURNS_PER_CASE = 1
MAX_APPROVED_REPLACEMENT_TEXT_CHARS = 200
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
RISK_DISPOSITIONS = ("supported", "not_applicable", "issue")

RISK_ASSESSMENT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "risk_id",
        "disposition",
        "issue_type",
        "severity",
        "draft_block_ids",
        "transcript_span_ids",
        "resolution",
        "approved_replacement_text",
    ],
    "properties": {
        "risk_id": {"type": "string"},
        "disposition": {"type": "string", "enum": list(RISK_DISPOSITIONS)},
        "issue_type": {
            "type": ["string", "null"],
            "enum": [*ISSUE_TYPES, None],
        },
        "severity": {
            "type": ["string", "null"],
            "enum": [*ISSUE_SEVERITIES, None],
        },
        "draft_block_ids": {
            "type": "array",
            "items": {"type": "string"},
        },
        "transcript_span_ids": {
            "type": "array",
            "items": {"type": "string"},
        },
        "resolution": {
            "type": "string",
            "minLength": 1,
            "maxLength": MAX_RESOLUTION_CHARS,
        },
        "approved_replacement_text": {
            "type": ["string", "null"],
            "maxLength": MAX_APPROVED_REPLACEMENT_TEXT_CHARS,
        },
    },
}

ADDITIONAL_ISSUE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "issue_type",
        "severity",
        "draft_block_ids",
        "transcript_span_ids",
        "resolution",
        "approved_replacement_text",
    ],
    "properties": {
        "issue_type": {"type": "string", "enum": list(ISSUE_TYPES)},
        "severity": {"type": "string", "enum": list(ISSUE_SEVERITIES)},
        "draft_block_ids": {
            "type": "array",
            "items": {"type": "string"},
        },
        "transcript_span_ids": {
            "type": "array",
            "items": {"type": "string"},
        },
        "resolution": {
            "type": "string",
            "minLength": 1,
            "maxLength": MAX_RESOLUTION_CHARS,
        },
        "approved_replacement_text": {
            "type": ["string", "null"],
            "maxLength": MAX_APPROVED_REPLACEMENT_TEXT_CHARS,
        },
    },
}

SUMMARY_REVIEW_AUDIT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["status", "risk_assessments", "additional_issues"],
    "properties": {
        "status": {"type": "string", "enum": ["pass", "revised", "fail"]},
        "risk_assessments": {
            "type": "array",
            "items": RISK_ASSESSMENT_SCHEMA,
        },
        "additional_issues": {
            "type": "array",
            "maxItems": MAX_ISSUES,
            "items": ADDITIONAL_ISSUE_SCHEMA,
        },
    },
}

SUMMARY_REVIEW_EDIT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["resolved_issue_ids", "final_markdown"],
    "properties": {
        "resolved_issue_ids": {
            "type": "array",
            "items": {"type": "string"},
        },
        "final_markdown": {"type": "string"},
    },
}

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

# approved_replacement_text is a narrow, Python-validated audit field: only
# these two issue types describe a bounded literal-text correction (a wrong
# number/unit, or a term normalized away from the source's own wording),
# where the model can name the single replacement string it is asserting is
# source-supported. It is never a general license to introduce new material;
# _validate_no_unsupported_new_material treats a declared value as a third,
# narrowly-scoped grounding source alongside the draft and transcript.
APPROVED_REPLACEMENT_TEXT_ISSUE_TYPES = frozenset(
    {"unsupported_precision", "unsupported_term_normalization"}
)


def aggregate_completion_metadata(
    attempts: list[dict],
    *,
    attempt_count: int,
) -> dict:
    """Keep final identity while summing known usage across contract attempts."""

    if (
        not isinstance(attempt_count, int)
        or isinstance(attempt_count, bool)
        or attempt_count < 1
    ):
        raise ValueError("Summary review attempt_count must be a positive integer")
    if not isinstance(attempts, list) or any(
        not isinstance(item, dict) for item in attempts
    ):
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


def parse_review_content(content: str) -> dict:
    if not isinstance(content, str):
        raise ValueError("Summary review response must contain JSON text")
    parsed = json.loads(content.strip())
    if not isinstance(parsed, dict):
        raise ValueError("Summary review response must be a JSON object")
    return parsed


def slugify_repair_detail_code(message: object) -> str:
    """Derive a machine-safe repair-turn detail code from a real error message.

    This is used only for the text shown to the model in a repair-turn prompt.
    It is deliberately independent from any terminal machine-readable failure
    code (for example ``SummaryReviewValidationError.code`` or
    ``ModelContractExhausted.failure_code``), which must remain unchanged for
    existing diagnostics/lookup consumers.
    """

    if not isinstance(message, str):
        return "invalid_contract"
    tokens = re.findall(r"[a-z0-9]+", message.casefold())
    if not tokens:
        return "invalid_contract"

    slug_tokens: list[str] = []
    length = 0
    for token in tokens:
        addition = len(token) if not slug_tokens else len(token) + 1
        if length + addition > 80:
            break
        slug_tokens.append(token)
        length += addition

    if not slug_tokens:
        return "invalid_contract"
    return "_".join(slug_tokens)


def _build_phase_repair_payload(
    payload: dict,
    previous_content: str | None,
    failure_code: str,
    *,
    phase: str,
) -> dict:
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
        repaired_messages.append({"role": "assistant", "content": previous_content})

    if phase == "AUDIT":
        instruction = (
            "Your previous AUDIT response failed strict machine validation with code "
            f"{failure_code}. Return the complete AUDIT response again and only the "
            "AUDIT response. Adjudicate every supplied R#### exactly once. Use only "
            "the supplied Python-owned R####, D####, and S#### IDs; do not invent, "
            "alter, add, or omit identities. Preserve the grounded inputs and fix only "
            "the bounded validation failure."
        )
    elif phase == "EDIT":
        instruction = (
            "Your previous EDIT response failed strict machine validation with code "
            f"{failure_code}. Return the complete EDIT response again and only the "
            "EDIT response. Preserve the validated audit and resolve exactly the "
            "supplied immutable obligation IDs, with no added, removed, or "
            "re-adjudicated obligations. Preserve all supplied request identities and "
            "fix only the bounded validation failure."
        )
    else:
        raise ValueError("Unknown summary review repair phase")

    repaired_messages.append({"role": "user", "content": instruction})
    repaired["messages"] = repaired_messages
    return repaired


def build_audit_repair_payload(
    payload: dict,
    previous_content: str | None,
    failure_code: str,
) -> dict:
    """Build one same-identity repair turn for an invalid AUDIT response."""

    return _build_phase_repair_payload(
        payload,
        previous_content,
        failure_code,
        phase="AUDIT",
    )


def build_edit_repair_payload(
    payload: dict,
    previous_content: str | None,
    failure_code: str,
) -> dict:
    """Build one same-identity repair turn for an invalid EDIT response."""

    return _build_phase_repair_payload(
        payload,
        previous_content,
        failure_code,
        phase="EDIT",
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


def _require_exact_fields(
    value: object,
    *,
    expected: set[str],
    label: str,
    optional: frozenset[str] = frozenset(),
) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    present = set(value)
    if not expected.issubset(present) or not present.issubset(expected | optional):
        raise ValueError(f"{label} fields do not match the contract")
    return value


def _validate_approved_replacement_text(
    value: object,
    *,
    issue_type: str | None,
) -> str | None:
    if value is None:
        return None
    if issue_type not in APPROVED_REPLACEMENT_TEXT_ISSUE_TYPES:
        raise ValueError(
            "approved_replacement_text is only permitted for "
            "unsupported_precision or unsupported_term_normalization issues"
        )
    if not isinstance(value, str) or not value.strip():
        raise ValueError("approved_replacement_text must be a non-empty string")
    if len(value) > MAX_APPROVED_REPLACEMENT_TEXT_CHARS:
        raise ValueError(
            "approved_replacement_text exceeds "
            f"{MAX_APPROVED_REPLACEMENT_TEXT_CHARS} characters"
        )
    return value


def _validate_resolution(value: object, *, field: str = "resolution") -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} is required")
    if len(value) > MAX_RESOLUTION_CHARS:
        raise ValueError(f"{field} exceeds {MAX_RESOLUTION_CHARS} characters")
    return value


def _validate_id_list(
    value: object,
    *,
    prefix: str,
    known: dict[str, dict],
    field: str,
) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array")

    unique: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"{field} must contain strings")
        if not re.fullmatch(rf"{re.escape(prefix)}\d{{4}}", item):
            raise ValueError(f"{field} contains an invalid ID")
        if item not in known:
            raise ValueError(f"{field} contains an unknown ID")
        unique.add(item)

    source_order = {item_id: index for index, item_id in enumerate(known)}
    return sorted(unique, key=source_order.__getitem__)


def _validate_issue_evidence(
    issue_type: str,
    draft_ids: list[str],
    span_ids: list[str],
) -> None:
    if issue_type == "coverage_omission":
        if draft_ids or not span_ids:
            raise ValueError(
                "coverage_omission requires transcript evidence and no draft evidence"
            )
        return

    if issue_type == "unsupported_claim_absent":
        if not draft_ids or span_ids:
            raise ValueError(
                "unsupported_claim_absent requires draft evidence and no transcript evidence"
            )
        return

    if issue_type in SEMANTIC_TYPES:
        if not draft_ids or not span_ids:
            raise ValueError(f"{issue_type} requires draft and transcript evidence")
        return

    if issue_type in STRUCTURAL_TYPES:
        if not draft_ids:
            raise ValueError(f"{issue_type} requires draft evidence")
        return

    if issue_type == "other":
        if not draft_ids and not span_ids:
            raise ValueError("other requires at least one evidence ID")
        return

    raise ValueError("Unknown summary review issue type")


def _validate_risk_assessments(
    assessments: object,
    review_context: dict,
) -> list[dict]:
    if not isinstance(assessments, list):
        raise ValueError("risk_assessments must be an array")

    risks = risks_by_id(review_context)
    blocks = draft_blocks_by_id(review_context)
    spans = transcript_spans_by_id(review_context)

    supplied: dict[str, dict] = {}
    expected_fields = {
        "risk_id",
        "disposition",
        "issue_type",
        "severity",
        "draft_block_ids",
        "transcript_span_ids",
        "resolution",
    }

    for raw in assessments:
        assessment = _require_exact_fields(
            raw,
            expected=expected_fields,
            label="risk assessment",
            optional=frozenset({"approved_replacement_text"}),
        )
        risk_id = assessment.get("risk_id")
        if not isinstance(risk_id, str) or not re.fullmatch(r"R\d{4}", risk_id):
            raise ValueError("risk assessment contains an invalid risk_id")
        if risk_id not in risks:
            raise ValueError("risk assessment contains an unknown risk_id")
        if risk_id in supplied:
            raise ValueError("risk assessment contains a duplicate risk_id")
        supplied[risk_id] = assessment

    if set(supplied) != set(risks):
        raise ValueError(
            "risk_assessments must adjudicate every Python-owned risk exactly once"
        )

    normalized: list[dict] = []
    for risk_id, risk in risks.items():
        assessment = supplied[risk_id]
        disposition = assessment.get("disposition")
        if disposition not in RISK_DISPOSITIONS:
            raise ValueError("Unknown risk disposition")

        issue_type = assessment.get("issue_type")
        severity = assessment.get("severity")
        resolution = _validate_resolution(assessment.get("resolution"))

        draft_ids = _validate_id_list(
            assessment.get("draft_block_ids"),
            prefix="D",
            known=blocks,
            field="draft_block_ids",
        )
        span_ids = _validate_id_list(
            assessment.get("transcript_span_ids"),
            prefix="S",
            known=spans,
            field="transcript_span_ids",
        )

        inventory_block_id = risk["draft_block_id"]

        if disposition == "supported":
            if issue_type is not None or severity is not None:
                raise ValueError("supported risk must not declare issue_type or severity")
            if draft_ids != [inventory_block_id]:
                raise ValueError(
                    "supported risk must reference exactly its inventory-owned draft block"
                )
            if not span_ids:
                raise ValueError("supported risk requires transcript evidence")

        elif disposition == "not_applicable":
            if issue_type is not None or severity is not None:
                raise ValueError(
                    "not_applicable risk must not declare issue_type or severity"
                )
            if draft_ids != [inventory_block_id]:
                raise ValueError(
                    "not_applicable risk must reference exactly its inventory-owned draft block"
                )
            if span_ids:
                raise ValueError(
                    "not_applicable risk must not reference transcript evidence"
                )

        else:
            if issue_type not in ISSUE_TYPES:
                raise ValueError("issue risk requires a known issue_type")
            if severity not in ISSUE_SEVERITIES:
                raise ValueError("issue risk requires a known severity")
            if inventory_block_id not in draft_ids:
                raise ValueError(
                    "issue risk must include its inventory-owned draft block"
                )
            _validate_issue_evidence(issue_type, draft_ids, span_ids)

        approved_replacement_text = _validate_approved_replacement_text(
            assessment.get("approved_replacement_text"),
            issue_type=issue_type,
        )

        normalized.append(
            {
                "risk_id": risk_id,
                "disposition": disposition,
                "issue_type": issue_type,
                "severity": severity,
                "draft_block_ids": draft_ids,
                "transcript_span_ids": span_ids,
                "resolution": resolution,
                "approved_replacement_text": approved_replacement_text,
            }
        )

    return normalized


def _validate_additional_issues(
    issues: object,
    review_context: dict,
) -> list[dict]:
    if not isinstance(issues, list):
        raise ValueError("additional_issues must be an array")
    if len(issues) > MAX_ISSUES:
        raise ValueError(f"additional_issues exceeds {MAX_ISSUES} items")

    blocks = draft_blocks_by_id(review_context)
    spans = transcript_spans_by_id(review_context)
    expected_fields = {
        "issue_type",
        "severity",
        "draft_block_ids",
        "transcript_span_ids",
        "resolution",
    }

    normalized: list[dict] = []
    for raw in issues:
        issue = _require_exact_fields(
            raw,
            expected=expected_fields,
            label="additional issue",
            optional=frozenset({"approved_replacement_text"}),
        )
        issue_type = issue.get("issue_type")
        severity = issue.get("severity")
        if issue_type not in ISSUE_TYPES:
            raise ValueError("additional issue requires a known issue_type")
        if severity not in ISSUE_SEVERITIES:
            raise ValueError("additional issue requires a known severity")

        draft_ids = _validate_id_list(
            issue.get("draft_block_ids"),
            prefix="D",
            known=blocks,
            field="draft_block_ids",
        )
        span_ids = _validate_id_list(
            issue.get("transcript_span_ids"),
            prefix="S",
            known=spans,
            field="transcript_span_ids",
        )
        _validate_issue_evidence(issue_type, draft_ids, span_ids)

        normalized.append(
            {
                "issue_type": issue_type,
                "severity": severity,
                "draft_block_ids": draft_ids,
                "transcript_span_ids": span_ids,
                "resolution": _validate_resolution(issue.get("resolution")),
                "approved_replacement_text": _validate_approved_replacement_text(
                    issue.get("approved_replacement_text"),
                    issue_type=issue_type,
                ),
            }
        )

    return normalized


def validate_audit_result(
    payload: dict,
    transcript: str,
    review_context: dict,
) -> dict:
    _require_exact_fields(
        payload,
        expected={"status", "risk_assessments", "additional_issues"},
        label="audit result",
    )

    if not isinstance(transcript, str):
        raise ValueError("Audit transcript must be a string")
    if not isinstance(review_context, dict):
        raise ValueError("Audit review_context must be an object")

    status = payload.get("status")
    if status not in {"pass", "revised", "fail"}:
        raise ValueError("Unknown audit status")

    risk_assessments = _validate_risk_assessments(
        payload.get("risk_assessments"),
        review_context,
    )
    additional_issues = _validate_additional_issues(
        payload.get("additional_issues"),
        review_context,
    )

    actionable_count = sum(
        item["disposition"] == "issue" for item in risk_assessments
    ) + len(additional_issues)

    if status == "pass" and actionable_count:
        raise ValueError("pass audit requires zero issues")
    if status == "revised" and actionable_count == 0:
        raise ValueError("revised audit requires at least one issue")

    return {
        "status": status,
        "risk_assessments": risk_assessments,
        "additional_issues": additional_issues,
    }


def build_edit_obligations(validated_audit: dict) -> list[dict]:
    if not isinstance(validated_audit, dict):
        raise ValueError("Validated audit must be an object")

    status = validated_audit.get("status")
    if status == "pass":
        return []
    if status == "fail":
        raise ValueError("fail audit cannot produce edit obligations")
    if status != "revised":
        raise ValueError("Unknown audit status")

    obligations: list[dict] = []

    for assessment in validated_audit.get("risk_assessments", []):
        if assessment.get("disposition") != "issue":
            continue
        obligations.append(
            {
                "issue_id": assessment["risk_id"],
                "issue_type": assessment["issue_type"],
                "severity": assessment["severity"],
                "draft_block_ids": list(assessment["draft_block_ids"]),
                "transcript_span_ids": list(assessment["transcript_span_ids"]),
                "resolution": assessment["resolution"],
                "approved_replacement_text": assessment.get(
                    "approved_replacement_text"
                ),
            }
        )

    for index, issue in enumerate(validated_audit.get("additional_issues", []), start=1):
        obligations.append(
            {
                "issue_id": f"I{index:04d}",
                "issue_type": issue["issue_type"],
                "severity": issue["severity"],
                "draft_block_ids": list(issue["draft_block_ids"]),
                "transcript_span_ids": list(issue["transcript_span_ids"]),
                "resolution": issue["resolution"],
                "approved_replacement_text": issue.get("approved_replacement_text"),
            }
        )

    if not obligations:
        raise ValueError("revised audit requires edit obligations")
    return obligations


# Closed inflection groups for the small fixed keyword vocabularies detected by
# _CERTAINTY_PATTERN and _RECOMMENDATION_PATTERN in summary_review_evidence.py.
# A surface in one of these groups is treated as source-supported when ANY other
# member of its own group is present verbatim in the draft or transcript -- this
# tolerates ordinary grammatical inflection of an already-supported word (e.g. the
# transcript's "recommend" supporting the edit's "recommended") without loosening
# the check for any word outside these closed, Python-owned lists.
_UNSUPPORTED_MATERIAL_INFLECTION_GROUPS = (
    frozenset({"recommend", "recommended"}),
    frozenset({"causes", "caused"}),
    frozenset({"proves", "proven"}),
)


def _unsupported_material_inflection_supported(
    surface: str,
    normalized_draft: str,
    normalized_transcript: str,
) -> bool:
    for group in _UNSUPPORTED_MATERIAL_INFLECTION_GROUPS:
        if surface not in group:
            continue
        return any(
            variant in normalized_draft or variant in normalized_transcript
            for variant in group
        )
    return False


def _dehyphenated(text: str) -> str:
    return text.replace("-", " ")


# Risk kinds (see summary_review_evidence.RISK_KINDS) a Python-declared
# approved_replacement_text may ground: the bounded literal-text correction
# an unsupported_precision/unsupported_term_normalization obligation names.
_APPROVED_REPLACEMENT_TOLERANT_KINDS = frozenset(
    {"numeric_precision", "named_or_technical_term"}
)


def _approved_replacement_text_supported(
    surface: str,
    kind: str,
    obligations: list[dict] | None,
) -> bool:
    if kind not in _APPROVED_REPLACEMENT_TOLERANT_KINDS or not obligations:
        return False
    for obligation in obligations:
        approved = obligation.get("approved_replacement_text")
        if not isinstance(approved, str):
            continue
        if normalize_newlines(approved).casefold() == surface:
            return True
    return False


def _validate_no_unsupported_new_material(
    final_markdown: str,
    draft: str,
    transcript: str,
    obligations: list[dict] | None = None,
) -> None:
    final_index = build_draft_block_index(final_markdown)
    final_inventory = build_risk_inventory(final_markdown, final_index)
    normalized_draft = normalize_newlines(draft).casefold()
    normalized_transcript = normalize_newlines(transcript).casefold()

    for risk in final_inventory["risks"]:
        surface = normalize_newlines(risk["surface"]).casefold()
        if surface in normalized_draft or surface in normalized_transcript:
            continue

        if risk["kind"] in (
            "certainty_or_causality",
            "recommendation_language",
        ) and _unsupported_material_inflection_supported(
            surface, normalized_draft, normalized_transcript
        ):
            continue

        if (
            risk["kind"] == "named_or_technical_term"
            and "-" in surface
            and not any(character.isdigit() for character in surface)
        ):
            dehyphenated_surface = _dehyphenated(surface)
            if dehyphenated_surface in _dehyphenated(
                normalized_draft
            ) or dehyphenated_surface in _dehyphenated(normalized_transcript):
                continue

        if _approved_replacement_text_supported(surface, risk["kind"], obligations):
            continue

        raise ValueError("revised edit contains unsupported new material")


_SYNTHESIS_HEADING_NORMALIZED_TEXTS = frozenset({"tl;dr", "tldr"})
_HEADING_LINE_PATTERN = re.compile(r"^#{1,6}\s+")


def _synthesis_content_block_ids(review_context: dict) -> set[str]:
    """Block IDs of the content directly under a canonical TL;DR heading.

    Recognition is structural and Python-owned (heading text only, never a
    model declaration): resolving a coverage_omission obligation may
    legitimately update the synthesis line(s) under such a heading to
    reflect newly restored coverage, without opening any other untargeted
    block to a silent edit.
    """
    blocks = draft_blocks_by_id(review_context)
    synthesis_ids: set[str] = set()
    in_synthesis_section = False
    for block_id, block in blocks.items():
        text = block["text"]
        if _HEADING_LINE_PATTERN.match(text):
            heading_text = _HEADING_LINE_PATTERN.sub("", text, count=1).strip().casefold()
            in_synthesis_section = heading_text in _SYNTHESIS_HEADING_NORMALIZED_TEXTS
            continue
        if in_synthesis_section:
            synthesis_ids.add(block_id)
    return synthesis_ids


def _validate_untargeted_blocks_preserved(
    final_markdown: str,
    review_context: dict,
    obligations: list[dict],
) -> None:
    targeted_block_ids = {
        block_id
        for obligation in obligations
        for block_id in obligation["draft_block_ids"]
    }
    if any(
        obligation["issue_type"] == "coverage_omission"
        for obligation in obligations
    ):
        targeted_block_ids |= _synthesis_content_block_ids(review_context)
    required_counts: dict[str, int] = {}
    for block_id, block in draft_blocks_by_id(review_context).items():
        if block_id in targeted_block_ids:
            continue
        text = block["text"]
        required_counts[text] = required_counts.get(text, 0) + 1

    final_counts: dict[str, int] = {}
    for block in build_draft_block_index(final_markdown)["blocks"]:
        text = block["text"]
        final_counts[text] = final_counts.get(text, 0) + 1

    for text, required_count in required_counts.items():
        if final_counts.get(text, 0) < required_count:
            raise ValueError(
                "revised edit violates non-regression for untargeted draft blocks"
            )


def _validate_targeted_obligations_resolved(
    final_markdown: str,
    review_context: dict,
    obligations: list[dict],
) -> None:
    final_blocks = {
        block["block_id"]: block
        for block in build_draft_block_index(final_markdown)["blocks"]
    }
    risks = risks_by_id(review_context)

    for obligation in obligations:
        issue_id = obligation["issue_id"]
        if not re.fullmatch(r"R\d{4}", issue_id):
            continue

        risk = risks[issue_id]
        inventory_block_id = risk["draft_block_id"]
        if inventory_block_id not in obligation["draft_block_ids"]:
            raise ValueError(
                "revised edit targeted obligation is detached from its Python-owned risk"
            )

        final_block = final_blocks.get(inventory_block_id)
        if final_block is None:
            continue
        surface = normalize_newlines(risk["surface"]).casefold()
        final_text = normalize_newlines(final_block["text"]).casefold()
        if surface in final_text:
            raise ValueError("revised edit leaves targeted obligation unresolved")


def validate_edit_result(
    payload: dict,
    validated_audit: dict,
    draft: str,
    transcript: str,
    review_context: dict,
) -> dict:
    _require_exact_fields(
        payload,
        expected={"resolved_issue_ids", "final_markdown"},
        label="edit result",
    )

    if validated_audit.get("status") == "fail":
        raise ValueError("fail audit cannot have an edit result")

    obligations = build_edit_obligations(validated_audit)
    expected_ids = [item["issue_id"] for item in obligations]

    resolved = payload.get("resolved_issue_ids")
    if not isinstance(resolved, list) or any(not isinstance(item, str) for item in resolved):
        raise ValueError("resolved_issue_ids must be an array of strings")
    if len(resolved) != len(set(resolved)):
        raise ValueError("resolved_issue_ids must not contain duplicates")
    if resolved != expected_ids:
        raise ValueError(
            "resolved_issue_ids must equal the Python-owned obligation set exactly"
        )

    final = payload.get("final_markdown")
    if not isinstance(final, str) or not final.strip():
        raise ValueError("edit final_markdown must be a non-empty string")

    normalized_final = normalize_newlines(final)
    normalized_draft = normalize_newlines(draft)

    if validated_audit.get("status") == "pass":
        if final != draft:
            raise ValueError("pass edit must preserve the draft byte-for-byte")
    elif validated_audit.get("status") == "revised":
        if normalized_final == normalized_draft:
            raise ValueError("revised edit must change the draft")
        validate_final_markdown(normalized_final)
        _validate_no_unsupported_new_material(
            normalized_final,
            normalized_draft,
            transcript,
            obligations,
        )
        _validate_untargeted_blocks_preserved(
            normalized_final,
            review_context,
            obligations,
        )
        _validate_targeted_obligations_resolved(
            normalized_final,
            review_context,
            obligations,
        )

    return {
        "resolved_issue_ids": list(resolved),
        "final_markdown": final,
    }


def accepted_final_markdown(
    audit: dict,
    edit: dict | None,
    draft: str,
) -> str | None:
    if audit.get("status") in {"pass", "revised"} and edit is not None:
        final = edit.get("final_markdown")
        if not isinstance(final, str):
            raise ValueError("Accepted V2 edit requires final_markdown")
        return normalize_newlines(final)
    return None
