from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import re
import unicodedata


_ALLOWED_PARTITIONS = {"development", "held_out"}
_ALLOWED_HUMAN_OUTCOMES = {
    "apple",
    "whisper",
    "custom",
    "needs_audio",
}
_REPRESENTATION_TOKEN = re.compile(r"[^\W\d_]+|\d+", re.UNICODE)
_CLOCK_DOTTED = re.compile(r"^\s*(\d{1,2})\.(\d{2})\s*$")
_CLOCK_SPLIT = re.compile(r"^\s*(\d{1,2})\s*,\s*(\d{2})\s*$")
_CLOCK_CONTEXT_CUE = re.compile(
    r"\b(?:starts?|begins?|commences?)\s+at\b|\b(?:a\.m\.|p\.m\.)\b",
    re.IGNORECASE,
)
_NON_CLOCK_CONTEXT = re.compile(
    r"\b(?:minutes?|seconds?|hours?|milliseconds?|"
    r"kg|kgs|kilograms?|lb|lbs|pounds?|grams?|mg|mcg|"
    r"percent|percentage)\b|%",
    re.IGNORECASE,
)


def _representation_tokens(text: str) -> list[str]:
    return [
        match.group(0).casefold()
        for match in _REPRESENTATION_TOKEN.finditer(text)
    ]


def _normalize_et_cetera(text: str) -> list[str]:
    tokens = _representation_tokens(text)
    normalized: list[str] = []
    index = 0
    while index < len(tokens):
        if (
            tokens[index] == "et"
            and index + 1 < len(tokens)
            and tokens[index + 1] == "cetera"
        ):
            normalized.append("etc")
            index += 2
            continue
        normalized.append(tokens[index])
        index += 1
    return normalized


def _clock_value(text: str) -> tuple[int, int, str] | None:
    for style, pattern in (
        ("dotted", _CLOCK_DOTTED),
        ("split", _CLOCK_SPLIT),
    ):
        match = pattern.fullmatch(text)
        if match is None:
            continue
        hour = int(match.group(1))
        minute = int(match.group(2))
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            return hour, minute, style
    return None


def _clock_context_proves_time(context: str) -> bool:
    if not context or _NON_CLOCK_CONTEXT.search(context):
        return False
    return bool(_CLOCK_CONTEXT_CUE.search(context))


def representation_equivalence(
    apple_text: str,
    whisper_text: str,
    *,
    apple_context: str = "",
    whisper_context: str = "",
) -> dict | None:
    """Return only mechanically provable, source-preserving equivalence."""

    apple_tokens = _representation_tokens(apple_text)
    whisper_tokens = _representation_tokens(whisper_text)
    apple_et_cetera = _normalize_et_cetera(apple_text)
    whisper_et_cetera = _normalize_et_cetera(whisper_text)

    if (
        apple_tokens != whisper_tokens
        and apple_et_cetera == whisper_et_cetera
        and "etc" in apple_et_cetera
    ):
        return {
            "equivalence_class": "et_cetera",
            "reason": "Spelled-out et cetera and etc preserve the same token sequence",
            "selection_policy": "preserve_primary_source",
        }

    apple_clock = _clock_value(apple_text)
    whisper_clock = _clock_value(whisper_text)
    if (
        apple_clock is not None
        and whisper_clock is not None
        and apple_clock[:2] == whisper_clock[:2]
        and apple_clock[2] != whisper_clock[2]
        and _clock_context_proves_time(apple_context)
        and _clock_context_proves_time(whisper_context)
    ):
        return {
            "equivalence_class": "clock_time",
            "reason": "Both sources encode the same context-proven clock time",
            "selection_policy": "preserve_primary_source",
        }

    return None


_BATCH_ALLOWED_KINDS = frozenset({
    "wording_difference",
    "transcription_difference",
})
_BATCH_ALLOWED_SEVERITY = "low"
_BATCH_ALLOWED_CATEGORY = "other"
_BATCH_ALLOWED_PRESERVATION_CLASS = "not_source_only"
_BATCH_ALLOWED_MERGE_ACTION = "review_kept_primary"


def batch_recommendation(item: dict) -> dict | None:
    """Return one Python-qualified low-risk source recommendation.

    Advisory triage is evidence only. This helper grants no transcript
    authority; it only marks a still-pending Human Review card as eligible
    for a later explicit human batch action.
    """

    if not isinstance(item, dict):
        return None

    required_types = {
        "id": int,
        "kind": str,
        "severity": str,
        "category": str,
        "apple_text": str,
        "whisper_text": str,
        "source_only": bool,
        "risk_reasons": list,
        "domain_terms": list,
        "citation_signal": bool,
        "preservation_class": str,
        "merge_action": str,
        "triage": dict,
    }
    for field, expected_type in required_types.items():
        if not isinstance(item.get(field), expected_type):
            return None

    if (
        item["kind"] not in _BATCH_ALLOWED_KINDS
        or item["severity"] != _BATCH_ALLOWED_SEVERITY
        or item["category"] != _BATCH_ALLOWED_CATEGORY
        or item["source_only"] is not False
        or item["risk_reasons"]
        or item["domain_terms"]
        or item["citation_signal"] is not False
        or item["preservation_class"] != _BATCH_ALLOWED_PRESERVATION_CLASS
        or item["merge_action"] != _BATCH_ALLOWED_MERGE_ACTION
        or item.get("anomaly") is not None
        or item.get("custom_edit") is not None
        or item.get("third_asr") is not None
        or item.get("representation_modified") is True
        or item.get("generation_stale") is True
    ):
        return None

    triage = item["triage"]
    source = {
        "recommend_apple": "apple",
        "recommend_whisper": "whisper",
    }.get(triage.get("recommendation"))
    if (
        triage.get("status") != "advisory"
        or triage.get("confidence") != "high"
        or source is None
        or triage.get("source") != source
    ):
        return None

    source_text = item.get(f"{source}_text")
    recommended_text = triage.get("text")
    if (
        not isinstance(source_text, str)
        or not isinstance(recommended_text, str)
        or recommended_text != source_text
    ):
        return None

    return {
        "id": item["id"],
        "source": source,
        "text": source_text,
        "reason": "high_confidence_exact_source_low_risk",
    }




ASSISTED_REVIEW_POLICY_VERSION = "human-review-assisted-v1"

_MAX_COMPARISON_TOKENS = 11

_COMPARISON_TOKEN = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*|\d+", re.UNICODE)


def comparison_tokens(text: str) -> list[str]:
    """Tokenize text for TASK-076 assisted-routing/scoring comparison.

    NFKC-normalizes and casefolds the text, then extracts letter/number
    tokens, keeping an internal apostrophe as part of its word (so
    "you're" is one token). Punctuation and whitespace are ignored.
    """

    normalized = unicodedata.normalize("NFKC", text).casefold()
    return _COMPARISON_TOKEN.findall(normalized)


_ASSISTED_ROUTING_ALLOWED_KINDS = frozenset({
    "wording_difference",
    "transcription_difference",
})
_ASSISTED_ROUTING_KIND_REASON_CODES = {
    "negation_mismatch": "protected_negation",
    "number_mismatch": "protected_protocol_number",
    "source_only": "source_only_semantic",
}
_ASSISTED_ROUTING_ALLOWED_CATEGORY = "other"
_ASSISTED_ROUTING_PROTECTED_REASON_BY_LABEL = {
    "negation": "protected_negation",
    "protocol_number": "protected_protocol_number",
    "unit": "protected_unit",
    "proper_name": "protected_proper_name",
    "citation": "protected_citation",
    "scientific_medical_term": "protected_domain_term",
    "supplement": "protected_domain_term",
    "training_term": "protected_domain_term",
    "exercise_name": "protected_domain_term",
}
_ASSISTED_ROUTING_REQUIRED_TYPES = {
    "kind": str,
    "category": str,
    "apple_text": str,
    "whisper_text": str,
    "source_only": bool,
    "risk_reasons": list,
    "domain_terms": list,
    "citation_signal": bool,
    "preservation_class": str,
    "merge_action": str,
}


def _malformed_routing_result() -> dict:
    return {
        "policy_version": ASSISTED_REVIEW_POLICY_VERSION,
        "eligible": False,
        "reason_codes": ["unknown_kind", "unknown_category"],
        "decision_scope": None,
        "token_counts": None,
    }


def _localization_invalid(item: dict) -> bool:
    """True if any present apple/whisper timestamp pair is malformed.

    A source whose start/end timestamp keys are both entirely absent is
    skipped rather than treated as invalid: not every card carries both
    sources' localization evidence.
    """

    for source in ("apple", "whisper"):
        start_key = f"{source}_start_timestamp"
        end_key = f"{source}_end_timestamp"
        if start_key not in item and end_key not in item:
            continue

        start = item.get(start_key)
        end = item.get(end_key)
        if (
            not isinstance(start, (int, float))
            or isinstance(start, bool)
            or not isinstance(end, (int, float))
            or isinstance(end, bool)
            or math.isnan(start)
            or math.isnan(end)
            or math.isinf(start)
            or math.isinf(end)
            or end < start
        ):
            return True

    return False


def assisted_routing(item: dict) -> dict:
    """Return TASK-076 assisted-routing/admission provenance for one card.

    This is deterministic workflow eligibility only. It never inspects
    compiler ``suggestion`` or advisory ``triage``, and it never returns a
    chosen source or text: human decision remains the sole transcript
    authority. Every known or malformed card shape gets a result with a
    policy version, an eligibility boolean, stable exclusion reason codes,
    the decision-scope identity used for comparison, and comparison-token
    counts (``None`` when the card shape itself is malformed).
    """

    if not isinstance(item, dict):
        return _malformed_routing_result()

    for field, expected_type in _ASSISTED_ROUTING_REQUIRED_TYPES.items():
        if not isinstance(item.get(field), expected_type):
            return _malformed_routing_result()

    focus = item.get("focus")
    if (
        isinstance(focus, dict)
        and isinstance(focus.get("apple_text"), str)
        and isinstance(focus.get("whisper_text"), str)
    ):
        decision_scope = "focus"
        apple_scope_text = focus["apple_text"]
        whisper_scope_text = focus["whisper_text"]
    else:
        decision_scope = "full"
        apple_scope_text = item["apple_text"]
        whisper_scope_text = item["whisper_text"]

    token_counts = {
        "apple": len(comparison_tokens(apple_scope_text)),
        "whisper": len(comparison_tokens(whisper_scope_text)),
    }

    reason_codes: list[str] = []

    kind_value = item["kind"]
    if kind_value in _ASSISTED_ROUTING_ALLOWED_KINDS:
        pass
    elif kind_value in _ASSISTED_ROUTING_KIND_REASON_CODES:
        reason_codes.append(_ASSISTED_ROUTING_KIND_REASON_CODES[kind_value])
    else:
        reason_codes.append("unknown_kind")

    category_value = item["category"]
    if category_value == _ASSISTED_ROUTING_ALLOWED_CATEGORY:
        pass
    elif category_value in _ASSISTED_ROUTING_PROTECTED_REASON_BY_LABEL:
        reason_codes.append(_ASSISTED_ROUTING_PROTECTED_REASON_BY_LABEL[category_value])
    else:
        reason_codes.append("unknown_category")

    for risk_reason in item["risk_reasons"]:
        reason_codes.append(
            _ASSISTED_ROUTING_PROTECTED_REASON_BY_LABEL.get(risk_reason, "unknown_category")
        )

    if item["domain_terms"]:
        reason_codes.append("protected_domain_term")

    if item["citation_signal"] is True:
        reason_codes.append("protected_citation")

    if item["source_only"] is True or item["preservation_class"] != "not_source_only":
        reason_codes.append("source_only_semantic")

    if item.get("anomaly") is not None:
        reason_codes.append("anomaly")

    if item["merge_action"] != "review_kept_primary":
        reason_codes.append("suspected_repetition")

    if item.get("custom_edit") is not None:
        reason_codes.append("custom_edit")

    if item.get("representation_modified") is True:
        reason_codes.append("representation_modified")

    if item.get("generation_stale") is True:
        reason_codes.append("stale_generation")

    if _localization_invalid(item):
        reason_codes.append("invalid_localization")

    if (
        token_counts["apple"] > _MAX_COMPARISON_TOKENS
        or token_counts["whisper"] > _MAX_COMPARISON_TOKENS
    ):
        reason_codes.append("span_over_limit")

    return {
        "policy_version": ASSISTED_REVIEW_POLICY_VERSION,
        "eligible": len(reason_codes) == 0,
        "reason_codes": reason_codes,
        "decision_scope": decision_scope,
        "token_counts": token_counts,
    }


def assisted_candidate(item: dict) -> bool:
    """Thin convenience wrapper: True only when assisted_routing admits item.

    This must not duplicate exclusion logic; it defers entirely to
    assisted_routing().
    """

    return bool(assisted_routing(item).get("eligible"))


def classify_review_anomaly(difference) -> dict | None:
    """Classify only explicit operational/alignment anomalies.

    This metadata is advisory routing evidence. It never chooses transcript
    text, changes a merge action, or turns Human Review into an automatic
    resolution.
    """

    for source in ("apple", "whisper"):
        start = getattr(difference, f"{source}_start_timestamp", None)
        end = getattr(difference, f"{source}_end_timestamp", None)
        if (
            isinstance(start, (int, float))
            and isinstance(end, (int, float))
            and end < start
        ):
            return {
                "kind": "audio_localization_unreliable",
                "reason": (
                    f"{source} audio interval ends before it starts; "
                    "localization evidence is unreliable"
                ),
            }

    if (
        getattr(difference, "source_only", False)
        and getattr(difference, "source_only_source", None)
        in {"apple", "whisper"}
    ):
        source = difference.source_only_source
        other_source = "whisper" if source == "apple" else "apple"
        changed_words = getattr(
            difference,
            f"changed_{source}_words",
            [],
        )
        other_words = getattr(
            difference,
            f"changed_{other_source}_words",
            [],
        )
        if (
            isinstance(changed_words, list)
            and isinstance(other_words, list)
            and len(changed_words) >= 40
            and not other_words
        ):
            return {
                "kind": "source_alignment_mismatch_candidate",
                "reason": (
                    f"{source} has a one-sided {len(changed_words)}-word "
                    "span; route as an alignment/operational anomaly "
                    "instead of asking the AI resolver to choose transcript truth"
                ),
            }

    return None


def load_golden_cases(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("unsupported golden fixture schema")

    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list):
        raise ValueError("golden fixture cases must be a list")

    cases: list[dict] = []
    case_ids: set[str] = set()
    group_partitions: dict[str, str] = {}

    for raw_case in raw_cases:
        if not isinstance(raw_case, dict):
            raise ValueError("golden case must be an object")

        case = dict(raw_case)

        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("case_id must be non-empty")
        if case_id in case_ids:
            raise ValueError(f"duplicate case_id: {case_id}")
        case_ids.add(case_id)

        provenance_group = case.get("provenance_group")
        if not isinstance(provenance_group, str) or not provenance_group:
            raise ValueError("provenance_group must be non-empty")

        partition = case.get("partition")
        if partition not in _ALLOWED_PARTITIONS:
            raise ValueError("unsupported partition")

        previous_partition = group_partitions.setdefault(
            provenance_group,
            partition,
        )
        if previous_partition != partition:
            raise ValueError("provenance_group may not cross partitions")

        if case.get("human_outcome") not in _ALLOWED_HUMAN_OUTCOMES:
            raise ValueError("unsupported human_outcome")

        if case.get("validated_truth", True) is not True:
            raise ValueError("case is not validated truth")

        if not isinstance(case.get("meaning_sensitive"), bool):
            raise ValueError("meaning_sensitive must be boolean")

        failure_class = case.get("failure_class")
        if not isinstance(failure_class, str) or not failure_class:
            raise ValueError("failure_class must be non-empty")

        expected = case.get("expected")
        if not isinstance(expected, dict):
            raise ValueError("expected routing metadata is required")

        for field in (
            "representation_equivalent",
            "hard_review",
            "anomaly",
        ):
            if not isinstance(expected.get(field), bool):
                raise ValueError(f"expected.{field} must be boolean")

        cases.append(case)

    return sorted(cases, key=lambda case: case["case_id"])


def _failure_class_counts(cases: list[dict]) -> dict[str, int]:
    return dict(
        sorted(
            Counter(
                case["failure_class"]
                for case in cases
            ).items()
        )
    )


def _candidate_representation_equivalent(case: dict) -> bool:
    return representation_equivalence(
        case["apple_text"],
        case["whisper_text"],
        apple_context=case.get("apple_context", ""),
        whisper_context=case.get("whisper_context", ""),
    ) is not None


def _partition_report(
    cases: list[dict],
    *,
    candidate: bool,
) -> dict:
    auto_flags = [
        bool(candidate and _candidate_representation_equivalent(case))
        for case in cases
    ]
    automatic = sum(auto_flags)
    correct = sum(
        auto and bool(case["expected"]["representation_equivalent"])
        for case, auto in zip(cases, auto_flags)
    )
    false = sum(
        auto and not bool(case["expected"]["representation_equivalent"])
        for case, auto in zip(cases, auto_flags)
    )

    return {
        "cases": len(cases),
        "unique_cases": len(cases),
        "unique_provenance_groups": len(
            {case["provenance_group"] for case in cases}
        ),
        "automatic_resolutions": automatic,
        "correct_auto_resolutions": correct,
        "false_auto_resolutions": false,
        "human_review": len(cases) - automatic,
        "burden_reduction": automatic,
        "labeled_anomalies": sum(
            bool(case["expected"]["anomaly"])
            for case in cases
        ),
        "hard_review_cases": sum(
            bool(case["expected"]["hard_review"])
            for case in cases
        ),
        "meaning_sensitive_cases": sum(
            bool(case["meaning_sensitive"])
            for case in cases
        ),
        "by_failure_class": _failure_class_counts(cases),
    }


def evaluate_policy_cases(
    cases: list[dict],
    *,
    candidate: bool,
) -> dict:
    partitions = {
        partition: _partition_report(
            [
                case
                for case in cases
                if case["partition"] == partition
            ],
            candidate=candidate,
        )
        for partition in sorted(_ALLOWED_PARTITIONS)
    }

    automatic = sum(
        partition["automatic_resolutions"]
        for partition in partitions.values()
    )
    correct = sum(
        partition["correct_auto_resolutions"]
        for partition in partitions.values()
    )
    false = sum(
        partition["false_auto_resolutions"]
        for partition in partitions.values()
    )

    return {
        "policy": "candidate" if candidate else "current",
        "cases": len(cases),
        "unique_cases": len(cases),
        "unique_provenance_groups": len(
            {case["provenance_group"] for case in cases}
        ),
        "automatic_resolutions": automatic,
        "correct_auto_resolutions": correct,
        "false_auto_resolutions": false,
        "human_review": len(cases) - automatic,
        "burden_reduction": automatic,
        "labeled_anomalies": sum(
            bool(case["expected"]["anomaly"])
            for case in cases
        ),
        "hard_review_cases": sum(
            bool(case["expected"]["hard_review"])
            for case in cases
        ),
        "meaning_sensitive_cases": sum(
            bool(case["meaning_sensitive"])
            for case in cases
        ),
        "by_failure_class": _failure_class_counts(cases),
        "advisory_evaluated_cases": 0,
        "correct_advisory_recommendations": 0,
        "false_advisory_recommendations": 0,
        "advisory_abstentions": 0,
        "partitions": partitions,
    }
