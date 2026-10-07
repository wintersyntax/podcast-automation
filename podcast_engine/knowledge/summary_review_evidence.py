"""Deterministic Python-owned evidence indexes for summary review."""

from __future__ import annotations

import hashlib
import json
import re


TRANSCRIPT_SPAN_ALGORITHM = "summary-review-transcript-span-v1"
TRANSCRIPT_SPAN_MAX_WORDS = 48
DRAFT_BLOCK_ALGORITHM = "summary-review-draft-block-v1"
RISK_INVENTORY_ALGORITHM = "summary-review-risk-v1"
MAX_RISK_ITEMS = 64
RISK_KINDS = (
    "numeric_precision",
    "named_or_technical_term",
    "certainty_or_causality",
    "recommendation_language",
    "absolute_quantifier",
)

_NUMERIC_PRECISION_PATTERN = re.compile(
    r"(?<!\w)(?:"
    r"\d+(?:\.\d+)?\s*(?:%|milliseconds?|seconds?|minutes?|hours?|days?|weeks?|"
    r"months?|years?|mcg|mg|kg|g|ml|mm|cm|km|bpm|hz|watts?|reps?|sets?)"
    r"|\d+\.\d+)(?!\w)",
    flags=re.IGNORECASE | re.UNICODE,
)
_TECHNICAL_TOKEN_PATTERN = re.compile(
    r"(?<![\w-])[\w]+(?:-[\w]+)*(?![\w-])",
    flags=re.UNICODE,
)
_CERTAINTY_PATTERN = re.compile(
    r"(?<!\w)(?:definitely|certainly|proves|proven|guarantees|causes|caused|"
    r"leads\s+to|results\s+in)(?!\w)",
    flags=re.IGNORECASE | re.UNICODE,
)
_RECOMMENDATION_PATTERN = re.compile(
    r"(?<!\w)(?:should|must|need\s+to|needs\s+to|recommend|recommended|best\s+to)(?!\w)",
    flags=re.IGNORECASE | re.UNICODE,
)
_ABSOLUTE_PATTERN = re.compile(
    r"(?<!\w)(?:always|never|all|none|every|everyone|nobody|no\s+one)(?!\w)",
    flags=re.IGNORECASE | re.UNICODE,
)


def normalize_newlines(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("Review text must be a string")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def build_transcript_span_index(transcript: str) -> dict:
    """Build stable bounded transcript spans with normalized-source offsets."""

    normalized = normalize_newlines(transcript)
    tokens = list(re.finditer(r"\S+", normalized, flags=re.UNICODE))
    spans = []
    for offset in range(0, len(tokens), TRANSCRIPT_SPAN_MAX_WORDS):
        group = tokens[offset : offset + TRANSCRIPT_SPAN_MAX_WORDS]
        start_char = group[0].start()
        end_char = group[-1].end()
        spans.append(
            {
                "span_id": f"S{len(spans) + 1:04d}",
                "start_char": start_char,
                "end_char": end_char,
                "text": normalized[start_char:end_char],
            }
        )

    index = {
        "algorithm": TRANSCRIPT_SPAN_ALGORITHM,
        "source_sha256": _sha256_text(normalized),
        "spans": spans,
    }
    return {**index, "index_sha256": _canonical_json_sha256(index)}


def build_draft_block_index(draft: str) -> dict:
    """Index normalized Markdown headings, list items, and prose paragraphs."""

    normalized = normalize_newlines(draft)
    lines = []
    offset = 0
    for line_with_ending in normalized.splitlines(keepends=True):
        text = line_with_ending[:-1] if line_with_ending.endswith("\n") else line_with_ending
        lines.append((offset, offset + len(text), text))
        offset += len(line_with_ending)

    heading_pattern = re.compile(r"^#{1,6}\s+")
    list_item_pattern = re.compile(r"^\s*(?:[-+*]|\d+[.)])\s+")
    units = []
    line_index = 0
    while line_index < len(lines):
        start_char, end_char, text = lines[line_index]
        if not text.strip():
            line_index += 1
            continue

        if heading_pattern.match(text):
            units.append((start_char, end_char))
            line_index += 1
            continue

        unit_end = end_char
        line_index += 1
        while line_index < len(lines):
            _, next_end, next_text = lines[line_index]
            if not next_text.strip():
                break
            if heading_pattern.match(next_text) or list_item_pattern.match(next_text):
                break
            unit_end = next_end
            line_index += 1
        units.append((start_char, unit_end))

    blocks = [
        {
            "block_id": f"D{number:04d}",
            "start_char": start_char,
            "end_char": end_char,
            "text": normalized[start_char:end_char],
        }
        for number, (start_char, end_char) in enumerate(units, start=1)
    ]
    index = {
        "algorithm": DRAFT_BLOCK_ALGORITHM,
        "source_sha256": _sha256_text(normalized),
        "blocks": blocks,
    }
    return {**index, "index_sha256": _canonical_json_sha256(index)}


_LIST_MARKER_PREFIX_PATTERN = re.compile(r"^(?:[-+*]|\d+[.)])$")
_SENTENCE_BOUNDARY_CHARACTERS = frozenset(".!?:")


def _at_sentence_start(block_text: str, match_start: int) -> bool:
    """True when a token opens its block, a sentence, or a list item.

    Ordinary sentence-initial capitalization (the first word of a block, or
    the first word after a list marker like ``"- "``/``"1. "`` or a
    sentence-ending ``. ! ? :``) is not itself evidence of a named or
    technical term -- nearly every sentence in English prose starts with a
    capital letter. This is deliberately structural (string position only,
    never the token's own meaning), so it never depends on which specific
    word happens to start the sentence.
    """

    prefix = block_text[:match_start].rstrip()
    if not prefix:
        return True
    if prefix[-1] in _SENTENCE_BOUNDARY_CHARACTERS:
        return True
    return bool(_LIST_MARKER_PREFIX_PATTERN.match(prefix))


def _technical_token_match(
    token: str,
    *,
    at_sentence_start: bool = False,
    is_heading: bool = False,
) -> bool:
    letters = [character for character in token if character.isalpha()]
    if not letters or "_" in token:
        return False
    if any(character.isdigit() for character in token):
        return True
    if "-" in token:
        return True
    if any(character.isupper() for character in token[1:]) and any(
        character.islower() for character in token
    ):
        return True

    # Structural (heading) text is a small, fixed, Python-validated set of
    # section titles that never changes between draft and final; detecting
    # named/technical terms there only adds inventory noise, not signal.
    if is_heading or len(letters) < 2:
        return False

    # An all-uppercase run of two or more letters is an acronym (DEXA, NASA)
    # regardless of where it falls in the sentence -- unlike ordinary
    # capitalization, unusual all-caps casing is itself the signal.
    if all(character.isupper() for character in letters):
        return True

    # A plain Title Case word (one leading capital, otherwise all lowercase)
    # is very often just an ordinary capitalized sentence-initial word (The,
    # This, It...); require it to also occur mid-sentence before treating it
    # as a candidate proper noun (a name, place, or brand) worth grounding.
    if (
        not at_sentence_start
        and letters[0].isupper()
        and all(character.islower() for character in letters[1:])
    ):
        return True

    return False


def _matches_overlap(left: dict, right: dict) -> bool:
    return (
        left["start_char"] < right["end_char"]
        and right["start_char"] < left["end_char"]
    )


def _reduce_same_kind_overlaps(matches: list[dict]) -> list[dict]:
    reduced = []
    groups: dict[tuple[str, str], list[dict]] = {}
    for match in matches:
        groups.setdefault((match["kind"], match["draft_block_id"]), []).append(match)

    for group in groups.values():
        pending = sorted(group, key=lambda item: (item["start_char"], item["end_char"]))
        while pending:
            component = [pending.pop(0)]
            component_end = component[0]["end_char"]
            while pending and pending[0]["start_char"] < component_end:
                candidate = pending.pop(0)
                component.append(candidate)
                component_end = max(component_end, candidate["end_char"])
            reduced.append(
                min(
                    component,
                    key=lambda item: (
                        -(item["end_char"] - item["start_char"]),
                        item["start_char"],
                        item["normalized_surface"],
                    ),
                )
            )
    return reduced


def _resolve_cross_kind_collisions(matches: list[dict]) -> list[dict]:
    priority = {kind: index for index, kind in enumerate(RISK_KINDS)}
    accepted = []
    by_block: dict[str, list[dict]] = {}
    for match in matches:
        by_block.setdefault(match["draft_block_id"], []).append(match)

    for group in by_block.values():
        selected = []
        for match in sorted(
            group,
            key=lambda item: (
                priority[item["kind"]],
                item["start_char"],
                item["end_char"],
            ),
        ):
            if any(_matches_overlap(match, existing) for existing in selected):
                continue
            selected.append(match)
        accepted.extend(selected)
    return accepted


def _is_pure_interrogative_block(text: str) -> bool:
    stripped = text.strip()
    if not stripped.endswith("?"):
        return False

    body = stripped[:-1]
    if re.search(r"[!?]|(?<!\d)\.(?!\d)", body):
        return False

    return bool(
        re.match(
            r"^(?:[-*+]\s+|\d+[.)]\s+)?"
            r"(?:who|what|when|where|why|how|which|"
            r"can|could|would|should|will|"
            r"is|are|am|was|were|"
            r"do|does|did|has|have|had|may|might|must)\b",
            stripped,
            flags=re.IGNORECASE,
        )
    )


def build_risk_inventory(draft: str, draft_index: dict) -> dict:
    """Build a bounded deterministic inventory of review-worthy draft surfaces."""

    normalized = normalize_newlines(draft)
    if draft_index != build_draft_block_index(draft):
        raise ValueError("Draft block index does not match draft")

    matches = []
    for block in draft_index["blocks"]:
        block_start = block["start_char"]
        block_text = block["text"]
        is_heading = bool(re.match(r"^#{1,6}\s+", block_text))
        is_interrogative = _is_pure_interrogative_block(block_text)

        detectors = [
            ("numeric_precision", _NUMERIC_PRECISION_PATTERN.finditer(block_text)),
            (
                "named_or_technical_term",
                (
                    match
                    for match in _TECHNICAL_TOKEN_PATTERN.finditer(block_text)
                    if _technical_token_match(
                        match.group(0),
                        at_sentence_start=_at_sentence_start(
                            block_text, match.start()
                        ),
                        is_heading=is_heading,
                    )
                ),
            ),
        ]
        if not is_heading and not is_interrogative:
            detectors.extend(
                (
                    ("certainty_or_causality", _CERTAINTY_PATTERN.finditer(block_text)),
                    ("recommendation_language", _RECOMMENDATION_PATTERN.finditer(block_text)),
                    ("absolute_quantifier", _ABSOLUTE_PATTERN.finditer(block_text)),
                )
            )
        for kind, detector_matches in detectors:
            for match in detector_matches:
                start_char = block_start + match.start()
                end_char = block_start + match.end()
                surface = normalized[start_char:end_char]
                matches.append(
                    {
                        "kind": kind,
                        "draft_block_id": block["block_id"],
                        "surface": surface,
                        "normalized_surface": surface.casefold(),
                        "start_char": start_char,
                        "end_char": end_char,
                    }
                )

    matches = _reduce_same_kind_overlaps(matches)
    matches = _resolve_cross_kind_collisions(matches)
    collapsed: dict[tuple[str, str, str], list[dict]] = {}
    for match in matches:
        key = (
            match["kind"],
            match["draft_block_id"],
            match["normalized_surface"],
        )
        collapsed.setdefault(key, []).append(match)

    block_order = {
        block["block_id"]: index for index, block in enumerate(draft_index["blocks"])
    }
    reduced_items = []
    for (kind, block_id, _normalized_surface), occurrences in collapsed.items():
        occurrences.sort(key=lambda item: (item["start_char"], item["end_char"]))
        first = occurrences[0]
        reduced_items.append(
            {
                "kind": kind,
                "draft_block_id": block_id,
                "surface": first["surface"],
                "start_char": first["start_char"],
                "end_char": first["end_char"],
                "occurrences": [
                    {
                        "start_char": occurrence["start_char"],
                        "end_char": occurrence["end_char"],
                    }
                    for occurrence in occurrences
                ],
            }
        )

    reduced_items.sort(
        key=lambda item: (
            block_order[item["draft_block_id"]],
            item["start_char"],
            item["kind"],
        )
    )
    if len(reduced_items) > MAX_RISK_ITEMS:
        raise ValueError("risk_inventory_overflow")

    risks = [
        {"risk_id": f"R{number:04d}", **item}
        for number, item in enumerate(reduced_items, start=1)
    ]
    inventory = {
        "algorithm": RISK_INVENTORY_ALGORITHM,
        "source_sha256": draft_index["source_sha256"],
        "draft_block_index_sha256": draft_index["index_sha256"],
        "risks": risks,
    }
    return {**inventory, "index_sha256": _canonical_json_sha256(inventory)}


def build_review_context(transcript: str, draft: str) -> dict:
    """Build all deterministic evidence objects for one review request."""

    draft_index = build_draft_block_index(draft)
    return {
        "transcript_span_index": build_transcript_span_index(transcript),
        "draft_block_index": draft_index,
        "risk_inventory": build_risk_inventory(draft, draft_index),
    }


def validate_review_context(context: dict, transcript: str, draft: str) -> dict:
    """Rebuild the context from source bytes and reject any mismatch."""

    if not isinstance(context, dict) or context != build_review_context(transcript, draft):
        raise ValueError("Review context does not match supplied transcript and draft")
    return context


def transcript_spans_by_id(context: dict) -> dict[str, dict]:
    return {
        span["span_id"]: span
        for span in context["transcript_span_index"]["spans"]
    }


def draft_blocks_by_id(context: dict) -> dict[str, dict]:
    return {
        block["block_id"]: block
        for block in context["draft_block_index"]["blocks"]
    }


def risks_by_id(context: dict) -> dict[str, dict]:
    return {
        risk["risk_id"]: risk for risk in context["risk_inventory"]["risks"]
    }
