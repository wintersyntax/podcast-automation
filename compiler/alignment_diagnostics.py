"""Shadow-only transcript-alignment diagnostics and bounded DP comparison.

This module observes alignment structure after the canonical compiler decision path.
It never chooses transcript text, changes resolver eligibility, or changes Human Review
routing.  A diagnostic signal means "inspect this region", not "alignment is wrong".
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Sequence


ALIGNMENT_DIAGNOSTICS_SCHEMA_VERSION = 1
ALIGNMENT_DIAGNOSTICS_MODE = "shadow"
MAX_DIAGNOSTIC_REGIONS = 20
DIAGNOSTIC_MERGE_GAP = 12
LARGE_CHANGE_TOKENS = 12
LARGE_LENGTH_SKEW = 8
FRAGMENTED_CHANGE_BLOCKS = 4
BENCHMARK_CONTEXT_TOKENS = 12
MAX_BENCHMARK_TOKENS_PER_SIDE = 160
MAX_BENCHMARK_CELLS = 25_600

Opcode = tuple[str, int, int, int, int]


@dataclass(frozen=True)
class ReviewSpan:
    """One already-built compiler difference used only for overlap telemetry."""

    difference_id: int
    apple_start: int
    apple_end: int
    whisper_start: int
    whisper_end: int
    review_required: bool


def disabled_alignment_diagnostics() -> dict[str, object]:
    """Return the explicit disabled/non-authoritative report contract."""

    return {
        "schema_version": ALIGNMENT_DIAGNOSTICS_SCHEMA_VERSION,
        "mode": "disabled",
        "stage": "post_decision_observability",
        "authoritative": False,
        "decision_effect": "none",
        "resolver_visibility": "none",
        "review_routing_effect": "none",
        "suspicious_region_count": 0,
        "comparison_count": 0,
        "regions": [],
    }


def _collapse_steps(steps: Sequence[str]) -> list[Opcode]:
    """Collapse token-level traceback steps into difflib-style opcodes."""

    if not steps:
        return []
    opcodes: list[Opcode] = []
    apple_index = whisper_index = 0
    current_tag: str | None = None
    start_apple = start_whisper = 0

    def flush() -> None:
        nonlocal current_tag, start_apple, start_whisper
        if current_tag is not None:
            opcodes.append(
                (current_tag, start_apple, apple_index, start_whisper, whisper_index)
            )
        current_tag = None

    for tag in steps:
        if tag != current_tag:
            flush()
            current_tag = tag
            start_apple = apple_index
            start_whisper = whisper_index
        if tag in {"equal", "replace", "delete"}:
            apple_index += 1
        if tag in {"equal", "replace", "insert"}:
            whisper_index += 1
    flush()
    return opcodes


def needleman_wunsch_opcodes(
    apple_values: Sequence[str],
    whisper_values: Sequence[str],
) -> list[Opcode]:
    """Return deterministic global-alignment opcodes for one bounded window.

    Scoring is intentionally simple because this is a comparison harness, not a
    production decision engine: exact match +2, substitution -1, gap -1.  Ties
    prefer diagonal alignment, then deletion, then insertion for reproducibility.
    """

    n = len(apple_values)
    m = len(whisper_values)
    if n * m > MAX_BENCHMARK_CELLS:
        raise ValueError("bounded Needleman-Wunsch window exceeds cell budget")

    scores = [[0] * (m + 1) for _ in range(n + 1)]
    trace = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        scores[i][0] = -i
        trace[i][0] = "delete"
    for j in range(1, m + 1):
        scores[0][j] = -j
        trace[0][j] = "insert"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            same = apple_values[i - 1] == whisper_values[j - 1]
            diagonal = scores[i - 1][j - 1] + (2 if same else -1)
            deletion = scores[i - 1][j] - 1
            insertion = scores[i][j - 1] - 1
            best = max(diagonal, deletion, insertion)
            scores[i][j] = best
            if diagonal == best:
                trace[i][j] = "equal" if same else "replace"
            elif deletion == best:
                trace[i][j] = "delete"
            else:
                trace[i][j] = "insert"

    steps: list[str] = []
    i, j = n, m
    while i or j:
        tag = trace[i][j]
        if not tag:
            tag = "delete" if i else "insert"
        steps.append(tag)
        if tag in {"equal", "replace", "delete"}:
            i -= 1
        if tag in {"equal", "replace", "insert"}:
            j -= 1
    steps.reverse()
    return _collapse_steps(steps)


def _opcode_metrics(opcodes: Sequence[Opcode]) -> dict[str, int]:
    changed = [opcode for opcode in opcodes if opcode[0] != "equal"]
    return {
        "matched_tokens": sum(i2 - i1 for tag, i1, i2, _, _ in opcodes if tag == "equal"),
        "change_blocks": len(changed),
        "changed_apple_tokens": sum(i2 - i1 for _, i1, i2, _, _ in changed),
        "changed_whisper_tokens": sum(j2 - j1 for _, _, _, j1, j2 in changed),
        "longest_change_span": max(
            (max(i2 - i1, j2 - j1) for _, i1, i2, j1, j2 in changed),
            default=0,
        ),
    }


def _difference_regions(opcodes: Sequence[Opcode]) -> list[list[Opcode]]:
    """Group nearby non-equal opcodes for diagnostic inspection."""

    regions: list[list[Opcode]] = []
    current: list[Opcode] = []
    for index, opcode in enumerate(opcodes):
        tag, i1, i2, j1, j2 = opcode
        if tag != "equal":
            current.append(opcode)
            continue
        equal_length = min(i2 - i1, j2 - j1)
        next_is_change = index + 1 < len(opcodes) and opcodes[index + 1][0] != "equal"
        if current and next_is_change and equal_length <= DIAGNOSTIC_MERGE_GAP:
            current.append(opcode)
        elif current:
            regions.append(current)
            current = []
    if current:
        regions.append(current)
    return regions


def _region_reasons(group: Sequence[Opcode]) -> tuple[list[str], dict[str, int]]:
    changed = [opcode for opcode in group if opcode[0] != "equal"]
    changed_apple = sum(i2 - i1 for _, i1, i2, _, _ in changed)
    changed_whisper = sum(j2 - j1 for _, _, _, j1, j2 in changed)
    change_blocks = len(changed)
    reasons: list[str] = []

    if changed_apple == 0 and changed_whisper >= LARGE_CHANGE_TOKENS:
        reasons.append("large_insert_span")
    elif changed_whisper == 0 and changed_apple >= LARGE_CHANGE_TOKENS:
        reasons.append("large_delete_span")
    elif changed_apple >= LARGE_CHANGE_TOKENS and changed_whisper >= LARGE_CHANGE_TOKENS:
        reasons.append("large_replace_span")

    if change_blocks >= FRAGMENTED_CHANGE_BLOCKS:
        reasons.append("high_conflict_fragmentation")
    if (
        max(changed_apple, changed_whisper) >= LARGE_CHANGE_TOKENS
        and abs(changed_apple - changed_whisper) >= LARGE_LENGTH_SKEW
    ):
        reasons.append("large_length_skew")

    return reasons, {
        "changed_apple_tokens": changed_apple,
        "changed_whisper_tokens": changed_whisper,
        "change_blocks": change_blocks,
    }




def _span_payload(start: int, end: int) -> dict[str, int | None]:
    """Render a token span without inventing word numbers for a zero-width gap."""

    if end <= start:
        return {
            "start_word": None,
            "end_word": None,
            "token_count": 0,
            "gap_after_word": start,
        }
    return {
        "start_word": start + 1,
        "end_word": end,
        "token_count": end - start,
        "gap_after_word": None,
    }

def _review_overlap_count(
    spans: Sequence[ReviewSpan],
    *,
    apple_start: int,
    apple_end: int,
    whisper_start: int,
    whisper_end: int,
) -> tuple[int, list[int]]:
    ids: list[int] = []
    for span in spans:
        apple_overlap = span.apple_start < apple_end and span.apple_end > apple_start
        whisper_overlap = span.whisper_start < whisper_end and span.whisper_end > whisper_start
        if (apple_overlap or whisper_overlap) and span.review_required:
            ids.append(span.difference_id)
    return len(ids), ids


def _bounded_comparison(
    apple_values: Sequence[str],
    whisper_values: Sequence[str],
    *,
    apple_start: int,
    apple_end: int,
    whisper_start: int,
    whisper_end: int,
) -> dict[str, object]:
    a0 = max(0, apple_start - BENCHMARK_CONTEXT_TOKENS)
    a1 = min(len(apple_values), apple_end + BENCHMARK_CONTEXT_TOKENS)
    w0 = max(0, whisper_start - BENCHMARK_CONTEXT_TOKENS)
    w1 = min(len(whisper_values), whisper_end + BENCHMARK_CONTEXT_TOKENS)
    apple_window = list(apple_values[a0:a1])
    whisper_window = list(whisper_values[w0:w1])

    payload: dict[str, object] = {
        "mode": "bounded_shadow_comparison",
        "authoritative": False,
        "decision_effect": "none",
        "apple_window": {"start_word": a0 + 1, "end_word": a1, "tokens": len(apple_window)},
        "whisper_window": {"start_word": w0 + 1, "end_word": w1, "tokens": len(whisper_window)},
    }
    if (
        len(apple_window) > MAX_BENCHMARK_TOKENS_PER_SIDE
        or len(whisper_window) > MAX_BENCHMARK_TOKENS_PER_SIDE
        or len(apple_window) * len(whisper_window) > MAX_BENCHMARK_CELLS
    ):
        payload["status"] = "skipped_window_too_large"
        return payload

    sequence_opcodes = SequenceMatcher(
        None, apple_window, whisper_window, autojunk=False
    ).get_opcodes()
    nw_opcodes = needleman_wunsch_opcodes(apple_window, whisper_window)
    sequence_metrics = _opcode_metrics(sequence_opcodes)
    nw_metrics = _opcode_metrics(nw_opcodes)
    payload.update(
        {
            "status": "compared",
            "bounded_sequence_matcher": sequence_metrics,
            "needleman_wunsch": nw_metrics,
            "indicators": {
                "nw_matched_token_delta": (
                    nw_metrics["matched_tokens"] - sequence_metrics["matched_tokens"]
                ),
                "nw_change_block_delta": (
                    nw_metrics["change_blocks"] - sequence_metrics["change_blocks"]
                ),
                "nw_longest_change_span_delta": (
                    nw_metrics["longest_change_span"]
                    - sequence_metrics["longest_change_span"]
                ),
            },
        }
    )
    return payload


def build_alignment_diagnostics_shadow(
    apple_values: Sequence[str],
    whisper_values: Sequence[str],
    production_opcodes: Sequence[Opcode],
    *,
    review_spans: Sequence[ReviewSpan] = (),
) -> dict[str, object]:
    """Build bounded, non-authoritative alignment diagnostics telemetry."""

    regions: list[dict[str, object]] = []
    comparison_count = 0
    for group in _difference_regions(production_opcodes):
        reasons, metrics = _region_reasons(group)
        if not reasons:
            continue
        apple_start = min(opcode[1] for opcode in group)
        apple_end = max(opcode[2] for opcode in group)
        whisper_start = min(opcode[3] for opcode in group)
        whisper_end = max(opcode[4] for opcode in group)
        review_count, review_ids = _review_overlap_count(
            review_spans,
            apple_start=apple_start,
            apple_end=apple_end,
            whisper_start=whisper_start,
            whisper_end=whisper_end,
        )
        comparison = _bounded_comparison(
            apple_values,
            whisper_values,
            apple_start=apple_start,
            apple_end=apple_end,
            whisper_start=whisper_start,
            whisper_end=whisper_end,
        )
        if comparison.get("status") == "compared":
            comparison_count += 1
        regions.append(
            {
                "reasons": reasons,
                "apple_span": _span_payload(apple_start, apple_end),
                "whisper_span": _span_payload(whisper_start, whisper_end),
                **metrics,
                "review_overlap_count": review_count,
                "review_difference_ids": review_ids,
                "comparison": comparison,
            }
        )
        if len(regions) >= MAX_DIAGNOSTIC_REGIONS:
            break

    return {
        "schema_version": ALIGNMENT_DIAGNOSTICS_SCHEMA_VERSION,
        "mode": ALIGNMENT_DIAGNOSTICS_MODE,
        "stage": "post_decision_observability",
        "authoritative": False,
        "decision_effect": "none",
        "resolver_visibility": "none",
        "review_routing_effect": "none",
        "current_alignment": "sequence_matcher_with_long_transcript_anchor_windows",
        "candidate_alignment": "bounded_needleman_wunsch",
        "suspicious_region_count": len(regions),
        "comparison_count": comparison_count,
        "regions": regions,
    }
