"""Locate the part of a Third-ASR clip transcript that covers one review card.

TASK-123: Third-ASR evidence is a transcript of a whole 12-15 s audio clip,
while a review card replaces only a few canonical words. Choosing "Use Third"
must therefore insert only the clip words that sit between the card's own
left and right context, never the whole clip (which would duplicate the
surrounding sentences in the canonical transcript).

The window is derived deterministically and fails closed: both anchors must
be found exactly in the clip transcript, in order; when an anchor repeats
(TASK-126, longer clips), the ordered pair whose window length is closest to
the card's own length is used only if no other pair is equally close.
Otherwise no window exists and the reviewer must use Custom/Edit instead. The result is
reviewer-facing evidence only; it never selects a source by itself.
"""

from __future__ import annotations

import re
import unicodedata

WINDOW_METHOD = "context-anchor-v2"
_ANCHOR_SIZES = (3, 2)
_MAX_EXTRA_WINDOW_TOKENS = 12
_CONTEXT_RADIUS_TOKENS = 30
_TOKEN = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)


def _normalized(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold().replace("’", "'")


def _tokens_with_spans(text: str) -> list[tuple[str, int, int]]:
    return [
        (_normalized(match.group(0)), match.start(), match.end())
        for match in _TOKEN.finditer(text)
    ]


def _token_values(text: str) -> list[str]:
    return [token for token, _, _ in _tokens_with_spans(text)]


def _split_context(context: str, span: str) -> tuple[str, str] | None:
    """Split a compiler context around its own span.

    The compiler centres each context on the span with a fixed token radius,
    so when the span text occurs more than once the occurrence whose left
    side is closest to that radius is the card's own span.
    """

    if not isinstance(context, str) or not isinstance(span, str) or not span.strip():
        return None
    positions = [match.start() for match in re.finditer(re.escape(span), context)]
    if not positions:
        return None
    best = min(
        positions,
        key=lambda position: abs(
            len(_token_values(context[:position])) - _CONTEXT_RADIUS_TOKENS
        ),
    )
    return context[:best], context[best + len(span):]


def _scope_contexts(item: dict, source: str) -> tuple[str, str, int] | None:
    """Return (left, right, span_token_count) for the decision scope."""

    context = item.get(f"{source}_context")
    span = item.get(f"{source}_text")
    split = _split_context(context, span)
    if split is None:
        return None
    left, right = split
    focus = item.get("focus") if isinstance(item.get("focus"), dict) else {}
    if focus.get("scope") == "partial":
        focus_text = focus.get(f"{source}_text")
        if not isinstance(focus_text, str) or not focus_text.strip():
            return None
        if span.count(focus_text) != 1:
            return None
        position = span.index(focus_text)
        left = left + span[:position]
        right = span[position + len(focus_text):] + right
        span = focus_text
    return left, right, len(_token_values(span))


def _occurrences(haystack: list[str], needle: list[str], start: int = 0) -> list[int]:
    return [
        index
        for index in range(start, len(haystack) - len(needle) + 1)
        if haystack[index:index + len(needle)] == needle
    ]


def _unique_occurrence(haystack: list[str], needle: list[str], start: int = 0) -> int | None:
    hits = _occurrences(haystack, needle, start)
    return hits[0] if len(hits) == 1 else None


def _anchor_pair(
    values: list[str],
    left_anchor: list[str],
    right_anchor: list[str],
    span_tokens: int,
) -> tuple[int, int, bool] | None:
    """Return (window_start, right_at, disambiguated) for one anchor size.

    The unambiguous case (each anchor occurs once, in order) is unchanged.
    TASK-126: when an anchor repeats inside a longer clip, every ordered
    left/right pair within the size limit is considered and the pair whose
    window length is closest to the card's own length wins -- but only if no
    other pair is equally close. A tie means no window.
    """

    size = len(left_anchor)
    left_hits = _occurrences(values, left_anchor)
    if len(left_hits) == 1:
        window_start = left_hits[0] + size
        right_at = _unique_occurrence(values, right_anchor, window_start)
        if right_at is not None and right_at - window_start <= span_tokens + _MAX_EXTRA_WINDOW_TOKENS:
            return window_start, right_at, False
    if not left_hits:
        return None
    candidates: list[tuple[int, int, int]] = []
    for left_at in left_hits:
        window_start = left_at + size
        for right_at in _occurrences(values, right_anchor, window_start):
            length = right_at - window_start
            if length > span_tokens + _MAX_EXTRA_WINDOW_TOKENS:
                break
            candidates.append((abs(length - span_tokens), window_start, right_at))
    if not candidates:
        return None
    candidates.sort()
    if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
        return None
    _, window_start, right_at = candidates[0]
    return window_start, right_at, len(candidates) > 1 or len(left_hits) > 1


def anchored_third_asr_window(item: dict) -> dict | None:
    """Return the Third-ASR words between the card's context anchors.

    The returned ``text`` preserves the Third-ASR casing and inner
    punctuation. ``None`` means the evidence cannot be aligned safely.
    """

    if not isinstance(item, dict):
        return None
    evidence = item.get("third_asr")
    third_text = evidence.get("text") if isinstance(evidence, dict) else None
    if not isinstance(third_text, str) or not third_text.strip():
        return None
    third = _tokens_with_spans(third_text)
    values = [token for token, _, _ in third]

    for source in ("apple", "whisper"):
        scope = _scope_contexts(item, source)
        if scope is None:
            continue
        left, right, span_tokens = scope
        left_tokens = _token_values(left)
        right_tokens = _token_values(right)
        for size in _ANCHOR_SIZES:
            if len(left_tokens) < size or len(right_tokens) < size:
                continue
            pair = _anchor_pair(values, left_tokens[-size:], right_tokens[:size], span_tokens)
            if pair is None:
                continue
            window_start, right_at, disambiguated = pair
            if right_at == window_start:
                text = ""
            else:
                start_char = third[window_start][1]
                end_char = third[right_at - 1][2]
                text = third_text[start_char:end_char]
            return {
                "text": text,
                "method": WINDOW_METHOD,
                "anchor_source": source,
                "anchor_tokens": size,
                "disambiguated": disambiguated,
            }
    return None
