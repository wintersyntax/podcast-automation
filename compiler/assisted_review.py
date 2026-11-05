"""Pure, non-authoritative evidence comparison for Human Review.

Callers must supply current, generation-validated Third-ASR evidence. This
module neither acquires evidence nor persists a human decision. Token ranges
are zero-based and end-exclusive within the normalized Third-ASR token list.
"""

from copy import deepcopy
from fractions import Fraction

from compiler.review_policy import (
    ASSISTED_REVIEW_POLICY_VERSION,
    assisted_routing,
    comparison_tokens,
)


MATCH_METHOD = "local-word-edit-v1"
_SOURCES = ("apple", "whisper")


def _distance(left: list[str], right: list[str]) -> int:
    previous = list(range(len(right) + 1))
    for row, token in enumerate(left, 1):
        current = [row]
        for column, other in enumerate(right, 1):
            current.append(min(
                current[-1] + 1,
                previous[column] + 1,
                previous[column - 1] + (token != other),
            ))
        previous = current
    return previous[-1]


def best_local_word_edit_match(source_text: str, audio_text: str) -> dict:
    """Return the best bounded token window; preserve both original texts.

    Exact rational ordering avoids floating-point ties changing the selected
    window. Only the audit/display score is converted to a JSON-safe float.
    Empty input or an audio sequence shorter than the minimum window has no
    match, represented by a zero score and null distance/range.
    """
    source = comparison_tokens(source_text)
    audio = comparison_tokens(audio_text)
    result = dict(method=MATCH_METHOD, score=0.0, edit_distance=None, token_range=None)
    if not source or not audio:
        return result
    best_key = None
    for length in range(max(1, len(source) - 4), min(len(audio), len(source) + 6) + 1):
        for start in range(len(audio) - length + 1):
            distance = _distance(source, audio[start:start + length])
            score = max(Fraction(0), 1 - Fraction(distance, max(len(source), length)))
            key = (-score, distance, start, length)
            if best_key is None or key < best_key:
                best_key = key
                result.update(score=float(score), edit_distance=distance,
                              token_range=[start, start + length])
    return result


def _score(match: dict, token_count: int) -> Fraction:
    if match["token_range"] is None:
        return Fraction(0)
    start, end = match["token_range"]
    return max(Fraction(0), 1 - Fraction(match["edit_distance"], max(token_count, end - start)))


def _triage(item: dict) -> dict:
    raw = item.get("triage")
    if raw is None:
        return {"disposition": "missing"}
    if not isinstance(raw, dict):
        return {"disposition": "malformed"}
    recommendation = raw.get("recommendation")
    confidence = raw.get("confidence")
    if (raw.get("status") != "advisory"
            or recommendation not in ("recommend_apple", "recommend_whisper", "likely_custom", "needs_audio")
            or confidence not in ("high", "medium", "low")):
        return {"disposition": "malformed"}
    source = {"recommend_apple": "apple", "recommend_whisper": "whisper"}.get(recommendation)
    if source is not None and (
        ("source" in raw and raw["source"] != source)
        or ("text" in raw and raw["text"] != item[f"{source}_text"])
    ):
        return {"disposition": "malformed"}
    return dict(disposition=recommendation, confidence=confidence, source=source)


def derive_assisted_state(item: dict) -> dict:
    """Derive an auditable UI preselection, never transcript authority.

    ``item.third_asr`` is absent/None before preparation. A successful current
    evidence object contains ``text`` (existing cached shape); explicit
    ``status`` may be ``complete`` or ``available``. Pending/processing status
    stays pending. Other statuses and malformed/empty evidence are unavailable.
    The caller owns cache/generation validation and storage lifecycle.
    """
    routing = assisted_routing(item)
    result = dict(policy_version=ASSISTED_REVIEW_POLICY_VERSION, routing=routing)
    if not routing["eligible"]:
        return result
    evidence = item.get("third_asr")
    advisory = _triage(item)
    result.update(
        state="audio_pending", recommendation=None, reason_codes=[],
        matches=None, margin=None, triage=advisory,
        third_asr=deepcopy(evidence), compiler_suggestion=deepcopy(item.get("suggestion")),
    )

    def finish(state: str, reason: str, source: str | None = None) -> dict:
        result.update(state=state, recommendation=source)
        result["reason_codes"].append(reason)
        return result

    if evidence is None:
        return finish("audio_pending", "audio_not_prepared")
    if isinstance(evidence, dict) and evidence.get("status") in ("pending", "processing"):
        return finish("audio_pending", "audio_preparation_pending")
    if (not isinstance(evidence, dict)
            or evidence.get("status") not in (None, "complete", "available")
            or not isinstance(evidence.get("text"), str)
            or not comparison_tokens(evidence["text"])):
        reason = "audio_evidence_unavailable"
        if isinstance(evidence, dict) and evidence.get("reason_code") == "budget_exhausted":
            reason = "budget_exhausted"
        return finish("audio_unavailable", reason)

    scope = item["focus"] if routing["decision_scope"] == "focus" else item
    texts = {source: scope[f"{source}_text"] for source in _SOURCES}
    tokens = {source: comparison_tokens(text) for source, text in texts.items()}
    matches = {source: best_local_word_edit_match(text, evidence["text"]) for source, text in texts.items()}
    scores = {source: _score(matches[source], len(tokens[source])) for source in _SOURCES}
    result["matches"] = matches
    result["margin"] = float(abs(scores["apple"] - scores["whisper"]))

    if advisory["disposition"] == "likely_custom":
        return finish("ambiguous_audio", "triage_likely_custom")
    if not all(tokens.values()):
        return finish("ambiguous_audio", "empty_source_alternative")
    if tokens["apple"] == tokens["whisper"]:
        return finish("ambiguous_audio", "identical_normalized_alternatives")
    if scores["apple"] == scores["whisper"]:
        if scores["apple"] == 0:
            return finish("neither_source_supported", "no_source_token_support")
        return finish("ambiguous_audio", "tied_audio_support")

    winner = "apple" if scores["apple"] > scores["whisper"] else "whisper"
    loser = "whisper" if winner == "apple" else "apple"
    margin = scores[winner] - scores[loser]
    # A score of exactly one proves a contiguous exact normalized phrase.
    exact = scores[winner] == 1 and scores[loser] < 1
    strong = (scores[winner] >= Fraction(85, 100)
              and scores[loser] <= Fraction(65, 100) and margin >= Fraction(20, 100))
    short = min(len(tokens[source]) for source in _SOURCES) <= 2
    if short and not exact:
        return finish("ambiguous_audio", "short_span_requires_exclusive_exact")
    audio_support = exact or strong
    triage_source = advisory.get("source")
    if audio_support and advisory.get("confidence") == "high" and triage_source == loser:
        return finish("evidence_conflict", "high_confidence_opposite_source")
    if triage_source == loser:
        result["reason_codes"].append("triage_disagreement")
    agreed = (scores[winner] >= Fraction(75, 100) and margin >= Fraction(15, 100)
              and advisory.get("confidence") == "high" and triage_source == winner)
    if audio_support or (not short and agreed):
        reason = "exclusive_exact_phrase" if exact else "strong_audio" if strong else "audio_triage_agreement"
        return finish(f"machine_supported_{winner}", reason, winner)
    return finish("ambiguous_audio", "insufficient_audio_support")
