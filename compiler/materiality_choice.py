"""TASK-133: which reading to keep, decided by evidence -- never by source.

User decision 2026-10-06/07: no source is preferred by default (Apple can
garble as badly as Whisper). For a card the materiality filter settles, the
reading to keep comes from the first decisive step, in this order:

1. ``advertisement`` -- the reading without the sponsor read;
2. ``third_asr``     -- the anchored Third-ASR words equal exactly one reading
                        (fillers ignored);
3. ``registry_term`` -- exactly one reading names a terminology-registry term;
4. ``judge``         -- all three judge answers name the same better reading;
5. ``fuller``        -- rule-settled card (fillers, restarts, function words,
                        same letters): the reading with more words, i.e. what
                        was actually said;
6. ``either``        -- the readings differ only in how the same letters or
                        numbers are written;
otherwise ``click_one``: nothing is decisive and the reviewer picks.

A card that stays with the reviewer gets a preselected ``proposal`` only on
strong evidence -- all three judge answers name the same reading and the
Third-ASR does not back the other one, or a number card where only one
source has words -- and the reviewer always confirms it.

Measured on the 293 audited decisions: Third-ASR 14/17, registry 1/1, judge
54/72, fuller 37/46 agreement with the reviewer's pick; proposals 31/44. Pure,
no I/O; shadow evidence only.
"""

from __future__ import annotations

import re

from .materiality import _strip_fillers, materiality_tokens, number_gap_source
from .terminology import TERMINOLOGY_REGISTRY

SOURCES = ("apple", "whisper")
RULE_FULLER_REASONS = frozenset({"fillers_restarts", "function_words", "same_letters"})
_TERM_TOKENS = tuple(
    tokens
    for tokens in {tuple(re.findall(r"[a-z0-9]+", entry.canonical.casefold())) for entry in TERMINOLOGY_REGISTRY}
    if tokens
)


def _core(text: str | None) -> list[str]:
    return _strip_fillers(materiality_tokens(text))


def third_asr_source(inputs: dict) -> str | None:
    """The only reading the anchored Third-ASR words equal, fillers ignored."""

    third = inputs.get("third")
    if not isinstance(third, str) or not third.strip():
        return None
    words = _core(third)
    hits = [source for source in SOURCES if _core(inputs[source]) == words]
    return hits[0] if len(hits) == 1 else None


def _terms(text: str | None) -> set[str]:
    """Registry terms in the text, matched on whole words ("RDL" not in "hurdle")."""

    plain = re.findall(r"[a-z0-9]+", (text or "").casefold())
    found: set[str] = set()
    for tokens in _TERM_TOKENS:
        joined = "".join(tokens)
        size = len(tokens)
        for start in range(len(plain)):
            # Spacing variants count: "pull down" and "pulldown" are one term.
            for width in range(1, size + 2):
                if "".join(plain[start:start + width]) == joined:
                    found.add(joined)
                    break
    return found


def registry_term_source(inputs: dict) -> str | None:
    """The only reading that names a registry term the other reading lacks."""

    terms = {source: _terms(inputs[source]) for source in SOURCES}
    only = [source for source in SOURCES if terms[source] - terms["whisper" if source == "apple" else "apple"]]
    return only[0] if len(only) == 1 else None


def unanimous_vote(votes: list[str] | None, expected: int = 3) -> str | None:
    if not votes or len(votes) != expected or len(set(votes)) != 1 or votes[0] not in SOURCES:
        return None
    return votes[0]


def choose_reading(inputs: dict, *, reason: str, votes: list[str] | None = None) -> tuple[str | None, str]:
    """``(use, step)`` for a card the filter settles; ``use`` is None for
    ``either`` (any reading) and ``click_one`` (the reviewer picks)."""

    if reason in ("use_apple", "use_whisper"):
        return reason.removeprefix("use_"), "advertisement"
    third = third_asr_source(inputs)
    if third:
        return third, "third_asr"
    term = registry_term_source(inputs)
    if term:
        return term, "registry_term"
    judged = unanimous_vote(votes)
    if judged:
        return judged, "judge"
    if reason in RULE_FULLER_REASONS:
        lengths = {source: len(materiality_tokens(inputs[source])) for source in SOURCES}
        if lengths["apple"] != lengths["whisper"]:
            return max(lengths, key=lengths.get), "fuller"
        return None, "either"
    if reason == "number_format":
        return None, "either"
    return None, "click_one"


def proposal(inputs: dict, *, reason: str, votes: list[str] | None = None) -> tuple[str | None, str | None]:
    """``(source, step)`` preselected for a card that stays with the reviewer."""

    if reason == "number":
        gap = number_gap_source(inputs)
        return (gap, "number_gap") if gap else (None, None)
    judged = unanimous_vote(votes)
    if judged is None:
        return None, None
    third = third_asr_source(inputs)
    if third and third != judged:
        return None, None
    return judged, "judge"
