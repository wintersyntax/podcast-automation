"""Sort pending Human Review cards into three tiers (TASK-127).

Tier A cards carry one machine-derived source proposal that a human may
confirm together with other tier-A cards in one explicit click; tier B cards
need listening (or offer a new Third-ASR reading); tier C cards are protected
(negation, number, unit, name, citation or any compiler risk reason) and have
no confirming evidence. The tier is derived deterministically and freshly
from the review item; it never records a decision, never changes transcript
authority, and never applies anything by itself.

Tier A rules (user decisions of 2026-09-30 and 2026-10-05):

- ``two_of_three``: the anchored Third-ASR window agrees word for word with
  exactly one source (hesitation fillers aside). The only route, together
  with ``two_of_three_disputed``, into tier A for a protected card.
- ``two_of_three_disputed`` (``review-tiers-v2``): Apple and Whisper share
  most words; for every disputed stretch between them, the Third-ASR words
  found between the same shared neighbours equal exactly one source's
  stretch, always the same source. Third-ASR slips in words both sources
  already agree on do not matter, because those words are not being decided.
- ``whisper_gap``: Whisper has no words here (fillers aside) while Apple has
  some, so there is no competing reading. Unprotected cards only.
- ``triage_high``: advisory triage recommends Apple or Whisper with high
  confidence. Unprotected cards only.

Conflicting signals (for example triage recommending the other source) never
produce tier A.

Colloquial forms (``review-tiers-v4``): the Third-ASR model writes
"going to" for a spoken "gonna", so on a card where Apple and Whisper
differ in a colloquial form ("gonna" / "going to", "wanna", "gotta",
"kinda", "sorta", "'cause" / "because") the third voice is not evidence.
If the colloquial form (plus spacing) is the only difference, the spoken
form is the proposal (``spoken_colloquial_form``, unprotected cards; the
reviewer chose it on every historical card); otherwise the card needs
listening (``colloquial_mixed``). Comparison tokens keep numbers whole (``2.5`` is not ``25``) and keep the
symbols ``% $ € £ °``, so a unit or amount can never agree by being dropped.
A card whose only blocker is ``audio_localization_unreliable`` may still reach
tier A when the anchored Third-ASR window agrees with a source that has
words: finding the card's context anchors in the clip proves the clip covers
the card.
"""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

from .third_asr_window import anchored_third_asr_window

REVIEW_TIER_POLICY_VERSION = "review-tiers-v4"

PROTECTED_CATEGORIES = frozenset(
    {"negation", "protocol_number", "unit", "proper_name", "citation"}
)

# Hesitation sounds only. Words such as "like", "so" or "and" can carry
# meaning, so they never count as filler here.
_FILLER_TOKENS = frozenset({"um", "uh", "uhm", "umm", "erm", "er", "ah", "hmm", "mhm", "mm"})
_FILLER_BIGRAMS = frozenset({("you", "know")})

_TOKEN = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)

_SOURCES = ("apple", "whisper")

# Comparison tokens for Third-ASR agreement: whole numbers (with decimal or
# thousands separators), words with inner apostrophes, and the unit/currency
# symbols that change meaning when dropped.
_COMPARE_TOKEN = re.compile(r"\d+(?:[.,]\d+)*|[^\W\d_]+(?:'[^\W\d_]+)*|[%$€£°]", re.UNICODE)
_LOCALIZATION_ANOMALY = "audio_localization_unreliable"
_ANCHOR_SIZES = (2, 1)


def _tokens(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold().replace("’", "'")
    return _TOKEN.findall(normalized)


def _without_fillers(tokens: list[str]) -> list[str]:
    kept: list[str] = []
    index = 0
    while index < len(tokens):
        if tuple(tokens[index:index + 2]) in _FILLER_BIGRAMS:
            index += 2
            continue
        if tokens[index] not in _FILLER_TOKENS:
            kept.append(tokens[index])
        index += 1
    return kept


# Colloquial and written forms of the same words (review-tiers-v4).
_COLLOQUIAL_FORMS = frozenset({"gonna", "wanna", "gotta", "kinda", "sorta", "cause", "'cause", "cuz"})
_COLLOQUIAL_WORDS = {"cause": "because", "'cause": "because", "cuz": "because"}
_COLLOQUIAL_BIGRAMS = {
    ("going", "to"): "gonna",
    ("want", "to"): "wanna",
    ("got", "to"): "gotta",
    ("kind", "of"): "kinda",
    ("sort", "of"): "sorta",
}


def _fold_colloquial(tokens: list[str]) -> list[str]:
    raw = [_COLLOQUIAL_WORDS.get(token, token) for token in tokens]
    folded: list[str] = []
    index = 0
    while index < len(raw):
        pair = tuple(raw[index:index + 2])
        if pair in _COLLOQUIAL_BIGRAMS:
            folded.append(_COLLOQUIAL_BIGRAMS[pair])
            index += 2
            continue
        folded.append(raw[index])
        index += 1
    return folded


def _compare_tokens(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold().replace("’", "'")
    return _without_fillers(_COMPARE_TOKEN.findall(normalized))


def _colloquial_dispute(texts: dict[str, str]) -> tuple[bool, str | None]:
    """(sources differ in a colloquial form, spoken source if that is all).

    The spoken source is returned only when folding the colloquial forms
    leaves the two readings equal up to spacing (no digits involved).
    """

    words = {source: _compare_tokens(texts[source]) for source in _SOURCES}
    counts = {source: sum(1 for token in words[source] if token in _COLLOQUIAL_FORMS) for source in _SOURCES}
    if counts["apple"] == counts["whisper"]:
        return False, None
    folded = {source: _fold_colloquial(words[source]) for source in _SOURCES}
    if _representation_only(folded["apple"], folded["whisper"]):
        return True, "apple" if counts["apple"] > counts["whisper"] else "whisper"
    return True, None


def _representation_only(left: list[str], right: list[str]) -> bool:
    """Same letters joined differently ("post war" / "postwar"), no digits."""

    joined_left, joined_right = "".join(left), "".join(right)
    return (
        joined_left == joined_right
        and not any(character.isdigit() for character in joined_left)
    )


def _unique_occurrence(values: list[str], pattern: list[str], start: int) -> int | None:
    found = None
    size = len(pattern)
    for index in range(start, len(values) - size + 1):
        if values[index:index + size] == pattern:
            if found is not None:
                return None
            found = index
    return found


def _disputed_agreement(apple: list[str], whisper: list[str], third: list[str]) -> str | None:
    """The one source whose every disputed stretch the third voice confirms.

    Shared Apple/Whisper words are the anchors: each disputed stretch is
    bounded by its shared neighbours (or by the anchored window edge), and
    the Third-ASR words between those neighbours must equal exactly one
    source's stretch. Any missing or ambiguous anchor means no agreement.
    """

    opcodes = SequenceMatcher(None, apple, whisper, autojunk=False).get_opcodes()
    verdicts: set[str] = set()
    cursor = 0
    for index, (tag, a_start, a_end, w_start, w_end) in enumerate(opcodes):
        if tag == "equal":
            continue
        apple_part, whisper_part = apple[a_start:a_end], whisper[w_start:w_end]
        if _representation_only(apple_part, whisper_part):
            continue
        previous = opcodes[index - 1] if index > 0 else None
        following = opcodes[index + 1] if index + 1 < len(opcodes) else None

        if previous is None:
            left = 0
        elif previous[0] != "equal":
            return None
        else:
            left_words = apple[previous[1]:previous[2]]
            left = None
            for size in _ANCHOR_SIZES:
                if len(left_words) >= size:
                    found = _unique_occurrence(third, left_words[-size:], cursor)
                    if found is not None:
                        left = found + size
                        break
            if left is None:
                return None

        if following is None:
            right = len(third)
        elif following[0] != "equal":
            return None
        else:
            right_words = apple[following[1]:following[2]]
            right = None
            for size in _ANCHOR_SIZES:
                if len(right_words) >= size:
                    found = _unique_occurrence(third, right_words[:size], left)
                    if found is not None:
                        right = found
                        break
            if right is None:
                return None

        heard = third[left:right]
        matches = [
            source
            for source, part in (("apple", apple_part), ("whisper", whisper_part))
            if heard == part
        ]
        if len(matches) != 1:
            return None
        verdicts.add(matches[0])
        cursor = right
    return verdicts.pop() if len(verdicts) == 1 else None


def _scoped_text(item: dict, source: str) -> str | None:
    focus = item.get("focus") if isinstance(item.get("focus"), dict) else {}
    if focus.get("scope") == "partial":
        value = focus.get(f"{source}_text")
    else:
        value = item.get(f"{source}_text")
    return value if isinstance(value, str) else None


def is_protected(item: dict) -> bool:
    risk = item.get("risk_reasons")
    return (
        item.get("category") in PROTECTED_CATEGORIES
        or (isinstance(risk, list) and bool(risk))
        or item.get("citation_signal") is True
    )


def _triage_source(item: dict) -> str | None:
    triage = item.get("triage")
    if not isinstance(triage, dict):
        return None
    if triage.get("status") != "advisory" or triage.get("confidence") != "high":
        return None
    return {"recommend_apple": "apple", "recommend_whisper": "whisper"}.get(
        triage.get("recommendation")
    )


def _third_agreement(item: dict, texts: dict[str, str]) -> tuple[str | None, str | None, bool]:
    """Return (agreeing source, reason, has_window) for the anchored window."""

    window = anchored_third_asr_window(item)
    if window is None:
        return None, None, False
    third = _compare_tokens(window["text"])
    words = {source: _compare_tokens(texts[source]) for source in _SOURCES}
    agreeing = [source for source in _SOURCES if words[source] == third]
    if len(agreeing) == 1:
        return agreeing[0], "two_of_three", True
    if agreeing:
        return None, None, True
    disputed = _disputed_agreement(words["apple"], words["whisper"], third)
    if disputed is not None:
        return disputed, "two_of_three_disputed", True
    return None, None, True


def _result(tier: str, reason: str, source: str | None = None, text: str | None = None) -> dict:
    return {
        "policy_version": REVIEW_TIER_POLICY_VERSION,
        "tier": tier,
        "reason": reason,
        "source": source,
        "text": text,
    }


def derive_review_tier(item: dict) -> dict:
    """Return the tier, the reason, and the tier-A source proposal if any."""

    if not isinstance(item, dict):
        return _result("C", "malformed_item")
    texts = {source: _scoped_text(item, source) for source in _SOURCES}
    protected = is_protected(item)
    if (
        any(text is None for text in texts.values())
        or item.get("custom_edit") is not None
        or item.get("generation_stale") is True
    ):
        return _result("C" if protected else "B", "no_safe_proposal")

    anomaly = item.get("anomaly")
    if anomaly is not None:
        kind = anomaly.get("kind") if isinstance(anomaly, dict) else None
        if kind == _LOCALIZATION_ANOMALY and not _colloquial_dispute(texts)[0]:
            source, reason, _ = _third_agreement(item, texts)
            triage_source = _triage_source(item)
            if (
                source is not None
                and _compare_tokens(texts[source])
                and triage_source in (None, source)
            ):
                return _result("A", reason, source, texts[source])
        return _result("C" if protected else "B", "no_safe_proposal")

    colloquial, spoken_source = _colloquial_dispute(texts)
    if colloquial:
        if spoken_source is not None and not protected:
            return _result("A", "spoken_colloquial_form", spoken_source, texts[spoken_source])
        return _result("C" if protected else "B", "colloquial_mixed")

    third_source, third_reason, has_window = _third_agreement(item, texts)
    triage_source = _triage_source(item)
    apple_words = _without_fillers(_tokens(texts["apple"]))
    whisper_words = _without_fillers(_tokens(texts["whisper"]))
    gap_source = "apple" if apple_words and not whisper_words else None

    proposals: list[tuple[str, str]] = []
    if third_source is not None:
        proposals.append((third_reason, third_source))
    if not protected:
        if gap_source is not None:
            proposals.append(("whisper_gap", gap_source))
        if triage_source is not None:
            proposals.append(("triage_high", triage_source))

    proposed_sources = {source for _, source in proposals}
    conflicting = bool(proposals) and (
        len(proposed_sources) > 1
        or (triage_source is not None and triage_source not in proposed_sources)
        # The third voice was asked and heard neither source.
        or (has_window and third_source is None)
    )
    if proposals and not conflicting:
        reason, source = proposals[0]
        return _result("A", reason, source, texts[source])
    if conflicting:
        return _result("C" if protected else "B", "conflicting_evidence")
    if has_window:
        return _result("B", "third_new_reading")
    if protected:
        return _result("C", "protected_without_confirmation")
    return _result("B", "needs_listening")


def review_tier_snapshot(item: dict) -> dict:
    """Bounded audit snapshot of the tier a decision was made in."""

    derived = derive_review_tier(item)
    return {
        "policy_version": derived["policy_version"],
        "tier": derived["tier"],
        "reason": derived["reason"],
        "source": derived["source"],
    }
