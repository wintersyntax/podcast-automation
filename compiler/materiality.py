"""TASK-133: is a Human Review card immaterial for the knowledge note?

The reviewer decided (2026-10-06) that a card only needs him when choosing
the wrong reading could change what ends up in the knowledge note: a value,
a claim or its polarity, an entity, a term with a different meaning. Fillers,
false starts, repetitions, contractions and harmless rewording -- including a
term or name spelled differently with the same meaning -- do not need him.
Numbers stay with the reviewer, except the same numbers written differently
("2am" / "2 a.m.", "forties" / "40s"; user decision 2026-10-07).

This module is the deterministic first layer. It is conservative by design:
a card is immaterial only when every word that differs between the two
scoped readings is a filler, an expanded contraction or colloquial form, a
restart (a repetition of the words right next to it), the same letters with
different spacing, a small swap of function words, or the same numbers
written differently. Any other digit or number word, any negation that is not an abandoned restart of a negated clause, and
any other content word leaves the card for the second layer (the materiality
judge) or the reviewer.

Calibrated on 293 audited decisions (10 episodes): 73 cards immaterial, none
of the material ones. It is pure and has no I/O.
"""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

from .review_tiers import _scoped_text
from .third_asr_window import _scope_contexts, anchored_third_asr_window

MATERIALITY_POLICY_VERSION = "materiality-v3"
CONTEXT_CHARS = 300
_RULE_CONTEXT_TOKENS = 12

_TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)*|\d+|[%$]")
NUMBER_WORDS = frozenset({
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
    "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    "hundred", "thousand", "million", "billion", "percent", "half", "quarter", "dozen",
    "first", "second", "third", "once", "twice",
})
NEGATIONS = frozenset({
    "no", "not", "never", "nothing", "nobody", "none", "nor", "nah", "cannot", "neither", "without",
})
FILLERS = frozenset({
    "um", "uh", "uhm", "umm", "erm", "er", "ah", "oh", "hmm", "mm", "mhm", "yeah", "yep", "yes",
    "okay", "ok", "so", "well", "right", "like", "just", "actually", "basically", "literally",
    "really", "and", "but", "or", "then", "anyway", "kinda", "sorta",
})
FILLER_PHRASES = frozenset({
    ("you", "know"), ("i", "mean"), ("kind", "of"), ("sort", "of"), ("i", "guess"),
    ("you", "see"), ("et", "cetera"), ("i", "think"),
})
# Function words whose swap does not change a claim. Deliberately excludes
# modals (can/should/must/...), negations, numbers and content verbs.
FUNCTION_WORDS = frozenset("""
a an the this that these those it its i me my we us our you your he him his she her they them their there here
is am are was were be been being have has had do does did
to of in on at for from with by about as into onto over up down out off than then if because so such too very
what which who whom when where how all any some each both also only even still yet
""".split())
MAX_FUNCTION_WORD_DIFFERENCE = 8
# A long stretch present in only one reading: at least this many words while
# the other reading has at most a third of them.
LONG_ONE_SIDED_WORDS = 25
# Sponsor reads that one source carries and the other does not (dynamic ads).
AD_MARKERS = (
    "brought to you by", "sponsored by", "promo code", "use code", "terms apply",
    "subject to credit approval",
)
_EXPAND = {
    "gonna": ("going", "to"), "wanna": ("want", "to"), "gotta": ("got", "to"), "cause": ("because",),
    "'cause": ("because",), "cuz": ("because",), "etc": ("et", "cetera"), "ok": ("okay",),
    "i'm": ("i", "am"), "you're": ("you", "are"), "we're": ("we", "are"), "they're": ("they", "are"),
    "it's": ("it", "is"), "that's": ("that", "is"), "there's": ("there", "is"), "what's": ("what", "is"),
    "he's": ("he", "is"), "she's": ("she", "is"), "i'll": ("i", "will"), "we'll": ("we", "will"),
    "you'll": ("you", "will"), "it'll": ("it", "will"), "i'd": ("i", "would"), "we'd": ("we", "would"),
    "you'd": ("you", "would"), "they'd": ("they", "would"), "i've": ("i", "have"), "we've": ("we", "have"),
    "you've": ("you", "have"), "they've": ("they", "have"), "let's": ("let", "us"),
}
# Negations of one verb family compare equal (can't/couldn't, no/not), so a
# tense change does not count as a polarity change.
_NEGATION_FAMILY = {
    "can't": "NEG_can", "couldn't": "NEG_can", "cannot": "NEG_can",
    "don't": "NEG_do", "doesn't": "NEG_do", "didn't": "NEG_do",
    "isn't": "NEG_be", "wasn't": "NEG_be", "aren't": "NEG_be", "weren't": "NEG_be",
    "haven't": "NEG_have", "hasn't": "NEG_have", "hadn't": "NEG_have",
    "won't": "NEG_will", "wouldn't": "NEG_will", "no": "NEG_no", "not": "NEG_no",
}


def materiality_tokens(text: str | None) -> list[str]:
    """Lower-case comparison tokens with contractions expanded."""

    text = unicodedata.normalize("NFKC", text or "").casefold().replace("’", "'")
    text = re.sub(r"(?<=[a-z])-(?=[a-z])", "", text)  # e-mail -> email
    tokens: list[str] = []
    for token in _TOKEN.findall(text):
        for part in _EXPAND.get(token, (token,)):
            tokens.append(_NEGATION_FAMILY.get(part, part))
    return tokens


def is_number_token(token: str) -> bool:
    return token.isdigit() or token in NUMBER_WORDS or token in {"%", "$"}


# ------------------------------------------------------------ number format
# Two readings that say the same numbers, only written differently. Both are
# put into one canonical form and compared token by token (fillers ignored).
# Deliberately conservative: "five" never equals "two three", "2017" never
# equals "17", "7.30" never equals "7, 30" and "twenty seventeen" stays a card.

_NUMBER_UNITS = {word: value for value, word in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen".split())}
_NUMBER_TENS = {word: 10 * value for value, word in enumerate(
    "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if word != "_"}
_NUMBER_DECADES = {word[:-1] + "ies": value for word, value in _NUMBER_TENS.items()}  # forties
_NUMBER_SCALES = {"hundred": 100, "thousand": 1000, "million": 1_000_000}
_NUMBER_UNIT_WORDS = {
    "kg": "kg", "kgs": "kg", "kilo": "kg", "kilos": "kg", "kilogram": "kg", "kilograms": "kg",
    "lb": "lb", "lbs": "lb", "pound": "lb", "pounds": "lb", "%": "percent",
}
_NUMBER_TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)*|\d+(?:_\d+)*|%")


def _number_format_tokens(text: str | None) -> list[str]:
    text = unicodedata.normalize("NFKC", text or "").casefold().replace("’", "'")
    text = re.sub(r"\b([ap])\.\s?m\b\.?", r"\1m", text)  # a.m. -> am
    text = re.sub(r"(\d),(\d{3})\b", r"\1\2", text)       # 1,000 -> 1000
    text = re.sub(r"(\d)[.:](\d)", r"\1_\2", text)         # 7.30 / 7:30 stay one token
    text = re.sub(r"(\d)([a-z%])", r"\1 \2", text)         # 2am -> 2 am, 40s -> 40 s
    return [_NUMBER_UNIT_WORDS.get(token, token) for token in _NUMBER_TOKEN.findall(text)]


def _number_words_to_digits(tokens: list[str]) -> list[str]:
    out: list[str] = []
    value: int | None = None
    last: str | None = None  # "unit", "ten" or "scale"

    def flush() -> None:
        nonlocal value, last
        if value is not None:
            out.append(str(value))
        value, last = None, None

    for index, token in enumerate(tokens):
        following = tokens[index + 1] if index + 1 < len(tokens) else None
        if token == "a" and following in _NUMBER_SCALES and value is None:
            value, last = 1, "unit"
        elif token in _NUMBER_DECADES:
            flush()
            out.extend([str(_NUMBER_DECADES[token]), "s"])
        elif token in _NUMBER_TENS:
            if last in (None, "scale"):
                value, last = (value or 0) + _NUMBER_TENS[token], "ten"
            else:
                flush()
                value, last = _NUMBER_TENS[token], "ten"
        elif token in _NUMBER_UNITS:
            number = _NUMBER_UNITS[token]
            if (last == "ten" and 1 <= number <= 9) or last == "scale":
                value, last = value + number, "unit"
            else:
                flush()
                value, last = number, "unit"
        elif token in _NUMBER_SCALES and value is not None and last != "scale":
            scale = _NUMBER_SCALES[token]
            if scale == 100:
                high, low = divmod(value, 100)
                value = high * 100 + low * 100 if high else value * 100
            else:
                value *= scale
            last = "scale"
        else:
            flush()
            out.append(token)
    flush()
    return out


def number_canonical(text: str | None) -> list[str]:
    """Canonical tokens for comparing how numbers are written; fillers ignored."""

    return _number_words_to_digits(_strip_fillers(_number_format_tokens(text)))


def same_numbers_written_differently(apple: str | None, whisper: str | None) -> bool:
    """Both readings say exactly the same words and numbers, only written differently."""

    canonical = number_canonical(apple)
    return (
        bool(canonical)
        and canonical == number_canonical(whisper)
        and any(token[0].isdigit() for token in canonical)
    )


def number_gap_source(inputs: dict) -> str | None:
    """For a number card: the only source with words where the other has none."""

    empty = [source for source in ("apple", "whisper") if not _strip_fillers(materiality_tokens(inputs[source]))]
    if len(empty) != 1:
        return None
    return "whisper" if empty[0] == "apple" else "apple"


def _is_negation(token: str) -> bool:
    return token.startswith("NEG_") or token in NEGATIONS or token.endswith("n't")


def _strip_fillers(tokens: list[str]) -> list[str]:
    kept: list[str] = []
    index = 0
    while index < len(tokens):
        if index + 1 < len(tokens) and (tokens[index], tokens[index + 1]) in FILLER_PHRASES:
            index += 2
            continue
        if tokens[index] not in FILLERS:
            kept.append(tokens[index])
        index += 1
    return kept


def _is_restart(full: list[str], start: int, end: int) -> bool:
    """The span ``full[start:end]`` repeats the words right before or after it."""

    span = full[start:end]
    size = len(span)
    if not size:
        return True
    if full[end:end + size] == span or full[max(0, start - size):start] == span:
        return True
    core = _strip_fillers(span)
    if not core:
        return True
    after = _strip_fillers(full[end:end + size + 6])
    if after[:len(core)] == core:
        return True
    before = _strip_fillers(full[max(0, start - size - 6):start])
    return before[-len(core):] == core


def _negated_restart(full: list[str], start: int, end: int, window: int = 4) -> bool:
    """An inserted negated fragment abandoned for a negated clause right next to it."""

    near = _strip_fillers(full[end:end + window]) + _strip_fillers(full[max(0, start - window):start])
    return any(_is_negation(token) for token in near)


def _interjection(text: str | None, right: str | None) -> bool:
    """A lone "No" used as an interjection: followed by punctuation."""

    raw = (text or "").strip()
    following = (right or "").lstrip()
    return raw.endswith((",", ".", "!", "?")) or following[:1] in {",", ".", "!", "?"}


def materiality_inputs(item: dict) -> dict | None:
    """The scoped readings, Whisper-side context and anchored Third-ASR text.

    ``third`` is the Third-ASR words between the card's context anchors, or
    ``None`` without aligned Third-ASR evidence. The judge never sees it; it
    only feeds the choice of the reading to keep.
    """

    apple, whisper = _scoped_text(item, "apple"), _scoped_text(item, "whisper")
    if apple is None or whisper is None:
        return None
    left = right = ""
    for source in ("whisper", "apple"):
        split = _scope_contexts(item, source)
        if split is not None:
            left, right = split[0], split[1]
            break
    window = anchored_third_asr_window(item) if isinstance(item.get("third_asr"), dict) else None
    return {
        "apple": apple,
        "whisper": whisper,
        "left": left[-CONTEXT_CHARS:],
        "right": right[:CONTEXT_CHARS],
        "third": window["text"] if window else None,
    }


def has_number(inputs: dict) -> bool:
    return any(
        is_number_token(token)
        for token in materiality_tokens(inputs["apple"]) + materiality_tokens(inputs["whisper"])
    )


def _one_sided_long(inputs: dict) -> str | None:
    """The source holding a long stretch the other reading (nearly) lacks."""

    lengths = {source: len(materiality_tokens(inputs[source])) for source in ("apple", "whisper")}
    longer = max(lengths, key=lengths.get)
    shorter = "whisper" if longer == "apple" else "apple"
    if lengths[longer] >= LONG_ONE_SIDED_WORDS and lengths[shorter] * 3 <= lengths[longer]:
        return longer
    return None


def advertisement_source(inputs: dict) -> str | None:
    """The source whose long one-sided stretch is a sponsor read, if any."""

    longer = _one_sided_long(inputs)
    if longer is None:
        return None
    text = unicodedata.normalize("NFKC", inputs[longer] or "").casefold()
    return longer if any(marker in text for marker in AD_MARKERS) else None


def rule_verdict(inputs: dict) -> tuple[str, str]:
    """Return ``(route, reason)`` for the deterministic layer.

    ``route`` is ``"advertisement"`` (one reading is a sponsor read the other
    lacks; ``reason`` names the reading to keep, ``use_apple`` or
    ``use_whisper``), ``"immaterial"`` (the rule settles the card),
    ``"reviewer"`` (numbers -- unless both readings say the same numbers
    written differently -- and long stretches only one source has: never
    automatic) or ``"judge"`` (the rule cannot tell).
    """

    ad = advertisement_source(inputs)
    if ad is not None:
        return "advertisement", "use_whisper" if ad == "apple" else "use_apple"
    apple, whisper = materiality_tokens(inputs["apple"]), materiality_tokens(inputs["whisper"])
    if any(is_number_token(token) for token in apple + whisper):
        if same_numbers_written_differently(inputs["apple"], inputs["whisper"]):
            return "immaterial", "number_format"
        return "reviewer", "number"
    if _one_sided_long(inputs) is not None:
        return "reviewer", "one_sided_long"
    if "".join(apple) == "".join(whisper):
        return "immaterial", "same_letters"
    left = materiality_tokens(inputs.get("left"))[-_RULE_CONTEXT_TOKENS:]
    right = materiality_tokens(inputs.get("right"))[:_RULE_CONTEXT_TOKENS]
    offset = len(left)
    sides = {
        "apple": (left + apple + right, inputs["apple"]),
        "whisper": (left + whisper + right, inputs["whisper"]),
    }
    leftover: list[str] = []
    for tag, a1, a2, w1, w2 in SequenceMatcher(None, apple, whisper, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        insertion = (a1 == a2) != (w1 == w2)
        for source, s1, s2 in (("apple", a1, a2), ("whisper", w1, w2)):
            full, raw = sides[source]
            core = _strip_fillers(full[offset + s1:offset + s2])
            if not core or _is_restart(full, offset + s1, offset + s2):
                continue
            if any(_is_negation(token) for token in core):
                if insertion and _negated_restart(full, offset + s1, offset + s2):
                    continue
                if insertion and core == ["NEG_no"] and _interjection(raw, inputs.get("right")):
                    continue
                return "judge", "negation"
            leftover.extend(core)
    if not leftover:
        return "immaterial", "fillers_restarts"
    if len(leftover) <= MAX_FUNCTION_WORD_DIFFERENCE and all(token in FUNCTION_WORDS for token in leftover):
        return "immaterial", "function_words"
    return "judge", "content"
