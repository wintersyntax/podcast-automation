"""Quantity parsing, in-quote validation, and SI conversion.

TASK-106 Phase A Task 2. Design authority:
docs/superpowers/specs/2026-09-24-grounded-knowledge-note-pipeline-design.md
Sections 4.5 (SI conversion) and 6 (numeric validation).

Numeric validation always compares the *spoken* value against the quote
(§6); SI conversion is a separate, purely cosmetic rendering step (§4.5)
that never changes what is validated against the quote.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP

# ---------------------------------------------------------------------------
# Number parsing
# ---------------------------------------------------------------------------

_WORD_NUMBERS: dict[str, Decimal] = {
    "zero": Decimal("0"),
    "one": Decimal("1"),
    "two": Decimal("2"),
    "three": Decimal("3"),
    "four": Decimal("4"),
    "five": Decimal("5"),
    "six": Decimal("6"),
    "seven": Decimal("7"),
    "eight": Decimal("8"),
    "nine": Decimal("9"),
    "ten": Decimal("10"),
    "eleven": Decimal("11"),
    "twelve": Decimal("12"),
    "thirteen": Decimal("13"),
    "fourteen": Decimal("14"),
    "fifteen": Decimal("15"),
    "sixteen": Decimal("16"),
    "seventeen": Decimal("17"),
    "eighteen": Decimal("18"),
    "nineteen": Decimal("19"),
    "twenty": Decimal("20"),
    "thirty": Decimal("30"),
    "forty": Decimal("40"),
    "fifty": Decimal("50"),
    "sixty": Decimal("60"),
    "seventy": Decimal("70"),
    "eighty": Decimal("80"),
    "ninety": Decimal("90"),
    "hundred": Decimal("100"),
}
_HALF = Decimal("0.5")
_QUARTER = Decimal("0.25")

# Deliberately vague quantity words: never parse to a specific number.
_VAGUE_QUANTITY_WORDS = frozenset({"couple", "few", "several", "some", "many"})

_DIGIT_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def parse_number(text: str) -> Decimal | None:
    """Parse a single numeric expression (digit or word form) to a Decimal.

    Handles digits, decimals, and thousands separators ("1,234.5"), plus a
    small set of spoken-number forms used in fitness/nutrition podcasts:
    cardinal words ("two"), halves ("half a gram", "a gram and a half",
    "one and a half"), and quarters ("a quarter").

    Returns None when the fragment is not a specific numeric value --
    including deliberately vague quantity words ("a couple", "a few").
    Does not parse ranges; see `parse_range`.
    """
    fragment = text.strip().lower()
    if not fragment:
        return None

    digit_match = _DIGIT_RE.fullmatch(fragment)
    if digit_match:
        return Decimal(fragment.replace(",", ""))

    words = [w for w in fragment.replace("-", " ").split() if w not in ("a", "an")]
    if not words:
        return None

    if any(word in _VAGUE_QUANTITY_WORDS for word in words):
        return None

    if words[0] == "half":
        return _HALF
    if words[0] == "quarter":
        return _QUARTER

    if "and" in words and words[-1] == "half":
        idx = words.index("and")
        head = words[:idx]
        if head and head[0] in _WORD_NUMBERS:
            return _WORD_NUMBERS[head[0]] + _HALF
        # "a gram and a half" -- the head is a noun, not a number; a bare
        # noun before "and a half" implies "one" (one gram and a half).
        return Decimal("1") + _HALF

    if len(words) == 1 and words[0] in _WORD_NUMBERS:
        return _WORD_NUMBERS[words[0]]

    return None


_RANGE_DASHES = ("–", "—")  # en dash, em dash


def parse_range(text: str) -> tuple[Decimal, Decimal] | None:
    """Parse a two-ended range ("15 to 30", "15-30", "15–30") to (low, high).

    Returns None when the fragment is not a range both of whose ends parse
    as numbers via `parse_number`.
    """
    fragment = text.strip()
    if not fragment:
        return None

    for dash in _RANGE_DASHES:
        if dash in fragment:
            left, _, right = fragment.partition(dash)
            return _both(left, right)

    parts = re.split(r"\s+to\s+", fragment, maxsplit=1)
    if len(parts) == 2:
        return _both(*parts)

    if re.fullmatch(r"\d[\d,]*(?:\.\d+)?-\d[\d,]*(?:\.\d+)?", fragment):
        left, right = fragment.split("-", 1)
        return _both(left, right)

    return None


def _both(left: str, right: str) -> tuple[Decimal, Decimal] | None:
    low = parse_number(left)
    high = parse_number(right)
    if low is None or high is None:
        return None
    return (low, high)


# ---------------------------------------------------------------------------
# Quote validation (design spec §6): the spoken value must appear in the quote.
# ---------------------------------------------------------------------------

_NUMERIC_FRAGMENT_RE = re.compile(
    r"\d[\d,]*(?:\.\d+)?(?:\s*(?:to|–|—|-)\s*\d[\d,]*(?:\.\d+)?)?"
)
_WORD_TOKEN_RE = re.compile(r"[A-Za-z']+")
_MAX_WORD_WINDOW = 5


def _candidate_fragments(quote: str) -> list[str]:
    fragments = [match.group(0) for match in _NUMERIC_FRAGMENT_RE.finditer(quote)]
    word_tokens = _WORD_TOKEN_RE.findall(quote.lower())
    for window in range(1, _MAX_WORD_WINDOW + 1):
        for start in range(len(word_tokens) - window + 1):
            fragments.append(" ".join(word_tokens[start:start + window]))
    return fragments


def _value_found_in_quote(value: Decimal, quote: str) -> bool:
    for fragment in _candidate_fragments(quote):
        parsed = parse_number(fragment)
        if parsed is not None and parsed == value:
            return True
        parsed_range = parse_range(fragment)
        if parsed_range is not None and value in parsed_range:
            return True
    return False


def quantity_in_quote(value, value_high, quote: str) -> bool:
    """True only when the spoken value (and `value_high`, if given) appears
    in the quote, as a digit literal or a recognized spoken-number phrase.
    """
    if value is None or quote is None:
        return False
    value = Decimal(str(value))
    if not _value_found_in_quote(value, quote):
        return False
    if value_high is not None:
        value_high = Decimal(str(value_high))
        if not _value_found_in_quote(value_high, quote):
            return False
    return True


# ---------------------------------------------------------------------------
# Free-number token scanning (design spec §5.5 / §6): bare digit or exact-
# quantity-word tokens in free text that assert a specific numeric claim.
#
# Shared by composition.py's `_free_number_violations` (a composed bullet
# may not smuggle a number past a `{q:...}` placeholder) and items.py's
# statement-quantity-fidelity check (an item's own `statement` may not
# assert a number its own `quantities` array doesn't declare, added
# 2026-09-27 -- Task 12 Step 6's dev-tuning found real extraction items
# where a number plainly stated in `quote`/`statement` ["half a set"] never
# made it into `quantities`, so composition correctly refused to cite it and
# the note silently dropped the fact). One tokenizer, used by both checks,
# so they can never drift out of sync with each other the way Step 6 found
# `extract.py`'s default prompt path had drifted from the decided version.
# ---------------------------------------------------------------------------

# Digit runs try the comma/decimal-aware alternative first (matching the
# same "\d[\d,]*(?:\.\d+)?" convention `_DIGIT_RE`/`_NUMERIC_FRAGMENT_RE`
# already use above) so a thousands-grouped number ("6,000") or a decimal
# ("3.5") is judged as the one number it is, not split into separate
# digit tokens at each comma/period -- found 2026-09-27 replaying this
# tokenizer against real `statement` text (which, unlike composed bullet
# text, routinely contains comma-grouped numbers): "6,000 to 8,000" was
# splitting into four bogus single/triple-digit tokens ("6", "000", "8",
# "000"), none of which matched the correctly-declared 6000/8000 entries.
# A clock-time literal ("5:30") tries first and is treated as the one
# concatenated number this codebase's own extraction convention stores it
# as (`quantities` holds `530`, not `5`/`30` separately -- see the real
# dev-set item this was found against). This trades away correctly
# handling the rare non-time colon-joined pair (a spoken ratio, "5:30")
# for correctly handling the common one, the same trade-off already made
# for comma-grouped thousands above.
_CLAIM_TOKEN_RE = re.compile(r"\d{1,2}:\d{2}|\d[\d,]*(?:\.\d+)?|[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*")

# Cardinal/exact-quantity words: prose asserting one of these outside an
# allowlisted structural use is making a specific numeric claim.
_NUMBER_WORDS = frozenset({
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty",
    "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred",
    "thousand", "million", "billion", "half", "quarter", "dozen",
})

# Ordinal/structural words that don't encode a fresh quantity claim (design
# spec §5.5: "except ordinal/structural words on an allowlist"). "one" and
# "two" joined 2026-09-26 from Task 12 Step 6's dev-set baseline measurement
# of composition bullet text -- a 100% false-positive rate for both words,
# always indefinite/referential ("one of the biggest myths", "the two of
# them"), never a fresh smuggled quantity. Reused as-is for the statement
# check below; not independently re-tuned against statement text.
_STRUCTURAL_ALLOWLIST = frozenset({
    "first", "second", "third", "fourth", "fifth", "sixth", "seventh",
    "eighth", "ninth", "tenth", "once", "single", "one", "two",
})

# "<noun/number> and a half|quarter" (design-note idiom, 2026-09-27): a bare
# "half"/"quarter" token would parse to 0.5/0.25, but this whole phrase is a
# single combined claim ("a scoop and a half" == 1.5) that `parse_number`
# already knows how to evaluate correctly (see its own "and a half" handling
# above). Consumed as one phrase-token here, before the plain per-word scan
# below, so the bare "half"/"quarter" inside it is never *also* emitted as a
# separate, wrongly-valued claim.
_AND_A_HALF_RE = re.compile(r"\b[A-Za-z]+\s+and\s+an?\s+(?:half|quarter)\b", re.IGNORECASE)


def numeric_claim_tokens(text: str) -> list[str]:
    """Digits or exact-quantity number words found in `text`.

    Judged per maximal alphanumeric token (letters/digits, internal hyphens
    joining them) rather than per character/word in isolation, so that a
    digit embedded in a mixed alphanumeric identifier -- "VO2", "GLP-1",
    "C20" -- is never flagged (it names a term, not a quantity), and a
    hyphenated compound ("zone-two") is judged on its parts, so an
    allowlisted part doesn't make the whole token un-allowlistable while a
    non-allowlisted number word inside a compound ("zone-three") still gets
    caught. A "<noun> and a half/quarter" phrase ("a scoop and a half") is
    consumed and returned as one phrase-token (`parse_number` evaluates it as
    a whole), not as a stray, wrongly-valued "half"/"quarter" word. A clock
    time ("5:30") returns as one concatenated digit token ("530"), matching
    how this codebase's own extraction stores a spoken time. Callers that
    need `{q:...}` placeholders excluded first (as composed bullet text has)
    strip them before calling this.
    """
    tokens: list[str] = []
    consumed: list[tuple[int, int]] = []
    for match in _AND_A_HALF_RE.finditer(text):
        tokens.append(match.group(0))
        consumed.append(match.span())

    remainder_parts: list[str] = []
    cursor = 0
    for start, end in consumed:
        remainder_parts.append(text[cursor:start])
        cursor = end
    remainder_parts.append(text[cursor:])
    remainder = " ".join(remainder_parts)

    for token in _CLAIM_TOKEN_RE.findall(remainder):
        if ":" in token:
            tokens.append(token.replace(":", ""))
            continue
        has_digit = any(ch.isdigit() for ch in token)
        has_alpha = any(ch.isalpha() for ch in token)
        if has_digit and has_alpha:
            continue  # mixed alphanumeric identifier (VO2, GLP-1, C20) -- not a quantity
        if has_digit:
            tokens.append(token)
            continue
        for part in token.split("-"):
            lower = part.lower()
            if lower in _NUMBER_WORDS and lower not in _STRUCTURAL_ALLOWLIST:
                tokens.append(part)
    return tokens


# ---------------------------------------------------------------------------
# SI conversion (design spec §4.5)
# ---------------------------------------------------------------------------

_LB_TO_KG = Decimal("0.45359237")
_G_PER_LB_TO_G_PER_KG = Decimal("2.20462262")
_OZ_TO_G = Decimal("28.349523125")
_MILE_TO_KM = Decimal("1.609344")
_FT_TO_M = Decimal("0.3048")
_IN_TO_CM = Decimal("2.54")

_SI_CONVERSIONS: dict[str, tuple[str, Decimal]] = {
    "lb": ("kg", _LB_TO_KG),
    "lbs": ("kg", _LB_TO_KG),
    "pound": ("kg", _LB_TO_KG),
    "pounds": ("kg", _LB_TO_KG),
    "g/lb": ("g/kg", _G_PER_LB_TO_G_PER_KG),
    "gram per pound": ("g/kg", _G_PER_LB_TO_G_PER_KG),
    "grams per pound": ("g/kg", _G_PER_LB_TO_G_PER_KG),
    "gram per pound of bodyweight": ("g/kg", _G_PER_LB_TO_G_PER_KG),
    "grams per pound of bodyweight": ("g/kg", _G_PER_LB_TO_G_PER_KG),
    "per pound of bodyweight": ("g/kg", _G_PER_LB_TO_G_PER_KG),
    "oz": ("g", _OZ_TO_G),
    "ounce": ("g", _OZ_TO_G),
    "ounces": ("g", _OZ_TO_G),
    "mile": ("km", _MILE_TO_KM),
    "miles": ("km", _MILE_TO_KM),
    "ft": ("m", _FT_TO_M),
    "feet": ("m", _FT_TO_M),
    "foot": ("m", _FT_TO_M),
    "in": ("cm", _IN_TO_CM),
    "inch": ("cm", _IN_TO_CM),
    "inches": ("cm", _IN_TO_CM),
}

_FAHRENHEIT_UNITS = frozenset({"f", "°f", "fahrenheit", "degrees fahrenheit"})

_PASS_THROUGH_UNITS = frozenset({
    "kcal", "g", "kg", "mg", "ml",
    "reps", "sets", "rir", "%", "percent",
    "hours", "hour", "minutes", "minute",
})


@dataclass(frozen=True)
class QuantityConversion:
    """Result of converting a spoken quantity to SI per design spec §4.5."""

    converted_value: Decimal | None
    converted_value_high: Decimal | None
    si_unit: str
    rendered: str
    unknown_unit: bool


def _round_2_sig_figs(value: Decimal) -> Decimal:
    if value == 0:
        return Decimal("0")
    sign = Decimal(1) if value > 0 else Decimal(-1)
    magnitude = abs(value)
    quantize_exp = magnitude.adjusted() - 1
    quantum = Decimal(1).scaleb(quantize_exp)
    rounded = (magnitude / quantum).to_integral_value(rounding=ROUND_HALF_UP) * quantum
    return sign * rounded


def _convert_fahrenheit(value: Decimal | None) -> Decimal | None:
    if value is None:
        return None
    return _round_2_sig_figs((value - Decimal("32")) * Decimal("5") / Decimal("9"))


def _format_spoken(value, value_high, unit_as_spoken: str) -> str:
    if value_high is not None:
        return f"{value}-{value_high} {unit_as_spoken}"
    return f"{value} {unit_as_spoken}"


def _render_converted(converted, converted_high, si_unit, value, value_high, unit_as_spoken) -> str:
    spoken = _format_spoken(value, value_high, unit_as_spoken)
    if converted_high is not None:
        return f"~{converted}-{converted_high} {si_unit} (said: {spoken})"
    return f"~{converted} {si_unit} (said: {spoken})"


def _render_passthrough(value, value_high, unit_as_spoken: str) -> str:
    if value_high is not None:
        return f"{value}-{value_high} {unit_as_spoken}"
    return f"{value} {unit_as_spoken}"


def convert_quantity(value, value_high, unit_as_spoken: str) -> QuantityConversion:
    """Convert a spoken quantity to SI per design spec §4.5.

    `kcal, g, kg, mg, ml, reps, sets, RIR, %, hours, minutes` pass through
    unchanged. Converted values are rounded to 2 significant figures and
    rendered with a leading `~`. Both ends of a range are converted. An
    unrecognized unit is rendered exactly as spoken and flagged via
    `unknown_unit=True` rather than silently guessed.
    """
    value = Decimal(str(value)) if value is not None else None
    value_high = Decimal(str(value_high)) if value_high is not None else None
    unit_key = unit_as_spoken.strip().lower()

    if unit_key in _FAHRENHEIT_UNITS:
        converted = _convert_fahrenheit(value)
        converted_high = _convert_fahrenheit(value_high)
        si_unit = "°C"
        rendered = _render_converted(converted, converted_high, si_unit, value, value_high, unit_as_spoken)
        return QuantityConversion(converted, converted_high, si_unit, rendered, False)

    if unit_key in _SI_CONVERSIONS:
        si_unit, factor = _SI_CONVERSIONS[unit_key]
        converted = _round_2_sig_figs(value * factor) if value is not None else None
        converted_high = _round_2_sig_figs(value_high * factor) if value_high is not None else None
        rendered = _render_converted(converted, converted_high, si_unit, value, value_high, unit_as_spoken)
        return QuantityConversion(converted, converted_high, si_unit, rendered, False)

    if unit_key in _PASS_THROUGH_UNITS:
        rendered = _render_passthrough(value, value_high, unit_as_spoken)
        return QuantityConversion(value, value_high, unit_as_spoken, rendered, False)

    # Unknown unit: render exactly as spoken, flagged for review rather than
    # silently passed through or guessed at.
    rendered = _render_passthrough(value, value_high, unit_as_spoken)
    return QuantityConversion(value, value_high, unit_as_spoken, rendered, True)
