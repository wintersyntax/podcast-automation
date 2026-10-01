"""Python-owned SI conversion for single-pass knowledge notes (TASK-118).

The writer never converts units. It writes every non-SI quantity exactly as
spoken inside double braces -- ``{{5 lb}}``, ``{{200-230 lb}}``,
``{{1 g/lb}}``, ``{{6 feet}}`` -- and this module renders it as
``~2.3 kg (said: 5 lb)``. Imperial quantities the writer forgot to brace are
converted too, and every such repair is reported so it stays visible.

Conversion factors and 2-significant-figure rounding come from
``notes_v2.quantities.convert_quantity``; volume units it does not know are
handled here with the same rounding.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP

from ..notes_v2.quantities import convert_quantity

UNIT_ALIASES = {
    "kilo": "kg",
    "kilos": "kg",
    "kilogram": "kg",
    "kilograms": "kg",
    "gram": "g",
    "grams": "g",
    "g per pound": "g/lb",
    "g per lb": "g/lb",
    "grams per lb": "g/lb",
    "g/lb of bodyweight": "g/lb",
    "g per pound of bodyweight": "g/lb",
    "g per pound of body weight": "g/lb",
    "grams per pound of body weight": "g/lb",
    "calorie": "kcal",
    "calories": "kcal",
    "degrees f": "°f",
    "degrees": "°f",
}

_VOLUME = {
    "gallon": ("L", Decimal("3.785411784")),
    "gallons": ("L", Decimal("3.785411784")),
    "quart": ("L", Decimal("0.946352946")),
    "quarts": ("L", Decimal("0.946352946")),
    "fl oz": ("ml", Decimal("29.5735295625")),
    "fluid ounce": ("ml", Decimal("29.5735295625")),
    "fluid ounces": ("ml", Decimal("29.5735295625")),
    "cup": ("ml", Decimal("236.5882365")),
    "cups": ("ml", Decimal("236.5882365")),
}

_PASS_THROUGH = frozenset({
    "kcal", "g", "kg", "mg", "ml", "l", "reps", "sets", "rir", "rpe", "%",
    "percent", "hours", "hour", "minutes", "minute",
})

# A quantity may be a mixed number ("5 5/8"), a fraction ("5/8") or a plain
# number ("1,000", "2.5"). Fractions come first so "5/8" is never split into
# "5" and a separate "8 inch".
_NUMBER = r"\d+\s+\d+/\d+|\d+/\d+|\d[\d,]*(?:\.\d+)?"
BRACE_RE = re.compile(
    r"\{\{\s*(" + _NUMBER + r")(?:\s*[-–]\s*(" + _NUMBER + r"))?\s*([^}]*?)\s*\}\}"
)
LOOSE_IMPERIAL_RE = re.compile(
    r"(?<![\w.~/])(" + _NUMBER + r")(?:\s*[-–]\s*(" + _NUMBER + r"))?\s?"
    r"(lbs?|pounds?|oz|ounces?|miles?|feet|foot|ft|inches|inch|gallons?|quarts?|°F|degrees fahrenheit)\b"
)
SAID_RE = re.compile(r"(\(said: [^)]*\))")
APPROX_WORD_RE = re.compile(r"\b(about|around|roughly|approximately|approx\.|circa|nearly|almost) ~", re.IGNORECASE)
FEET_INCHES_RE = re.compile(
    r"\{\{\s*(\d+)\s*(?:feet|foot|ft)\s*(?:and\s*)?(\d+(?:\.\d+)?)\s*(?:inches|inch|in)\s*\}\}"
)


@dataclass
class UnitReport:
    """Every repair or unknown unit met while converting one note."""

    unbraced_imperial: list[str] = field(default_factory=list)
    unknown_units: list[str] = field(default_factory=list)


def _n(value: Decimal | str) -> str:
    return f"{float(value):g}"


def _decimal(text: str) -> Decimal:
    """Parse a plain, fractional ("5/8") or mixed ("5 5/8") quantity."""

    text = text.replace(",", "").strip()
    whole = Decimal(0)
    if " " in text:
        head, text = text.split(None, 1)
        whole = Decimal(head)
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        return whole + Decimal(numerator) / Decimal(denominator)
    return whole + Decimal(text)


def _spoken(text: str) -> str:
    text = " ".join(text.replace(",", "").split())
    return text if "/" in text else _n(text)


def _round2(value: Decimal) -> Decimal:
    if value == 0:
        return Decimal(0)
    quantum = Decimal(1).scaleb(abs(value).adjusted() - 1)
    return (value / quantum).to_integral_value(rounding=ROUND_HALF_UP) * quantum


def _convert(value: str, value_high: str | None, unit: str, report: UnitReport, original: str) -> str:
    key = UNIT_ALIASES.get(unit.lower(), unit)
    spoken = f"{_spoken(value)}–{_spoken(value_high)} {unit}" if value_high else f"{_spoken(value)} {unit}"
    low_value = _decimal(value)
    high_value = _decimal(value_high) if value_high else None
    if key.lower() in _VOLUME:
        si_unit, factor = _VOLUME[key.lower()]
        low = _round2(low_value * factor)
        high = _round2(high_value * factor) if high_value is not None else None
        converted = f"~{_n(low)}–{_n(high)}" if high is not None else f"~{_n(low)}"
        return f"{converted} {si_unit} (said: {spoken})"
    if key.lower() in _PASS_THROUGH:
        return f"{_spoken(value)}–{_spoken(value_high)} {key}" if value_high else f"{_spoken(value)} {key}"
    result = convert_quantity(low_value, high_value, key)
    if result.unknown_unit or result.converted_value is None:
        report.unknown_units.append(original)
        return spoken
    low = _n(result.converted_value)
    converted = f"~{low}–{_n(result.converted_value_high)}" if value_high else f"~{low}"
    return f"{converted} {result.si_unit} (said: {spoken})"


def to_si(text: str, report: UnitReport) -> str:
    """Convert braced (and stray unbraced imperial) quantities in ``text``."""

    def feet_inches(match: re.Match) -> str:
        metres = (Decimal(match.group(1)) * 12 + Decimal(match.group(2))) * Decimal("0.0254")
        return f"~{_n(metres.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))} m (said: {match.group(1)} feet {match.group(2)} inches)"

    text = FEET_INCHES_RE.sub(feet_inches, text)
    text = BRACE_RE.sub(
        lambda m: _convert(m.group(1), m.group(2), m.group(3).strip(), report, m.group(0)),
        text,
    )

    def loose(match: re.Match) -> str:
        report.unbraced_imperial.append(match.group(0))
        return _convert(match.group(1), match.group(2), match.group(3), report, match.group(0))

    parts = SAID_RE.split(text)
    text = "".join(part if SAID_RE.fullmatch(part) else LOOSE_IMPERIAL_RE.sub(loose, part) for part in parts)
    # "about ~23 kg" says "approximately" twice; keep the writer's word.
    return APPROX_WORD_RE.sub(r"\1 ", text)
