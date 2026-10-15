"""Compare Apple and Whisper transcripts and produce review artifacts.

The compiler aligns normalized words globally before it reports differences. This
avoids sentence-boundary drift when the two sources use different punctuation.
It has no third-party dependencies and can be used as a CLI or imported.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable, Sequence

if __package__:
    from .alignment_diagnostics import (
        ReviewSpan,
        build_alignment_diagnostics_shadow,
        disabled_alignment_diagnostics,
    )
    from .terminology import DOMAIN_GLOSSARY, terminology_registry_manifest
else:  # Preserve direct ``python compiler/transcript.py`` compatibility.
    from alignment_diagnostics import (
        ReviewSpan,
        build_alignment_diagnostics_shadow,
        disabled_alignment_diagnostics,
    )
    from terminology import DOMAIN_GLOSSARY, terminology_registry_manifest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures" / "transcript_compiler"
DEFAULT_APPLE = FIXTURES_DIR / "apple-transcript.txt"
DEFAULT_WHISPER = FIXTURES_DIR / "whisper-transcript.txt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "var" / "compiler-reports"
LONG_TRANSCRIPT_WORDS = 6_000
ANCHOR_NGRAM_WORDS = 12
MIN_HALLUCINATION_RUNS = 3
MAX_HALLUCINATION_PHRASE_WORDS = 8
WHISPER_CONFIDENT_LOGPROB = -1.0
WHISPER_HIGH_NO_SPEECH_PROB = 0.6
MAX_RESOLVER_ITEMS = 40
MAX_TRIAGE_ITEMS = 80
TRIAGE_REASON_MAX_LENGTH = 150  # TASK-076: tightened from 400 so triage's
# worst-case per-item completion size is provable and small; see
# docs/superpowers/specs/2026-09-16-human-review-evidence-assisted-adjudication-design.md
# "Task 3 worst-case cost-estimate finding".
MIN_METADATA_ALIGNMENT_COVERAGE = 0.95
EPISODE_LOCAL_MEMORY_SCHEMA_VERSION = 1
TERMINOLOGY_RETRIEVAL_SCHEMA_VERSION = 1
MAX_EPISODE_LOCAL_MEMORY_OCCURRENCES = 5
MAX_HUMAN_EDIT_EXPANSION_WORDS = 3

TIMESTAMP_LINE = re.compile(
    r"^\s*(?:\[)?\d{1,3}:\d{2}(?::\d{2})?(?:[.,]\d{1,3})?(?:\])?\s*$"
)
TIMESTAMP_VALUE = re.compile(
    r"^\s*\[?(?P<value>\d{1,3}:\d{2}(?::\d{2})?(?:[.,]\d{1,3})?)\]?\s*$"
)
TOKEN_PATTERN = re.compile(
    r"\d+(?:[.,:/]\d+)+(?:[^\W\d_]+)?|\d+|%|[^\W\d_]+(?:['’][^\W_]+)*",
    re.UNICODE,
)
SPACE_PATTERN = re.compile(r"\s+")
PUNCTUATION_COLLISION = re.compile(r"([,.;:!?])\s+([,.;:!?])")

NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16",
    "seventeen": "17", "eighteen": "18", "nineteen": "19",
    "twenty": "20", "thirty": "30", "forty": "40", "fifty": "50",
    "sixty": "60", "seventy": "70", "eighty": "80", "ninety": "90",
}
NUMBER_SCALES = {
    "hundred": 100,
    "thousand": 1_000,
    "million": 1_000_000,
    "billion": 1_000_000_000,
    "trillion": 1_000_000_000_000,
}
FRACTION_DENOMINATORS = {
    "half": 2,
    "halves": 2,
    "third": 3,
    "thirds": 3,
    "quarter": 4,
    "quarters": 4,
    "fourth": 4,
    "fourths": 4,
    "fifth": 5,
    "fifths": 5,
    "sixth": 6,
    "sixths": 6,
    "seventh": 7,
    "sevenths": 7,
    "eighth": 8,
    "eighths": 8,
    "ninth": 9,
    "ninths": 9,
    "tenth": 10,
    "tenths": 10,
}
STRONG_FILLER_WORDS = {
    # Only unambiguous hesitation forms are eligible for automatic deletion.
    # Ambiguous short forms such as ER, Ah, or mm can be meaningful scientific
    # or measurement evidence and must be preserved/reviewed instead.
    "um", "uh", "hmm", "mmm", "uhh", "umm",
}
UNIT_ALIASES = {
    "kg": "kg", "kgs": "kg", "kilogram": "kg", "kilograms": "kg",
    "lb": "lb", "lbs": "lb", "pound": "lb", "pounds": "lb",
    "g": "g", "gram": "g", "grams": "g",
    "mg": "mg", "milligram": "mg", "milligrams": "mg",
    "mcg": "mcg", "μg": "mcg", "ug": "mcg", "microgram": "mcg",
    "micrograms": "mcg",
    "kcal": "kcal", "kcals": "kcal", "calorie": "kcal", "calories": "kcal",
    "cal": "cal", "ml": "ml", "milliliter": "ml", "milliliters": "ml",
    "liter": "l", "liters": "l", "litre": "l", "litres": "l",
    "percent": "%", "percentage": "%", "%": "%",
}
CITATION_CUES = {
    "study", "studies", "research", "researcher", "researchers", "paper",
    "trial", "trials", "review", "meta", "analysis", "authors", "author",
    "published", "journal", "et", "al",
}
RESOLVER_CATEGORIES = {
    "protocol_number", "unit", "exercise_name", "supplement", "training_term",
    "citation", "proper_name", "negation", "scientific_medical_term",
}
NEGATIONS = {
    "no", "not", "never", "none", "neither", "nor", "cannot", "can't",
    "couldn't", "didn't", "doesn't", "don't", "hadn't", "hasn't", "haven't",
    "isn't", "shouldn't", "wasn't", "weren't", "won't", "wouldn't",
}


@dataclass(frozen=True)
class Token:
    """A normalized word plus its character offsets in cleaned source text."""

    value: str
    original: str
    start: int
    end: int


@dataclass
class Difference:
    """One aligned difference that may need review."""

    id: int
    kind: str
    severity: str
    reason: str
    apple_start_word: int
    apple_end_word: int
    whisper_start_word: int
    whisper_end_word: int
    apple_text: str
    whisper_text: str
    apple_context: str
    whisper_context: str
    changed_apple_words: list[str]
    changed_whisper_words: list[str]
    local_similarity: float
    apple_repetition: bool
    whisper_repetition: bool
    whisper_suspected_repetition: bool
    whisper_avg_logprob: float | None
    whisper_no_speech_prob: float | None
    whisper_start_timestamp: float | None
    whisper_end_timestamp: float | None
    apple_start_timestamp: float | None
    apple_end_timestamp: float | None
    domain_terms: list[str]
    citation_signal: bool
    resolver_category: str
    source_only: bool
    source_only_source: str | None
    risk_reasons: list[str]
    preservation_class: str
    review_required: bool
    selected_source: str
    selected_text: str | None
    merge_action: str
    selection_reason: str


@dataclass
class ComparisonResult:
    """Serializable result returned by :func:`compile_transcripts`."""

    status: str
    recommended_source: str
    similarity: float
    matched_words: int
    apple_words: int
    whisper_words: int
    apple_coverage: float
    whisper_coverage: float
    apple_quality: float
    whisper_quality: float
    high_risk: int
    medium_risk: int
    low_risk: int
    compiler_edits: int
    auto_resolved: int
    review_required: int
    source_only_summary: dict[str, int]
    terminology_registry: dict[str, object]
    episode_local_memory: dict[str, object]
    terminology_retrieval: dict[str, object]
    alignment_diagnostics: dict[str, object]
    differences: list[Difference]
    compiled_transcript: str

    def to_dict(self) -> dict:
        data = asdict(self)
        data.pop("compiled_transcript", None)
        return data


def load_text(path: str | Path) -> str:
    """Load UTF-8 transcript text and give a useful error for a missing file."""

    transcript_path = Path(path)
    try:
        return transcript_path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Transcript not found: {transcript_path}") from error


def clean_transcript(text: str, *, remove_timestamps: bool = False) -> str:
    """Remove timestamp-only lines and normalize whitespace, retaining punctuation."""

    lines = text.replace("\ufeff", "").splitlines()
    if remove_timestamps:
        lines = [line for line in lines if not TIMESTAMP_LINE.fullmatch(line)]
    return SPACE_PATTERN.sub(" ", " ".join(lines)).strip()


def _timestamp_to_seconds(value: str) -> float:
    """Convert a Apple timestamp line to seconds."""

    match = TIMESTAMP_VALUE.fullmatch(value)
    if not match:
        raise ValueError(f"Invalid timestamp: {value!r}")
    parts = match.group("value").replace(",", ".").split(":")
    seconds = float(parts[-1])
    if len(parts) >= 2:
        seconds += int(parts[-2]) * 60
    if len(parts) == 3:
        seconds += int(parts[0]) * 3600
    return seconds


def _clean_apple_with_timestamps(
    text: str, *, remove_timestamps: bool
) -> tuple[str, list[float | None]]:
    """Clean Apple text and preserve an approximate timestamp per word.

    Apple exports commonly put a timestamp on its own line before a caption
    block.  The cleaned transcript intentionally omits those lines, while the
    returned list lets review records retain a jump-back-to-audio hint.
    """

    if not remove_timestamps:
        clean = clean_transcript(text)
        return clean, [None] * len(tokenize(clean))

    entries: list[tuple[str, float | None]] = []
    active_timestamp: float | None = None
    for line in text.replace("\ufeff", "").splitlines():
        if TIMESTAMP_LINE.fullmatch(line):
            active_timestamp = _timestamp_to_seconds(line)
        elif line.strip():
            entries.append((line, active_timestamp))

    clean = SPACE_PATTERN.sub(" ", " ".join(line for line, _ in entries)).strip()
    timestamps: list[float | None] = []
    for index, (line, start) in enumerate(entries):
        words = tokenize(line)
        if not words:
            continue
        end = next(
            (
                candidate
                for _, candidate in entries[index + 1 :]
                if candidate is not None and candidate != start
            ),
            None,
        )
        if start is None or end is None or end <= start:
            timestamps.extend([start] * len(words))
            continue
        interval = (end - start) / len(words)
        timestamps.extend(start + position * interval for position in range(len(words)))

    # A malformed caption line should never make the compiler fail; it merely
    # loses timestamp hints for that input.
    if len(timestamps) != len(tokenize(clean)):
        timestamps = [None] * len(tokenize(clean))
    return clean, timestamps


def normalize_token(value: str) -> str:
    normalized = value.casefold().replace("’", "'")
    if re.fullmatch(r"\d{1,3}(?:,\d{3})+", normalized):
        normalized = normalized.replace(",", "")
    elif re.fullmatch(r"\d+,\d+", normalized):
        normalized = normalized.replace(",", ".")
    return NUMBER_WORDS.get(normalized, normalized)


def tokenize(text: str) -> list[Token]:
    """Tokenize text while retaining offsets into the cleaned source."""

    return [
        Token(normalize_token(match.group(0)), match.group(0), match.start(), match.end())
        for match in TOKEN_PATTERN.finditer(text)
    ]


def _comparison_values(tokens: Sequence[Token]) -> list[str]:
    """Return alignment keys without changing the source tokens or audit text.

    A percentage unit is interchangeable only when immediately attached to a
    numeric mention. This lets alignment retain the number and its unit as one
    deterministic expression while leaving bare ``percent`` evidence
    conservative and reviewable.
    """

    values = [token.value for token in tokens]
    for index, value in enumerate(values):
        if (
            value in {"percent", "percentage", "%"}
            and index > 0
            and bool(_numbers([values[index - 1]]))
        ):
            values[index] = "%"
    return values


def _load_source_metadata(
    metadata: str | Path | dict | None,
) -> tuple[
    list[str],
    list[float | None],
    list[float | None],
    list[float | None],
    list[float | None],
]:
    """Read optional transcript JSON as words, confidence, and timestamps.

    Supports faster-whisper style ``segments[].words[]`` objects as well as
    segment-only JSON.  Word probabilities are converted to logprobs so every
    confidence source uses the same threshold.  Unknown shapes are ignored
    rather than making the text compiler unusable.
    """

    if metadata is None:
        return [], [], [], [], []
    if isinstance(metadata, (str, Path)):
        try:
            payload = json.loads(Path(metadata).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Could not read Whisper metadata: {error}") from error
    elif isinstance(metadata, dict):
        payload = metadata
    else:
        raise ValueError("source metadata must be a JSON path or dictionary")

    segments = payload.get("segments", []) if isinstance(payload, dict) else []
    if not isinstance(segments, list):
        return [], [], [], [], []
    values: list[str] = []
    logprobs: list[float | None] = []
    no_speech: list[float | None] = []
    starts: list[float | None] = []
    ends: list[float | None] = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        segment_logprob = segment.get("avg_logprob")
        segment_no_speech = segment.get("no_speech_prob")
        segment_start = segment.get("start")
        segment_end = segment.get("end")
        words = segment.get("words")
        if not isinstance(words, list):
            words = [{"word": segment.get("text", "")}]
        for word in words:
            if not isinstance(word, dict):
                continue
            word_text = str(word.get("word", word.get("text", "")))
            word_logprob = word.get("avg_logprob", word.get("logprob", segment_logprob))
            if word_logprob is None and word.get("probability") is not None:
                probability = float(word["probability"])
                word_logprob = math.log(max(probability, 1e-12))
            try:
                normalized_logprob = float(word_logprob) if word_logprob is not None else None
            except (TypeError, ValueError):
                normalized_logprob = None
            try:
                normalized_no_speech = (
                    float(word.get("no_speech_prob", segment_no_speech))
                    if word.get("no_speech_prob", segment_no_speech) is not None
                    else None
                )
            except (TypeError, ValueError):
                normalized_no_speech = None
            try:
                normalized_start = float(word.get("start", segment_start))
            except (TypeError, ValueError):
                normalized_start = None
            try:
                normalized_end = float(word.get("end", segment_end))
            except (TypeError, ValueError):
                normalized_end = None
            for token in tokenize(word_text):
                values.append(token.value)
                logprobs.append(normalized_logprob)
                no_speech.append(normalized_no_speech)
                starts.append(normalized_start)
                ends.append(normalized_end)
    return values, logprobs, no_speech, starts, ends


def _empty_metadata_alignment(
    size: int,
) -> tuple[
    list[float | None],
    list[float | None],
    list[float | None],
    list[float | None],
]:
    empty = [None] * size
    return empty, empty.copy(), empty.copy(), empty.copy()


def _average_metadata(values: Sequence[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else None


def _first_metadata(values: Sequence[float | None]) -> float | None:
    return next((value for value in values if value is not None), None)


def _last_metadata(values: Sequence[float | None]) -> float | None:
    return next((value for value in reversed(values) if value is not None), None)


def _decimal_metadata_match(
    transcript_value: str, metadata_values: Sequence[str]
) -> bool:
    """Return True for a decimal split across Whisper word objects.

    Faster Whisper can expose ``1.4`` as two word objects, for example ``1``
    and ``.4``. Tokenizing those objects independently produces ``1`` and
    ``4`` even though the rendered transcript correctly tokenizes as ``1.4``.

    Only this narrow numeric split is merged. Arbitrary replacement spans stay
    unaligned.
    """

    match = re.fullmatch(r"(\d+)\.(\d+)", transcript_value)

    return bool(
        match
        and list(metadata_values) == [
            match.group(1),
            match.group(2),
        ]
    )


def _align_source_metadata(
    tokens: Sequence[Token], metadata: str | Path | dict | None
) -> tuple[
    list[float | None],
    list[float | None],
    list[float | None],
    list[float | None],
]:
    """Attach source metadata with a conservative token-alignment fallback.

    Exact token identity remains the fast path.

    When metadata tokenization differs slightly from the rendered transcript,
    equal spans are aligned and known decimal splits such as ``1.4`` versus
    ``1`` + ``4`` are merged.

    Ambiguous spans stay ``None``. If fewer than 95% of transcript tokens can
    be aligned confidently, all metadata is discarded rather than risking a
    shifted confidence/timestamp mapping.
    """

    values, logprobs, no_speech, starts, ends = _load_source_metadata(metadata)

    target_values = [
        token.value
        for token in tokens
    ]

    if not values or not tokens:
        return _empty_metadata_alignment(
            len(tokens)
        )

    # Fast path: the common case remains zero-cost beyond the old behaviour.
    if (
        len(values) == len(tokens)
        and values == target_values
    ):
        return (
            logprobs,
            no_speech,
            starts,
            ends,
        )

    aligned_logprobs: list[float | None] = [
        None
    ] * len(tokens)

    aligned_no_speech: list[float | None] = [
        None
    ] * len(tokens)

    aligned_starts: list[float | None] = [
        None
    ] * len(tokens)

    aligned_ends: list[float | None] = [
        None
    ] * len(tokens)

    aligned_tokens = 0

    for tag, i1, i2, j1, j2 in _alignment_opcodes(
        target_values,
        values,
    ):
        if tag == "equal":
            length = i2 - i1

            aligned_logprobs[i1:i2] = (
                logprobs[j1:j2]
            )

            aligned_no_speech[i1:i2] = (
                no_speech[j1:j2]
            )

            aligned_starts[i1:i2] = (
                starts[j1:j2]
            )

            aligned_ends[i1:i2] = (
                ends[j1:j2]
            )

            aligned_tokens += length

            continue

        # Faster Whisper can represent a decimal as two word objects:
        #
        # rendered transcript:  1.4
        # word metadata:        1   .4
        # token metadata:       1    4
        #
        # Merge only this deterministic case.
        if (
            tag == "replace"
            and i2 - i1 == 1
            and _decimal_metadata_match(
                target_values[i1],
                values[j1:j2],
            )
        ):
            aligned_logprobs[i1] = (
                _average_metadata(
                    logprobs[j1:j2]
                )
            )

            aligned_no_speech[i1] = (
                _average_metadata(
                    no_speech[j1:j2]
                )
            )

            aligned_starts[i1] = (
                _first_metadata(
                    starts[j1:j2]
                )
            )

            aligned_ends[i1] = (
                _last_metadata(
                    ends[j1:j2]
                )
            )

            aligned_tokens += 1

    coverage = (
        aligned_tokens
        / len(tokens)
    )

    if (
        coverage
        < MIN_METADATA_ALIGNMENT_COVERAGE
    ):
        return _empty_metadata_alignment(
            len(tokens)
        )

    return (
        aligned_logprobs,
        aligned_no_speech,
        aligned_starts,
        aligned_ends,
    )


def _range_average(values: Sequence[float | None], start: int, end: int) -> float | None:
    selected = [value for value in values[start:end] if value is not None]
    return round(sum(selected) / len(selected), 4) if selected else None


def _range_timestamp(
    timestamps: Sequence[float | None], start: int, end: int
) -> tuple[float | None, float | None]:
    if not timestamps:
        return None, None
    start_index = min(max(start, 0), len(timestamps) - 1)
    end_index = min(max(end - 1, 0), len(timestamps) - 1)
    return timestamps[start_index], timestamps[end_index]


def _slice_text(text: str, tokens: Sequence[Token], start: int, end: int) -> str:
    if start >= end or not tokens:
        return ""
    return text[tokens[start].start : tokens[end - 1].end].strip()


def _context_text(
    text: str,
    tokens: Sequence[Token],
    start: int,
    end: int,
    radius: int = 10,
) -> str:
    if not tokens:
        return ""
    context_start = max(0, start - radius)
    context_end = min(len(tokens), max(end, start + 1) + radius)
    return _slice_text(text, tokens, context_start, context_end)


def _collapse_adjacent(words: Iterable[str]) -> list[str]:
    collapsed: list[str] = []
    for word in words:
        if not collapsed or collapsed[-1] != word:
            collapsed.append(word)
    return collapsed


def _format_number(value: float) -> str:
    return str(int(value)) if value.is_integer() else str(round(value, 6))


def _numbers(words: Iterable[str]) -> list[str]:
    """Return canonical numeric mentions from digits and common spoken forms.

    Token normalization already maps single number words to digits.  This
    parser deliberately handles adjacent normalized values as spoken groups,
    including scales, decimals, fractions, and the common "twenty twenty-four"
    year construction.  It is conservative: false positives cause review, not
    a silent merge.
    """

    values = list(words)
    numbers: list[str] = []
    index = 0
    numeric = re.compile(r"\d+(?:\.\d+)?")
    fraction = re.compile(r"\d+(?:\.\d+)?/\d+(?:\.\d+)?")
    while index < len(values):
        word = values[index]

        # Token normalization turns spoken "one" into "1". In expressions
        # such as "the next one", however, it is a pronoun rather than a
        # numeric protocol value and must not create a number mismatch.
        if (
            word == "1"
            and index > 0
            and values[index - 1] == "next"
        ):
            index += 1
            continue

        if word in {"a", "an"} and index + 1 < len(values):
            following = values[index + 1]
            if following in FRACTION_DENOMINATORS:
                numbers.append(_format_number(1 / FRACTION_DENOMINATORS[following]))
                index += 2
                continue
        if fraction.fullmatch(word):
            numerator, denominator = word.split("/", 1)
            if denominator != "0":
                numbers.append(_format_number(float(numerator) / float(denominator)))
            index += 1
            continue
        if word not in {"a", "an"} and not numeric.fullmatch(word):
            index += 1
            continue

        # Adjacent rendered number tokens can be an ASR formatting split
        # (``34 81``), a spoken construction (``seventeen sixty``), or two
        # distinct values.  Their intent cannot be known from this source
        # alone, so retain each token instead of summing it.  Cross-source
        # equivalence below may join them only when the competing evidence
        # proves the exact value.
        if numeric.fullmatch(word):
            run_end = index
            while run_end < len(values) and numeric.fullmatch(values[run_end]):
                run_end += 1
            run = values[index:run_end]
            following = values[run_end] if run_end < len(values) else ""
            followed_by_fraction = (
                following == "and"
                and (
                    (run_end + 1 < len(values) and values[run_end + 1] in FRACTION_DENOMINATORS)
                    or (
                        run_end + 2 < len(values)
                        and values[run_end + 1] in {"a", "an"}
                        and values[run_end + 2] in FRACTION_DENOMINATORS
                    )
                )
            )
            if following in NUMBER_SCALES or following == "point" or following in FRACTION_DENOMINATORS or followed_by_fraction:
                pass
            elif (
                len(run) == 3
                and run[:2] == ["20", "20"]
                and run[2].isdigit()
            ):
                numbers.append(str(2000 + int(run[1]) + int(run[2])))
                index = run_end
                continue
            else:
                numbers.extend(
                    str(int(part)) if part.isdigit() else _format_number(float(part))
                    for part in run
                )
                index = run_end
                continue

        start = index
        total = 0.0
        current = 0.0
        consumed = False
        preceded_by_indefinite_article = False
        while index < len(values):
            part = values[index]
            if part in {"a", "an"} and not consumed:
                following = values[index + 1] if index + 1 < len(values) else ""
                if following in NUMBER_SCALES:
                    current = 1.0
                    consumed = True
                    index += 1
                    continue
                break
            if part in {"a", "an"} and consumed:
                following = values[index + 1] if index + 1 < len(values) else ""
                if following in FRACTION_DENOMINATORS:
                    preceded_by_indefinite_article = True
                index += 1
                continue
            if part in {"and", "a", "an"} and consumed:
                index += 1
                continue
            if part == "point" and consumed:
                decimal_digits: list[str] = []
                index += 1
                while index < len(values) and values[index].isdigit():
                    decimal_digits.append(values[index])
                    index += 1
                if decimal_digits:
                    current += float(f"0.{''.join(decimal_digits)}")
                break
            if part in FRACTION_DENOMINATORS:
                denominator = FRACTION_DENOMINATORS[part]
                if preceded_by_indefinite_article:
                    total += current + 1.0 / denominator
                else:
                    total += (current or 1.0) / denominator
                current = 0.0
                consumed = True
                index += 1
                break
            if part in NUMBER_SCALES:
                scale = NUMBER_SCALES[part]
                current = max(1.0, current) * scale
                if scale >= 1_000:
                    total += current
                    current = 0.0
                consumed = True
                index += 1
                continue
            if numeric.fullmatch(part):
                value = float(part)
                # Spoken years such as "twenty twenty-four" normalize to
                # 20, 20, 4.  Treat that form as 2024 instead of 44.
                if (
                    part == "20"
                    and index + 1 < len(values)
                    and values[index + 1] == "20"
                    and index + 2 < len(values)
                    and values[index + 2].isdigit()
                ):
                    current += 2000 + int(values[index + 1]) + int(values[index + 2])
                    consumed = True
                    index += 3
                    continue
                current += value
                consumed = True
                index += 1
                continue
            break
        if consumed:
            numbers.append(_format_number(total + current))
        elif index == start:
            index += 1
    return numbers


def _numeric_component_groups(words: Sequence[str]) -> list[list[str]] | None:
    """Return simple digit groups, preserving range boundaries.

    This intentionally accepts only bare digit tokens, ``to``, and canonical
    units.  Number scales and other words fall back to the ordinary parser;
    they must not be guessed into a compact value.
    """

    groups: list[list[str]] = [[]]
    saw_numeric = False
    for word in words:
        if re.fullmatch(r"\d+", word):
            groups[-1].append(word)
            saw_numeric = True
        elif word == "to":
            if not groups[-1]:
                return None
            groups.append([])
        elif word not in UNIT_ALIASES:
            return None
    if not saw_numeric or not groups[-1]:
        return None
    return groups


def _compact_numeric_group_values(parts: Sequence[str]) -> set[str]:
    """Return exact compact readings for one bare-digit ASR group.

    The output is deliberately only a *candidate* set.  It becomes an
    equivalence only when the opposing source supplies exactly one matching
    value.  This permits ``34 81`` -> ``3481`` and the spoken grouping
    ``twenty one twelve`` -> ``2112`` without globally reinterpreting either
    sequence when no counterpart proves it.
    """

    if not parts or any(len(part) > 1 and part.startswith("0") for part in parts):
        return set()

    values: set[str] = set()

    def visit(index: int, rendered: list[str]) -> None:
        if index == len(parts):
            candidate = "".join(rendered)
            if candidate and not candidate.startswith("0"):
                values.add(str(int(candidate)))
            return

        visit(index + 1, rendered + [parts[index]])
        if (
            index + 1 < len(parts)
            and int(parts[index]) in {20, 30, 40, 50, 60, 70, 80, 90}
            and 0 < int(parts[index + 1]) < 10
        ):
            visit(
                index + 2,
                rendered + [str(int(parts[index]) + int(parts[index + 1]))],
            )

    visit(0, [])
    return values


def _canonical_non_numeric_words(words: Sequence[str]) -> list[str]:
    """Normalize units while excluding proven spoken-number syntax."""

    canonical: list[str] = []
    for index, word in enumerate(words):
        if re.fullmatch(r"\d+", word) or word == "to" or word in NUMBER_SCALES:
            continue

        # In forms such as "a thousand five hundred", the article belongs to
        # the spoken numeric expression rather than the surrounding semantics.
        if (
            word in {"a", "an"}
            and index + 1 < len(words)
            and words[index + 1] in NUMBER_SCALES
        ):
            continue

        # Preserve ordinary spoken "and". Ignore it only when it is provably
        # connecting the remainder of a scale-based number.
        if (
            word == "and"
            and index > 0
            and words[index - 1] in NUMBER_SCALES
            and index + 1 < len(words)
            and re.fullmatch(r"\d+", words[index + 1])
        ):
            continue

        canonical.append(
            f"unit:{UNIT_ALIASES[word]}" if word in UNIT_ALIASES else word
        )

    return canonical


def _numeric_or_unit_equivalent(
    apple_words: Sequence[str], whisper_words: Sequence[str]
) -> bool:
    """Prove only formatting-level numeric or unit equivalence.

    Both the non-numeric wording and the ordered range boundaries must match.
    Bare split digits are compacted solely against a matching competing value;
    ambiguous scale-based speech is intentionally left for review.
    """

    apple_non_numeric = _canonical_non_numeric_words(apple_words)
    whisper_non_numeric = _canonical_non_numeric_words(whisper_words)
    if apple_non_numeric != whisper_non_numeric:
        return False

    apple_numbers = _numbers(apple_words)
    whisper_numbers = _numbers(whisper_words)
    if apple_numbers and whisper_numbers and apple_numbers == whisper_numbers:
        return True

    if not apple_numbers and not whisper_numbers:
        return any(word.startswith("unit:") for word in apple_non_numeric)

    apple_groups = _numeric_component_groups(apple_words)
    whisper_groups = _numeric_component_groups(whisper_words)
    if apple_groups is None or whisper_groups is None or len(apple_groups) != len(whisper_groups):
        return False

    return all(
        bool(_compact_numeric_group_values(apple_group) & _compact_numeric_group_values(whisper_group))
        for apple_group, whisper_group in zip(apple_groups, whisper_groups)
    )


def _measurement_units(words: Sequence[str]) -> set[str]:
    """Return normalized units that occur close to a numeric mention."""

    units: set[str] = set()
    for index, word in enumerate(words):
        unit = UNIT_ALIASES.get(word)
        if unit and _numbers(words[max(0, index - 4) : index]):
            units.add(unit)
    return units


def _nearby_values(
    tokens: Sequence[Token], start: int, end: int, *, radius: int = 5
) -> list[str]:
    return [
        token.value
        for token in tokens[max(0, start - radius) : min(len(tokens), end + radius)]
    ]


def _domain_glossary_matches(words: Sequence[str]) -> list[str]:
    """Find exact or close domain terms without changing the transcript."""

    matches: list[str] = []
    for terms in DOMAIN_GLOSSARY.values():
        for term in terms:
            term_words = [token.value for token in tokenize(term)]
            if not term_words:
                continue
            for size in range(max(1, len(term_words) - 1), len(term_words) + 2):
                for start in range(0, len(words) - size + 1):
                    candidate = list(words[start : start + size])
                    if candidate == term_words:
                        matches.append(term)
                        break
                    if len("".join(term_words)) >= 4 and _word_similarity(
                        candidate, term_words
                    ) >= 0.82:
                        matches.append(term)
                        break
                if term in matches:
                    break
    return matches


def _citation_signal(words: Sequence[str]) -> bool:
    """Identify study/year references that should never be silently trusted."""

    has_year = any(re.fullmatch(r"(?:19|20)\d{2}", word) for word in words)
    has_cue = bool(set(words) & CITATION_CUES)
    has_et_al = any(
        first == "et" and second == "al"
        for first, second in zip(words, words[1:])
    )
    return has_et_al or (has_year and has_cue)


def _token_near_domain_term(
    changed_word: str,
    term_word: str,
) -> bool:
    """Return True when a changed token plausibly belongs to a domain term.

    This is intentionally token-local. A glossary term merely appearing a few
    words away must not turn an unrelated wording difference into a domain
    mismatch.
    """

    if changed_word == term_word:
        return True

    if not changed_word or not term_word:
        return False

    similarity = SequenceMatcher(
        None,
        changed_word,
        term_word,
        autojunk=False,
    ).ratio()

    if (
        min(len(changed_word), len(term_word)) >= 4
        and similarity >= 0.80
    ):
        return True

    # Preserve short acronym near-misses such as RPE -> RP.
    if (
        min(len(changed_word), len(term_word)) >= 2
        and max(len(changed_word), len(term_word)) <= 4
        and (
            changed_word.startswith(term_word)
            or term_word.startswith(changed_word)
        )
    ):
        return True

    return False


def _domain_terms_for_difference(
    changed_words: Sequence[str],
    nearby_words: Sequence[str],
) -> list[str]:
    """Return nearby glossary matches anchored to an actually changed token."""

    candidates = _domain_glossary_matches(
        nearby_words
    )

    anchored: list[str] = []

    for term in candidates:
        term_words = [
            token.value
            for token in tokenize(term)
        ]

        if any(
            _token_near_domain_term(
                changed_word,
                term_word,
            )
            for changed_word in changed_words
            for term_word in term_words
        ):
            anchored.append(term)

    return anchored


def _citation_difference_signal(
    changed_words: Sequence[str],
    nearby_words: Sequence[str],
) -> bool:
    """Return True only when the changed span itself carries citation meaning.

    A year or citation cue five words away from an unrelated contraction must
    not reclassify that contraction as a citation mismatch.
    """

    changed = list(changed_words)
    nearby = list(nearby_words)

    changed_has_year = any(
        re.fullmatch(r"(?:19|20)\d{2}", word)
        for word in changed
    )

    changed_has_cue = bool(
        set(changed)
        & CITATION_CUES
    )

    nearby_has_year = any(
        re.fullmatch(r"(?:19|20)\d{2}", word)
        for word in nearby
    )

    nearby_has_cue = bool(
        set(nearby)
        & CITATION_CUES
    )

    changed_has_et_al = any(
        first == "et"
        and second == "al"
        for first, second in zip(
            changed,
            changed[1:],
        )
    )

    return (
        changed_has_et_al
        or (
            changed_has_year
            and nearby_has_cue
        )
        or (
            changed_has_cue
            and nearby_has_year
        )
    )


def _resolver_category(
    *,
    kind: str,
    domain_terms: Sequence[str],
    citation_signal: bool,
) -> str:
    if citation_signal:
        return "citation"
    if kind == "unit_mismatch":
        return "unit"
    if kind == "number_mismatch":
        return "protocol_number"
    if kind == "negation_mismatch":
        return "negation"
    if kind == "source_only_proper_name":
        return "proper_name"
    for category, terms in DOMAIN_GLOSSARY.items():
        if any(term in terms for term in domain_terms):
            return category
    return "other"


def _source_only_semantic_risk(
    words: Sequence[str],
    *,
    domain_terms: Sequence[str],
    citation_signal: bool,
    proper_name_signal: bool,
    low_whisper_acoustic_confidence: bool,
) -> list[str]:
    """Return deterministic reasons that one-sided evidence merits review.

    This intentionally favors review for recognized meaning-sensitive evidence.
    An unrecognized span is *not* unsafe to preserve: risk controls escalation,
    while preservation remains the default for non-noise source-only text.
    """

    reasons: list[str] = []
    if _numbers(words):
        reasons.append("number_or_date")
    if any(word in UNIT_ALIASES for word in words):
        reasons.append("measurement_unit")
    if _negations(words):
        reasons.append("negation")
    if citation_signal or any(term in DOMAIN_GLOSSARY["citation"] for term in domain_terms):
        reasons.append("citation_or_known_name")
    if proper_name_signal:
        reasons.append("proper_name_or_entity")
    if domain_terms:
        reasons.extend(
            f"domain_{category}"
            for category, terms in DOMAIN_GLOSSARY.items()
            if category != "citation" and any(term in terms for term in domain_terms)
        )
    if low_whisper_acoustic_confidence:
        reasons.append("low_whisper_acoustic_confidence")
    return list(dict.fromkeys(reasons))


def _source_only_proper_name_signal(
    tokens: Sequence[Token],
    source_text: str,
) -> bool:
    """Detect conservative person/entity signals in one-sided raw source text.

    A single sentence-initial capital is deliberately insufficient. We require
    two adjacent title-cased tokens, an honorific followed by a title-cased
    token, or an uncommon all-caps entity token. This is escalation-only: it
    never supplies evidence that the source is correct.
    """

    originals = [token.original.strip() for token in tokens]
    title_case = [
        bool(re.fullmatch(r"[A-Z][a-z]+(?:['’][A-Z][a-z]+)?", value))
        for value in originals
    ]
    if (
        "." not in source_text
        and any(first and second for first, second in zip(title_case, title_case[1:]))
    ):
        return True
    if any(
        token.value in {"dr", "prof", "mr", "mrs", "ms"} and next_is_title
        for token, next_is_title in zip(tokens, title_case[1:])
    ):
        return True
    return any(
        len(value) >= 3 and value.isalpha() and value.isupper()
        for value in originals
    )


def _low_whisper_acoustic_confidence(
    average_logprob: float | None,
    no_speech_probability: float | None,
) -> bool:
    """Reuse Whisper's existing confidence thresholds for a review decision."""

    return (
        no_speech_probability is not None
        and no_speech_probability >= WHISPER_HIGH_NO_SPEECH_PROB
    ) or (
        average_logprob is not None
        and average_logprob < WHISPER_CONFIDENT_LOGPROB
    )


def _source_only_kind_and_category(
    risk_reasons: Sequence[str],
    domain_terms: Sequence[str],
    citation_signal: bool,
) -> tuple[str, str]:
    """Select a focused resolver category for high-risk one-sided evidence."""

    if "citation_or_known_name" in risk_reasons or citation_signal:
        return "citation_mismatch", "citation"
    if "proper_name_or_entity" in risk_reasons:
        return "source_only_proper_name", "proper_name"
    if "measurement_unit" in risk_reasons:
        return "unit_mismatch", "unit"
    if "number_or_date" in risk_reasons:
        return "number_mismatch", "protocol_number"
    if "negation" in risk_reasons:
        return "negation_mismatch", "negation"
    if "low_whisper_acoustic_confidence" in risk_reasons:
        return "source_only_low_whisper_acoustic_confidence", "other"
    category = _resolver_category(
        kind="source_only_domain_term",
        domain_terms=domain_terms,
        citation_signal=citation_signal,
    )
    return "domain_term_mismatch", category


def _source_only_is_confirmed_filler(item: Difference) -> bool:
    """Return whether one-sided text is safe to discard as pure hesitation.

    Filler deletion is deliberately narrower than risk classification.  A
    semantic/domain/entity signal always wins, and all-caps raw tokens are
    preserved because short scientific abbreviations can normalize to ordinary
    hesitation spellings.
    """

    if not item.source_only:
        return False

    source_text = (
        item.apple_text
        if item.source_only_source == "apple"
        else item.whisper_text
    )
    source_tokens = tokenize(source_text)
    source_words = [token.value for token in source_tokens]
    if not source_words or any(word not in STRONG_FILLER_WORDS for word in source_words):
        return False

    if any(
        len(token.original) > 1 and token.original.isupper()
        for token in source_tokens
    ):
        return False

    semantic_risk = {
        reason
        for reason in item.risk_reasons
        if reason not in {
            "low_whisper_acoustic_confidence",
            "suspected_repetition",
        }
    }
    if semantic_risk or item.domain_terms or item.citation_signal:
        return False

    return True


def _anonymous_resolver_kind(kind: str) -> str:
    """Remove provider identity from model-facing difference labels."""

    if kind == "source_only_low_whisper_acoustic_confidence":
        return "source_only_low_acoustic_confidence"
    if kind in {"whisper_addition", "whisper_omission"}:
        return "source_only_difference"
    return kind


def _anonymous_resolver_risk_reasons(reasons: Sequence[str]) -> list[str]:
    """Return model-facing risk labels without Apple/Whisper identity."""

    return [
        "low_acoustic_confidence"
        if reason == "low_whisper_acoustic_confidence"
        else reason
        for reason in reasons
    ]


def _negations(words: Iterable[str]) -> set[str]:
    values = list(words)
    found: set[str] = set()
    for index, word in enumerate(values):
        following = values[index + 1] if index + 1 < len(values) else ""
        if word == "no" and following in {"cebo", "sebo"}:
            continue
        if word in NEGATIONS:
            found.add(word)
    return found


def _word_similarity(apple_words: Sequence[str], whisper_words: Sequence[str]) -> float:
    if not apple_words and not whisper_words:
        return 1.0
    return round(
        SequenceMatcher(None, apple_words, whisper_words, autojunk=False).ratio(), 4
    )


def _classify_difference(
    apple_words: Sequence[str], whisper_words: Sequence[str]
) -> tuple[str, str, str]:
    """Return ``(kind, severity, reason)`` for unmatched normalized words."""

    yt_numbers = _numbers(apple_words)
    wh_numbers = _numbers(whisper_words)
    yt_negations = _negations(apple_words)
    wh_negations = _negations(whisper_words)
    largest_side = max(len(apple_words), len(whisper_words))

    if _numeric_or_unit_equivalent(apple_words, whisper_words):
        return (
            "numeric_unit_equivalent",
            "low",
            "Numeric formatting and canonical unit identity are equivalent",
        )

    has_numbers_on_both_sides = bool(yt_numbers and wh_numbers)
    substantial_number_omission = largest_side >= 4
    if (
        yt_numbers != wh_numbers
        and (yt_numbers or wh_numbers)
        and (has_numbers_on_both_sides or substantial_number_omission)
    ):
        return (
            "number_mismatch",
            "high",
            f"Different numbers: Apple={yt_numbers or ['none']}, "
            f"Whisper={wh_numbers or ['none']}",
        )
    if yt_negations != wh_negations and (yt_negations or wh_negations):
        return (
            "negation_mismatch",
            "high",
            "A negation appears in only one transcript and may reverse the meaning",
        )
    if _collapse_adjacent(apple_words) == _collapse_adjacent(whisper_words):
        return (
            "repeated_word",
            "medium",
            "An adjacent repetition may be intentional speech and needs review",
        )
    if not whisper_words:
        severity = "high" if largest_side >= 40 else "medium" if largest_side >= 10 else "low"
        return (
            "whisper_omission",
            severity,
            f"Whisper is missing {len(apple_words)} aligned word(s)",
        )
    if not apple_words:
        severity = "high" if largest_side >= 40 else "medium" if largest_side >= 10 else "low"
        return (
            "whisper_addition",
            severity,
            f"Whisper contains {len(whisper_words)} extra aligned word(s)",
        )

    similarity = _word_similarity(apple_words, whisper_words)
    if similarity >= 0.65 or largest_side <= 2:
        return (
            "wording_difference",
            "low",
            "Likely spelling, name, contraction, or minor transcription variation",
        )
    severity = "medium" if largest_side >= 4 else "low"
    return (
        "transcription_difference",
        severity,
        "The aligned wording differs enough to require a context check",
    )


def _difference_groups(
    opcodes: Sequence[tuple[str, int, int, int, int]], *, merge_gap: int
) -> list[list[tuple[str, int, int, int, int]]]:
    """Merge nearby word edits without merging across strong alignment anchors."""

    groups: list[list[tuple[str, int, int, int, int]]] = []
    current: list[tuple[str, int, int, int, int]] = []
    for index, opcode in enumerate(opcodes):
        tag, i1, i2, j1, j2 = opcode
        if tag != "equal":
            current.append(opcode)
            continue
        equal_length = min(i2 - i1, j2 - j1)
        next_is_difference = index + 1 < len(opcodes) and opcodes[index + 1][0] != "equal"
        if current and next_is_difference and equal_length <= merge_gap:
            current.append(opcode)
        elif current:
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups


def _build_differences(
    apple_text: str,
    whisper_text: str,
    apple_tokens: Sequence[Token],
    whisper_tokens: Sequence[Token],
    opcodes: Sequence[tuple[str, int, int, int, int]],
    *,
    merge_gap: int,
    whisper_logprobs: Sequence[float | None],
    whisper_no_speech: Sequence[float | None],
    whisper_starts: Sequence[float | None],
    whisper_ends: Sequence[float | None],
    apple_timestamps: Sequence[float | None],
) -> list[Difference]:
    if __package__:
        from .review_policy import representation_equivalence
    else:
        from review_policy import representation_equivalence

    differences: list[Difference] = []
    groups = _difference_groups(opcodes, merge_gap=merge_gap)

    for difference_id, group in enumerate(groups, start=1):
        y_start = min(opcode[1] for opcode in group)
        y_end = max(opcode[2] for opcode in group)
        w_start = min(opcode[3] for opcode in group)
        w_end = max(opcode[4] for opcode in group)
        changed_yt: list[str] = []
        changed_wh: list[str] = []
        apple_changed_ranges: list[tuple[int, int]] = []
        whisper_changed_ranges: list[tuple[int, int]] = []
        for tag, i1, i2, j1, j2 in group:
            if tag == "equal":
                continue
            changed_yt.extend(token.value for token in apple_tokens[i1:i2])
            changed_wh.extend(token.value for token in whisper_tokens[j1:j2])
            if i1 < i2:
                apple_changed_ranges.append((i1, i2))
            if j1 < j2:
                whisper_changed_ranges.append((j1, j2))

        kind, severity, reason = _classify_difference(changed_yt, changed_wh)
        apple_nearby = _nearby_values(apple_tokens, y_start, y_end)
        whisper_nearby = _nearby_values(whisper_tokens, w_start, w_end)
        apple_units = _measurement_units(apple_nearby)
        whisper_units = _measurement_units(whisper_nearby)
        if apple_units and whisper_units and apple_units != whisper_units:
            kind = "unit_mismatch"
            severity = "high"
            reason = (
                f"Different measurement units: Apple={sorted(apple_units)}, "
                f"Whisper={sorted(whisper_units)}"
            )
        apple_domain_nearby = _nearby_values(
            apple_tokens,
            y_start,
            y_end,
            radius=2,
        )
        whisper_domain_nearby = _nearby_values(
            whisper_tokens,
            w_start,
            w_end,
            radius=2,
        )

        domain_terms = list(
            dict.fromkeys(
                _domain_terms_for_difference(
                    changed_yt,
                    apple_domain_nearby,
                )
                + _domain_terms_for_difference(
                    changed_wh,
                    whisper_domain_nearby,
                )
            )
        )

        apple_citation_nearby = _nearby_values(
            apple_tokens,
            y_start,
            y_end,
            radius=2,
        )
        whisper_citation_nearby = _nearby_values(
            whisper_tokens,
            w_start,
            w_end,
            radius=2,
        )

        citation_signal = (
            _citation_difference_signal(
                changed_yt,
                apple_citation_nearby,
            )
            or _citation_difference_signal(
                changed_wh,
                whisper_citation_nearby,
            )
        )

        source_only_source = (
            "apple" if changed_yt and not changed_wh
            else "whisper" if changed_wh and not changed_yt
            else None
        )
        source_tokens = (
            apple_tokens[y_start:y_end]
            if source_only_source == "apple"
            else whisper_tokens[w_start:w_end]
            if source_only_source == "whisper"
            else ()
        )
        source_span_text = (
            _slice_text(apple_text, apple_tokens, y_start, y_end)
            if source_only_source == "apple"
            else _slice_text(whisper_text, whisper_tokens, w_start, w_end)
            if source_only_source == "whisper"
            else ""
        )
        whisper_avg_logprob = _range_average(whisper_logprobs, w_start, w_end)
        whisper_no_speech_prob = _range_average(whisper_no_speech, w_start, w_end)
        source_only_suspected_repetition = (
            source_only_source is not None
            and _has_suspected_repeat_run(changed_yt or changed_wh)
        )
        risk_reasons = _source_only_semantic_risk(
            changed_yt or changed_wh,
            domain_terms=domain_terms,
            citation_signal=citation_signal,
            proper_name_signal=_source_only_proper_name_signal(
                source_tokens,
                source_span_text,
            ),
            low_whisper_acoustic_confidence=(
                source_only_source == "whisper"
                and _low_whisper_acoustic_confidence(
                    whisper_avg_logprob,
                    whisper_no_speech_prob,
                )
            ),
        ) if source_only_source else []
        if source_only_suspected_repetition:
            risk_reasons.append("suspected_repetition")

        if source_only_source and risk_reasons:
            kind, resolver_category = _source_only_kind_and_category(
                risk_reasons,
                domain_terms,
                citation_signal,
            )
            semantic_risk = any(
                reason not in {
                    "low_whisper_acoustic_confidence",
                    "suspected_repetition",
                }
                for reason in risk_reasons
            )
            severity = "high" if semantic_risk else "medium"
            reason = (
                f"{source_only_source.title()}-only evidence contains "
                + ", ".join(risk_reasons)
            )
        elif citation_signal and kind in {"wording_difference", "transcription_difference"}:
            kind = "citation_mismatch"
            severity = "medium"
            reason = "A study citation, researcher name, or study year differs"
        elif domain_terms and kind in {"wording_difference", "transcription_difference"}:
            kind = "domain_term_mismatch"
            severity = "medium"
            reason = "A fitness-domain term differs and should be checked against the glossary"
        if not (source_only_source and risk_reasons):
            resolver_category = _resolver_category(
                kind=kind,
                domain_terms=domain_terms,
                citation_signal=citation_signal,
            )

        representation = representation_equivalence(
            _slice_text(apple_text, apple_tokens, y_start, y_end),
            _slice_text(whisper_text, whisper_tokens, w_start, w_end),
            apple_context=_context_text(
                apple_text,
                apple_tokens,
                y_start,
                y_end,
                radius=30,
            ),
            whisper_context=_context_text(
                whisper_text,
                whisper_tokens,
                w_start,
                w_end,
                radius=30,
            ),
        )
        if (
            representation is not None
            and source_only_source is None
            and not citation_signal
            and (
                not domain_terms
                # TASK-131: a spacing-only variant of a domain term ("lat
                # pull down" / "lat pulldown") names the same term.
                or representation["equivalence_class"] == "compound_spacing"
            )
        ):
            kind = "representation_equivalent"
            severity = "low"
            reason = str(representation["reason"])
            resolver_category = "other"

        apple_repetition = bool(apple_changed_ranges) and all(
            _span_is_adjacent_duplicate(apple_tokens, start, end)
            for start, end in apple_changed_ranges
        )
        whisper_repetition = bool(whisper_changed_ranges) and all(
            _span_is_adjacent_duplicate(whisper_tokens, start, end)
            for start, end in whisper_changed_ranges
        )
        if (apple_repetition or whisper_repetition) and severity == "low":
            severity = "medium"
            reason = "An adjacent repetition may be intentional speech and needs review"
        differences.append(
            Difference(
                id=difference_id,
                kind=kind,
                severity=severity,
                reason=reason,
                apple_start_word=y_start + 1,
                apple_end_word=y_end,
                whisper_start_word=w_start + 1,
                whisper_end_word=w_end,
                apple_text=_slice_text(apple_text, apple_tokens, y_start, y_end),
                whisper_text=_slice_text(whisper_text, whisper_tokens, w_start, w_end),
                apple_context=_context_text(
                    apple_text,
                    apple_tokens,
                    y_start,
                    y_end,
                    radius=30,
                ),
                whisper_context=_context_text(
                    whisper_text,
                    whisper_tokens,
                    w_start,
                    w_end,
                    radius=30,
                ),
                changed_apple_words=changed_yt,
                changed_whisper_words=changed_wh,
                local_similarity=_word_similarity(changed_yt, changed_wh),
                apple_repetition=apple_repetition,
                whisper_repetition=whisper_repetition,
                whisper_suspected_repetition=_has_suspected_repeat_run(changed_wh),
                whisper_avg_logprob=whisper_avg_logprob,
                whisper_no_speech_prob=whisper_no_speech_prob,
                whisper_start_timestamp=_range_timestamp(
                    whisper_starts, w_start, w_end
                )[0],
                whisper_end_timestamp=_range_timestamp(
                    whisper_ends, w_start, w_end
                )[1],
                apple_start_timestamp=_range_timestamp(
                    apple_timestamps, y_start, y_end
                )[0],
                apple_end_timestamp=_range_timestamp(
                    apple_timestamps, y_start, y_end
                )[1],
                domain_terms=domain_terms,
                citation_signal=citation_signal,
                resolver_category=resolver_category,
                source_only=source_only_source is not None,
                source_only_source=source_only_source,
                risk_reasons=risk_reasons,
                preservation_class="pending_classification" if source_only_source else "not_source_only",
                review_required=False,
                selected_source="",
                selected_text=None,
                merge_action="",
                selection_reason="",
            )
        )
    return differences


def _span_is_adjacent_duplicate(
    tokens: Sequence[Token], start: int, end: int
) -> bool:
    """Return True when a differing span repeats the phrase beside it."""

    if start >= end:
        return False
    values = [token.value for token in tokens]
    fragment = values[start:end]
    size = len(fragment)
    previous = values[max(0, start - size) : start]
    following = values[end : end + size]
    return previous == fragment or following == fragment


def _has_suspected_repeat_run(words: Sequence[str]) -> bool:
    """Detect a repeated short multi-word phrase that merits review.

    We intentionally do not treat a single repeated word ("no no no") as a
    hallucination: that is common and meaningful conversational emphasis. A
    phrase must repeat three or more times before we flag it, but repetition
    alone is not affirmative proof that the content is a hallucination.
    """

    max_size = min(MAX_HALLUCINATION_PHRASE_WORDS, len(words) // MIN_HALLUCINATION_RUNS)
    for size in range(2, max_size + 1):
        for start in range(0, len(words) - size * MIN_HALLUCINATION_RUNS + 1):
            phrase = list(words[start : start + size])
            runs = 1
            cursor = start + size
            while words[cursor : cursor + size] == phrase:
                runs += 1
                cursor += size
            if runs >= MIN_HALLUCINATION_RUNS:
                return True
    return False


def _joined_word_preference(
    apple_words: Sequence[str], whisper_words: Sequence[str]
) -> str | None:
    """Prefer a readable split when one source contains an obvious run-on word."""

    if not apple_words or not whisper_words:
        return None
    if "".join(apple_words) != "".join(whisper_words):
        return None
    if len(apple_words) == len(whisper_words):
        return None

    apple_longest = max(map(len, apple_words))
    whisper_longest = max(map(len, whisper_words))
    if (
        len(apple_words) < len(whisper_words)
        and apple_longest >= 9
        and min(map(len, whisper_words)) >= 5
    ):
        return "whisper"
    if (
        len(whisper_words) < len(apple_words)
        and whisper_longest >= 9
        and min(map(len, apple_words)) >= 5
    ):
        return "apple"
    return None


def _adjacent_repetitions(words: Sequence[str]) -> int:
    return sum(first == second for first, second in zip(words, words[1:]))


def _assign_merge_decisions(
    differences: Sequence[Difference],
    apple_tokens: Sequence[Token],
    whisper_tokens: Sequence[Token],
    *,
    primary_source: str,
) -> None:
    """Select a source for every aligned difference using conservative rules."""

    for item in differences:
        item.review_required = False
        apple_duplicate = item.apple_repetition
        whisper_duplicate = item.whisper_repetition
        apple_suspected_repetition = _has_suspected_repeat_run(
            item.changed_apple_words
        )

        if item.source_only and _source_only_is_confirmed_filler(item):
            other_source = (
                "whisper" if item.source_only_source == "apple" else "apple"
            )
            item.selected_source = other_source
            item.merge_action = (
                f"discarded_{item.source_only_source}_confirmed_filler"
            )
            item.preservation_class = "discarded_confirmed_filler"
            item.selection_reason = (
                "Source-only span contains only unambiguous strong hesitation "
                "filler and no semantic/domain/entity signal"
            )
            continue

        if item.source_only and item.risk_reasons:
            item.selected_source = primary_source
            semantic_risk = any(
                reason not in {
                    "low_whisper_acoustic_confidence",
                    "suspected_repetition",
                }
                for reason in item.risk_reasons
            )
            if semantic_risk:
                item.merge_action = "review_source_only_semantic_risk"
                item.preservation_class = "awaiting_review"
                item.review_required = True
                item.selection_reason = (
                    "Source-only evidence has independent semantic risk and is held "
                    "for focused review"
                )
                continue
            if "suspected_repetition" in item.risk_reasons:
                item.merge_action = "review_source_only_suspected_repetition"
                item.preservation_class = "awaiting_review"
                item.review_required = True
                item.selection_reason = (
                    "A repeated source-only phrase may be real speech and cannot be "
                    "silently removed"
                )
                continue
            # Acoustic confidence is observability, not semantic evidence.  Keep
            # the one-sided span exactly as ordinary preserved evidence while its
            # low-confidence reason remains in the JSON report.
            item.selected_source = item.source_only_source
            item.merge_action = "preserved_source_only_low_whisper_confidence"
            item.preservation_class = "preserved_low_whisper_confidence"
            item.selection_reason = (
                "Low Whisper acoustic confidence alone does not delete preserved "
                "source-only evidence or require Human Review"
            )
            continue

        if item.kind in {"number_mismatch", "negation_mismatch"}:
            item.selected_source = primary_source
            item.merge_action = "review_kept_primary"
            item.review_required = True
            item.selection_reason = (
                "Meaning-sensitive conflict; automatic replacement is disabled"
            )
            continue

        if item.whisper_suspected_repetition and not apple_suspected_repetition:
            item.selected_source = primary_source
            item.merge_action = "review_suspected_whisper_repetition"
            item.review_required = True
            item.selection_reason = (
                "Whisper contains a repeated multi-word phrase, which is insufficient "
                "evidence to discard it automatically"
            )
            continue

        if apple_suspected_repetition and not item.whisper_suspected_repetition:
            item.selected_source = primary_source
            item.merge_action = "review_suspected_apple_repetition"
            item.review_required = True
            item.selection_reason = (
                "Apple contains a repeated multi-word phrase, which is insufficient "
                "evidence to discard it automatically"
            )
            continue

        if item.source_only:
            item.selected_source = item.source_only_source
            item.merge_action = f"preserved_{item.source_only_source}_content"
            item.preservation_class = "preserved"
            item.selection_reason = (
                "Ordinary source-only evidence is preserved regardless of length "
                "or primary-source selection"
            )
            continue

        if item.kind == "representation_equivalent":
            item.selected_source = primary_source
            item.merge_action = "equivalent_representation"
            item.selection_reason = item.reason
            continue

        if item.kind == "numeric_unit_equivalent":
            item.selected_source = primary_source
            item.merge_action = "equivalent_numeric_unit_formatting"
            item.selection_reason = (
                "Both sources prove the same ordered numeric values and canonical units"
            )
            continue

        if apple_duplicate or whisper_duplicate:
            item.selected_source = primary_source
            item.merge_action = "review_kept_primary"
            item.review_required = True
            item.selection_reason = (
                "A one-off adjacent repetition may be deliberate speech; it was not "
                "silently removed"
            )
            continue

        if item.changed_apple_words and not item.changed_whisper_words:
            item.selected_source = "apple"
            item.merge_action = "kept_apple_content"
            item.selection_reason = (
                "Apple contains aligned words missing from the Whisper transcript"
            )
            continue

        joined_preference = _joined_word_preference(
            item.changed_apple_words, item.changed_whisper_words
        )
        if joined_preference:
            item.selected_source = joined_preference
            item.merge_action = "split_run_on_word"
            item.selection_reason = "Selected the readable split form of a run-on word"
            continue

        if item.severity in {"high", "medium"}:
            item.selected_source = primary_source
            item.merge_action = "review_kept_primary"
            item.review_required = True
            item.selection_reason = (
                "High/medium-risk transcript evidence requires explicit adjudication; "
                "Whisper acoustic confidence is weak evidence and cannot resolve it"
            )
            continue

        if item.whisper_avg_logprob is not None:
            no_speech = item.whisper_no_speech_prob
            if (
                item.whisper_avg_logprob >= WHISPER_CONFIDENT_LOGPROB
                and (no_speech is None or no_speech < WHISPER_HIGH_NO_SPEECH_PROB)
            ):
                item.selected_source = "whisper"
                item.merge_action = "preferred_confident_whisper"
                item.selection_reason = (
                    "Whisper word confidence is high and no-speech probability is low"
                )
                continue
            if no_speech is not None and no_speech >= WHISPER_HIGH_NO_SPEECH_PROB:
                item.selected_source = "apple"
                item.merge_action = "rejected_low_speech_whisper"
                item.selection_reason = (
                    "Whisper reports a high no-speech probability for this fragment"
                )
                continue

        item.selected_source = primary_source
        item.merge_action = (
            "review_kept_primary"
            if item.severity in {"high", "medium"}
            else "kept_primary"
        )
        item.review_required = item.merge_action == "review_kept_primary"
        item.selection_reason = (
            "No deterministic quality advantage; retained the primary source"
        )


def _character_span(
    text: str, tokens: Sequence[Token], start: int, end: int
) -> tuple[int, int]:
    if start < end:
        return tokens[start].start, tokens[end - 1].end
    if start < len(tokens):
        position = tokens[start].start
    else:
        position = len(text)
    return position, position


def _fit_inserted_text(text: str, start: int, end: int, replacement: str) -> str:
    """Add only the boundary spaces needed for an inserted fragment."""

    replacement = replacement.strip()
    if not replacement or start != end:
        return replacement
    prefix = "" if start == 0 or text[start - 1].isspace() else " "
    suffix = "" if start >= len(text) or text[start].isspace() else " "
    return prefix + replacement + suffix


def _normalize_compiled_spacing(text: str) -> str:
    """Clean whitespace and punctuation collisions left by token-span edits."""

    text = SPACE_PATTERN.sub(" ", text).strip()
    while PUNCTUATION_COLLISION.search(text):
        text = PUNCTUATION_COLLISION.sub(r"\2", text)
    return re.sub(r"\s+([,.;:!?])", r"\1", text)


def render_review_merge_preview(
    context_text: str,
    *,
    span_start: int,
    span_end: int,
    replacement: str,
) -> str:
    """Render one read-only review edit with canonical compiler spacing rules."""

    if (
        not isinstance(span_start, int)
        or not isinstance(span_end, int)
        or span_start < 0
        or span_end < span_start
        or span_end > len(context_text)
    ):
        raise ValueError("Invalid review preview span")
    if not isinstance(replacement, str):
        raise ValueError("Review preview replacement must be text")

    fitted = _fit_inserted_text(
        context_text,
        span_start,
        span_end,
        replacement,
    )
    merged = (
        context_text[:span_start]
        + fitted
        + context_text[span_end:]
    )
    return _normalize_compiled_spacing(merged)


def _build_consensus_transcript(
    apple_text: str,
    whisper_text: str,
    apple_tokens: Sequence[Token],
    whisper_tokens: Sequence[Token],
    differences: Sequence[Difference],
    *,
    primary_source: str,
    edit_overrides: dict[int, tuple[int, int, str]] | None = None,
) -> tuple[str, int]:
    """Patch safe secondary-source fragments into the selected primary text."""

    if primary_source == "apple":
        base_text = apple_text
        base_tokens = apple_tokens
    else:
        base_text = whisper_text
        base_tokens = whisper_tokens

    edit_overrides = edit_overrides or {}
    edits: list[tuple[int, int, str]] = []
    for item in differences:
        override = edit_overrides.get(item.id)
        if override is None and item.selected_source == primary_source:
            continue
        if override is not None:
            start, end, replacement = override
        elif primary_source == "apple":
            start = item.apple_start_word - 1
            end = item.apple_end_word
            replacement = item.selected_text or item.whisper_text
        else:
            start = item.whisper_start_word - 1
            end = item.whisper_end_word
            replacement = item.selected_text or item.apple_text
        char_start, char_end = _character_span(base_text, base_tokens, start, end)
        replacement = _fit_inserted_text(
            base_text, char_start, char_end, replacement
        )
        edits.append((char_start, char_end, replacement))

    compiled = base_text
    for char_start, char_end, replacement in sorted(edits, reverse=True):
        compiled = compiled[:char_start] + replacement + compiled[char_end:]
    return _normalize_compiled_spacing(compiled), len(edits)


def _repetition_rate(tokens: Sequence[Token]) -> float:
    if len(tokens) < 2:
        return 0.0
    repetitions = sum(
        first.value == second.value for first, second in zip(tokens, tokens[1:])
    )
    return repetitions / len(tokens)


def _punctuation_score(text: str, tokens: Sequence[Token]) -> float:
    if not tokens:
        return 0.0
    sentence_marks = sum(text.count(mark) for mark in ".?!")
    return min(1.0, sentence_marks / max(1.0, len(tokens) * 0.02))


def _looks_like_auto_captions(text: str, tokens: Sequence[Token]) -> bool:
    """Return true when a sufficiently long source has no sentence punctuation."""

    return len(tokens) >= 20 and not re.search(r"[.?!](?=\s|$)", text)


def _source_quality(
    text: str,
    tokens: Sequence[Token],
    *,
    matched_words: int,
    longest_source: int,
) -> float:
    if not tokens:
        return 0.0
    coverage = matched_words / len(tokens)
    completeness = len(tokens) / max(1, longest_source)
    punctuation_weight = 0 if _looks_like_auto_captions(text, tokens) else 15
    score = (
        coverage * 20
        + completeness * 60
        + _punctuation_score(text, tokens) * punctuation_weight
        + (1 - min(1.0, _repetition_rate(tokens) * 20)) * 5
    )
    maximum = 85 + punctuation_weight
    return round(max(0.0, min(100.0, score / maximum * 100)), 2)


def _anchor_pairs(
    apple_values: Sequence[str], whisper_values: Sequence[str]
) -> list[tuple[int, int]]:
    """Return ordered unique n-gram anchors for long transcript alignment."""

    def positions(values: Sequence[str]) -> dict[tuple[str, ...], int | None]:
        found: dict[tuple[str, ...], int | None] = {}
        for index in range(len(values) - ANCHOR_NGRAM_WORDS + 1):
            ngram = tuple(values[index : index + ANCHOR_NGRAM_WORDS])
            found[ngram] = index if ngram not in found else None
        return found

    apple_positions = positions(apple_values)
    whisper_positions = positions(whisper_values)
    pairs = sorted(
        (apple_index, whisper_positions[ngram])
        for ngram, apple_index in apple_positions.items()
        if apple_index is not None and whisper_positions.get(ngram) is not None
    )
    if not pairs:
        return []

    tails: list[int] = []
    tail_indices: list[int] = []
    previous: list[int | None] = [None] * len(pairs)
    for index, (_, whisper_index) in enumerate(pairs):
        position = bisect.bisect_left(tails, whisper_index)
        if position:
            previous[index] = tail_indices[position - 1]
        if position == len(tails):
            tails.append(whisper_index)
            tail_indices.append(index)
        else:
            tails[position] = whisper_index
            tail_indices[position] = index
    selected: list[tuple[int, int]] = []
    cursor: int | None = tail_indices[-1]
    while cursor is not None:
        selected.append(pairs[cursor])
        cursor = previous[cursor]
    selected.reverse()

    non_overlapping: list[tuple[int, int]] = []
    apple_end = whisper_end = 0
    for apple_index, whisper_index in selected:
        if apple_index >= apple_end and whisper_index >= whisper_end:
            non_overlapping.append((apple_index, whisper_index))
            apple_end = apple_index + ANCHOR_NGRAM_WORDS
            whisper_end = whisper_index + ANCHOR_NGRAM_WORDS
    return non_overlapping


def _alignment_opcodes(
    apple_values: Sequence[str], whisper_values: Sequence[str]
) -> list[tuple[str, int, int, int, int]]:
    """Globally align short input and anchor/window-align long input."""

    if max(len(apple_values), len(whisper_values)) < LONG_TRANSCRIPT_WORDS:
        return SequenceMatcher(None, apple_values, whisper_values, autojunk=False).get_opcodes()

    anchors = _anchor_pairs(apple_values, whisper_values)
    if not anchors:
        return SequenceMatcher(None, apple_values, whisper_values, autojunk=False).get_opcodes()

    opcodes: list[tuple[str, int, int, int, int]] = []
    apple_cursor = whisper_cursor = 0
    for apple_anchor, whisper_anchor in anchors:
        window = SequenceMatcher(
            None,
            apple_values[apple_cursor:apple_anchor],
            whisper_values[whisper_cursor:whisper_anchor],
            autojunk=False,
        )
        opcodes.extend(
            (
                tag,
                i1 + apple_cursor,
                i2 + apple_cursor,
                j1 + whisper_cursor,
                j2 + whisper_cursor,
            )
            for tag, i1, i2, j1, j2 in window.get_opcodes()
        )
        opcodes.append(
            (
                "equal",
                apple_anchor,
                apple_anchor + ANCHOR_NGRAM_WORDS,
                whisper_anchor,
                whisper_anchor + ANCHOR_NGRAM_WORDS,
            )
        )
        apple_cursor = apple_anchor + ANCHOR_NGRAM_WORDS
        whisper_cursor = whisper_anchor + ANCHOR_NGRAM_WORDS
    window = SequenceMatcher(
        None,
        apple_values[apple_cursor:],
        whisper_values[whisper_cursor:],
        autojunk=False,
    )
    opcodes.extend(
        (
            tag,
            i1 + apple_cursor,
            i2 + apple_cursor,
            j1 + whisper_cursor,
            j2 + whisper_cursor,
        )
        for tag, i1, i2, j1, j2 in window.get_opcodes()
    )
    return opcodes


def compile_transcripts(
    apple_text: str,
    whisper_text: str,
    *,
    primary: str = "auto",
    merge_gap: int = 3,
    apple_has_timestamps: bool = True,
    apple_metadata: str | Path | dict | None = None,
    whisper_metadata: str | Path | dict | None = None,
    resolver_resolutions: Sequence[dict] | None = None,
    episode_local_memory_shadow: bool = True,
    alignment_diagnostics_shadow: bool = True,
) -> ComparisonResult:
    """Align, compare, classify, and select a recommended transcript.

    ``apple_metadata`` and ``whisper_metadata`` optionally provide segment or
    word timestamps. Whisper confidence values affect only otherwise unresolved
    wording decisions, never number or negation conflicts.
    Pass accepted entries returned by :func:`validate_resolver_response` as
    ``resolver_resolutions`` to apply them against freshly rebuilt token spans.
    """

    if primary not in {"auto", "apple", "whisper"}:
        raise ValueError("primary must be 'auto', 'apple', or 'whisper'")
    if merge_gap < 0:
        raise ValueError("merge_gap must be zero or greater")

    apple_clean, apple_timestamps = _clean_apple_with_timestamps(
        apple_text, remove_timestamps=apple_has_timestamps
    )
    whisper_clean = clean_transcript(whisper_text)
    apple_tokens = tokenize(apple_clean)
    whisper_tokens = tokenize(whisper_clean)
    if not apple_tokens:
        raise ValueError("Apple transcript contains no words")
    if not whisper_tokens:
        raise ValueError("Whisper transcript contains no words")

    apple_values = _comparison_values(apple_tokens)
    whisper_values = _comparison_values(whisper_tokens)
    _, _, apple_starts, _ = _align_source_metadata(apple_tokens, apple_metadata)
    if any(timestamp is not None for timestamp in apple_starts):
        apple_timestamps = apple_starts
    whisper_logprobs, whisper_no_speech, whisper_starts, whisper_ends = _align_source_metadata(
        whisper_tokens, whisper_metadata
    )
    opcodes = _alignment_opcodes(apple_values, whisper_values)
    matched_words = sum(i2 - i1 for tag, i1, i2, _, _ in opcodes if tag == "equal")
    differences = _build_differences(
        apple_clean,
        whisper_clean,
        apple_tokens,
        whisper_tokens,
        opcodes,
        merge_gap=merge_gap,
        whisper_logprobs=whisper_logprobs,
        whisper_no_speech=whisper_no_speech,
        whisper_starts=whisper_starts,
        whisper_ends=whisper_ends,
        apple_timestamps=apple_timestamps,
    )

    longest_source = max(len(apple_tokens), len(whisper_tokens))
    apple_quality = _source_quality(
        apple_clean, apple_tokens, matched_words=matched_words, longest_source=longest_source
    )
    whisper_quality = _source_quality(
        whisper_clean, whisper_tokens, matched_words=matched_words, longest_source=longest_source
    )
    recommended_source = primary
    if primary == "auto":
        recommended_source = "apple" if apple_quality >= whisper_quality else "whisper"

    _assign_merge_decisions(
        differences,
        apple_tokens,
        whisper_tokens,
        primary_source=recommended_source,
    )
    episode_local_memory = (
        _build_episode_local_memory_shadow(
            differences,
            apple_tokens,
            whisper_tokens,
            opcodes,
        )
        if episode_local_memory_shadow
        else {
            "schema_version": EPISODE_LOCAL_MEMORY_SCHEMA_VERSION,
            "mode": "disabled",
            "authoritative": False,
            "decision_effect": "none",
            "evidence_contract": "aligned_equal_same_episode_only",
            "eligible_difference_count": 0,
            "candidate_form_count": 0,
            "episode_confirmed_form_count": 0,
            "prior_confirmed_form_count": 0,
            "one_sided_signal_count": 0,
            "signal_count": 0,
            "signals": [],
        }
    )
    edit_overrides = _apply_validated_resolver_resolutions(
        differences,
        resolver_resolutions,
        primary_source=recommended_source,
        apple_text=apple_clean,
        whisper_text=whisper_clean,
        apple_tokens=apple_tokens,
        whisper_tokens=whisper_tokens,
    )
    compiled_transcript, compiler_edits = _build_consensus_transcript(
        apple_clean,
        whisper_clean,
        apple_tokens,
        whisper_tokens,
        differences,
        primary_source=recommended_source,
        edit_overrides=edit_overrides,
    )

    # Alignment diagnostics run only after the canonical decision surface has
    # already been built. They observe suspicious regions and compare a bounded
    # Needleman-Wunsch candidate, but they cannot affect merge/resolver/review
    # behavior or the compiled transcript.
    alignment_diagnostics = (
        build_alignment_diagnostics_shadow(
            apple_values,
            whisper_values,
            opcodes,
            review_spans=[
                ReviewSpan(
                    difference_id=item.id,
                    apple_start=max(0, item.apple_start_word - 1),
                    apple_end=item.apple_end_word,
                    whisper_start=max(0, item.whisper_start_word - 1),
                    whisper_end=item.whisper_end_word,
                    review_required=item.review_required,
                )
                for item in differences
            ],
        )
        if alignment_diagnostics_shadow
        else disabled_alignment_diagnostics()
    )

    counts = {
        severity: sum(item.severity == severity for item in differences)
        for severity in ("high", "medium", "low")
    }
    auto_resolved = sum(
        item.merge_action
        not in {"kept_primary", "review_kept_primary"}
        for item in differences
    )
    review_required = sum(item.review_required for item in differences)
    status = "review" if review_required else "pass"
    source_only_summary = {
        "apple_only": sum(item.source_only_source == "apple" for item in differences),
        "whisper_only": sum(item.source_only_source == "whisper" for item in differences),
        "preserved": sum(
            item.preservation_class
            in {"preserved", "preserved_low_whisper_confidence"}
            for item in differences
        ),
        "low_confidence_only_preserved": sum(
            item.preservation_class == "preserved_low_whisper_confidence"
            for item in differences
        ),
        "discarded_confirmed_noise": sum(
            item.preservation_class == "discarded_confirmed_filler"
            for item in differences
        ),
        "high_risk": sum(
            item.source_only and item.severity == "high" for item in differences
        ),
        "review_required": sum(
            item.source_only and item.review_required for item in differences
        ),
    }
    return ComparisonResult(
        status=status,
        recommended_source=recommended_source,
        similarity=round(2 * matched_words / (len(apple_tokens) + len(whisper_tokens)), 6),
        matched_words=matched_words,
        apple_words=len(apple_tokens),
        whisper_words=len(whisper_tokens),
        apple_coverage=round(matched_words / len(apple_tokens), 6),
        whisper_coverage=round(matched_words / len(whisper_tokens), 6),
        apple_quality=apple_quality,
        whisper_quality=whisper_quality,
        high_risk=counts["high"],
        medium_risk=counts["medium"],
        low_risk=counts["low"],
        compiler_edits=compiler_edits,
        auto_resolved=auto_resolved,
        review_required=review_required,
        source_only_summary=source_only_summary,
        terminology_registry=terminology_registry_manifest(),
        episode_local_memory=episode_local_memory,
        terminology_retrieval={
            "schema_version": TERMINOLOGY_RETRIEVAL_SCHEMA_VERSION,
            "mode": "disabled",
            "stage": "post_decision_observability",
            "authoritative": False,
            "decision_effect": "none",
            "resolver_visibility": "none",
            "review_routing_effect": "none",
            "candidate_registry_status": "not_attached",
            "signals": [],
        },
        alignment_diagnostics=alignment_diagnostics,
        differences=differences,
        compiled_transcript=compiled_transcript,
    )


def resolver_response_schema() -> dict:
    """Return the compact JSON contract expected from a small resolver model."""

    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["resolutions"],
        "properties": {
            "resolutions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "id",
                        "resolved_value",
                        "corrected_text",
                        "category",
                        "confidence",
                        "flag_for_human",
                        "note",
                    ],
                    "properties": {
                        "id": {"type": "integer"},
                        "resolved_value": {
                            "enum": ["source_a", "source_b", "neither", "unclear"],
                        },
                        "corrected_text": {"type": ["string", "null"]},
                        "category": {"enum": sorted(RESOLVER_CATEGORIES)},
                        "confidence": {"enum": ["high", "medium", "low"]},
                        "flag_for_human": {"type": "boolean"},
                        "note": {"type": "string"},
                    },
                },
            },
        },
    }


def _is_is_contraction_equivalent(
    apple_text: str,
    whisper_text: str,
) -> bool:
    """Recognize a narrow ``X is`` <-> ``X's`` equivalence.

    This prevents examples such as ``creatine is`` versus ``creatine's`` from
    becoming AI-review work merely because the shared noun is in the domain
    glossary.
    """

    def values(text: str) -> list[str]:
        return [
            token.value
            for token in tokenize(text)
        ]

    apple = values(apple_text)
    whisper = values(whisper_text)

    def matches(
        expanded: Sequence[str],
        contracted: Sequence[str],
    ) -> bool:
        if (
            len(expanded) != 2
            or expanded[1] != "is"
            or len(contracted) != 1
        ):
            return False

        return (
            contracted[0]
            == expanded[0] + "'s"
        )

    return (
        matches(apple, whisper)
        or matches(whisper, apple)
    )


def _closest_domain_focus(
    source_text: str,
    changed_words: Sequence[str],
    domain_terms: Sequence[str],
) -> str | None:
    """Return the exact source token most closely tied to a glossary term."""

    if not source_text:
        return None

    changed = set(
        changed_words
    )

    if not changed:
        return None

    term_words = [
        token.value
        for term in domain_terms
        for token in tokenize(term)
    ]

    if not term_words:
        return None

    candidates = [
        token
        for token in tokenize(source_text)
        if token.value in changed
    ]

    best_token = None
    best_score = 0.0

    for token in candidates:
        for term_word in term_words:
            score = SequenceMatcher(
                None,
                token.value,
                term_word,
                autojunk=False,
            ).ratio()

            if score > best_score:
                best_score = score
                best_token = token

    # RPE -> RP is 0.8, creatine -> creating is comfortably above this.
    if (
        best_token is None
        or best_score < 0.65
    ):
        return None

    return best_token.original


def _resolver_focus(
    item: Difference,
) -> tuple[str, str, str]:
    """Return Apple focus, Whisper focus, and focus scope.

    Domain-term conflicts can contain unrelated wording differences inside the
    same alignment group. In those cases the resolver should adjudicate only
    the glossary-related token, not the whole multi-error span.
    """

    if item.resolver_category in {
        "supplement",
        "training_term",
        "exercise_name",
    }:
        apple_focus = _closest_domain_focus(
            item.apple_text,
            item.changed_apple_words,
            item.domain_terms,
        )

        whisper_focus = _closest_domain_focus(
            item.whisper_text,
            item.changed_whisper_words,
            item.domain_terms,
        )

        if (
            apple_focus
            and whisper_focus
            and (
                apple_focus != item.apple_text
                or whisper_focus != item.whisper_text
            )
        ):
            return (
                apple_focus,
                whisper_focus,
                "partial",
            )

    return (
        item.apple_text,
        item.whisper_text,
        "full",
    )


def _replace_focus_once(
    source_text: str,
    old_focus: str,
    new_focus: str,
) -> str | None:
    """Replace one exact focus occurrence while preserving surrounding source."""

    if not old_focus:
        return None

    source_folded = source_text.casefold()
    focus_folded = old_focus.casefold()

    if source_folded.count(
        focus_folded
    ) != 1:
        return None

    start = source_folded.index(
        focus_folded
    )

    end = start + len(
        old_focus
    )

    return (
        source_text[:start]
        + new_focus
        + source_text[end:]
    )


def _normalized_words(text: str) -> list[str]:
    """Normalize supplied text for exact local-evidence checks only."""

    return [token.value for token in tokenize(text)]


def _contains_word_sequence(haystack: str, needle: str) -> int:
    """Count exact normalized phrase occurrences without plausibility matching."""

    source_words = _normalized_words(haystack)
    needle_words = _normalized_words(needle)
    if not needle_words or len(needle_words) > len(source_words):
        return 0
    return sum(
        source_words[index : index + len(needle_words)] == needle_words
        for index in range(len(source_words) - len(needle_words) + 1)
    )


def _aligned_equal_occurrences(
    focus: str,
    apple_tokens: Sequence[Token],
    whisper_tokens: Sequence[Token],
    opcodes: Sequence[tuple[str, int, int, int, int]],
    *,
    exclude_apple: tuple[int, int],
    exclude_whisper: tuple[int, int],
) -> list[dict[str, int | str]]:
    """Return exact same-episode confirmations from aligned equal regions.

    The search is intentionally strict. A candidate counts only when its
    normalized token sequence appears inside an alignment region already
    proven equal between Apple and Whisper. This is a shadow evidence signal;
    it does not authorize a transcript edit.
    """

    focus_tokens = tokenize(focus)
    focus_values = _comparison_values(focus_tokens)
    if not focus_values:
        return []

    apple_values = _comparison_values(apple_tokens)
    occurrences: list[dict[str, int | str]] = []
    apple_exclude_start, apple_exclude_end = exclude_apple
    whisper_exclude_start, whisper_exclude_end = exclude_whisper

    for tag, i1, i2, j1, j2 in opcodes:
        if tag != "equal":
            continue
        run_length = i2 - i1
        if run_length < len(focus_values):
            continue
        for offset in range(run_length - len(focus_values) + 1):
            apple_start = i1 + offset
            apple_end = apple_start + len(focus_values)
            whisper_start = j1 + offset
            whisper_end = whisper_start + len(focus_values)
            if apple_values[apple_start:apple_end] != focus_values:
                continue
            overlaps_dispute = (
                apple_start < apple_exclude_end
                and apple_end > apple_exclude_start
            ) or (
                whisper_start < whisper_exclude_end
                and whisper_end > whisper_exclude_start
            )
            if overlaps_dispute:
                continue
            relation = (
                "before"
                if apple_end <= apple_exclude_start
                and whisper_end <= whisper_exclude_start
                else "after"
                if apple_start >= apple_exclude_end
                and whisper_start >= whisper_exclude_end
                else "elsewhere"
            )
            occurrences.append(
                {
                    "apple_start_word": apple_start + 1,
                    "apple_end_word": apple_end,
                    "whisper_start_word": whisper_start + 1,
                    "whisper_end_word": whisper_end,
                    "relation": relation,
                }
            )
            if len(occurrences) >= MAX_EPISODE_LOCAL_MEMORY_OCCURRENCES:
                return occurrences
    return occurrences


def _build_episode_local_memory_shadow(
    differences: Sequence[Difference],
    apple_tokens: Sequence[Token],
    whisper_tokens: Sequence[Token],
    opcodes: Sequence[tuple[str, int, int, int, int]],
) -> dict[str, object]:
    """Build read-only Level-2 same-episode corroboration telemetry.

    Only differences that currently require review and belong to the bounded
    resolver taxonomy are inspected. The resulting signal is deliberately
    non-authoritative and is never read by merge or resolver acceptance logic.
    """

    signals: list[dict[str, object]] = []
    candidate_forms = 0
    episode_confirmed_forms = 0
    prior_confirmed_forms = 0
    one_sided_signals = 0

    for item in differences:
        if not item.review_required or item.resolver_category not in RESOLVER_CATEGORIES:
            continue
        apple_focus, whisper_focus, focus_scope = _resolver_focus(item)
        forms: list[dict[str, object]] = []
        for source, focus in (("apple", apple_focus), ("whisper", whisper_focus)):
            if not focus:
                continue
            candidate_forms += 1
            occurrences = _aligned_equal_occurrences(
                focus,
                apple_tokens,
                whisper_tokens,
                opcodes,
                exclude_apple=(item.apple_start_word - 1, item.apple_end_word),
                exclude_whisper=(item.whisper_start_word - 1, item.whisper_end_word),
            )
            episode_confirmed = bool(occurrences)
            prior_confirmed = any(
                occurrence["relation"] == "before" for occurrence in occurrences
            )
            episode_confirmed_forms += int(episode_confirmed)
            prior_confirmed_forms += int(prior_confirmed)
            forms.append(
                {
                    "source": source,
                    "text": focus,
                    "normalized_words": _comparison_values(tokenize(focus)),
                    "episode_confirmed": episode_confirmed,
                    "prior_confirmed": prior_confirmed,
                    "occurrences": occurrences,
                }
            )

        confirmed_sources = [
            str(form["source"])
            for form in forms
            if form["episode_confirmed"]
        ]
        signal = (
            "one_side_confirmed"
            if len(confirmed_sources) == 1
            else "both_sides_confirmed"
            if len(confirmed_sources) > 1
            else "none"
        )
        if signal == "one_side_confirmed":
            one_sided_signals += 1
        if signal != "none":
            signals.append(
                {
                    "difference_id": item.id,
                    "category": item.resolver_category,
                    "severity": item.severity,
                    "source_only": item.source_only,
                    "focus_scope": focus_scope,
                    "signal": signal,
                    "confirmed_sources": confirmed_sources,
                    "forms": forms,
                }
            )

    return {
        "schema_version": EPISODE_LOCAL_MEMORY_SCHEMA_VERSION,
        "mode": "shadow",
        "authoritative": False,
        "decision_effect": "none",
        "evidence_contract": "aligned_equal_same_episode_only",
        "eligible_difference_count": sum(
            item.review_required and item.resolver_category in RESOLVER_CATEGORIES
            for item in differences
        ),
        "candidate_form_count": candidate_forms,
        "episode_confirmed_form_count": episode_confirmed_forms,
        "prior_confirmed_form_count": prior_confirmed_forms,
        "one_sided_signal_count": one_sided_signals,
        "signal_count": len(signals),
        "signals": signals,
    }


def _independently_corroborated(
    focus: str,
    own_context: str,
    other_context: str,
) -> bool:
    """Require a repeat outside the disputed occurrence or in the other source."""

    return (
        _contains_word_sequence(own_context, focus) >= 2
        or _contains_word_sequence(other_context, focus) >= 1
    )


def _glossary_similarity(source_text: str, candidate: str) -> float:
    """Return deterministic lexical similarity; this is not world knowledge."""

    source_words = _normalized_words(source_text)
    candidate_words = _normalized_words(candidate)
    if not source_words or not candidate_words:
        return 0.0
    return _word_similarity(source_words, candidate_words)


def _allowed_glossary_candidates(
    item: Difference,
    apple_focus: str,
    whisper_focus: str,
) -> list[str]:
    """Return a small per-item candidate list anchored in its changed tokens."""

    terms = DOMAIN_GLOSSARY.get(item.resolver_category, ())
    candidates: list[tuple[float, str]] = []
    for term in terms:
        apple_score = _glossary_similarity(apple_focus, term)
        whisper_score = _glossary_similarity(whisper_focus, term)
        if (
            term in item.domain_terms
            or (apple_focus and whisper_focus and min(apple_score, whisper_score) >= 0.82)
        ):
            candidates.append((max(apple_score, whisper_score), term))
    candidates.sort(key=lambda value: (-value[0], value[1].casefold()))
    return [term for _, term in candidates[:3]]


def _glossary_candidates_with_two_sided_support(
    candidates: Sequence[str],
    apple_focus: str,
    whisper_focus: str,
) -> list[str]:
    """Keep only candidates that both supplied readings lexically support."""

    return [
        term
        for term in candidates
        if apple_focus
        and whisper_focus
        and _glossary_similarity(apple_focus, term) >= 0.82
        and _glossary_similarity(whisper_focus, term) >= 0.82
    ]


def build_resolver_batch(
    result: ComparisonResult,
    episode_id: str,
    *,
    max_items: int = MAX_RESOLVER_ITEMS,
) -> dict:
    """Build the minimal, high-value batch for a small transcript resolver.

    This deliberately returns data only; the compiler does not make a network
    call or give a model permission to rewrite an episode. The authoritative
    system prompt belongs to the configured OpenRouter preset; this function
    returns only episode-specific data and the response contract.
    """

    if not episode_id:
        raise ValueError("episode_id must not be empty")
    if max_items < 1:
        raise ValueError("max_items must be at least one")

    candidates = [
        difference
        for difference in result.differences
        if difference.review_required
        and difference.resolver_category in RESOLVER_CATEGORIES
        and not _is_is_contraction_equivalent(
            difference.apple_text,
            difference.whisper_text,
        )
    ]

    priority = {
        "protocol_number": 0,
        "unit": 1,
        "negation": 2,
        "citation": 3,
        "proper_name": 4,
        "supplement": 5,
        "exercise_name": 6,
        "training_term": 7,
        "scientific_medical_term": 8,
    }

    candidates.sort(
        key=lambda item: (
            priority[item.resolver_category],
            item.id,
        )
    )

    items = []
    source_mappings: dict[str, dict[str, str]] = {}
    bypassed_items: list[dict[str, int | str]] = []
    deferred_items: list[dict[str, int | str]] = []

    for item in candidates:
        (
            focus_apple_text,
            focus_whisper_text,
            focus_scope,
        ) = _resolver_focus(item)

        apple_first = item.id % 2 == 1
        source_a = "apple" if apple_first else "whisper"
        source_b = "whisper" if apple_first else "apple"
        source_values = {
            "apple": {
                "text": item.apple_text,
                "context": item.apple_context,
                "focus": focus_apple_text,
                "changed_words": item.changed_apple_words,
                "start_timestamp": item.apple_start_timestamp,
                "end_timestamp": item.apple_end_timestamp,
            },
            "whisper": {
                "text": item.whisper_text,
                "context": item.whisper_context,
                "focus": focus_whisper_text,
                "changed_words": item.changed_whisper_words,
                "start_timestamp": item.whisper_start_timestamp,
                "end_timestamp": item.whisper_end_timestamp,
            },
        }
        corroborated = {
            source: _independently_corroborated(
                values["focus"],
                values["context"],
                source_values["whisper" if source == "apple" else "apple"]["context"],
            )
            for source, values in source_values.items()
        }
        source_only_corroborated = (
            corroborated[item.source_only_source]
            if item.source_only_source
            else False
        )
        allowed_candidates = _allowed_glossary_candidates(
            item,
            focus_apple_text,
            focus_whisper_text,
        )
        supported_glossary_candidates = _glossary_candidates_with_two_sided_support(
            allowed_candidates,
            focus_apple_text,
            focus_whisper_text,
        )
        # A model may never turn a one-off reading into an automatic transcript
        # edit. Glossary corrections have their own two-sided evidence gate.
        requires_corroboration = True

        resolver_item = {
                "id": item.id,
                "kind": _anonymous_resolver_kind(item.kind),
                "severity": item.severity,
                "category": item.resolver_category,
                "source_only": item.source_only,
                "source_only_side": (
                    "source_a"
                    if item.source_only_source == source_a
                    else "source_b"
                    if item.source_only_source == source_b
                    else None
                ),
                "risk_reasons": _anonymous_resolver_risk_reasons(item.risk_reasons),
                "focus_scope": focus_scope,
                "source_a_text": source_values[source_a]["text"],
                "source_b_text": source_values[source_b]["text"],
                "source_a_review_context": source_values[source_a]["context"],
                "source_b_review_context": source_values[source_b]["context"],
                "focus_source_a_text": source_values[source_a]["focus"],
                "focus_source_b_text": source_values[source_b]["focus"],
                "changed_source_a_words": source_values[source_a]["changed_words"],
                "changed_source_b_words": source_values[source_b]["changed_words"],
                "source_a_metadata": {
                    "start_timestamp": source_values[source_a]["start_timestamp"],
                    "end_timestamp": source_values[source_a]["end_timestamp"],
                },
                "source_b_metadata": {
                    "start_timestamp": source_values[source_b]["start_timestamp"],
                    "end_timestamp": source_values[source_b]["end_timestamp"],
                },
                "allowed_glossary_candidates": allowed_candidates,
                "textually_supported_glossary_candidates": supported_glossary_candidates,
                "requires_textual_corroboration": requires_corroboration,
                "textually_supported_sides": [
                    side
                    for side, source in (("source_a", source_a), ("source_b", source_b))
                    if corroborated[source]
                ],
                "source_only_textually_corroborated": source_only_corroborated,
            }
        source_choice_eligible = bool(
            resolver_item["textually_supported_sides"]
        ) and (
            not item.source_only
            or source_only_corroborated
        )
        glossary_eligible = bool(supported_glossary_candidates)
        if not (source_choice_eligible or glossary_eligible):
            bypassed_items.append(
                {
                    "id": item.id,
                    "reason": "structurally_ineligible_no_acceptance_path",
                }
            )
            continue
        if len(items) >= max_items:
            deferred_items.append(
                {
                    "id": item.id,
                    "reason": "resolver_batch_limit",
                }
            )
            continue
        source_mappings[str(item.id)] = {
            "source_a": source_a,
            "source_b": source_b,
        }
        items.append(resolver_item)
    return {
        "response_schema": resolver_response_schema(),
        # Mapping remains local: the provider receives only anonymous sides.
        "source_mappings": source_mappings,
        "bypassed_items": bypassed_items,
        "deferred_items": deferred_items,
        "batch": {
            "episode_id": episode_id,
            "diff_items": items,
        },
    }


def triage_response_schema() -> dict:
    """Return the strict advisory-only transcript triage response contract."""

    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "triage": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "id": {"type": "integer"},
                        "recommendation": {
                            "type": "string",
                            "enum": [
                                "recommend_apple",
                                "recommend_whisper",
                                "likely_custom",
                                "needs_audio",
                            ],
                        },
                        "confidence": {
                            "type": "string",
                            "enum": ["high", "medium", "low"],
                        },
                        "reason": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": TRIAGE_REASON_MAX_LENGTH,
                        },
                    },
                    "required": [
                        "id",
                        "recommendation",
                        "confidence",
                        "reason",
                    ],
                },
            },
        },
        "required": ["triage"],
    }


def build_triage_batch(
    result: ComparisonResult,
    *,
    exclude_ids: set[int] | None = None,
    max_items: int = MAX_TRIAGE_ITEMS,
) -> dict:
    """Build bounded source-grounded evidence for advisory review triage.

    This includes ordinary pending compiler differences that the automatic
    resolver never sees. The returned contract has no acceptance authority.
    """

    if max_items < 1:
        raise ValueError("max_items must be at least one")

    excluded = {
        difference_id
        for difference_id in (exclude_ids or set())
        if isinstance(difference_id, int)
    }
    candidates = sorted(
        (
            difference
            for difference in result.differences
            if difference.review_required
            and difference.id not in excluded
        ),
        key=lambda item: item.id,
    )

    items: list[dict] = []
    deferred_ids: list[int] = []

    for item in candidates:
        if len(items) >= max_items:
            deferred_ids.append(item.id)
            continue

        items.append(
            {
                "id": item.id,
                "kind": item.kind,
                "severity": item.severity,
                "category": item.resolver_category,
                "source_only": item.source_only,
                "apple_text": item.apple_text,
                "whisper_text": item.whisper_text,
                "apple_context": item.apple_context,
                "whisper_context": item.whisper_context,
                "changed_apple_words": list(item.changed_apple_words),
                "changed_whisper_words": list(item.changed_whisper_words),
                "apple_metadata": {
                    "start_timestamp": item.apple_start_timestamp,
                    "end_timestamp": item.apple_end_timestamp,
                },
                "whisper_metadata": {
                    "start_timestamp": item.whisper_start_timestamp,
                    "end_timestamp": item.whisper_end_timestamp,
                },
            }
        )

    return {
        "response_schema": triage_response_schema(),
        "deferred_ids": deferred_ids,
        "batch": {
            "diff_items": items,
        },
    }


def validate_resolver_response(batch: dict, response: dict) -> dict[str, list[dict]]:
    """Accept only conservative model resolutions and return review rejections.

    A high-confidence Apple/Whisper choice may use only that source's span.
    A ``neither`` correction is allowed only when it exactly matches a supplied
    domain-glossary term.  Applying accepted patches is intentionally left to
    the pipeline after it retains source-to-compiled offset mappings.
    """

    items = {
        item["id"]: item
        for item in batch.get("batch", {}).get("diff_items", [])
        if isinstance(item, dict) and isinstance(item.get("id"), int)
    }
    source_mappings = batch.get("source_mappings", {})
    accepted: list[dict] = []
    review: list[dict] = []
    resolutions = response.get("resolutions", []) if isinstance(response, dict) else []
    if not isinstance(resolutions, list):
        return {"accepted": accepted, "review": [{"reason": "invalid_resolutions"}]}

    seen_ids: set[int] = set()
    for resolution in resolutions:
        if not isinstance(resolution, dict) or resolution.get("id") in seen_ids:
            review.append({"resolution": resolution, "reason": "invalid_or_duplicate_id"})
            continue
        difference_id = resolution.get("id")
        seen_ids.add(difference_id)
        item = items.get(difference_id)
        choice = resolution.get("resolved_value")
        confidence = resolution.get("confidence")
        corrected_text = resolution.get("corrected_text")
        if (
            item is None
            or choice not in {"source_a", "source_b", "neither", "unclear"}
            or resolution.get("category") != item.get("category")
            or confidence != "high"
            or resolution.get("flag_for_human") is not False
        ):
            review.append({"id": difference_id, "reason": "needs_human_review"})
            continue

        if choice in {"source_a", "source_b"}:
            mapping = source_mappings.get(str(difference_id), {})
            source = mapping.get(choice)
            if source not in {"apple", "whisper"}:
                review.append({"id": difference_id, "reason": "missing_source_mapping"})
                continue
            if (
                item.get("requires_textual_corroboration")
                and choice not in item.get("textually_supported_sides", [])
            ):
                review.append(
                    {"id": difference_id, "reason": "unsupported_textual_evidence"}
                )
                continue
            if (
                item.get("source_only")
                and not item.get("source_only_textually_corroborated")
            ):
                review.append(
                    {"id": difference_id, "reason": "source_only_requires_human_review"}
                )
                continue
            focus_scope = item.get(
                "focus_scope",
                "full",
            )

            if focus_scope == "partial":
                expected = item[
                    f"focus_{choice}_text"
                ]
            else:
                expected = item[
                    f"{choice}_text"
                ]

            if corrected_text not in {
                None,
                expected,
            }:
                review.append(
                    {
                        "id": difference_id,
                        "reason": "source_text_mismatch",
                    }
                )
                continue

            accepted_item = {
                "id": difference_id,
                "source": source,
                "text": expected,
                "scope": focus_scope,
            }

            if focus_scope == "partial":
                accepted_item.update(
                    {
                        "focus_apple_text": item[
                            "focus_source_a_text"
                            if mapping.get("source_a") == "apple"
                            else "focus_source_b_text"
                        ],
                        "focus_whisper_text": item[
                            "focus_source_a_text"
                            if mapping.get("source_a") == "whisper"
                            else "focus_source_b_text"
                        ],
                    }
                )

            accepted.append(
                accepted_item
            )

            continue

        normalized_text = (
            SPACE_PATTERN.sub(" ", corrected_text).casefold()
            if isinstance(corrected_text, str)
            else ""
        )
        allowed_glossary = {
            SPACE_PATTERN.sub(" ", term).casefold()
            for term in item.get("allowed_glossary_candidates", [])
            if isinstance(term, str)
        }
        supported_glossary = {
            SPACE_PATTERN.sub(" ", term).casefold()
            for term in item.get("textually_supported_glossary_candidates", [])
            if isinstance(term, str)
        }
        if (
            choice == "neither"
            and normalized_text in allowed_glossary
            and normalized_text in supported_glossary
        ):
            accepted.append(
                {
                    "id": difference_id,
                    "source": "glossary",
                    "text": corrected_text,
                    "allowed_glossary_candidates": item.get(
                        "allowed_glossary_candidates", []
                    ),
                }
            )
        else:
            review.append({"id": difference_id, "reason": "unresolved_or_unsupported"})
    return {"accepted": accepted, "review": review}


def _apply_validated_resolver_resolutions(
    differences: Sequence[Difference],
    resolutions: Sequence[dict] | None,
    *,
    primary_source: str,
    apple_text: str,
    whisper_text: str,
    apple_tokens: Sequence[Token],
    whisper_tokens: Sequence[Token],
) -> dict[int, tuple[int, int, str]]:
    """Overlay validated resolver choices onto freshly rebuilt diff spans.

    Full-scope decisions behave as before.

    Partial decisions create a source-grounded hybrid: preserve the primary
    source's wording for the full aligned span and replace only the focused
    token with the validated Apple/Whisper token.

    Expanded human decisions may include at most a few adjacent primary-source
    words. Their fresh source slice is revalidated exactly before an override
    can reach the consensus builder.
    """

    edit_overrides: dict[int, tuple[int, int, str]] = {}
    if not resolutions:
        return edit_overrides

    items = {
        item.id: item
        for item in differences
    }

    for resolution in resolutions:
        if not isinstance(
            resolution,
            dict,
        ):
            continue

        item = items.get(
            resolution.get("id")
        )

        source = resolution.get(
            "source"
        )

        text = resolution.get(
            "text"
        )

        scope = resolution.get(
            "scope",
            "full",
        )

        if (
            item is None
            or not isinstance(
                text,
                str,
            )
        ):
            continue

        if scope == "expanded":
            if (
                source != "human"
                or resolution.get("reviewed_by") != "human"
                or resolution.get("edit_base_source") != primary_source
            ):
                continue

            left = resolution.get("expand_left_words")
            right = resolution.get("expand_right_words")
            if (
                isinstance(left, bool)
                or isinstance(right, bool)
                or not isinstance(left, int)
                or not isinstance(right, int)
                or left < 0
                or right < 0
                or left > MAX_HUMAN_EDIT_EXPANSION_WORDS
                or right > MAX_HUMAN_EDIT_EXPANSION_WORDS
                or (left == 0 and right == 0)
            ):
                continue

            base_text = apple_text if primary_source == "apple" else whisper_text
            base_tokens = apple_tokens if primary_source == "apple" else whisper_tokens
            start = (
                item.apple_start_word - 1
                if primary_source == "apple"
                else item.whisper_start_word - 1
            )
            end = (
                item.apple_end_word
                if primary_source == "apple"
                else item.whisper_end_word
            )
            if start >= end:
                continue

            expanded_start = start - left
            expanded_end = end + right
            if (
                expanded_start < 0
                or expanded_end > len(base_tokens)
                or expanded_start >= expanded_end
            ):
                continue

            expected_base_text = resolution.get("expected_base_text")
            if (
                not isinstance(expected_base_text, str)
                or _slice_text(
                    base_text,
                    base_tokens,
                    expanded_start,
                    expanded_end,
                )
                != expected_base_text
            ):
                continue

            overlaps_other_difference = False
            for other in differences:
                if other.id == item.id:
                    continue
                other_start = (
                    other.apple_start_word - 1
                    if primary_source == "apple"
                    else other.whisper_start_word - 1
                )
                other_end = (
                    other.apple_end_word
                    if primary_source == "apple"
                    else other.whisper_end_word
                )
                if other_start < other_end:
                    overlaps = (
                        other_start < expanded_end
                        and other_end > expanded_start
                    )
                else:
                    overlaps = (
                        expanded_start <= other_start <= expanded_end
                    )
                if overlaps:
                    overlaps_other_difference = True
                    break
            if overlaps_other_difference:
                continue

            item.selected_source = "human"
            item.selected_text = text
            item.merge_action = "human_bounded_expanded_edit"
            item.review_required = False
            item.selection_reason = (
                "Applied an audited human Custom/Edit over a bounded "
                "freshly revalidated primary-source range"
            )
            edit_overrides[item.id] = (
                expanded_start,
                expanded_end,
                text,
            )
            continue

        if scope == "partial":
            if source not in {"apple", "whisper", "human"}:
                continue

            if (
                source == "human"
                and resolution.get("reviewed_by") != "human"
            ):
                continue

            base_text = (
                item.apple_text
                if primary_source == "apple"
                else item.whisper_text
            )

            old_focus = resolution.get(
                f"focus_{primary_source}_text"
            )

            if not isinstance(
                old_focus,
                str,
            ):
                continue

            hybrid = _replace_focus_once(
                base_text,
                old_focus,
                text,
            )

            if hybrid is None:
                continue

            if hybrid == base_text:
                item.selected_source = (
                    primary_source
                )
                item.selected_text = None
            else:
                item.selected_source = (
                    "hybrid"
                )
                item.selected_text = hybrid

            item.merge_action = (
                "resolver_validated_focus_choice"
            )
            item.review_required = False

            item.selection_reason = (
                "Applied a validated source-grounded "
                "resolver decision only to the focused conflict"
            )

            continue

        if (
            source == "apple"
            and text == item.apple_text
        ):
            item.selected_source = "apple"

        elif (
            source == "whisper"
            and text == item.whisper_text
        ):
            item.selected_source = "whisper"

        elif (
            source == "glossary"
            and SPACE_PATTERN.sub(" ", text).casefold()
            in {
                SPACE_PATTERN.sub(" ", candidate).casefold()
                for candidate in resolution.get("allowed_glossary_candidates", [])
                if isinstance(candidate, str)
            }
        ):
            item.selected_source = "glossary"
            item.selected_text = text

        elif (
            source == "human"
            and resolution.get("reviewed_by") == "human"
        ):
            item.selected_source = "human"
            item.selected_text = text

        else:
            continue

        item.merge_action = (
            "resolver_validated_choice"
        )
        item.review_required = False

        item.selection_reason = (
            "Applied a validated, source-grounded resolver decision"
        )

    return edit_overrides


def _escape_table(value: str, limit: int = 180) -> str:
    compact = SPACE_PATTERN.sub(" ", value).replace("|", "\\|").strip()
    return compact if len(compact) <= limit else compact[: limit - 1] + "…"


def _format_timestamp(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    minutes, remainder = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    prefix = f"{hours}:" if hours else ""
    return f"{prefix}{minutes:02d}:{remainder:02d}"


def _format_timestamp_range(start: float | None, end: float | None) -> str:
    if start is None:
        return "—"
    formatted_start = _format_timestamp(start)
    if end is None or int(end) == int(start):
        return formatted_start
    return f"{formatted_start}–{_format_timestamp(end)}"


def render_markdown_report(
    result: ComparisonResult,
    *,
    apple_path: str | Path,
    whisper_path: str | Path,
    max_issues: int = 100,
) -> str:
    """Render a compact human-readable report; JSON retains every difference."""

    review_items = [item for item in result.differences if item.review_required][:max_issues]
    lines = [
        "# Transcript compiler report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Status: **{result.status.upper()}**",
        f"Recommended transcript: **{result.recommended_source}**",
        "",
        "## Summary",
        "",
        "| Metric | Apple | Whisper |",
        "| --- | ---: | ---: |",
        f"| Words | {result.apple_words:,} | {result.whisper_words:,} |",
        f"| Coverage by shared words | {result.apple_coverage:.2%} | {result.whisper_coverage:.2%} |",
        f"| Source quality | {result.apple_quality:.2f}/100 | {result.whisper_quality:.2f}/100 |",
        "",
        f"Global aligned similarity: **{result.similarity:.2%}** "
        f"({result.matched_words:,} exact normalized words)",
        "",
        f"- High risk: **{result.high_risk}**",
        f"- Medium risk: **{result.medium_risk}**",
        f"- Low risk: **{result.low_risk}**",
        f"- Consensus edits applied: **{result.compiler_edits}**",
        f"- Differences resolved by deterministic rules: **{result.auto_resolved}**",
        "",
        "Punctuation, capitalization, whitespace, and equivalent single-word numbers "
        "are ignored during alignment.",
        "",
        "## Items requiring review",
        "",
    ]
    if not review_items:
        lines.append("No high- or medium-risk differences were found.")
    else:
        lines.extend([
            "| # | Risk | Type | Selected | Apple time | Whisper time | Apple | Whisper |",
            "| ---: | --- | --- | --- | --- | --- | --- | --- |",
        ])
        for item in review_items:
            lines.append(
                f"| {item.id} | {item.severity} | {item.kind} | "
                f"{item.selected_source} | "
                f"{_format_timestamp_range(item.apple_start_timestamp, item.apple_end_timestamp)} | "
                f"{_format_timestamp_range(item.whisper_start_timestamp, item.whisper_end_timestamp)} | "
                f"{_escape_table(item.apple_text or '—')} | "
                f"{_escape_table(item.whisper_text or '—')} |"
            )

    omitted = result.review_required - len(review_items)
    if omitted > 0:
        lines.extend(["", f"{omitted} additional review item(s) are in the JSON report."])
    lines.extend([
        "",
        "## Output decision",
        "",
        f"The consensus transcript uses `{result.recommended_source}` for structure "
        f"and applies {result.compiler_edits} safe edit(s) from the other source. "
        "Potential number and negation changes are not silently rewritten; every "
        "selection and reason is recorded in the JSON report.",
        "",
        f"- Apple input: `{apple_path}`",
        f"- Whisper input: `{whisper_path}`",
        "",
    ])
    return "\n".join(lines)


def write_outputs(
    result: ComparisonResult,
    *,
    output_dir: str | Path,
    apple_path: str | Path,
    whisper_path: str | Path,
    max_issues: int = 100,
) -> dict[str, Path]:
    """Write Markdown, JSON, and recommended transcript outputs."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = {
        "markdown": destination / "compiler-report.md",
        "json": destination / "compiler-report.json",
        "transcript": destination / "compiled-transcript.txt",
    }
    paths["markdown"].write_text(
        render_markdown_report(
            result,
            apple_path=apple_path,
            whisper_path=whisper_path,
            max_issues=max_issues,
        ),
        encoding="utf-8",
    )
    payload = result.to_dict()
    payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    payload["inputs"] = {"apple": str(apple_path), "whisper": str(whisper_path)}
    paths["json"].write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    paths["transcript"].write_text(
        result.compiled_transcript.rstrip() + "\n", encoding="utf-8"
    )
    return paths


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Globally align and compare Apple and Whisper transcripts."
    )
    parser.add_argument("apple", nargs="?", default=str(DEFAULT_APPLE))
    parser.add_argument("whisper", nargs="?", default=str(DEFAULT_WHISPER))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--primary", choices=("auto", "apple", "whisper"), default="auto"
    )
    parser.add_argument("--merge-gap", type=int, default=3)
    parser.add_argument(
        "--apple-metadata",
        help="Optional Apple transcript JSON with segment timestamps",
    )
    parser.add_argument(
        "--whisper-metadata",
        help="Optional Whisper JSON with segments/words confidence values",
    )
    parser.add_argument("--max-report-issues", type=int, default=100)
    parser.add_argument(
        "--apple-keep-timestamp-lines",
        action="store_true",
        help="Keep timestamp-only lines in the Apple input",
    )
    parser.add_argument(
        "--fail-on", choices=("none", "high", "medium"), default="none"
    )
    return parser


def _should_fail(result: ComparisonResult, fail_on: str) -> bool:
    if fail_on == "high":
        return result.high_risk > 0
    if fail_on == "medium":
        return result.high_risk > 0 or result.medium_risk > 0
    return False


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        result = compile_transcripts(
            load_text(args.apple),
            load_text(args.whisper),
            primary=args.primary,
            merge_gap=args.merge_gap,
            apple_has_timestamps=not args.apple_keep_timestamp_lines,
            apple_metadata=args.apple_metadata,
            whisper_metadata=args.whisper_metadata,
        )
        paths = write_outputs(
            result,
            output_dir=args.output_dir,
            apple_path=args.apple,
            whisper_path=args.whisper,
            max_issues=args.max_report_issues,
        )
    except (OSError, ValueError) as error:
        parser.exit(1, f"error: {error}\n")

    print(
        f"Status: {result.status.upper()} | similarity: {result.similarity:.2%} | "
        f"high: {result.high_risk} | medium: {result.medium_risk} | low: {result.low_risk}"
    )
    print(f"Recommended source: {result.recommended_source}")
    print(
        f"Consensus edits: {result.compiler_edits} | "
        f"auto-resolved differences: {result.auto_resolved}"
    )
    for label, path in paths.items():
        print(f"{label.capitalize()}: {path}")
    return 2 if _should_fail(result, args.fail_on) else 0


if __name__ == "__main__":
    sys.exit(main())
