"""Deterministic coverage detectors for single-pass notes (TASK-119).

Design: ``docs/KNOWLEDGE_GENERATION.md``
section 7. No model calls, no network, no GCS access. Three detectors run on the
canonical compiled transcript against the checked note:

1. **Exercise-advice spots.** A ~25-word chunk counts when it names an exercise
   and an execution cue appears within one chunk either side, or (weaker tier)
   names a muscle with a cue in the same chunk. Adjacent spots merge. A spot is
   covered only when a note unit is anchored within ``NEAR_CHARS`` of it *and*
   that unit names one of the spot's exercises (or muscles).
2. **Quantity inventory.** Digits with a unit (or a year) and number words with
   a unit whose value appears nowhere in the note.
3. **Coverage map.** ~``WINDOW_CHARS`` stretches of the normalized transcript with
   no anchored note unit.

The fill trigger is computed before merging: the fill runs when at least one
uncovered exercise-advice spot or unit-bearing number exists anywhere, including
inside an uncovered window. Uncovered windows alone never trigger it.

Terms come from the versioned lexicon ``config/knowledge/exercise-lexicon-v1.json``;
TASK-120 operator additions are passed in as ``extra_terms``.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .checks import TranscriptIndex, normalize

LEXICON_PATH = Path(__file__).resolve().parents[3] / "config" / "knowledge" / "exercise-lexicon-v1.json"
LEXICON_SCHEMA_VERSION = "exercise-lexicon-v1"
COVERAGE_POLICY_VERSION = "note-coverage-v1"

CHUNK_WORDS = 25
NEAR_CHARS = 400
WINDOW_CHARS = 2000
QUANTITY_RADIUS = 150
MERGE_GAP = 50
MAX_CANDIDATES = 40
TERM_KINDS = ("exercises", "muscles", "cues", "units")

NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30,
    "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100,
    "thousand": 1000,
}
_UNIT_FOLLOW_RE = re.compile(
    r"(and a half )?(\w+ )?(percent|kg|kilos?|grams?|pounds?|reps?|sets?|minutes?|hours?|days?|weeks?|calories|steps)\b"
)
_TYPE_PRIORITY = {"exercise_advice": 0, "quantity": 1, "uncovered_window": 2}


class LexiconError(ValueError):
    """The lexicon file is missing, malformed or of another schema version."""


@dataclass(frozen=True)
class Lexicon:
    exercises: tuple[str, ...]
    muscles: tuple[str, ...]
    cues: tuple[str, ...]
    units: tuple[str, ...]
    sha256: str

    def pattern(self, kind: str) -> re.Pattern:
        terms = sorted({re.escape(term) for term in getattr(self, kind)}, key=len, reverse=True)
        return re.compile(r"\b(?:" + "|".join(terms) + r")\b") if terms else re.compile(r"(?!x)x")


def load_lexicon(path: Path = LEXICON_PATH, extra_terms: dict | None = None) -> Lexicon:
    """Load the repository lexicon, optionally merged with operator terms."""

    try:
        raw = path.read_bytes()
        data = json.loads(raw)
    except (OSError, ValueError) as exc:
        raise LexiconError(f"cannot read exercise lexicon {path}: {exc}") from exc
    if not isinstance(data, dict) or data.get("schema_version") != LEXICON_SCHEMA_VERSION:
        raise LexiconError("exercise lexicon has an unexpected schema version")
    merged: dict[str, list[str]] = {}
    for kind in TERM_KINDS:
        values = data.get(kind)
        if not isinstance(values, list) or not all(isinstance(v, str) and v.strip() for v in values):
            raise LexiconError(f"exercise lexicon field {kind!r} is malformed")
        extra = (extra_terms or {}).get(kind) or []
        if not isinstance(extra, list) or not all(isinstance(v, str) for v in extra):
            raise LexiconError(f"operator lexicon field {kind!r} is malformed")
        merged[kind] = sorted({normalize(v) for v in [*values, *extra] if normalize(v)})
    digest = hashlib.sha256(json.dumps(merged, sort_keys=True).encode("utf-8")).hexdigest()
    return Lexicon(*(tuple(merged[kind]) for kind in TERM_KINDS), sha256="sha256:" + digest)


@dataclass
class Candidate:
    id: str
    type: str
    start: int
    end: int
    detail: str
    excerpt: str = ""
    raw_start: int = 0
    raw_end: int = 0
    terms: list[str] = field(default_factory=list)

    @property
    def excerpt_sha256(self) -> str:
        return "sha256:" + hashlib.sha256(normalize(self.excerpt).encode("utf-8")).hexdigest()

    def as_record(self) -> dict:
        return {
            "id": self.id, "type": self.type, "detail": self.detail, "excerpt": self.excerpt,
            "excerpt_sha256": self.excerpt_sha256, "raw_start": self.raw_start, "raw_end": self.raw_end,
            "terms": list(self.terms),
        }


@dataclass
class CoverageReport:
    candidates: list[Candidate]
    total: int
    triggered: bool
    raw_counts: dict[str, int]
    lexicon_sha256: str
    policy_version: str = COVERAGE_POLICY_VERSION

    def as_record(self) -> dict:
        return {
            "policy_version": self.policy_version,
            "lexicon_sha256": self.lexicon_sha256,
            "triggered": self.triggered,
            "raw_counts": dict(self.raw_counts),
            "total": self.total,
            "candidates": [c.as_record() for c in self.candidates],
        }


def anchored_units(note: dict) -> list[tuple[str, str]]:
    """(text, anchor) for every anchored unit of a ``knowledge-note-v3`` note."""

    units: list[tuple[str, str]] = []
    for section in note.get("sections", []):
        units += [(b["text"], b["anchor"]) for b in section.get("bullets", [])]
        for p in section.get("protocols", []):
            text = " ".join(str(p.get(k, "")) for k in ("name", "what", "dose_parameters", "when_for_whom", "caveat"))
            units.append((text, p["anchor"]))
    for study in note.get("research_discussed", []):
        text = " ".join(str(study.get(k, "")) for k in ("authors_year", "design", "sample", "duration", "result", "host_comment"))
        units.append((text, study["anchor"]))
    units += [(u["text"], u["anchor"]) for u in note.get("also_discussed", [])]
    units += [(u["reference_as_heard"], u["anchor"]) for u in note.get("sources_mentioned", [])]
    return units


def _chunks(text: str) -> list[tuple[int, int]]:
    words, spans, position, chunk, start = text.split(), [], 0, [], 0
    for word in words:
        if not chunk:
            start = position
        chunk.append(word)
        position += len(word) + 1
        if len(chunk) >= CHUNK_WORDS:
            spans.append((start, position - 1))
            chunk = []
    if chunk:
        spans.append((start, position - 1))
    return spans


def exercise_spots(text: str, lexicon: Lexicon) -> list[dict]:
    exercise_re, cue_re, muscle_re = lexicon.pattern("exercises"), lexicon.pattern("cues"), lexicon.pattern("muscles")
    chunks = _chunks(text)
    flags = []
    for i, (a, b) in enumerate(chunks):
        piece = text[a:b]
        exercises = set(exercise_re.findall(piece))
        context = text[chunks[max(0, i - 1)][0]: chunks[min(len(chunks) - 1, i + 1)][1]]
        cues = set(cue_re.findall(context))
        if not exercises:
            muscles, cues_here = set(muscle_re.findall(piece)), set(cue_re.findall(piece))
            if muscles and cues_here:
                exercises, cues = muscles, cues_here
        flags.append((a, b, exercises, cues) if exercises and cues else None)
    spots: list[dict] = []
    for flag in flags:
        if flag is None:
            continue
        if spots and flag[0] - spots[-1]["end"] <= 2:
            spots[-1]["end"] = flag[1]
            spots[-1]["exercises"] |= flag[2]
            spots[-1]["cues"] |= flag[3]
        else:
            spots.append({"start": flag[0], "end": flag[1], "exercises": set(flag[2]), "cues": set(flag[3])})
    return spots


def quantity_mentions(text: str, lexicon: Lexicon, raw: str = "", raw_index: list[int] | None = None) -> list[dict]:
    """Unit-bearing numbers in normalized ``text``.

    With the raw transcript and its index, a decimal or thousands separator
    that normalization turned into a space ("0.35" -> "0 35", "1,500" ->
    "1 500") is read back, so the value is 0.35 or 1500 rather than 0 and 35.
    """

    unit_re = lexicon.pattern("units")
    tokens, offsets, position = text.split(), [], 0
    for token in tokens:
        offsets.append(position)
        position += len(token) + 1
    found = []
    skip_next = False
    for i, token in enumerate(tokens):
        if skip_next:
            skip_next = False
            continue
        if token.isdigit():
            value = float(token)
            following = tokens[i + 1] if i + 1 < len(tokens) else ""
            if raw and raw_index and following.isdigit():
                gap = raw[raw_index[offsets[i] + len(token) - 1] + 1: raw_index[offsets[i + 1]]]
                if gap == ".":
                    value = float(f"{token}.{following}")
                    skip_next = True
                elif gap == "," and len(following) == 3:
                    value = float(token + following)
                    skip_next = True
            if skip_next:
                tokens_after = tokens[i + 2:i + 5]
            else:
                tokens_after = tokens[i + 1:i + 4]
            has_unit = bool(unit_re.match(" ".join(tokens_after[:2]))) or bool(
                _UNIT_FOLLOW_RE.match(" ".join(tokens_after))
            )
            if not has_unit and not 1900 <= value <= 2100:
                continue
            found.append({"pos": offsets[i], "value": value, "token": token if not skip_next else f"{value:g}"})
            continue
        elif token in NUMBER_WORDS:
            value = float(NUMBER_WORDS[token])
            following = tokens[i + 1] if i + 1 < len(tokens) else ""
            if following in NUMBER_WORDS and NUMBER_WORDS[following] < 10 and value >= 20:
                value += NUMBER_WORDS[following]
        else:
            continue
        has_unit = bool(unit_re.match(" ".join(tokens[i + 1:i + 3]))) or bool(
            _UNIT_FOLLOW_RE.match(" ".join(tokens[i + 1:i + 4]))
        )
        if token in NUMBER_WORDS and not has_unit:
            continue  # bare number words ("one of the") are too noisy
        if token.isdigit() and not has_unit and not 1900 <= value <= 2100:
            continue
        found.append({"pos": offsets[i], "value": value, "token": token})
    return found


def note_numbers(text: str) -> set[float]:
    values: set[float] = set()
    for match in re.finditer(r"\d[\d,]*(?:\.\d+)?", text):
        try:
            values.add(float(match.group().replace(",", "")))
        except ValueError:
            pass
    for word in normalize(text).split():
        if word in NUMBER_WORDS:
            values.add(float(NUMBER_WORDS[word]))
    return values


def find_candidates(
    note: dict,
    transcript: str,
    lexicon: Lexicon | None = None,
    *,
    max_candidates: int = MAX_CANDIDATES,
) -> CoverageReport:
    lexicon = lexicon or load_lexicon()
    index = TranscriptIndex(transcript)
    text, raw_index = index.text, index.raw_index
    units = anchored_units(note)
    placed = []
    for unit_text, anchor in units:
        offset, _ = index.locate(anchor)
        if offset is not None:
            placed.append((offset, normalize(unit_text)))
    note_text = " ".join(t for t, _ in units) + " " + " ".join(note.get("tldr", [])) + " " + " ".join(note.get("takeaways", []))
    note_text += " " + " ".join(f"{row.get('topic', '')} {row.get('value', '')}" for row in note.get("numbers", []))
    values_in_note = note_numbers(note_text)

    raw: list[Candidate] = []
    for spot in exercise_spots(text, lexicon):
        near = [t for p, t in placed if spot["start"] - NEAR_CHARS <= p <= spot["end"] + NEAR_CHARS]
        if any(re.search(r"\b" + re.escape(term) + r"\b", t) for t in near for term in spot["exercises"]):
            continue
        raw.append(Candidate("", "exercise_advice", spot["start"], spot["end"],
                             "exercises: " + ", ".join(sorted(spot["exercises"])) + "; cues: " + ", ".join(sorted(spot["cues"])),
                             terms=sorted(spot["exercises"])))
    for quantity in quantity_mentions(text, lexicon, transcript, raw_index):
        if quantity["value"] in values_in_note:
            continue
        raw.append(Candidate("", "quantity", max(0, quantity["pos"] - QUANTITY_RADIUS),
                             min(len(text), quantity["pos"] + QUANTITY_RADIUS),
                             f"number {quantity['token']} not found anywhere in the note", terms=[quantity["token"]]))
    raw_counts = {kind: sum(c.type == kind for c in raw) for kind in ("exercise_advice", "quantity")}
    triggered = bool(raw)  # computed before merging into uncovered windows
    for start in range(0, len(text), WINDOW_CHARS):
        if not any(start <= p < start + WINDOW_CHARS for p, _ in placed):
            raw.append(Candidate("", "uncovered_window", start, min(len(text), start + WINDOW_CHARS),
                                 "no note point anchored in this passage"))
    raw_counts["uncovered_window"] = sum(c.type == "uncovered_window" for c in raw)

    raw.sort(key=lambda c: (_TYPE_PRIORITY[c.type], c.start))
    merged: list[Candidate] = []
    for candidate in raw:
        last = merged[-1] if merged else None
        if last and last.type == candidate.type and candidate.start <= last.end + MERGE_GAP:
            last.end = max(last.end, candidate.end)
            last.detail += " | " + candidate.detail
            last.terms = sorted(set(last.terms) | set(candidate.terms))
        else:
            merged.append(candidate)
    windows = [c for c in merged if c.type == "uncovered_window"]
    merged = [c for c in merged if c.type == "uncovered_window"
              or not any(w.start <= c.start < w.end for w in windows)]
    merged.sort(key=lambda c: (_TYPE_PRIORITY[c.type], c.start))
    for number, candidate in enumerate(merged, start=1):
        candidate.id = f"c{number:02d}"
        if raw_index:
            candidate.raw_start = raw_index[min(candidate.start, len(raw_index) - 1)]
            candidate.raw_end = raw_index[min(candidate.end, len(raw_index) - 1)] + 1
            candidate.excerpt = transcript[candidate.raw_start:candidate.raw_end].strip()
    return CoverageReport(merged[:max_candidates], len(merged), triggered, raw_counts, lexicon.sha256)
