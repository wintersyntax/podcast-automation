"""Deterministic, non-destructive checks for single-pass notes (TASK-118).

Every anchored unit (bullet, protocol, study, also-discussed line, source)
must quote the transcript: its verbatim ``anchor`` is located in the
normalized compiled transcript, exactly or by a bounded fuzzy match that
tolerates small ASR-copy slips. A unit that fails a check is removed from
the note and reported -- the note itself is never held for a single bad
unit. Only a note that loses more than ``MAX_DROPPED_FRACTION`` of its
anchored units, or keeps no section at all, is rejected as a whole.
"""

from __future__ import annotations

import copy
import difflib
import re
import unicodedata
from dataclasses import dataclass, field

FUZZY_MIN_RATIO = 0.85
MAX_DROPPED_FRACTION = 0.2
STUB_MAX_WORDS = 5

# Sponsor reads and plugs must never reach a note (design §2); the writer is
# told to omit them, and this is the deterministic backstop.
AD_RE = re.compile(
    r"\b(?:discount code|promo code|coupon code|use (?:the )?code|sponsored by|our sponsor|"
    r"today'?s sponsor|brought to you by|affiliate link|patreon)\b",
    re.IGNORECASE,
)


class NoteRejectedError(RuntimeError):
    """The note lost too much content to the checks to be published."""


@dataclass
class DroppedUnit:
    path: str
    reason: str
    text: str


@dataclass
class CheckReport:
    anchored_units: int = 0
    anchored_dropped: int = 0
    fuzzy_anchors: int = 0
    dropped: list[DroppedUnit] = field(default_factory=list)


def normalize_with_index(raw: str) -> tuple[str, list[int]]:
    """Lower-case ASCII letters/digits with single spaces, plus the raw
    index of every normalized character (for excerpts and audio lookup)."""

    raw = unicodedata.normalize("NFKC", raw)
    out: list[str] = []
    index: list[int] = []
    last_space = True
    for position, char in enumerate(raw):
        lowered = char.lower()
        if ("a" <= lowered <= "z") or ("0" <= lowered <= "9"):
            out.append(lowered)
            index.append(position)
            last_space = False
        elif not last_space:
            out.append(" ")
            index.append(position)
            last_space = True
    if out and out[-1] == " ":
        out.pop()
        index.pop()
    return "".join(out), index


def normalize(text: str) -> str:
    return normalize_with_index(text)[0]


class TranscriptIndex:
    """Normalized transcript with exact and bounded fuzzy anchor lookup."""

    def __init__(self, transcript: str):
        self.text, self.raw_index = normalize_with_index(transcript)
        self._words = self.text.split()
        self._offsets: list[int] = []
        position = 0
        for word in self._words:
            self._offsets.append(position)
            position += len(word) + 1

    def locate(self, anchor: str) -> tuple[int | None, bool]:
        """Return (normalized offset, fuzzy?) for an anchor, or (None, False)."""

        needle = normalize(anchor)
        if not needle:
            return None, False
        found = self.text.find(needle)
        if found >= 0:
            return found, False
        words = needle.split()
        width = len(words)
        if width < 4:
            return None, False
        heads = set(words[:3])
        best_ratio, best_offset = 0.0, None
        for start in range(0, max(1, len(self._words) - width + 1)):
            if self._words[start] not in heads:
                continue
            ratio = difflib.SequenceMatcher(None, words, self._words[start:start + width]).ratio()
            if ratio > best_ratio:
                best_ratio, best_offset = ratio, self._offsets[start]
        if best_ratio >= FUZZY_MIN_RATIO:
            return best_offset, True
        return None, False


def _is_stub(text: str) -> bool:
    words = text.split()
    return len(words) <= STUB_MAX_WORDS and not re.search(r"\d|\{\{", text)


def apply_checks(note: dict, transcript: str) -> tuple[dict, CheckReport]:
    """Return a checked copy of ``note`` and a report of what was removed.

    Removes: units whose anchor is not in the transcript, any text that
    carries ad/sponsor wording, and label-only stub bullets (five words or
    fewer, no number) that the writer sometimes emits after a full bullet.
    Raises ``NoteRejectedError`` when more than ``MAX_DROPPED_FRACTION`` of
    the anchored units are removed or no section keeps a bullet.
    """

    checked = copy.deepcopy(note)
    report = CheckReport()
    index = TranscriptIndex(transcript)

    def keep_anchored(unit: dict, path: str, text: str) -> bool:
        report.anchored_units += 1
        if AD_RE.search(text):
            report.anchored_dropped += 1
            report.dropped.append(DroppedUnit(path, "ad_or_sponsor", text))
            return False
        offset, fuzzy = index.locate(unit["anchor"])
        if offset is None:
            report.anchored_dropped += 1
            report.dropped.append(DroppedUnit(path, "anchor_not_in_transcript", text))
            return False
        report.fuzzy_anchors += int(fuzzy)
        return True

    for s_index, section in enumerate(checked["sections"]):
        kept = []
        for b_index, bullet in enumerate(section["bullets"]):
            path = f"sections[{s_index}].bullets[{b_index}]"
            if _is_stub(bullet["text"]):
                report.dropped.append(DroppedUnit(path, "label_only_bullet", bullet["text"]))
                continue
            if keep_anchored(bullet, path, bullet["text"]):
                kept.append(bullet)
        section["bullets"] = kept
        section["protocols"] = [
            protocol
            for p_index, protocol in enumerate(section["protocols"])
            if keep_anchored(
                protocol,
                f"sections[{s_index}].protocols[{p_index}]",
                " ".join(protocol[k] for k in ("name", "what", "dose_parameters", "when_for_whom", "caveat")),
            )
        ]
    checked["sections"] = [s for s in checked["sections"] if s["bullets"] or s["protocols"]]

    for key, text_of in (
        ("research_discussed", lambda u: " ".join(u[k] for k in ("authors_year", "design", "sample", "duration", "result", "host_comment"))),
        ("also_discussed", lambda u: u["text"]),
        ("sources_mentioned", lambda u: u["reference_as_heard"]),
    ):
        checked[key] = [
            unit for u_index, unit in enumerate(checked[key])
            if keep_anchored(unit, f"{key}[{u_index}]", text_of(unit))
        ]

    for key in ("tldr", "takeaways", "follow_up"):
        kept_texts = []
        for t_index, text in enumerate(checked[key]):
            if AD_RE.search(text):
                report.dropped.append(DroppedUnit(f"{key}[{t_index}]", "ad_or_sponsor", text))
            else:
                kept_texts.append(text)
        checked[key] = kept_texts
    kept_rows = []
    for n_index, row in enumerate(checked["numbers"]):
        if AD_RE.search(f"{row['topic']} {row['value']}"):
            report.dropped.append(DroppedUnit(f"numbers[{n_index}]", "ad_or_sponsor", row["value"]))
        else:
            kept_rows.append(row)
    checked["numbers"] = kept_rows

    if not checked["sections"]:
        raise NoteRejectedError("no section kept a bullet after checks")
    if report.anchored_units and report.anchored_dropped / report.anchored_units > MAX_DROPPED_FRACTION:
        raise NoteRejectedError(
            f"{report.anchored_dropped} of {report.anchored_units} anchored units failed checks"
        )
    return checked, report
