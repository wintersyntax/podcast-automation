"""Deterministic Markdown body for single-pass knowledge notes (TASK-118).

Pure Python; the same checked note always renders to the same bytes. The
body carries no frontmatter and no H1 -- ``frontmatter.render`` adds both
concerns around it exactly as for earlier summary bodies.

Where a claim comes from (user decision 2026-10-15, replacing the
2026-09-29 ▰▱ meter) is an icon plus a word at the start of every bullet
that claims or recommends something::

    🔬 **Study** — …    🏋️ **Coaches** — …
    👤 **One person** — …    💬 **Opinion** — …

``· unsure`` is appended inside the label when the speaker hedged.
Descriptive bullets (basis ``none``) carry no label, and no legend is needed.
"""

from __future__ import annotations

import re

from .units import SAID_RE, UnitReport, to_si

# Bump whenever the rendered body for the same checked note changes (SI
# conversion, formatting). Stored notes are then re-rendered from the
# manifest's checked note without a model call.
# v2 (2026-09-30): fractions and mixed numbers ("5/8 inch") convert as one
# quantity; v1 rendered "5/8 inch" as "5/~20 cm (said: 8 inch)".
# v3 (2026-09-30): clock/split times ("2:55") bold as one value, "about ~"
# keeps a single approximation marker, "calorie(s)" is treated as kcal.
# v4 (2026-10-14): converted values of 100 or more keep whole-number
# precision ("230 pounds" -> "~104 kg"; v3 rounded it to "~100 kg").
# v5 (2026-10-15): evidence labels are an icon plus a word ("🔬 **Study**")
# instead of the ▰▱ meter, and the final note opens with the episode title
# as H1 (added by ``frontmatter.render``).
# v6 (2026-10-15): the protocol block's Evidence line uses the same icon and
# word; v5 concatenated the new (icon, word) tuple with a string and failed
# on hedged protocols (and printed the raw tuple for unhedged ones).
RENDER_VERSION = "note-render-v6"
LEGACY_RENDER_VERSION = "note-render-v1"

EVIDENCE_LABELS = {
    "research": ("🔬", "Study"),
    "coaching_experience": ("🏋️", "Coaches"),
    "personal_experience": ("👤", "One person"),
    "opinion": ("💬", "Opinion"),
}
NOT_STATED = {"", "not stated", "not specified", "none", "n/a"}

_NUMBER_RE = re.compile(
    r"(?<![\w*~/:-])(~?\d+(?:[.,]\d+)*(?::\d{2})*(?:\s?[–-]\s?\d+(?:[.,]\d+)*(?::\d{2})*)?"
    r"(?:\s?(?:%|g/kg|kg|km|cm|mg|ml|kcal|°C|L|g|m|reps?|sets?|RIR|RPE|hours?|minutes?|min|days?|weeks?|months?|years?|steps))?)"
    r"(?![\w*:])"
)


def evidence_label(basis: str, hedged: bool) -> str:
    label = EVIDENCE_LABELS.get(basis)
    if label is None:
        return ""
    icon, word = label
    return f"{icon} **{word}{' · unsure' if hedged else ''}** — "


def _bold_numbers(text: str) -> str:
    def bold(match: re.Match) -> str:
        value = match.group(1)
        trailing = ""
        while value and value[-1] in ",.":
            trailing = value[-1] + trailing
            value = value[:-1]
        return f"**{value}**{trailing}" if value else match.group(0)

    return _NUMBER_RE.sub(bold, text)


def format_text(text: str, report: UnitReport) -> str:
    """SI-convert, then bold numbers outside the ``(said: …)`` originals."""

    converted = to_si(text.strip(), report)
    return "".join(
        part if SAID_RE.fullmatch(part) else _bold_numbers(part)
        for part in SAID_RE.split(converted)
    )


def _stated(value: str) -> bool:
    return value.strip().lower() not in NOT_STATED


def _study_line(study: dict, report: UnitReport) -> str:
    details = [
        format_text(study[key], report)
        for key in ("design", "sample", "duration")
        if _stated(study[key])
    ]
    line = f"- {evidence_label(study['basis'], study['hedged'])}**{study['authors_year'].strip()}**"
    line += " — " + (", ".join(details) + " → " if details else "") + format_text(study["result"], report)
    if _stated(study["host_comment"]):
        line += f" *Hosts:* {format_text(study['host_comment'], report)}"
    return line


def _protocol_block(protocol: dict, report: UnitReport) -> list[str]:
    def field(value: str) -> str:
        return format_text(value, report) if _stated(value) else "not specified"

    evidence = ""
    label = EVIDENCE_LABELS.get(protocol["basis"])
    if label is not None:
        icon, word = label
        evidence = f"{icon} {word}{' · unsure' if protocol['hedged'] else ''}"
    return [
        f"> **📋 Protocol: {protocol['name'].strip()}**",
        f"> **What:** {field(protocol['what'])}",
        f"> **Dose / parameters:** {field(protocol['dose_parameters'])}",
        f"> **When / for whom:** {field(protocol['when_for_whom'])}",
        f"> **Evidence:** {evidence or 'not specified'}",
        f"> **Caveat:** {format_text(protocol['caveat'], report) if _stated(protocol['caveat']) else 'none stated'}",
    ]


def render_body(note: dict, report: UnitReport | None = None) -> str:
    """Render a checked ``knowledge-note-v3`` note as a Markdown body."""

    report = report if report is not None else UnitReport()
    lines = ["## TL;DR", format_text(" ".join(note["tldr"]), report), ""]

    for section in note["sections"]:
        lines.append(f"## {section['title'].strip()}")
        if section["bottom_line"].strip():
            lines.append(f"*{format_text(section['bottom_line'], report)}*")
        lines.append("")
        lines.extend(
            f"- {evidence_label(b['basis'], b['hedged'])}{format_text(b['text'], report)}"
            for b in section["bullets"]
        )
        for protocol in section["protocols"]:
            lines.append("")
            lines.extend(_protocol_block(protocol, report))
        lines.append("")

    if note["research_discussed"]:
        lines.append("## Research discussed")
        lines.extend(_study_line(study, report) for study in note["research_discussed"])
        lines.append("")
    if note["also_discussed"]:
        lines.append("## Also discussed")
        lines.extend(f"- {format_text(item['text'], report)}" for item in note["also_discussed"])
        lines.append("")
    if note["numbers"]:
        lines += ["## Numbers & protocols", "", "| Topic | Value / protocol | Evidence |", "|---|---|---|"]
        for row in note["numbers"]:
            icon, word = EVIDENCE_LABELS.get(row["basis"], ("", ""))
            label = f"{icon} {word}".strip()
            topic = row["topic"].replace("|", "/").strip()
            value = to_si(row["value"], report).replace("|", "/").strip()
            lines.append(f"| {topic} | {value} | {label} |")
        lines.append("")
    if note["takeaways"]:
        lines.append("## Takeaways")
        lines.extend(f"- {format_text(text, report)}" for text in note["takeaways"])
        lines.append("")
    if note["sources_mentioned"]:
        lines.append("## Sources mentioned")
        lines.extend(
            f"- {source['reference_as_heard'].strip()} — as heard"
            for source in note["sources_mentioned"]
        )
        lines.append("")
    if note["follow_up"]:
        lines.append("## Follow up")
        lines.extend(f"- {format_text(text, report)}" for text in note["follow_up"])
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"
