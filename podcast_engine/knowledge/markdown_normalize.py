"""Conservative, semantics-free layout normalization for summary Markdown."""

from __future__ import annotations

import re


_FENCE_RE = re.compile(r"^[ \t]*(`{3,}|~{3,})(.*)$")
_HEADING_RE = re.compile(r"^[ \t]*(#{1,6})[ \t]+(.+?)[ \t]*$")
_LIST_ITEM_RE = re.compile(r"^[ \t]*[-*+][ \t]+")
_INLINE_HEADING_GAP_RE = re.compile(r"(?<=\S)[ \t]{2,}(?=#{1,6}[ \t]+)")
_INLINE_BULLET_AFTER_PUNCT_RE = re.compile(r"(?<=[.!?:;)])[ \t]{2,}(?=-[ \t]+)")
_INLINE_BOLD_BULLET_RE = re.compile(r"(?<=\S)[ \t]{2,}(?=-[ \t]+\*\*)")


def _fence_marker(line: str) -> tuple[str, int, str] | None:
    match = _FENCE_RE.fullmatch(line)
    if match is None:
        return None
    marker = match.group(1)
    return marker[0], len(marker), match.group(2)


def _split_accidental_inline_markdown(line: str) -> list[str]:
    leading = line[: len(line) - len(line.lstrip(" \t"))]
    starts_as_list = _LIST_ITEM_RE.match(line) is not None

    repaired = _INLINE_HEADING_GAP_RE.sub("\n", line)
    repaired = _INLINE_BULLET_AFTER_PUNCT_RE.sub("\n", repaired)
    repaired = _INLINE_BOLD_BULLET_RE.sub("\n", repaired)
    parts = repaired.split("\n")

    if starts_as_list and leading:
        for index in range(1, len(parts)):
            if parts[index].startswith(("- ", "* ", "+ ")):
                parts[index] = leading + parts[index]
    return parts


def normalize_markdown_layout(markdown: str) -> str:
    """Normalize Markdown layout without interpreting or rewriting its meaning.

    The normalizer only repairs whitespace-level presentation defects that are
    mechanically recognizable: accidental indentation before headings or
    top-level bullets, inline headings/bullets separated by large whitespace,
    trailing whitespace, and missing/excess blank lines around headings and
    list blocks. Fenced code content is treated as opaque.
    """

    text = markdown.replace("\r\n", "\n").replace("\r", "\n")
    source_lines = text.split("\n")

    logical_lines: list[tuple[str, bool]] = []
    fence_char: str | None = None
    fence_len = 0

    for raw_line in source_lines:
        marker = _fence_marker(raw_line)
        if fence_char is not None:
            logical_lines.append((raw_line, True))
            if marker is not None:
                char, length, suffix = marker
                if char == fence_char and length >= fence_len and not suffix.strip():
                    fence_char = None
                    fence_len = 0
            continue

        if marker is not None:
            char, length, _ = marker
            logical_lines.append((raw_line.rstrip(), True))
            fence_char = char
            fence_len = length
            continue

        line = raw_line.rstrip()
        for part in _split_accidental_inline_markdown(line):
            heading = _HEADING_RE.fullmatch(part)
            if heading is not None:
                part = f"{heading.group(1)} {heading.group(2).strip()}"
            logical_lines.append((part, False))

    output: list[str] = []
    pending_blank = False
    previous_kind: str | None = None
    list_base_indent: int | None = None
    in_opaque_fence = False
    opaque_fence_char: str | None = None
    opaque_fence_len = 0

    for raw_line, opaque in logical_lines:
        if opaque:
            marker = _fence_marker(raw_line)
            if not in_opaque_fence:
                if output and output[-1] != "" and pending_blank:
                    output.append("")
                pending_blank = False
                output.append(raw_line)
                if marker is not None:
                    opaque_fence_char, opaque_fence_len, _ = marker
                    in_opaque_fence = True
                previous_kind = "other"
                list_base_indent = None
                continue

            output.append(raw_line)
            if marker is not None:
                char, length, suffix = marker
                if (
                    char == opaque_fence_char
                    and length >= opaque_fence_len
                    and not suffix.strip()
                ):
                    in_opaque_fence = False
                    opaque_fence_char = None
                    opaque_fence_len = 0
                    previous_kind = "other"
                    list_base_indent = None
            continue

        if raw_line.strip() == "":
            pending_blank = True
            continue

        line = raw_line
        heading = _HEADING_RE.fullmatch(line)
        is_list_item = _LIST_ITEM_RE.match(line) is not None

        if heading is not None:
            line = f"{heading.group(1)} {heading.group(2).strip()}"
            kind = "heading"
            list_base_indent = None
        elif is_list_item:
            kind = "list"
            leading = len(line) - len(line.lstrip(" \t"))
            if previous_kind != "list" or list_base_indent is None:
                list_base_indent = leading
            elif leading < list_base_indent:
                list_base_indent = leading

            relative_indent = max(0, leading - list_base_indent)
            line = " " * relative_indent + line.lstrip(" \t")
        else:
            kind = "other"
            list_base_indent = None

        need_blank = False
        if output:
            if pending_blank:
                need_blank = True
            if kind == "heading" or previous_kind == "heading":
                need_blank = True
            if kind == "list" and previous_kind not in {None, "list", "heading"}:
                need_blank = True
            if previous_kind == "list" and kind not in {"list", "heading"}:
                need_blank = True

        if need_blank and output[-1] != "":
            output.append("")

        output.append(line)
        pending_blank = False
        previous_kind = kind

    while output and output[-1] == "":
        output.pop()

    return "\n".join(output) + "\n" if output else ""
