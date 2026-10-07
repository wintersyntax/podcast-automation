"""Deterministic structural validation for summary Markdown."""

from __future__ import annotations

import re


_EMPTY_HEADING_RE = re.compile(r"^#{1,6}$")
_LEVEL_TWO_HEADING_RE = re.compile(r"^##[ \t]+(.+?)(?:[ \t]+#+)?[ \t]*$")
_YAML_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*[ \t]*:")
_FENCE_RE = re.compile(r"^(`{3,}|~{3,})(.*)$")


def _fence_marker(line: str) -> tuple[str, int, str] | None:
    match = _FENCE_RE.fullmatch(line.strip())
    if match is None:
        return None
    marker = match.group(1)
    return marker[0], len(marker), match.group(2)


def _has_leading_yaml_frontmatter(lines: list[str]) -> bool:
    if not lines or lines[0].strip() != "---":
        return False

    saw_yaml_key = False
    for line in lines[1:]:
        stripped = line.strip()
        if stripped == "---":
            return saw_yaml_key
        if stripped.startswith("#"):
            return False
        if _YAML_KEY_RE.match(stripped):
            saw_yaml_key = True
    return False


def _is_whole_output_fenced(lines: list[str]) -> bool:
    if len(lines) < 2:
        return False
    opening = _fence_marker(lines[0])
    closing = _fence_marker(lines[-1])
    if opening is None or closing is None:
        return False
    open_char, open_len, _ = opening
    close_char, close_len, close_suffix = closing
    return open_char == close_char and close_len >= open_len and not close_suffix.strip()


def normalize_and_validate_summary_body(content: object) -> str:
    """Return accepted Markdown using the pre-guard byte normalization contract.

    This guard intentionally owns only machine-provable formatting invariants.
    It does not score semantic fidelity, coverage, names, numbers, or importance.
    """

    if not isinstance(content, str):
        raise ValueError("Podcast summary response must contain Markdown body text")

    normalized = content.strip()
    if not normalized:
        raise ValueError("Podcast summary response must contain Markdown body text")

    lines = normalized.splitlines()
    if _has_leading_yaml_frontmatter(lines):
        raise ValueError("Podcast summary response must not contain YAML frontmatter")
    if _is_whole_output_fenced(lines):
        raise ValueError("Podcast summary response must not be wrapped in a Markdown code fence")

    seen_level_two_sections: set[str] = set()
    fence_char: str | None = None
    fence_len = 0

    for line in lines:
        marker = _fence_marker(line)
        if fence_char is not None:
            if marker is not None:
                char, length, suffix = marker
                if char == fence_char and length >= fence_len and not suffix.strip():
                    fence_char = None
                    fence_len = 0
            continue

        if marker is not None:
            fence_char, fence_len, _ = marker
            continue

        stripped = line.strip()
        if _EMPTY_HEADING_RE.fullmatch(stripped):
            raise ValueError("Podcast summary response contains an empty Markdown heading")

        match = _LEVEL_TWO_HEADING_RE.fullmatch(stripped)
        if match is None:
            continue
        title = " ".join(match.group(1).split()).casefold()
        if title in seen_level_two_sections:
            raise ValueError(
                f"Podcast summary response contains duplicate level-2 section: {match.group(1)!r}"
            )
        seen_level_two_sections.add(title)

    if fence_char is not None:
        raise ValueError("Podcast summary response contains an unclosed Markdown code fence")

    return normalized + "\n"
