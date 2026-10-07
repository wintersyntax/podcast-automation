"""Deterministic Markdown rendering for TASK-106 Phase A Task 9.

Design authority:
docs/superpowers/specs/2026-09-24-grounded-knowledge-note-pipeline-design.md
§4 (note format v2), §4.2-§4.5 (evidence markers, protocol block, numbers,
SI conversion), §7 (rendering), §8.1-§8.2 (failure-policy rendering).

Pure Python, no model calls, no network, no GCS access. Rendering the same
input twice gives identical bytes. `render_note` assumes its input already
passed Task 7's `validate_composition` and (when a review/fix pipeline ran)
Task 8's `decide` -- a malformed `{q:...}` placeholder or an out-of-range
quantity index is a programming error upstream, not a data condition this
module recovers from; it raises `ValueError` rather than guessing.

Design decisions made here, where the spec's §4.1 format example and its
surrounding prose left a gap or a literal inconsistency (the same posture
Tasks 7-8 took: pick the most defensible reading, document it, move on):

1. **Frontmatter is passed through untouched.** `render_note` takes the
   existing note's frontmatter block (including its `---` delimiters,
   trailing whitespace ignored) as an opaque string and reproduces it
   byte-for-byte before the regenerated body -- it never parses or rebuilds
   it. This is the simplest reading of "existing frontmatter preserved" and
   matches how `podcast_engine/knowledge/tags.py`'s `patch_local_markdown`
   already treats frontmatter elsewhere in this codebase: touch only what
   you must, leave the rest as bytes.
2. **Bold-wrapping a rendered quantity.** Design spec §4.5 shows
   ``**~<SI value>** (said: <original>)`` -- bold covers only the converted
   value, not the "(said: ...)" clause. `quantities.convert_quantity(...)
   .rendered` already produces that whole string (with or without a
   "(said: ...)" suffix depending on whether the unit was actually SI
   -converted). `_bold_quantity` splits on the literal `" (said: "` marker
   and bolds only what precedes it, so a passthrough value (no conversion,
   no "said" clause) is bolded whole.
3. **A protocol block's `Caveat` field falls back to "none stated", not
   "not specified".** Spec §4.3's prose says every unstated protocol field
   renders as `not specified`, but §4.1's own format example shows
   `**Caveat:** … | none stated` -- a different fallback word specifically
   for Caveat. Since `none_stated` is the sentinel this whole design already
   uses everywhere else for "the source genuinely stated no qualifier" (the
   `conditions` field, §5.1-§5.2), and a caveat is conceptually that same
   kind of qualifier, this module follows §4.1's literal example: `Caveat`
   falls back to `"none stated"`; `What`, `Dose / parameters`, and
   `When / for whom` fall back to `"not specified"` per §4.3.
4. **Protocol fields render verbatim, never SI-converted.** §4.3: "Every
   field is copied by Python from item fields... The composer cannot fill
   it." A protocol item's `protocol.dose`/`when_for_whom`/etc. are
   extraction-stage free text (as spoken), never touched by composition's
   `{q:...}` placeholder mechanism (that only exists in composed bullet/
   TL;DR/takeaway text) and never SI-converted -- SI conversion applies to
   an item's structured `quantities` list, not to `protocol`'s prose
   fields.
5. **A protocol item's blockquote name is its `statement`.** The item
   schema has no dedicated "protocol name" field; `statement` is the one
   free-text field every item carries, so ``📋 Protocol: <statement>`` uses
   it.
6. **A bullet's evidence marker uses its `primary_item_id` when set,
   otherwise its first cited item.** Spec §4.2: "When a bullet cites items
   with different bases, Python renders the basis of the item marked
   primary by the composer." `primary_item_id` is nullable in the
   `note-composition-v1` schema (Task 7) even when a bullet cites more than
   one item, so a deterministic fallback is needed for that case: the first
   `item_ids` entry, in composition order. The ❔-hedged suffix is added
   whenever *any* cited item is hedged, per §4.2's "if the cited items
   disagree in hedged, the bullet is rendered hedged."
7. **`research_discussed` gets a resolved-reference prefix.** §4.1's format
   line ``**<Resolved or as-heard study reference>** — design, sample,
   finding as described...`` bolds a *resolved source reference* at the
   start -- something the model cannot know (source resolution is Task 6's
   deterministic Python), so this module, not composition, prepends it:
   the resolved title of the first `source_mention`/`study_description`
   item the unit cites (via `resolutions_by_item_id`, falling back to the
   item's `source_mention.title_as_heard` when no resolution was supplied),
   followed by " — " and the unit's own composed text.
8. **"Numbers & protocols" is one row per quantity-bearing item and one row
   per protocol item, scoped to items cited within topic sections** (not
   `research_discussed`/`also_discussed`/`takeaways`/`follow_up`, which are
   prose, not itemized facts) -- deduplicated by item id within a section.
   `Topic` is the owning section's title; `Basis` is the same marker a
   bullet would carry. A protocol row's `Value / protocol` cell condenses
   `what`/`dose`/`when_for_whom` (never `caveat`, already in its own
   blockquote) into one line; a quantities row joins every quantity on the
   item with "; ".
9. **"Sources mentioned" is omitted when it would be empty.** Spec §4.1
   lists exactly four sections as "omitted when empty" (Research discussed,
   Follow up, Also discussed, Numbers & protocols) and doesn't name Sources
   mentioned -- but an empty heading with no list under it serves no reader,
   and the same omit-when-empty principle plainly extends to it. TL;DR,
   topic sections, and Takeaways are never considered for omission here:
   they are structurally guaranteed non-empty upstream (composition always
   writes 3-5 TL;DR sentences and at least one topic section; an empty
   Takeaways list is a composition/prompt concern, not a rendering one).
10. **A removed bullet becomes a collapsible `<details>` block** under its
    section, holding its own cited quote -- §8.2 says "with its cited quote
    collapsed below," and MinkNote (Obsidian) renders raw HTML in Markdown,
    so `<details>`/`<summary>` gives genuine click-to-expand behavior. The
    quote shown is the removed bullet's primary (or first) cited item's
    `quote`.
"""

from __future__ import annotations

import re

from .quantities import convert_quantity

# ---------------------------------------------------------------------------
# Constants (design spec §4.2, §4.3, §4.1's legend line)
# ---------------------------------------------------------------------------

NOT_SPECIFIED = "not specified"
NONE_STATED_CAVEAT = "none stated"

_EVIDENCE_MARKERS = {
    "research": "🔬",
    "coaching_experience": "👥",
    "personal_experience": "👤",
    "opinion": "💭",
}

_SOURCE_LABELS = {
    "show_notes": "✔︎ show notes",
    "auto_matched": "✔︎ auto-matched",
    "as_heard": "as heard",
    "corrected": "✎ corrected",
}

LEGEND_LINE = (
    "Basis: 🔬 research cited · 👥 coaching/clinical experience · "
    "👤 personal experience · 💭 opinion/speculation · ❔ speaker hedged. "
    "🔬 means the speaker cites research, not that the research was verified."
)

_SOURCE_KINDS = frozenset({"source_mention", "study_description"})

_PLACEHOLDER_RE = re.compile(r"\{q:([A-Za-z0-9_\-]+):(\d+)\}")


# ---------------------------------------------------------------------------
# Quantity rendering
# ---------------------------------------------------------------------------

def _bold_quantity(rendered: str) -> str:
    """Bold only the value portion of a `convert_quantity(...).rendered`
    string, per design spec §4.5's `**~<SI value>** (said: <original>)`.
    """
    marker = " (said: "
    if marker in rendered:
        head, tail = rendered.split(marker, 1)
        return f"**{head}**{marker}{tail}"
    return f"**{rendered}**"


def _render_quantity(item: dict, index: int) -> str:
    quantities = item.get("quantities") or []
    if index < 0 or index >= len(quantities):
        raise ValueError(
            f"quantity index {index} out of range for item {item.get('item_id')!r} "
            f"(has {len(quantities)} quantities)"
        )
    q = quantities[index]
    conversion = convert_quantity(q.get("value"), q.get("value_high"), q.get("unit_as_spoken", ""))
    return _bold_quantity(conversion.rendered)


def render_placeholders(text: str, items_by_id: dict[str, dict]) -> str:
    """Replace every `{q:<item_id>:<n>}` placeholder in `text` with its
    bold-wrapped SI rendering. Raises `ValueError` on an unknown item id --
    Task 7's `validate_composition` guarantees every placeholder in
    already-validated composition text resolves, so this is a defensive
    check against malformed input reaching this module directly, not a
    condition callers are expected to handle.
    """

    def _replace(match: re.Match) -> str:
        item_id, index = match.group(1), int(match.group(2))
        item = items_by_id.get(item_id)
        if item is None:
            raise ValueError(f"placeholder cites unknown item id {item_id!r}")
        return _render_quantity(item, index)

    return _PLACEHOLDER_RE.sub(_replace, text)


# ---------------------------------------------------------------------------
# Evidence markers (design spec §4.2)
# ---------------------------------------------------------------------------

def _item_marker(item: dict) -> str:
    marker = _EVIDENCE_MARKERS.get(item.get("evidence_basis"), "")
    if item.get("hedged"):
        marker += "❔"
    return marker


def _bullet_marker(bullet: dict, items_by_id: dict[str, dict]) -> str:
    """The design spec §4.2 marker for one bullet: the primary-cited item's
    basis (or the first cited item's, when no primary is set), plus ❔ when
    *any* cited item is hedged. Markers never appear on a non-asserting
    (`asserts: false`) bullet.
    """
    if not bullet.get("asserts"):
        return ""
    item_ids = bullet.get("item_ids") or []
    if not item_ids:
        return ""
    primary_id = bullet.get("primary_item_id") or item_ids[0]
    primary_item = items_by_id.get(primary_id, {})
    marker = _EVIDENCE_MARKERS.get(primary_item.get("evidence_basis"), "")
    any_hedged = any(items_by_id.get(iid, {}).get("hedged") for iid in item_ids)
    if any_hedged:
        marker += "❔"
    return marker


def _unit_marker(item_ids: list[str], items_by_id: dict[str, dict]) -> str:
    """The marker for a plain `{"text","item_ids"}` unit (research_discussed
    entries) -- no `primary_item_id` concept, so the first cited item's
    basis is used, with ❔ when any cited item is hedged.
    """
    if not item_ids:
        return ""
    first_item = items_by_id.get(item_ids[0], {})
    marker = _EVIDENCE_MARKERS.get(first_item.get("evidence_basis"), "")
    if any(items_by_id.get(iid, {}).get("hedged") for iid in item_ids):
        marker += "❔"
    return marker


# ---------------------------------------------------------------------------
# Source resolution labels (design spec §4.1 "Sources mentioned")
# ---------------------------------------------------------------------------

def _resolve_reference(item_id: str, items_by_id: dict[str, dict], resolutions_by_item_id: dict) -> tuple[str, str]:
    """Returns (reference_title, status) for one source item, falling back
    to the item's `source_mention.title_as_heard` (status "as_heard") when
    no `SourceResolution` was supplied for it.
    """
    item = items_by_id.get(item_id, {})
    resolution = resolutions_by_item_id.get(item_id)
    if resolution is not None:
        title = resolution.title or (item.get("source_mention") or {}).get("title_as_heard") or "untitled"
        return title, resolution.status
    title = (item.get("source_mention") or {}).get("title_as_heard") or item.get("statement") or "untitled"
    return title, "as_heard"


# ---------------------------------------------------------------------------
# Protocol block (design spec §4.3)
# ---------------------------------------------------------------------------

def _render_protocol_block(item_id: str, items_by_id: dict[str, dict]) -> str:
    item = items_by_id.get(item_id, {})
    protocol = item.get("protocol") or {}
    what = protocol.get("what") or NOT_SPECIFIED
    dose = protocol.get("dose") or NOT_SPECIFIED
    when_for_whom = protocol.get("when_for_whom") or NOT_SPECIFIED
    caveat = protocol.get("caveat") or NONE_STATED_CAVEAT
    basis = _item_marker(item)
    name = item.get("statement") or "protocol"
    lines = [
        f"> **📋 Protocol: {name}**",
        f"> **What:** {what}",
        f"> **Dose / parameters:** {dose}",
        f"> **When / for whom:** {when_for_whom}",
        f"> **Basis:** {basis}",
        f"> **Caveat:** {caveat}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Removed-bullet notes (design spec §8.2)
# ---------------------------------------------------------------------------

def _render_removed_note(bullet: dict, items_by_id: dict[str, dict]) -> str:
    item_ids = bullet.get("item_ids") or []
    primary_id = bullet.get("primary_item_id") or (item_ids[0] if item_ids else None)
    quote = items_by_id.get(primary_id, {}).get("quote", "") if primary_id else ""
    return (
        "<details>\n"
        "<summary>Removed 1 unverified point</summary>\n\n"
        f"> {quote}\n\n"
        "</details>"
    )


# ---------------------------------------------------------------------------
# Bullets and units
# ---------------------------------------------------------------------------

def _render_bullet(unit_id: str, bullet: dict, items_by_id: dict[str, dict], removed_unit_ids: frozenset[str]) -> str | None:
    """Returns the rendered bullet line, or `None` when this bullet was
    removed by review (its caller then renders the removed-note block
    instead)."""
    if unit_id in removed_unit_ids:
        return None
    text = render_placeholders(bullet["text"], items_by_id)
    marker = _bullet_marker(bullet, items_by_id)
    suffix = f" {marker}" if marker else ""
    return f"- {text}{suffix}"


def _render_unit_bullet(text: str, item_ids: list[str], items_by_id: dict[str, dict]) -> str:
    rendered = render_placeholders(text, items_by_id)
    marker = _unit_marker(item_ids, items_by_id)
    suffix = f" {marker}" if marker else ""
    return f"- {rendered}{suffix}"


def _render_research_discussed_bullet(unit: dict, items_by_id: dict[str, dict], resolutions_by_item_id: dict) -> str:
    text = render_placeholders(unit["text"], items_by_id)
    item_ids = unit.get("item_ids") or []
    source_item_id = next(
        (iid for iid in item_ids if items_by_id.get(iid, {}).get("kind") in _SOURCE_KINDS),
        None,
    )
    marker = _unit_marker(item_ids, items_by_id)
    suffix = f" {marker}" if marker else ""
    if source_item_id is not None:
        title, _status = _resolve_reference(source_item_id, items_by_id, resolutions_by_item_id)
        return f"- **{title}** — {text}{suffix}"
    return f"- {text}{suffix}"


# ---------------------------------------------------------------------------
# Numbers & protocols table (design spec §4.4)
# ---------------------------------------------------------------------------

def _protocol_table_cell(item: dict) -> str:
    protocol = item.get("protocol") or {}
    what = protocol.get("what") or NOT_SPECIFIED
    dose = protocol.get("dose") or NOT_SPECIFIED
    when_for_whom = protocol.get("when_for_whom") or NOT_SPECIFIED
    return f"{what} — {dose}, {when_for_whom}"


def _build_numbers_table(composition: dict, items_by_id: dict[str, dict]) -> list[str]:
    rows: list[str] = []
    for section in composition.get("sections", []):
        title = section.get("title", "")
        seen: set[str] = set()
        cited_ids = [iid for bullet in section.get("bullets", []) for iid in bullet.get("item_ids", [])]
        for item_id in cited_ids + list(section.get("protocol_item_ids", [])):
            if item_id in seen:
                continue
            seen.add(item_id)
            item = items_by_id.get(item_id)
            if item is None:
                continue
            basis = _item_marker(item)
            if item.get("kind") == "protocol":
                rows.append(f"| {title} | {_protocol_table_cell(item)} | {basis} |")
            elif item.get("quantities"):
                values = "; ".join(
                    _render_quantity(item, i) for i in range(len(item["quantities"]))
                )
                rows.append(f"| {title} | {values} | {basis} |")
    return rows


# ---------------------------------------------------------------------------
# Sources mentioned (design spec §4.1, §5.3's resolution labels)
# ---------------------------------------------------------------------------

def _collect_cited_item_ids(composition: dict) -> list[str]:
    ids: list[str] = []
    for unit in composition.get("tldr", []):
        ids.extend(unit.get("item_ids", []))
    for section in composition.get("sections", []):
        for bullet in section.get("bullets", []):
            ids.extend(bullet.get("item_ids", []))
        ids.extend(section.get("protocol_item_ids", []))
    for key in ("research_discussed", "also_discussed", "takeaways", "follow_up"):
        for unit in composition.get(key, []):
            ids.extend(unit.get("item_ids", []))
    return ids


def _build_sources_section(composition: dict, items_by_id: dict[str, dict], resolutions_by_item_id: dict) -> list[str]:
    seen: set[str] = set()
    lines: list[str] = []
    for item_id in _collect_cited_item_ids(composition):
        if item_id in seen:
            continue
        item = items_by_id.get(item_id)
        if item is None or item.get("kind") not in _SOURCE_KINDS:
            continue
        seen.add(item_id)
        title, status = _resolve_reference(item_id, items_by_id, resolutions_by_item_id)
        label = _SOURCE_LABELS.get(status, _SOURCE_LABELS["as_heard"])
        lines.append(f"- {title} — {label}")
    return lines


# ---------------------------------------------------------------------------
# Top-level assembly
# ---------------------------------------------------------------------------

def _render_section(section: dict, items_by_id: dict[str, dict], removed_unit_ids: frozenset[str], section_index: int) -> str:
    lines = [f"## {section['title']}"]
    bullet_lines: list[str] = []
    for bullet_index, bullet in enumerate(section.get("bullets", [])):
        unit_id = f"sections[{section_index}].bullets[{bullet_index}]"
        rendered = _render_bullet(unit_id, bullet, items_by_id, removed_unit_ids)
        if rendered is None:
            bullet_lines.append(_render_removed_note(bullet, items_by_id))
        else:
            bullet_lines.append(rendered)
    if bullet_lines:
        lines.append("\n".join(bullet_lines))
    for item_id in section.get("protocol_item_ids", []):
        lines.append(_render_protocol_block(item_id, items_by_id))
    return "\n\n".join(lines)


def render_note(
    existing_frontmatter: str,
    composition: dict,
    items_by_id: dict[str, dict],
    *,
    resolutions_by_item_id: dict | None = None,
    removed_unit_ids: list[str] | None = None,
    footer_line: str | None = None,
) -> str:
    """Render one `note-composition-v1` payload (plus the items it cites)
    to the design spec §4 Markdown format. Pure and re-runnable: the same
    input always produces identical bytes.

    `existing_frontmatter` is reproduced verbatim (see module docstring,
    decision 1). `resolutions_by_item_id` maps a `source_mention`/
    `study_description` item id to a `sources.SourceResolution`; an item
    with no entry falls back to its `source_mention.title_as_heard` as
    "as heard". `removed_unit_ids` and `footer_line` are the two §8.1/§8.2
    outputs of `review.decide` -- passed as plain values rather than that
    dataclass to keep this module's only import `quantities.convert_quantity`.
    """
    resolutions_by_item_id = resolutions_by_item_id or {}
    removed = frozenset(removed_unit_ids or [])

    blocks = [existing_frontmatter.rstrip()]

    tldr_text = " ".join(render_placeholders(u["text"], items_by_id) for u in composition.get("tldr", []))
    blocks.append(f"## TL;DR\n{tldr_text}")

    for index, section in enumerate(composition.get("sections", [])):
        blocks.append(_render_section(section, items_by_id, removed, index))

    research = composition.get("research_discussed", [])
    if research:
        lines = [_render_research_discussed_bullet(u, items_by_id, resolutions_by_item_id) for u in research]
        blocks.append("## Research discussed\n" + "\n".join(lines))

    also = composition.get("also_discussed", [])
    if also:
        lines = [_render_unit_bullet(u["text"], u.get("item_ids", []), items_by_id) for u in also]
        blocks.append("## Also discussed\n" + "\n".join(lines))

    table_rows = _build_numbers_table(composition, items_by_id)
    if table_rows:
        header = "| Topic | Value / protocol | Basis |\n|---|---|---|"
        blocks.append("## Numbers & protocols\n" + header + "\n" + "\n".join(table_rows))

    takeaways = composition.get("takeaways", [])
    takeaway_lines = [_render_unit_bullet(u["text"], u.get("item_ids", []), items_by_id) for u in takeaways]
    blocks.append("## Takeaways\n" + ("\n".join(takeaway_lines) if takeaway_lines else ""))

    sources_lines = _build_sources_section(composition, items_by_id, resolutions_by_item_id)
    if sources_lines:
        blocks.append("## Sources mentioned\n" + "\n".join(sources_lines))

    follow_up = composition.get("follow_up", [])
    if follow_up:
        lines = [_render_unit_bullet(u["text"], u.get("item_ids", []), items_by_id) for u in follow_up]
        blocks.append("## Follow up\n" + "\n".join(lines))

    blocks.append(f"---\n{LEGEND_LINE}")

    if footer_line:
        blocks.append(footer_line)

    return "\n\n".join(blocks).rstrip() + "\n"
