# Knowledge-note gap check — prompt v1

TASK-106 Phase A, `gap-check-v1` schema version. Design authority:
`docs/superpowers/specs/2026-09-24-grounded-knowledge-note-pipeline-design.md`
§5.7 (optional gap check, missing-item mode) and §11.6 (gap-check decision).

You are checking one already-extracted section of a fitness/nutrition podcast
transcript for **missing-item gaps**: important in-scope content that the
extraction pass did not capture in any item. You see this section's text, the
episode's title and description for orientation, and the list of items
already accepted for this section (their `statement` and `quote`) — never the
rest of the transcript, and never any other section.

This is the missing-item mode only. You are not asked to check whether an
existing item's `conditions` field is complete or accurate — do not report
those as gaps.

## What counts as a gap

Report a gap only when the section's own text states something that:

- is in scope (training, bodybuilding, hypertrophy, nutrition, supplements,
  recovery, injury, competition; or sleep/work/stress/scheduling/travel/
  finances as they concern living and training around the sport; or identity/
  discipline/motivation/long-term relationship with lifting — the same scope
  rules extraction itself uses), and
- is not covered, even partially, by any of the already-accepted items listed
  above (a different phrasing of the same fact is NOT a gap; a genuinely
  distinct fact, number, name, or explicit answer that no accepted item
  mentions at all IS a gap).

Do not report a gap for out-of-scope content (ads, sponsor reads, unrelated
chat, relationships/dating/family advice, content-creation/podcast-production
talk) even if no item covers it — that content is correctly excluded, not
missing.

If nothing in this section qualifies as a gap, return an empty
`"missing_items"` list. Do not invent a gap to fill space.

## Quote requirement

Every gap's `quote` must be **verbatim, contiguous** text copied from this
section — exact enough that a plain string search finds it in the section
text. `quote_occurrence` is 1-indexed: if the exact `quote` string appears
more than once in the section, say which occurrence this gap refers to. A gap
whose quote does not locate exactly in the section text is discarded before
it ever reaches extraction, so an approximate or paraphrased quote wastes the
check entirely — copy the words exactly as spoken, do not summarize them.

## Output

Return **only** JSON matching this shape — no prose before or after:

```json
{
  "section_id": "<the section_id given to you>",
  "missing_items": [
    {
      "quote": "exact contiguous transcript text this gap covers",
      "quote_occurrence": 1,
      "reason": "one sentence: what specific fact/number/name is missing and why it matters"
    }
  ]
}
```
