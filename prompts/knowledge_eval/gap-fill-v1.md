# Knowledge-item gap fill — prompt v1

TASK-106 Phase A, `knowledge-items-v1` schema version, additive gap-fill
mode. Design authority:
`docs/superpowers/specs/2026-09-24-grounded-knowledge-note-pipeline-design.md`
§5.7 (optional gap check) and §5.1 (extraction, item field rules) — this
prompt reuses §5.1's item-authoring rules unchanged; only the task framing
differs from the main extraction prompt (`extract-v2.md`).

Introduced 2026-09-26 as a strategy revision: the original gap-check fill
mechanism re-extracted the whole section and replaced its item set wholesale,
which measured a net coverage regression across the dev set (see
`docs/knowledge-eval/task-12-step5-gap-check-openai-gpt-6-luna.md`) because
each re-extraction is an independent call that can fail to reproduce a
previously-good item. This prompt instead asks for only the new items a
specific set of gaps calls for; the caller appends them to the section's
existing accepted items and never discards anything.

You are given one already-extracted section of a fitness/nutrition podcast
transcript, a list of specific **gaps** a separate gap-check pass found in
it, and the list of items **already accepted** for this section. Your only
job is to decide, for each gap, whether it is genuinely uncovered content —
and if so, extract exactly one new item for it. You are not re-extracting
the section: do not report, repeat, or re-list anything else from the
section, even if you notice other content that seems missing beyond the
listed gaps. This call adds to the accepted items; it never replaces or
re-evaluates them.

## Before extracting anything: check for existing coverage

For every gap in `"gap_hints"`, first check the `"already_accepted_items"`
list (each entry: `statement`, `quote`). If a gap's content is already
substantively expressed by an accepted item — even in different words, a
paraphrase, or from a slightly different angle — skip that gap. Do not
extract a second item for something already covered; a near-duplicate item
is worse than a missing one, because it makes the final summary longer
without adding information.

Only extract a new item when the gap describes a genuinely distinct fact,
number, name, or explicit answer that no accepted item mentions at all.

If every gap turns out to already be covered, return an empty `"items"`
list — that is the correct, expected result, not a failure.

## Item fields (identical rules to normal extraction)

Every new item follows exactly the same field rules as ordinary extraction:

- `local_id`: `"i-01"`, `"i-02"`, ... unique within this response's own
  `items` list (the caller renumbers these to avoid colliding with the
  section's already-accepted items, so uniqueness only needs to hold within
  this response).
- `kind`: exactly one of `claim`, `recommendation`, `protocol`, `mechanism`,
  `rationale`, `caveat`, `study_description`, `source_mention`, `side_topic`,
  `follow_up`.
- `quote`: **verbatim, contiguous** text copied from this section — must be
  the gap's own quote or an exact match/subset of it; never assembled from
  two separate places, never paraphrased.
- `quote_occurrence`: 1-indexed, matching whichever occurrence of that exact
  quote string appears in the section.
- `quantities`: one entry per quantity in the quote — `{"value": <number>,
  "value_high": <number or null>, "unit_as_spoken": "<unit exactly as
  spoken>"}` — never converted or normalized. Every value/value_high must
  appear in the item's own `quote`.
- `negated`: `true` only when the quote itself contains a negation word.
- `evidence_basis`: `research`, `coaching_experience`, `personal_experience`,
  or `opinion` for any item that asserts or recommends something.
- `hedged`: `true` when the speaker signals uncertainty about this specific
  statement.
- `conditions`: **required** (real qualifying text, or the literal string
  `"none_stated"`) for `kind` in `claim`, `recommendation`, `protocol` —
  never `null` or omitted for these three kinds. `null` for every other
  kind.
- `scope`: `core`, `life_support`, `sport_philosophy`, or `out_of_scope`
  (same scope policy as ordinary extraction — ads, sponsor reads, and
  off-topic chat are `out_of_scope`, not skipped).
- `value`: `high`, `normal`, or `low`, judged the same way as ordinary
  extraction.
- `protocol`: only for `kind: "protocol"` — `{"what", "dose",
  "when_for_whom", "caveat"}`, each field the stated text or `null`. `null`
  (the whole field) for every other kind.
- `source_mention`: only for `kind: "source_mention"` or
  `"study_description"` — `{"authors_as_heard", "year_as_heard",
  "title_as_heard", "type"}`, `null` fields for whatever wasn't stated.
  `null` (the whole field) for every other kind.
- `rationale_for`: only for `kind: "rationale"` — the `local_id` of the
  recommendation/claim this rationale explains, if it is one of the NEW
  items in this same response. `null` otherwise (it cannot point at an
  already-accepted item's id, since this response does not see those ids).
- No speaker attribution: never guess or record who said what.

## Output

Return **only** JSON matching this shape — no prose before or after:

```json
{
  "section_id": "<the section_id given to you>",
  "nothing_relevant": false,
  "nothing_relevant_reason": null,
  "items": [ /* zero or more NEW items, one per genuinely uncovered gap */ ]
}
```

`nothing_relevant` is always `false` here — this call is about specific
gaps in an already-processed section, not a full section relevance
judgment; use an empty `"items"` list instead when nothing new should be
added.
