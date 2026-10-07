# Knowledge-item extraction — prompt v3

TASK-106 Phase A, `knowledge-items-v1` schema version. Design authority:
`docs/superpowers/specs/2026-09-24-grounded-knowledge-note-pipeline-design.md`
§2 (locked decisions) and §5.1 (extraction).

You are extracting atomic, quote-anchored knowledge items from one section of a
fitness/nutrition podcast transcript. You see **only this section's text** plus
the episode's title and description for orientation — never the rest of the
transcript, and never any other section. Every claim you make must be provably
grounded in this section's own text.

## Completeness — extract every distinct detail, not just the headline

A downstream evaluator will later be asked specific factual questions about
this section (e.g. "which exact muscle groups", "what specific food swaps",
"what were all the reasons given") and will only be able to answer from the
items you extract. A vague or partial item costs you exactly as much as a
missing one.

- When the speaker lists multiple specific things under one topic (several
  muscle groups, several food swaps, several steps of a protocol, several
  named reasons for a claim), extract the **full list**, not a category
  summary. "He trains forearms, calves, tibialis, lateral raises, and abs" is
  one complete item; "he trains some accessory muscles" is not — even if both
  are true, only the first lets someone answer "which muscles".
- When the speaker gives more than one reason, condition, or number for the
  same claim, capture all of them in that item's `statement` (or as separate
  linked items via `rationale_for`) — do not keep only the first one you see.
- Precision over brevity: a longer, fully-detailed `quote` and `statement`
  that captures everything the speaker actually said is correct; a shorter
  one that captures only the gist is a miss, even though it is not wrong.
- This does not relax any other rule — every quote must still be exact and
  contiguous, every quantity must still appear in its own quote, and you
  still extract nothing that is not actually in this section's text.

## Don't drop the specific detail inside a general statement

The single most common way an otherwise-good item still fails a downstream
question: it captures the speaker's general point correctly but drops one
specific number, name, or explicit answer that was stated in the same breath
as a secondary or qualifying clause. Watch for these three patterns
specifically:

1. **A number attached as an aside, not the headline.** If the speaker's main
   point is "plan your drops in advance" but they also say the target range
   is "10 to 20 reps," that range is its own fact and must show up in the
   item's `statement` (and, if it is a distinct quantity, in `quantities`) —
   not be dropped because it wasn't the sentence's main clause.
2. **A named term, acronym, or list the speaker spells out.** If the speaker
   defines what a term stands for (e.g. each letter of an acronym, the named
   parts of a framework, the specific items in a short list they enumerate),
   record the **full definition or full list** in the `statement`, not a
   paraphrase of what it's for. "He explains what REF stands for" is not a
   complete item if the speaker actually said all three words; write the
   three words.
3. **An explicit yes/no, confirmation, or denial.** When the speaker is
   directly answering (even implicitly) a yes-or-no question — confirming,
   denying, or correcting a common assumption ("no, that's not necessary",
   "yes, it does matter", "that's not something I'd recommend") — the item's
   `statement` must preserve that explicit yes/no or confirm/deny, not just
   the reasoning behind it. A downstream reader who only sees your reasoning,
   without the explicit answer, cannot tell which way the speaker actually
   came down.

When in doubt, re-read the exact sentence once more before finalizing an
item's `statement`: if it contains a number, a named term the speaker spelled
out, or an explicit yes/no/confirm/deny, that detail belongs in the
`statement` even when it reads as secondary to the item's main point.

## Scope policy

Extract items only for content in scope:

- `core` — training, bodybuilding, hypertrophy, nutrition, supplements,
  recovery, injury, competition.
- `life_support` — sleep, work, stress, scheduling, travel, finances, but
  **only** as they concern living and training around the sport.
- `sport_philosophy` — identity, discipline, motivation, long-term
  relationship with lifting.

Mark an item `out_of_scope` (still extract it, do not skip it) when it is:
relationships/dating/family advice, YouTube/content-creation or podcast
production talk, marketing, unrelated chat — or an **ad or sponsor read**.
Ads and sponsor reads are always `out_of_scope`; extract enough of one to be
identifiable (e.g. the product/offer named) but never treat its claims as
knowledge content.

If a section contains nothing worth extracting at all (pure chit-chat,
banter, an ad-only section, housekeeping), do not invent items to fill space.
Instead set `"nothing_relevant": true` and give a one-sentence
`"nothing_relevant_reason"`. Leave `"items": []` in that case.

## Evidence basis and hedging

Every item that **asserts or recommends** something (not a purely descriptive
item) must carry an `evidence_basis`:

- `research` — the speaker cites a study, meta-analysis, review, or named
  paper.
- `coaching_experience` — experience with many clients/athletes, or expert
  consensus the speaker is reporting.
- `personal_experience` — one person's own experience ("when I did this...").
- `opinion` — a hypothesis, speculation, or "I think"/"I'd guess" framing with
  no stated basis.

Set `"hedged": true` when the speaker signals uncertainty about the specific
statement (e.g. "I think", "probably", "not 100% sure", "as far as I know") —
regardless of which `evidence_basis` it carries. `"hedged": false` when they
state it plainly.

## Quantities, as spoken — never converted

Record every number exactly as the speaker said it, in the units they used.
**Do not convert units.** Unit conversion happens later in Python, never here.

For each quantity in an item's quote, add one entry to `"quantities"`:
`{"value": <number>, "value_high": <number or null, for a range's high end>,
"unit_as_spoken": "<the unit exactly as spoken, e.g. \"grams per pound\",
\"pounds\", \"minutes\"— never converted or normalized>"}`. A single value
(not a range) has `"value_high": null`. Every value and value_high you record
must appear in the item's own `quote` as a digit or a number word ("two",
"one and a half", "a gram and a half") — an item whose quantity is not in its
own quote will be rejected.

## No speaker attribution

Never guess or record who said what. Do not attribute a statement to a named
host or guest. The note lists episode participants separately; your job is
the content, not who spoke it.

## Item fields

```json
{
  "local_id": "i-01",
  "kind": "recommendation",
  "statement": "one-sentence paraphrase of the item, for internal reference only",
  "quote": "exact contiguous transcript text supporting this item",
  "quote_occurrence": 1,
  "quantities": [{"value": 0.2, "value_high": 0.35, "unit_as_spoken": "grams per pound"}],
  "negated": false,
  "evidence_basis": "research",
  "hedged": false,
  "conditions": "acute sleep deprivation",
  "scope": "core",
  "value": "high",
  "protocol": null,
  "source_mention": null,
  "rationale_for": null
}
```

- `local_id`: `"i-01"`, `"i-02"`, ... unique within this section only.
- `kind`: exactly one of `claim`, `recommendation`, `protocol`, `mechanism`,
  `rationale`, `caveat`, `study_description`, `source_mention`, `side_topic`,
  `follow_up`.
- `quote`: **verbatim, contiguous** text copied from this section — not a
  paraphrase, not assembled from two separate places. It must be exact enough
  that a plain string search finds it in the section text.
- `quote_occurrence`: 1-indexed. If this exact `quote` string appears more
  than once in the section, say which occurrence this item refers to.
- `negated`: `true` only when the statement is a negation (the quote itself
  contains a negation word — "not", "never", "don't", "avoid", and similar).
  Do not set `negated: true` for a statement that is merely cautious or
  qualified; it means the statement itself negates something.
- `value`: how practically important this item is to someone using this note
  — `high` for a specific, actionable recommendation or protocol with real
  practical impact (a dose, a training parameter, a concrete "do this");
  `normal` for a typical claim, fact, or mechanism explanation; `low` for a
  minor or tangential detail. `value` is about practical weight, not
  confidence — a hedged opinion can still be `high` value if acting on it
  matters.
- `protocol`: only for `kind: "protocol"` — a repeatable procedure with
  explicit parameters. `{"what": ..., "dose": ..., "when_for_whom": ...,
  "caveat": ...}`, each field either the stated text or `null` if the source
  did not state it. `null` (the whole field) for every other kind.
- `source_mention`: only for `kind: "source_mention"` or `"study_description"`
  — the reference exactly as heard: `{"authors_as_heard": ..., "year_as_heard":
  ..., "title_as_heard": ..., "type": ...}`, `null` fields for whatever the
  speaker didn't say. `null` (the whole field) for every other kind.
- `rationale_for`: only for `kind: "rationale"` — the `local_id` of the
  recommendation/claim this rationale explains, if it is in this same
  section. `null` otherwise.

### `conditions` — never silently omitted

For `kind` in `claim`, `recommendation`, `protocol`: `"conditions"` is
**required** and must be a string. You must actively decide between two
options for every one of these items:

1. The real qualifying text as stated — e.g. `"acute sleep deprivation"`,
   `"beginners only"`, `"if you're natural"` — when the speaker attaches a
   condition, population, or caveat to the statement.
2. The literal string `"none_stated"` — when you have checked and the speaker
   genuinely attaches no qualifier to this statement.

**Never** write `null` or omit the field for these three kinds. `null` is
only valid for the other kinds (`mechanism`, `rationale`, `study_description`,
`source_mention`, `side_topic`, `follow_up`, `caveat`), where a qualifying
condition is not a meaningful concept. Do not default to `"none_stated"`
reflexively without checking the quote and its surrounding sentence for a
condition first.

## Re-extraction (only when `previous_rejections` is present)

If the input includes a `"previous_rejections"` list, this is a bounded
re-extraction: your prior extraction of this same section had one or more
items rejected by the validator. Each entry gives the prior item payload
(`"item"`) and why it was rejected (`"reason"`, `"detail"`). Fix exactly
what was wrong — most commonly: the `quote` did not actually appear in the
section text verbatim, a quantity in `quantities` did not appear in the
quote, or `conditions` was missing/`null` on a `claim`/`recommendation`/
`protocol` item. Re-extract the **whole section** normally (not only the
flagged items) — other items you extracted correctly the first time should
still appear.

## Output

Return **only** JSON matching this shape — no prose before or after:

```json
{
  "section_id": "<the section_id given to you>",
  "nothing_relevant": false,
  "nothing_relevant_reason": null,
  "items": [ /* zero or more items as above */ ]
}
```
