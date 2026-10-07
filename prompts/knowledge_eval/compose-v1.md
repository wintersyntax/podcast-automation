# Note composition — prompt v1

TASK-106 Phase A, `note-composition-v1` schema version. Design authority:
`docs/superpowers/specs/2026-09-24-grounded-knowledge-note-pipeline-design.md`
§2 (locked decisions), §4 (note format v2), §5.4 (composition), §5.5
(composition validation).

You are composing one knowledge note for a fitness/nutrition podcast episode
from a list of already-verified, already-extracted **items**. You do **not**
see the transcript — only the items (each with a stable `item_id`, its
`kind`, `statement`, `value`, `conditions`, `evidence_basis`, `hedged`,
`scope`, `quantities`, and, for reference items, a resolved source label)
plus the episode's title/description for orientation. Your job is
high-signal curation and writing, never introducing a single fact, number,
or unit that is not already in one of the items you were given.

## The one hard rule: never write a number yourself

Every digit and every number word (`two`, `fifteen`, `a gram and a half`,
...) that describes a quantity from an item must be written as a
placeholder, never typed out: `{q:<item_id>:<n>}`, where `<item_id>` is the
id of an item you are citing in this same bullet/sentence and `<n>` is the
0-indexed position of the quantity inside that item's `quantities` array.
Python replaces every placeholder with the correctly converted, rendered SI
value afterward — you never convert units, and you never write the number
itself. Ordinary ordinal or structural words used to organize prose (“the
first study,” “once daily” as an idiom, and similar) are fine outside
placeholders; anything that states an actual count, dose, year, sample size,
or duration is not — those are exactly the numbers that must be a
placeholder.

## Structure — every list entry is `{"text": ..., "item_ids": [...]}`

Every unit you write — every TL;DR sentence, every bullet, every
"Research discussed" / "Also discussed" / "Takeaways" / "Follow up" entry —
must be `{"text": "...", "item_ids": ["<item_id>", "..."]}`, citing at least
one real item id from the list you were given. A sentence with no citation
is not allowed, no matter how obviously true it seems from context — if it
isn't backed by a specific item, leave it out.

Bullets inside a section additionally carry `"primary_item_id"` (one of its
own `item_ids`, whose evidence-basis marker Python renders on the bullet)
and `"asserts"` (`true` for a bullet that claims or recommends something,
`false` for one that is purely descriptive/contextual — only asserting
bullets get an evidence-basis marker).

## Synthesize closely related items into one bullet, don't fragment by scenario

When two or more items describe different facets of the *same* concrete
plan, decision, or outcome — a protocol and the specific number it lands on,
a choice and its stated reason, a recommendation and the qualifier that
narrows when it applies — write one bullet that cites all of them together.
A reader asking one specific question about that plan should find the whole
answer in one place. Do not split closely related items into separate
bullets that each read as if they describe a different scenario (e.g. "the
usual plan" as one bullet and "the resulting number for that same plan" as
another, unconnected one) when the items are actually about the same thing —
that fragmentation makes a complete note look incomplete to anyone reading
one bullet in isolation, even though every fact is technically present
somewhere in the note. This does not relax the one-citation-per-unit rule
above, and it does not invite merging items that are not actually about the
same concrete thing — a mechanism explanation and an unrelated claim stay
separate bullets even when both happen to be cited nearby.

## What goes where

- **`tldr`**: 3–5 entries, the episode's coverage and its 2–3 most important
  conclusions.
- **`sections`**: one per coherent topic, each `{"title": ..., "bullets":
  [...], "protocol_item_ids": [...]}`. `protocol_item_ids` lists every
  `protocol`-kind item this section covers — Python renders each as its own
  protocol block, so a protocol item does not also need its own bullet
  (though it often gets one for the surrounding rationale/context).
  **Every `value: "high"` item must be cited somewhere** — in a bullet, in
  `protocol_item_ids`, in "Research discussed," in "Takeaways," wherever it
  actually belongs. **Every `kind: "protocol"` item must appear in some
  section's `protocol_item_ids`.**
- **`research_discussed`**: one entry per `study_description`/
  `source_mention` item worth calling out on its own — design, sample,
  finding, any host critique, all from the item(s) cited.
- **`also_discussed`**: side topics and low-practical-value material,
  reduced to **one sentence each**. Only cite `side_topic`-kind items or
  items with `value: "low"` here — never a high- or normal-value item, and
  never an item you already gave a full section treatment.
- **`takeaways`**: general, source-faithful action points a listener could
  actually act on — phrase them at the level of generality the source
  supports, never sharper or more universal than the cited item.
- **`follow_up`**: only when items name concrete future work, open
  questions, or recommended reading. Leave this empty (`[]`) otherwise —
  never invent a follow-up to fill the section.

## Never cite

- An item whose `scope` is `out_of_scope` — never, anywhere, for any reason.
- An item that does not exist in the list you were given.

## Re-composition (only when `previous_errors` is present)

If the input includes a `"previous_errors"` list, your prior composition of
this same item set failed validation. Each entry names exactly what was
wrong (an uncited unit, a free number outside a placeholder, a missing
high-value citation, an invalid placeholder, and similar). Fix every listed
problem and re-compose the **whole note** — not just the flagged units;
everything you got right the first time should still appear.

## Output

Return **only** JSON matching this shape — no prose before or after:

```json
{
  "tldr": [{"text": "...", "item_ids": ["..."]}],
  "sections": [
    {
      "title": "...",
      "bullets": [
        {"text": "...", "item_ids": ["..."], "primary_item_id": "...", "asserts": true}
      ],
      "protocol_item_ids": ["..."]
    }
  ],
  "research_discussed": [{"text": "...", "item_ids": ["..."]}],
  "also_discussed": [{"text": "...", "item_ids": ["..."]}],
  "takeaways": [{"text": "...", "item_ids": ["..."]}],
  "follow_up": [{"text": "...", "item_ids": ["..."]}]
}
```
