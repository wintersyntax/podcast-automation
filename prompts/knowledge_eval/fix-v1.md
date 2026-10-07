# Note targeted fix — prompt v1

TASK-106 Phase A, `note-fix-v1` schema version. Design authority:
`docs/superpowers/specs/2026-09-24-grounded-knowledge-note-pipeline-design.md`
§5.6.2 (targeted fix).

You are rewriting a **small, specific set of flagged units** from an
already-composed note — never the whole note. Each entry in
`flagged_units` gives you the unit's current `text`, the `item_ids` it
cites, why review flagged it (`reason`), and the same `cited_items`
(`statement`/`quote`) context composition originally had. Rewrite each
flagged unit so its text is fully supported by its cited items and no
longer has the problem named in `reason` — nothing else changes.

## The same rules as composition

- Write only what the cited items support: no new facts, no changed
  certainty, no dropped conditions, no broadened or narrowed scope.
- Never write a number or unit yourself. Any quantity stays a
  `{q:<item_id>:<n>}` placeholder, exactly as in composition — never a
  literal digit or number word outside one.
- You may adjust which items a unit cites (`item_ids`) if the fix requires
  it (e.g. restoring a dropped condition by citing an item that states it),
  but every id you use must be one that could plausibly appear in this
  note — never invent one.

## Transcript excerpt (only for a `condition_dropped` fix)

A flagged unit whose `reason` is about a dropped condition may additionally
carry `transcript_excerpt` — a short, bounded slice of the original
transcript around the cited item's quote. This is the **only** point in the
whole pipeline where you see transcript text, and only because the item's
own qualifying condition genuinely needs to be recovered from source. Use
it only to find the missing qualifier; do not pull in any other fact from
it, and do not cite it directly — the fixed unit must still be grounded
only in its cited items' `statement`/`quote`.

## Output

Return **only** JSON matching this shape — no prose before or after, and
only for the units you were given:

```json
{
  "fixed_units": [
    {
      "unit_id": "sections[0].bullets[1]",
      "text": "...",
      "item_ids": ["..."],
      "primary_item_id": "...",
      "asserts": true
    }
  ]
}
```

`primary_item_id` and `asserts` only apply to a bullet (a `unit_id` under
`sections[...].bullets[...]`); for a `tldr[...]` or `takeaways[...]` unit,
set both to `null`.
