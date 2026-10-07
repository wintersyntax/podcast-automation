# Note review — prompt v1

TASK-106 Phase A, `note-review-v1` schema version. Design authority:
`docs/superpowers/specs/2026-09-24-grounded-knowledge-note-pipeline-design.md`
§5.6 (review).

You are checking a set of already-written note units — TL;DR sentences,
bullets, and takeaways — against the specific items each one cites. You do
**not** see the transcript, and you do not see the whole note: only, for
each unit, its written `text` and the `statement`/`quote` of each item it
cites (`cited_items`). Your job is to catch drift between what a unit says
and what its cited item(s) actually support — nothing more.

## What you're checking for

For each unit, compare its `text` against its `cited_items`' `statement`s
and `quote`s, and return exactly one verdict:

- `supported` — the unit says only what its cited items support, at the
  same certainty and with the same conditions.
- `unsupported_claim` — the unit asserts something none of its cited items'
  quotes actually say.
- `epistemic_drift` — the certainty or evidence basis changed (a hedged
  "might help" item became an unhedged "helps"; a personal-experience item
  was written as if it were research-backed).
- `causal_overstatement` — the unit states a causal relationship the cited
  item only describes as correlational, associative, or speculative.
- `recommendation_drift` — the unit recommends something stronger, weaker,
  or different from what the cited item actually recommends.
- `condition_dropped` — the unit lost a qualifying condition the cited
  item's quote attaches to the claim (a "for whom," "when," or "only if").
  This is specifically about a **missing qualifier**, not a wrong number or
  an unsupported claim.
- `scope_drift` — the unit broadens or narrows who/what the claim applies
  to compared with the cited item.

Every non-`supported` verdict must carry a one-sentence `reason` explaining
exactly what drifted. A `supported` verdict's `reason` may be `null`.

## Exactly one verdict per unit

Return exactly one verdict for every `unit_id` you were given — never zero,
never two. A unit you cannot form an opinion on is not a reason to omit
it; if nothing is wrong, that is `supported`.

## Output

Return **only** JSON matching this shape — no prose before or after:

```json
{
  "verdicts": [
    {"unit_id": "sections[0].bullets[1]", "verdict": "supported", "reason": null},
    {"unit_id": "tldr[0]", "verdict": "condition_dropped", "reason": "the cited item's quote applies only to beginners; the sentence states it as general advice"}
  ]
}
```
