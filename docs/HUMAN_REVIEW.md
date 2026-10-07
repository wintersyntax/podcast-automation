# Human Review

Human Review is the explicit authority boundary for transcript conflicts that cannot be resolved safely from deterministic evidence or a bounded resolver.

The Flask review application is durable and generation-aware. It presents source spans, transcript context, bounded audio, optional Third-ASR evidence and current policy evidence, but every canonical change still comes from an explicit human decision.

## Current review flow

1. **Quick cards** — materiality evidence groups low-impact cards into a 10% control sample, fast Apple/Whisper choices, and one-at-a-time proposals. Choices are staged in the browser, can be revisited, and are saved together only after a final check list.
2. **Settled by the filter** — once the control sample is complete, the remaining filter-settled cards can be accepted with one explicit click. They are still persisted as individual audited human decisions.
3. **Full review** — cards without safe quick treatment stay in the ordinary listening/edit flow. Numbers and other protected semantics are intentionally conservative.
4. **Recompile** — only durable decisions can advance the episode.

The quick-card page shows the disputed span in transcript context with the currently selected reading, uses consistent Apple/Whisper source styling, and counts active review time only while the page is visible and focused. A saved session appends a bounded `materiality_review_log` entry so later tuning can measure overrides, time, settled-list inspection and reviewer notes.

## Advisory evidence

Third-ASR evidence can be prefetched before the review notification and refreshed in the page for eligible cards. The older A/B/C review tiers, Batch and Assisted Review remain available under advanced controls. They are advisory only: no model, third transcript, tier or materiality verdict can silently write canonical transcript text.

## Reliability properties

- review queues are durable and generation-aware;
- stale review state is detected rather than silently reused;
- review audio is bounded to the relevant evidence window;
- recompile requests carry correlation identifiers;
- authentication can be enabled for deployed review services;
- notification delivery is best-effort and never substitutes for durable state;
- operator scripts can report review telemetry and materiality-session history without changing review decisions.

The same application can run locally for development or as a separate Cloud Run service.
