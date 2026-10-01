# Human Review

Human Review is the explicit boundary for transcript conflicts that cannot be resolved safely from deterministic evidence or a bounded resolver.

The Flask review application presents the competing source spans, alignment context, optional audio evidence and the current review state. Decisions are persisted before a controlled worker recompile is requested.

Key properties:

- review queues are durable and generation-aware;
- stale review state is detected rather than silently reused;
- review audio is bounded to the relevant evidence window;
- recompile requests carry correlation identifiers;
- authentication can be enabled for deployed review services;
- notification delivery is best-effort and never substitutes for durable state.

The same application can run locally for development or as a separate Cloud Run service.
