# Knowledge generation

A completed transcript is transformed into a durable knowledge note through a second guarded pipeline.

1. Generate a Markdown summary draft from the canonical compiled transcript.
2. Normalize layout without changing semantics.
3. Derive deterministic transcript spans, draft blocks and risk items.
4. Run an independent transcript-grounded reviewer through AUDIT and EDIT contracts.
5. Validate evidence coverage, edit obligations and non-regression in Python.
6. Generate structured metadata under a strict schema.
7. Construct YAML frontmatter deterministically in Python.
8. Publish the final Markdown note only after the accepted review chain is valid.

The summary model does not author arbitrary frontmatter. The independent reviewer cannot invent evidence IDs or silently use outside knowledge to repair uncertain transcript wording.
