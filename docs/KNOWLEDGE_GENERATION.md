# Knowledge generation

A completed canonical transcript is transformed into a durable knowledge note through one of two guarded paths. `PODCAST_KNOWLEDGE_WRITER_PRESET` selects the single-pass writer when it is set.

## Single-pass writer

1. Send the whole compiled transcript, bounded episode context and tag vocabulary to a verified writer preset whose system prompt matches `prompts/knowledge/note-writer-v1.md`.
2. Receive a structured `knowledge-note-v3` note with topics, people and tag proposals. Each physical request reserves from the episode AI budget; an invalid structure permits one more request, while a truncated response does not.
3. Validate the structure in Python. Remove individual units with unsupported transcript anchors, sponsor content or label-only stubs; reject the whole note if too many anchored units fail or no section remains.
4. Convert marked quantities, render the Markdown body and resolve tags deterministically. Store the checked structured note and spend/provenance record in the existing knowledge manifest.
5. Construct frontmatter in Python and publish only after the checks pass. A cached checked note can be rendered again without a model call when the renderer changes.

## Summary-review path

When no writer preset is configured, the earlier path remains available only with an active summary-review preset:

1. Generate a Markdown summary draft from the canonical compiled transcript.
2. Normalize layout without changing semantics.
3. Derive deterministic transcript spans, draft blocks and risk items.
4. Run an independent transcript-grounded reviewer through AUDIT and EDIT contracts.
5. Validate evidence coverage, edit obligations and non-regression in Python.
6. Generate structured metadata under a strict schema.
7. Construct YAML frontmatter deterministically in Python.
8. Publish the final Markdown note only after the accepted review chain is valid.

Neither path lets a model author arbitrary frontmatter. The independent reviewer cannot invent evidence IDs or silently use outside knowledge to repair uncertain transcript wording.
