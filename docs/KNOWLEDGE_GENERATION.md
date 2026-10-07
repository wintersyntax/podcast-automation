# Knowledge generation

A completed canonical transcript is transformed into a durable knowledge note through the current single-pass `knowledge-note-v3` path. A real deployment must configure a verified `PODCAST_KNOWLEDGE_WRITER_PRESET` whose system prompt matches `prompts/knowledge/note-writer-v1.md`.

## Current production path

1. Send the whole compiled transcript, bounded episode context and tag vocabulary to the verified writer preset.
2. Receive one structured `knowledge-note-v3` response containing the note plus topics, people and tag proposals. Each physical request reserves from the episode AI budget; an invalid structure permits one more request, while a truncated response does not.
3. Validate the full structure in Python. Individual units whose transcript anchors cannot be supported, sponsor/ad content and label-only stubs are removed; the whole note fails only when the bounded publication checks say it is no longer trustworthy.
4. Convert marked quantities to SI in Python, render the Markdown body deterministically and resolve tags against the controlled registry.
5. Store the checked structured note together with bounded spend/provenance evidence in the existing knowledge manifest.
6. Construct YAML frontmatter in Python and publish only after the checks pass. A cached checked note can be rendered again without another model call when renderer-only behavior changes.

The model never authors arbitrary frontmatter, controls publication state, or silently repairs uncertain transcript wording. Source-truth decisions remain upstream in the compiler and Human Review.

## Compatibility code

The curated codebase still contains older summary, summary-review and metadata modules plus their regression tests because the private production project has not yet physically removed that compatibility surface. They are **not part of the current production deployment or the portfolio architecture shown in this repository**. New knowledge notes use the single-pass writer described above.
