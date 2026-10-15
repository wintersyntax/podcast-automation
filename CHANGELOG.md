# Development changelog

Selected milestones from the project's development. These entries describe how the software evolved; they are **not** separate public releases or a reconstructed Git history.

## October 2026 — More reliable, clearer output

- Replaced the legacy multi-stage summary path with a single-pass, transcript-grounded knowledge-note writer and deterministic checks.
- Added a read-only PodcastOps dashboard and checksum-based local vault-copy receipts.
- Improved Human Review with materiality-first quick choices, bounded review-audio handling and durable decision replay.
- Added republished-episode detection, per-episode recovery from transcription-provider outages and more focused regression checks.
- Simplified evidence labels to 🔬 Study, 🏋️ Coaches, 👤 One person and 💬 Opinion.

## Early October 2026 — Review and observability

- Introduced synthetic, credential-free previews of the dashboard, Human Review and generated Markdown notes.
- Strengthened independent transcript comparison, AI-budget accounting and review traceability.

## September 2026 — Core pipeline

- Established resumable RSS episode processing, separate transcript sources and evidence-first reconciliation.
- Added explicit Human Review for unresolved material transcription conflicts and structured, searchable Markdown output.

The current public edition includes a selected, sanitized implementation of these features. For implementation details, see [Architecture](docs/ARCHITECTURE.md) and [Knowledge generation](docs/KNOWLEDGE_GENERATION.md).
