# Development history

Podcast Automation grew from a small feed-to-summary worker into a resumable, evidence-aware podcast processing system. The major decisions below explain **what changed, why it changed, and what the project gained**.

This is a **selected engineering timeline**, not a list of tagged releases. The public portfolio snapshot is prepared as a single root commit named `Initialize`; the history below is reconstructed from contemporaneous development notes and verified source commits. Some experiments described here were later retired and are **not** features of the current public edition.

## October 31, 2026 — Public source refresh

- Integrated reviewed product changes through private source `72677eee1e9e8059490d367edb41a9ffebe7b11f`, preserving disabled example configuration and synthetic demonstrations.
- Included optional note-coverage fill, episode-index concurrency fixes, duplicate status display, newer materiality rules, bounded Third-ASR retries and idempotent terminology backfill.
- Added a separate CC BY-SA 4.0 notice for the exercise lexicon and a narrowly scoped Gitleaks exception for one synthetic episode key.
- Retained the exclusion of production evidence, personal configuration, credentials, private CI tooling and private Git history.

## August 23, 2026 — The first working pipeline

The original question was straightforward: *could a podcast episode be picked up automatically and turned into something useful to read?*

- Built a Python/Flask worker packaged for Docker and Cloud Run, with an RSS reader, episode discovery and an audio downloader.
- Connected Whisper transcription to an OpenRouter-powered summarizer, then assembled an end-to-end episode processing and export flow.
- Split the pipeline from its HTTP entry point and added a standalone one-run worker so processing would not depend on an interactive app.
- Introduced basic episode state and duplicate checks, initially in a much simpler form than today's durable lifecycle.

**Turning point:** a working demo exposed the next problem: producing a fluent summary is easy compared with proving that the transcript it came from is faithful.

## August 24 — Experimenting with a second transcript: YouTube

The first attempt at independent corroboration used **YouTube captions as an experimental second transcript** beside Whisper. This was a short-lived comparison route, not the long-term production design.

- Added YouTube transcript extraction, cleanup and normalization, along with fixture-based compiler tests.
- Introduced a transcript compiler with global alignment, mismatch detection and machine-readable conflict reports.
- Moved episode state and generated artifacts toward **Google Cloud Storage** so processing could survive short-lived Cloud Run executions.
- Switched episode identity toward RSS GUIDs instead of relying on titles, and expanded support to multiple podcasts.

**What the experiment taught us:** an independent source is useful only if the system can align different wordings and report disagreements without silently declaring one source correct.

## August 25–26 — Apple + Whisper and the first Human Review

YouTube extraction was removed from the active source path. **Apple Podcasts transcripts and Whisper** became the two canonical sources where Apple text was available.

- Built Apple transcript acquisition and parsing, including handling malformed or delayed subtitle artifacts.
- Added bounded Whisper audio chunks, overlap deduplication, timestamp stitching and alignment tolerant of tokenization differences.
- Made transcript differences explicit: numbers, units, names and negations could not be casually rewritten into a more fluent reading.
- Created the first dedicated **Human Review** workflow, with stored decisions and a reviewer-facing web interface.
- Began converting a loose summarizer into a versioned knowledge pipeline with Python-owned metadata, tags and YAML frontmatter.

**Turning point:** the system stopped treating an AI-generated transcript or summary as its own source of truth. Ambiguous evidence had to be preserved for a person to resolve.

## August 27–31 — From a script to a recoverable service

The next challenge was operating reliably when cloud jobs restart, an Apple transcript arrives late, or a local notes folder is temporarily unavailable.

- Consolidated podcast processing in the cloud and limited the macOS agent to Apple credential maintenance and local knowledge/vault synchronization.
- Added resumption of older incomplete episodes, not just the newest RSS item; kept durable artifacts reusable when their identity and provenance matched.
- Standardized canonical episode keys and storage contracts, removed redundant state copies and tightened idempotent ingestion.
- Added monitoring, review notifications and a small Slack acknowledgement boundary.
- Hardened access to stored transcripts and review artifacts using narrower service permissions and producer fingerprints.

**Result:** a restart could resume verified work instead of blindly re-downloading and retranscribing, and local synchronization no longer had to orchestrate the entire pipeline.

## September 1–16 — Making the output more grounded

Once episode processing was dependable, the focus moved to whether a knowledge note actually represented what a speaker said.

- Iterated on structured summaries, readable Markdown, navigation across long transcripts and deterministic metadata formatting.
- Built evaluation fixtures and checks for unsupported precision, incorrect technical terms, missing coverage and source-grounding mistakes.
- Refined compiler and review boundaries so low-risk notation differences could be handled mechanically while meaningful disagreement still required review.
- Improved the review lifecycle, replay/recompile tracking and resilience when an advisory model returned malformed or partial results.

Several experimental summary-review approaches were evaluated during this period. They informed the acceptance checks, but the older multi-stage review path was **eventually removed**, not kept as an additional current product feature.

**Lesson:** confidence in a polished-looking note is not evidence of accuracy. Validation must be explicit, reproducible and anchored in the transcript.

## September 17–28 — Controlling AI cost and reducing review work

Adding more AI calls was not automatically better. The project needed bounded spending and a clearer separation between **suggestion** and **decision**.

- Introduced episode-level AI budget admission and settlement with durable spend/provenance records.
- Explored a third, independent ASR reading for disputed audio spans rather than paying to retranscribe entire episodes.
- Added guarded proposals and quicker review interactions without allowing advisory evidence to rewrite canonical transcript text.
- Continued testing number and terminology handling, review replay and difficult long-form audio cases.

**Turning point:** automation became useful for *preparing evidence and reducing clicks*, while the reviewer retained authority over unresolved material changes.

## September 29 – October 1 — A new approach to knowledge notes

The earlier summarization pipeline had accumulated too many stages. A simpler alternative was tested: **one structured, transcript-grounded writer response**, followed by deterministic Python checks.

- Introduced `knowledge-note-v3`: research discussed, topic and people proposals, training/nutrition protocols, takeaways and sources in one structured response.
- Checked each claimed unit against transcript anchors, dropping unsupported or promotional content rather than inventing a replacement.
- Converted marked physical quantities to SI, resolved controlled tags, built frontmatter and rendered stable Markdown in Python.
- Bound model requests to the same per-episode budget and repeatability rules used elsewhere.

**Design decision:** let the model organize content, but let deterministic code decide what can be published. The single-pass writer later replaced the legacy summary, summary-review and metadata stages entirely.

## October 2–7 — Evidence-first review and better ASR

Real review cases showed that many disagreements were harmless formatting, fillers or sponsor text, while a few changed actual claims.

- Added a materiality filter to distinguish low-impact wording differences from conflicting numbers, entities, polarity and domain terminology.
- Built **quick review cards**, protected detailed review cases, control samples and an explicit save/confirmation step.
- Prefetched bounded Third-ASR clips when another listening pass could add useful evidence; the extra source remained advisory.
- Moved hosted ASR from an initial Whisper model to **Qwen3-ASR** after comparing missed speech and transcription behavior.
- Improved chunk-boundary timestamps and producer identity so stale transcripts from a different model/stitch policy were not accidentally reused.
- Introduced transcript-based **republished episode detection** to avoid repeating downstream work for a substantially duplicated recording.

**Result:** less repetitive review work, stronger protection for disagreements that matter, and cleaner separation of independent transcript sources.

## October 8–10 — Seeing what happened to each episode

A worker that completes a job is not the same as proof that a particular episode reached the final notes library.

- Added a read-only **PodcastOps** episode dashboard showing transcription, compilation, review, note and delivery states independently.
- Introduced checksum-bound **Mac Vault copy receipts**, accepted only when they matched the current canonical note bytes.
- Kept uncertain, missing or stale delivery evidence visibly uncertain rather than displaying a misleading success.
- Hardened replay of compatible Human Review decisions and allowed feedback to be saved without requiring another transcription decision.

**Design principle:** observability should describe what can actually be verified. A local copy receipt does not prove that a remote iCloud upload finished.

## October 12–15 — Reliability, simplification and a clearer output format

The final phase represented in this portfolio is more about eliminating failure modes and obsolete paths than adding more AI features.

- Deferred an episode during transient transcription-provider outages while allowing unrelated episodes to continue.
- Made small ASR word-timestamp overlaps at chunk seams recoverable while keeping large regressions invalid.
- Retried transient AI-budget contention during parallel Third-ASR evidence preparation without losing the hard budget boundary.
- Used audio from the exact transcription source for review clips, preventing dynamic ad insertions from shifting the listening context.
- Retired the older multi-stage summary and experimental note pipelines in favor of the single-pass writer.
- Simplified knowledge-note presentation: each note opens with the episode title, and claim-basis labels are now **🔬 Study**, **🏋️ Coaches**, **👤 One person** or **💬 Opinion**. A speaker's hedge adds `· unsure`.
- Improved test repeatability and made the public portfolio run separate publication-safety, dependency and application checks.

**Where that leaves the project:** a production-derived workflow built around durable state, independent evidence, deliberate human decisions and inspectable Markdown — with local synthetic demos that let readers explore the key interfaces without production credentials.

## What is included in this public edition?

The public repository includes curated podcast application code, a compact Python test suite, safe example configuration and synthetic demonstrations of [PodcastOps](docs/DEMO_PODCASTOPS.md), [Human Review](docs/DEMO_HUMAN_REVIEW.md) and [knowledge-note output](docs/DEMO_KNOWLEDGE_NOTE.md).

Historical experiments, private operations and real episode data are **not** bundled simply because they appear in the timeline. See the [README](README.md) for the current feature overview and [Security and privacy](docs/SECURITY_AND_PRIVACY.md) for the publication boundary.
