# Public extraction scope and upstream status

This repository is a **curated portfolio extraction**, not a live, byte-for-byte mirror of the private production project. It deliberately demonstrates core engineering boundaries with safe configuration, synthetic fixtures and public tests.

**Comparison date:** 11 October 2026. The public baseline was curated on 7 October; the private system continued to evolve on 8–10 October. A passing public CI run proves the published snapshot's test suite, **not** that every later private production feature has been copied here.

## What the public source demonstrates

| Area | Included in the published extraction |
| --- | --- |
| Pipeline | Resumable RSS/audio ingestion, independent Apple and OpenRouter Qwen3-ASR source handling, provenance and spend controls |
| Compiler | Bounded alignment and reconciliation, conservative conflict authority, Third-ASR evidence and materiality logic |
| Human Review | Materiality-first quick cards, explicit settled-card acceptance, detailed review and synthetic interactive demo |
| Knowledge | Single-pass structured `knowledge-note-v3` writer, deterministic validation/rendering and a synthetic rendered note |
| Integration | Separate worker/review services, optional Slack and Mac helpers, example configuration and curated regression tests |\n| PodcastOps demo UI | The production frontend template is publicly inspectable through an offline, read-only synthetic server; the private GCS-backed status projection and authentication are not ported |

## Verified later private-production milestones — not fully ported here

| Area | Added or proven in private production after the public baseline |
| --- | --- |
| Republished episodes | Versioned, evidence-based duplicate classification before Apple acquisition or subsequent processing, with an explicit operator override |
| Review durability | Legacy-safe versioned review queue identity, bounded full-span decision replay and independently persisted post-review feedback notes |
| PodcastOps | Authenticated read-only episode/card status and delivery dashboard, separated from the review queue; optional evidence failures do not hide episode status |
| Mac Vault delivery | Checksum-bound, idempotent local-copy receipts; the dashboard distinguishes `summary ready` from `Mac copy confirmed`. A local receipt is **not** proof of remote iCloud upload |
| Operational verification | Production demonstrations of duplicate suppression, completed review replay, independent feedback saving and scheduled Vault Sync recovery |

These rows describe **upstream production progress**, not missing-file bugs in the portfolio build. The synthetic PodcastOps demo added later includes only the presentation template and fabricated API response; it is not a port of the production backend. They must not be described as implemented in the checked-in public source until the corresponding sanitized implementation and regression tests are deliberately ported and pass this repository's CI.

## Media and demo boundary

- `assets/demo-human-review.gif` is the tracked Human Review animation; its launcher uses fabricated podcast/transcript content.
- `assets/demo-knowledge-note.png` is the tracked static preview, with the corresponding inspectable synthetic Markdown at `examples/demo-knowledge-note.md`.
- The knowledge-note guide's asset is a **PNG**, not a second published GIF. Do not imply that an uncommitted or separately recorded GIF has been shipped.
- The two demos do not require production credentials, real podcast material or live cloud APIs.

## Publication and verification rules

Never copy the private repository wholesale. Before porting a private change, review the exact source and tests for production resource names, account identifiers, personal paths, private podcast configuration, raw transcripts, evaluation cases, credentials and endpoint URLs. Substitute synthetic fixtures where needed; retain deterministic and human-approval authority boundaries.

The public GitHub Actions workflow runs the curated test suite on the public checkout. Any subsequent source update needs its own public PR and successful CI. Private-repository acceptance cannot substitute for public-repository verification, nor can public CI prove the separate private production deployment.

For system design and demo details, see [Architecture](ARCHITECTURE.md), [Human Review](HUMAN_REVIEW.md), [Knowledge generation](KNOWLEDGE_GENERATION.md) and [Security and privacy](SECURITY_AND_PRIVACY.md).
