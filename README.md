# Podcast Automation

[![Portfolio CI](https://github.com/wintersyntax/podcast-automation/actions/workflows/portfolio-ci.yml/badge.svg)](https://github.com/wintersyntax/podcast-automation/actions/workflows/portfolio-ci.yml)
![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-2ea44f)
![Status: Active WIP](https://img.shields.io/badge/Status-Active%20WIP-f59e0b)

A resumable automation pipeline for evidence-heavy **exercise science, training and nutrition podcasts**. It combines independent transcript sources, preserves disagreements as evidence, routes ambiguous cases to human review, and produces durable structured knowledge notes.

This repository is a curated portfolio version of a larger private production project. Deployment-specific identifiers, personal paths, real podcast feed configuration and transcript-derived evaluation corpora are excluded or replaced with synthetic examples.

## Engineering highlights

- resumable RSS-driven episode processing with durable GCS-backed state
- independent Apple Podcasts and Whisper transcript acquisition
- transcript alignment and conservative multi-source reconciliation
- deterministic evidence/provenance checks around bounded AI-assisted resolution
- explicit Human Review for material unresolved conflicts
- episode-level AI budget admission, settlement and provenance tracking
- independent transcript-grounded summary review before publication
- deterministic metadata/frontmatter construction for Markdown knowledge notes
- separate Cloud Run worker and Human Review service boundaries
- optional Slack signalling and macOS credential/sync helpers

## Domain focus

The system is tuned for long-form exercise and nutrition content where small transcription errors can change the meaning of the material being captured. The knowledge pipeline is designed to preserve and structure details such as:

- studies and papers mentioned by the speakers, including authors/year/name as heard, design, sample, duration and reported findings
- exercise selection, setup, execution and technique
- training variables, protocols, numerical recommendations and their conditions
- nutrition, supplementation, recovery and evidence-quality discussions
- practical takeaways while retaining whether a claim came from research, coaching experience, personal experience or another stated basis
- named sources and references for later retrieval

This makes the project intentionally more domain-aware than a generic podcast summarizer: the goal is to turn technical exercise/nutrition conversations into searchable notes without silently strengthening, normalizing or inventing what the speakers said.

## Why multiple transcript sources?

A fluent transcript is not necessarily a trustworthy transcript. Differences involving numbers, units, negations, names, citations or domain terminology can materially change meaning.

This system keeps independent sources separate, compares their evidence, resolves only what can be justified, and stops for human judgment when confidence is insufficient.

## Architecture

```mermaid
flowchart LR
    RSS[RSS feeds] --> Worker[Cloud Run worker]
    Worker --> Audio[Episode audio]
    Audio --> Whisper[Whisper transcription]
    Worker --> Apple[Apple transcript acquisition]

    Whisper --> GCS[(Google Cloud Storage)]
    Apple --> GCS

    GCS --> Compiler[Transcript compiler]
    Compiler --> Resolver[Bounded resolver]

    Resolver -->|material ambiguity| Review[Human Review]
    Review -->|decision + recompile| Worker

    Compiler -->|resolved| Draft[Summary draft]
    Draft --> SummaryReview[Independent grounded review]
    SummaryReview --> Note[Structured Markdown note]
    Note --> GCS
    GCS --> Sync[Optional local vault sync]
```

The important boundary is authority: model output can assist inside explicit contracts, but deterministic code owns identity, evidence IDs, budget accounting, state transitions, validation and publication gates. Unresolved speech evidence remains unresolved until a human decides it.

## Pipeline at a glance

1. Discover the latest configured RSS episode and resume older incomplete work.
2. Download episode audio to ephemeral worker storage.
3. Generate a chunked Whisper transcript with provenance.
4. Acquire an independent Apple transcript when available.
5. Store both source artifacts separately.
6. Align and compare the sources.
7. Apply deterministic equivalence/corroboration rules.
8. Use a bounded resolver only for eligible conflicts.
9. Route remaining material disagreements to Human Review.
10. Recompile after durable review decisions.
11. Generate a summary draft from the canonical transcript.
12. Run an independent transcript-grounded review.
13. Construct metadata/frontmatter deterministically and publish the final note.

## Human Review

The same Flask review application can run locally or as a separate service. Review state is durable and generation-aware; recompile requests carry correlation identifiers so a later worker execution can continue the exact episode/review generation.

![Human Review demo](assets/demo-human-review.gif)

*The synthetic demo shows a high-risk study sample-size conflict (**42 vs 40 participants**) and a deliberately simplified exercise-name conflict (**Romanian deadlift vs Roman deadlift**). The exercise-name example is pedagogical: earlier compiler and bounded-resolution stages are intended to remove many obvious or low-risk differences before Human Review. The important boundary is that protected disagreements such as negations, protocol numbers, units, citations and other domain-sensitive mismatches are not silently normalized away.*

Current Human Review work is focused on reducing how many low-risk cases require a manual decision while keeping protected categories fail-closed. High-confidence, exact-source, low-risk recommendations are treated differently from protected cases, and the ongoing optimization is about making that boundary more selective without weakening it.

Synthetic fixtures in `tests/fixtures/synthetic/` demonstrate the same review boundary without publishing real podcast transcript material.

## Knowledge-note output

The final knowledge artifact is static, so a GIF is not necessary here. The repository includes both a visual preview and a fully synthetic Markdown example rendered from the production note schema and deterministic renderer:

![Synthetic knowledge-note output](assets/demo-knowledge-note.png)

**[Open the generated knowledge-note example](examples/demo-knowledge-note.md)**

It shows the parts this project is designed to preserve for exercise/nutrition podcasts: research discussed, evidence-basis labels, training/nutrition protocols, exercise technique, numerical recommendations, takeaways, tags and sources mentioned.

The evidence meter describes **what basis the speaker gave for a claim**, not an independent rating of whether the claim is true or whether a cited study is high quality:

- **▰▰▰ Study** — the speaker explicitly cites research, a study, review or paper.
- **▰▰▱ Coaches** — coaching experience across multiple clients/athletes, or expert consensus reported by the speaker.
- **▰▱▱ One person** — one person's own experience or anecdote.
- **▱▱▱ Opinion** — hypothesis, speculation, or an "I think / I'd guess" claim without another stated basis.
- **· unsure** — appended when the speaker themselves hedges or expresses uncertainty.
- Purely descriptive bullets carry no evidence-basis label.

A **Study** label therefore means "research was cited in the episode"; it does **not** mean the system independently verified the paper, methodology or conclusion.

The screenshot provides a quick visual overview, while the rendered Markdown remains the primary inspectable artifact because it can be read, searched and reviewed directly.

## Repository layout

```text
podcast_engine/        production orchestration, state, review and knowledge pipeline
compiler/              transcript alignment, comparison and adjudication logic
macos_agent/           optional local credential maintenance and sync helpers
slack_ack/             small signed-request acknowledgement service
security/              narrow IAM/reference policy helpers
prompts/               versioned knowledge/evaluation prompts
scripts/               bounded operator and maintenance tools
tests/                 curated unit/integration tests plus synthetic fixtures
config/                safe example/runtime policy configuration
docs/                  public architecture and deployment documentation
examples/launchd/      sanitized macOS launch-agent examples
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Transcript pipeline](docs/TRANSCRIPT_PIPELINE.md)
- [Human Review](docs/HUMAN_REVIEW.md)
- [Synthetic Human Review demo](docs/DEMO_HUMAN_REVIEW.md)
- [Synthetic knowledge-note demo](docs/DEMO_KNOWLEDGE_NOTE.md)
- [Knowledge generation](docs/KNOWLEDGE_GENERATION.md)
- [Setup](docs/SETUP.md)
- [Deployment](docs/DEPLOYMENT.md)
- [Security and privacy](docs/SECURITY_AND_PRIVACY.md)

## Configuration

The checked-in podcast configuration is intentionally disabled and synthetic. Start from `config/podcasts.example.json` and `.env.example`.

The committed OpenRouter preset lock is also an example identity, not a production preset. A real deployment must provide its own verified preset/version configuration.

## Public-repository safety

This repository intentionally contains no production API keys, OAuth secrets, Slack secrets/webhooks, Apple bearer tokens, production GCP project/bucket identifiers, personal account allowlists, private filesystem paths, real deployment feed configuration or real transcript-derived evaluation corpus.

## Future direction

The current pipeline is designed to produce durable, source-grounded knowledge notes that can later support semantic search and retrieval-augmented generation (RAG). A future phase could index the structured notes, studies, exercises, recommendations, topics and cited sources so questions can be answered across the podcast library while preserving links back to the underlying evidence.

The goal would not be to replace the grounded note pipeline, but to use it as the trusted retrieval layer for cross-episode search and synthesis.

## Development approach

This project was developed with substantial **AI-assisted software development**. AI tools were used throughout architecture exploration, implementation, test generation and repair, code review, and iterative design. The domain goals, workflow requirements, acceptance criteria, evaluation, integration, and final decisions were shaped iteratively through that human-AI process; this repository is not presented as if every implementation or design idea was independently authored from scratch.

## Project status

**Active work in progress.** The private production project is working, while this public repository is a cleaned portfolio extraction of that system.

Recent work focused on optimizing the summary / knowledge-note workflow. The current development focus is Human Review: reducing the number of low-risk transcript differences that require a manual decision, while preserving strict handling for cases where a transcription error could materially change meaning — especially negations, protocol numbers, units, citations, source-only semantics, anomalies, and domain-sensitive terminology.

The next phase after Human Review optimization is the retrieval layer described above: semantic search and RAG over the source-grounded knowledge notes.


## License

Released under the [MIT License](LICENSE).
