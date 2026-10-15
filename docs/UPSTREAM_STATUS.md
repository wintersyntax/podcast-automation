# Public extraction scope and upstream status

**Audited private source:** `wintersyntax/podcast-worker` at `d8678469077dd4b644fcc950562e10fd11ee730e` (15 October 2026).
**Public base:** `wintersyntax/podcast-automation` at `2e869bb057b21c1d795f09ca5754a94cdbac4938`.
This repository is an independently tested, curated **derivative**, not an automatic mirror, a production deployment, or a transfer of private Git history.

## Included in the October 15 extraction

| Area | Public source state |
| --- | --- |
| RSS / Apple / OpenRouter ASR | Current lifecycle, provider-defer/resume behavior, versioned duplicate classification, conservative transcript alignment, bounded retries and spend accounting |
| Human Review | Current materiality/Third-ASR evidence, guarded review decisions and replay, feedback persistence, bounded review-audio cleanup |
| Knowledge notes | Single-pass `knowledge-note-v3` writer, transcript-anchored validation, deterministic rendering, SI quantities, tags and frontmatter |
| Operations | Read-only PodcastOps projection, vault receipt helpers and local sync logic, plus separately isolated synthetic episode dashboard |
| Regression tests | Current sanitized tests for those paths, synthetic fixtures and original public demonstration tests |
| CI | Public-specific Publication safety, hash-locked Dependency and Application checkpoints |

## Removed from the previous public snapshot

The previous public extraction retained an older summary/metadata implementation and review-benchmark compatibility files, even though the private source had removed them. The refresh deletes **69 obsolete production, test, benchmark and prompt files** that are absent from the current private podcast product. The live note path now uses `podcast_engine/knowledge/orchestration.py`, `notes_v3`, `episode_inputs.py` and `quantities.py`.

The private development-tool split (TASK-147, 14 October) removed the Cline Host/console, Semantic Auditor, TypingMind bridge, their policy hooks, node contracts and historical planning evidence into **a separate private AI-assisted development-workflow project**. Those development tools were **not part of this public podcast product**, and this extraction does not import them. Runtime AI-assisted transcript evidence/resolution remains part of the podcast pipeline under its separate constraints.

## Deliberately not published

- Private repository history, production ledger/tasks, archived planning and operational evidence.
- Live RSS feeds, real transcripts, reviewer comments, corpus-evaluation documents and account-specific configuration.
- API credentials, OAuth/client secrets, personal emails or paths, real Cloud resource names, bucket names, review service URLs, IAM bindings and deploy access.
- Self-hosted Mac runner registration, supervisor control, exact-main provenance scripts and private CI operating runbooks. The public repository keeps its existing GitHub Actions portfolio job; its checkpoints are not a claim of private CI equivalence.
- The authenticated **deployment** of the private PodcastOps UI. The public `ops_dashboard.py` source is not connected to production data, while `scripts/demo_ops_dashboard.py` always serves invented data on loopback.

## Sanitization and verification

The extraction replaced production resource defaults with placeholders, anonymized an authentication email default and review tests, retained the disabled `config/podcasts.json` and example OpenRouter lock, and preserved public-only synthetic demos and media.

`scripts/check_public_release.py` rejects known credential patterns, internal/private paths, real resource identifiers and personalized defaults from all tracked text files; `Portfolio CI` also verifies hash-locked dependencies and the public application suite. Human review of the patch remains mandatory: a pattern-based scanner cannot prove that every sensitive datum is absent.

The production transcript pipeline is **not** publicly runnable with the committed placeholders. Readers can run the synthetic Human Review, PodcastOps and knowledge-note demos without production credentials. A private Main acceptance success never substitutes for public PR CI or public merge acceptance.

For further context: [Architecture](ARCHITECTURE.md), [Knowledge generation](KNOWLEDGE_GENERATION.md), [Synthetic PodcastOps demo](DEMO_PODCASTOPS.md), and [Security and privacy](SECURITY_AND_PRIVACY.md).
