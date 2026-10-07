# Transcript pipeline

The transcript pipeline is designed around disagreement preservation rather than choosing whichever transcript reads more fluently.

1. Discover the latest configured RSS episode and resume older incomplete episodes.
2. Download the audio to ephemeral worker storage.
3. Produce the logical Whisper source through OpenRouter `qwen/qwen3-asr-1.7b` in bounded MP3 chunks with word/segment timestamps, explicit producer provenance, AI-budget reservations and bounded HTTP 429 backoff. The parser normalizes Qwen's segment-join timestamp backsteps only within a strict bound and restores word boundaries when provider words omit leading spaces; larger timestamp regressions still fail closed.
4. Acquire an independent Apple transcript when available.
5. Store source artifacts separately with provenance.
6. Align and compare the two sources.
7. Apply deterministic equivalence and corroboration rules where they are safe, including representation-only spacing/hyphenation equivalence that does not collapse semantically different words.
8. Route eligible bounded conflicts through the constrained resolver.
9. For remaining review cards, optionally prefetch bounded Third-ASR evidence and record materiality evidence. These stages are advisory and do not select canonical text.
10. Send unresolved material conflicts to Human Review.
11. Recompile only after review decisions become durable.

Typical high-risk disagreements include values, units, negations, names, citations and domain terminology. Long source-only stretches and ambiguous anchoring also stay conservative.

The compiler intentionally prefers unresolved evidence over an unsupported correction.
