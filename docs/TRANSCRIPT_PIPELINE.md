# Transcript pipeline

The transcript pipeline is designed around disagreement preservation rather than choosing whichever transcript reads more fluently.

1. Discover the latest configured RSS episode and resume older incomplete episodes.
2. Download the audio to ephemeral worker storage.
3. Produce a Whisper transcript in bounded chunks with timestamp-preserving overlap handling.
4. Acquire an independent Apple transcript when available.
5. Store source artifacts separately with provenance.
6. Align and compare the two sources.
7. Apply deterministic equivalence and corroboration rules where they are safe.
8. Route eligible bounded conflicts through a constrained resolver.
9. Send unresolved material conflicts to Human Review.
10. Recompile only after review decisions become durable.

Typical high-risk disagreements include numbers, units, negations, names, citations and domain terminology.

The compiler intentionally prefers unresolved evidence over an unsupported correction.
