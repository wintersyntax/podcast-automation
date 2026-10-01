# Synthetic knowledge-note demo

This demo shows the final knowledge artifact produced by the podcast automation pipeline using only invented exercise/nutrition content.

It deliberately demonstrates the domain-specific output that matters for this project:

- research/studies with design, sample, duration and reported result;
- evidence-strength labels;
- exercise execution and technique;
- practical nutrition/training protocols;
- numerical values and protocol tables;
- takeaways, topics, tags and sources mentioned.

The demo runs the synthetic note through the production `knowledge-note-v3` schema, transcript-anchor checks, deterministic Markdown renderer, and frontmatter builder. It does not call a model or any cloud service.

## Evidence-basis legend

The visual meter records the **stated basis of the claim in the episode**, not an independent truth or study-quality score:

- **▰▰▰ Study** — research/study/review/paper cited by the speaker.
- **▰▰▱ Coaches** — coaching experience across multiple clients/athletes or reported expert consensus.
- **▰▱▱ One person** — personal experience from one person.
- **▱▱▱ Opinion** — speculation/hypothesis without another stated basis.
- **· unsure** — the speaker hedged the claim.
- No label — descriptive material that does not assert or recommend something.

A Study label does not mean the underlying research was independently verified by the pipeline.

## Run

```bash
python scripts/demo_knowledge_note.py
```

Open:

```text
http://127.0.0.1:8766
```

The script also writes the exact generated Markdown to:

```text
examples/demo-knowledge-note.md
```

## Suggested GIF

Record roughly 12-15 seconds:

1. start at the title, tags and TL;DR;
2. scroll through **Protein intake and timing** and its evidence labels;
3. pause briefly on **Research discussed** so the study design/sample/duration/result are readable;
4. continue through **Romanian deadlift execution**;
5. finish on **Numbers & protocols** / **Takeaways** / **Sources mentioned**.

The purpose is to show the useful final artifact, not every internal pipeline step.
