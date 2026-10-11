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

## Published portfolio asset

The published visual is a **static PNG**, not a GIF:

```text
assets/demo-knowledge-note.png
```

The exact inspectable synthetic Markdown example is:

```text
examples/demo-knowledge-note.md
```

A still image is appropriate for this static final artifact: readers can inspect the screenshot at a glance and open the Markdown for all sections, citations-as-heard, evidence labels, technique guidance and numerical protocols. The repo currently ships **one Human Review GIF and one knowledge-note PNG**; no second GIF is required or claimed.
