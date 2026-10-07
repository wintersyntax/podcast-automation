# Human Review portfolio demo

The repository includes a local synthetic launcher for recording or exploring the real Human Review frontend without using production data or cloud credentials.

## Run

```bash
python scripts/demo_human_review.py
```

Open:

```text
http://127.0.0.1:8765
```

Restart the process to reset the demo.

## What the synthetic episode demonstrates

The fixture mirrors the current materiality-first Human Review flow with six invented cards:

1. **Control sample** — a filter-settled contraction/wording difference that the reviewer explicitly checks.
2. **Click one** — an immaterial wording difference where either reading is acceptable and no evidence should choose for the reviewer.
3. **Confirm the proposal** — a domain-term conflict where synthetic Third-ASR evidence supports one reading.
4. **Two settled cards** — immaterial differences accepted together only after the control sample is saved.
5. **Protected full review** — a study sample-size conflict, **42 vs 40 participants**, with bounded Third-ASR evidence already present.

Every save still creates explicit synthetic human decisions. The demo backend never lets the materiality filter write canonical text on its own.

## README GIF: materiality-first flow

Target length: roughly 20–30 seconds.

1. Start on **Control sample** and keep the preselected Apple reading.
2. Advance through **Click one** and choose either source.
3. Show **Confirm the proposal** with the Third-ASR-backed Apple proposal and keep it.
4. Open **Check & save**, briefly show the staged list, then save the three decisions.
5. In **Settled by the filter**, optionally expand **Show the list**, then click **Accept 2**.
6. Open the remaining protected study-number card in **Full review**.
7. Show the Apple/Whisper disagreement, context, audio and cached Third-ASR evidence; choose **Apple** and confirm.
8. Finish on **Human review complete** and click **Recompile & continue**, leaving the last frame on **Worker started**.

The resulting asset should replace:

```text
assets/demo-human-review.gif
```

## Detailed evidence GIF

A second, shorter GIF can focus on the authority boundary rather than the whole queue:

1. progress to the protected **42 vs 40 participants** card;
2. show the two transcript readings and surrounding context;
3. show the audio control and cached Third-ASR evidence;
4. choose Apple explicitly and confirm the selection.

Suggested asset:

```text
assets/demo-human-review-evidence.gif
```

This second GIF is useful in `docs/HUMAN_REVIEW.md` because it makes the central rule visible: Third ASR and materiality evidence can help the reviewer, but only the explicit human choice becomes canonical.

## Safety

The demo backend is deliberately in-memory. It reuses the production Human Review frontend (`podcast_engine.review_web.PAGE`) but never contacts Google Cloud, OpenRouter, Apple, Slack, or a real podcast feed. All podcast names, transcript text, evidence and identifiers in the demo are synthetic.
