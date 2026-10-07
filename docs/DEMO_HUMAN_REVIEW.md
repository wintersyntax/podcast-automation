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

## Suggested GIF flow

The synthetic episode intentionally contains two domain-specific conflicts:

1. **Study sample size:** Apple says `42 participants`; Whisper says `40 participants`.
2. **Exercise name:** Apple says `Romanian deadlift`; Whisper says `Roman deadlift`.

For a compact portfolio recording:

1. start on the study-number conflict;
2. briefly show the two transcript sources and surrounding context;
3. optionally click **Run third ASR** to reveal an additional evidence source;
4. choose the supported source and click **Confirm selection**;
5. show progress moving to the exercise-name conflict;
6. choose the supported source and confirm it;
7. finish on **Human review complete** and click **Recompile & continue**, leaving the final frame on **Worker started**.

The demo backend is deliberately in-memory. It reuses the production Human Review frontend (`podcast_engine.review_web.PAGE`) but never contacts Google Cloud, OpenRouter, Apple, Slack, or a real podcast feed.
