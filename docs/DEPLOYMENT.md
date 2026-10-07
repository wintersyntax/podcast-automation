# Deployment notes

The production design uses separate deployable components:

- a Cloud Run Job for the main worker;
- a Cloud Run service for Human Review;
- a storage-finalize function for transcript ingest;
- an optional small Slack acknowledgement function;
- optional macOS launch agents for credential maintenance and local synchronization.

The checked-in Cloud Build files and launchd examples are templates. Replace placeholder project, bucket, account and filesystem values before use.

## Minimum configuration

At a minimum, a real deployment needs:

- a configured podcast RSS feed;
- a GCS bucket;
- Google Cloud credentials with narrowly scoped permissions;
- a guarded OpenRouter credential for the Whisper producer;
- Apple transcript acquisition configuration if that source is enabled;
- reviewer/knowledge model configuration if AI-assisted stages are enabled;
- a Third-ASR credential if automatic review evidence prefetch is enabled;
- a verified materiality-judge preset if materiality shadow/queue behavior is enabled.

The worker-side Third-ASR and materiality stages are optional and fail closed: missing credentials or verified preset evidence must not become an implicit transcript decision.

Do not treat the example values in this portfolio repository as production defaults.
