# Quick start

This public portfolio repository ships with disabled/example podcast configuration and no production credentials.

1. Create a Python virtual environment.
2. Install the hash-locked dependencies from `requirements.txt`.
3. Copy `.env.example` to an untracked `.env` and configure only the services you intend to use.
4. Replace `config/podcasts.json` with your own feed configuration or adapt `config/podcasts.example.json`.
5. Configure `PODCAST_GCS_BUCKET` and Google Application Default Credentials for cloud-backed operation.
6. Configure model credentials only for AI-assisted stages you explicitly enable.
7. Run focused unit tests before attempting a worker execution.

The repository intentionally does not contain a ready-to-use production OpenRouter preset identity. Replace the example preset lock with a verified preset version if you enable the bounded transcript resolver.

See [Architecture](ARCHITECTURE.md), [Deployment](DEPLOYMENT.md), and [Security and privacy](SECURITY_AND_PRIVACY.md) for the system boundaries.
