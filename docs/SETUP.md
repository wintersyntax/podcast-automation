# Quick start

This public portfolio repository ships with disabled/example podcast configuration and no production credentials.

1. Create a Python virtual environment.
2. Install the hash-locked dependencies from `requirements.txt`.
3. Copy `.env.example` to an untracked `.env` and configure only the services you intend to use.
4. Replace `config/podcasts.json` with your own feed configuration or adapt `config/podcasts.example.json`.
5. Configure `PODCAST_GCS_BUCKET` and Google Application Default Credentials for cloud-backed operation.
6. Configure model credentials only for AI-assisted stages you explicitly enable.
7. Run the public tests and publication check before attempting a worker execution: `python scripts/check_public_release.py` and `python -m unittest discover -s tests -p 'test_*.py'`.

The repository intentionally does not contain a ready-to-use production OpenRouter preset identity. Replace the example preset lock with verified versioned presets for any AI stage you enable, including the single-pass `PODCAST_KNOWLEDGE_WRITER_PRESET`. The older summary/metadata/reviewer presets were retired and are not configuration options for this release.

For a credential-free introduction, run the synthetic demos described in [Human Review](DEMO_HUMAN_REVIEW.md), [PodcastOps](DEMO_PODCASTOPS.md) and [Knowledge notes](DEMO_KNOWLEDGE_NOTE.md).

See [Architecture](ARCHITECTURE.md), [Deployment](DEPLOYMENT.md), and [Security and privacy](SECURITY_AND_PRIVACY.md) for the system boundaries.
