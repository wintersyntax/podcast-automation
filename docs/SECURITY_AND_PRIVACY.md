# Security and privacy

This portfolio repository intentionally excludes production secrets and deployment-specific identifiers.

## Not committed

- API keys or OAuth secrets
- Slack signing secrets or webhook URLs
- Apple bearer tokens
- production Google Cloud project or bucket identifiers
- personal account allowlists
- private local filesystem paths
- real podcast feed configuration from the private deployment
- transcript-derived evaluation corpora from real episodes

Configuration files in this repository use examples or placeholders. A real deployment should inject secrets through environment variables or a managed secret store and should use dedicated least-privilege service identities.

The Slack acknowledgement service verifies the signed raw request before parsing. The worker and review service keep notification failures non-authoritative: durable pipeline state is written independently of delivery success.
