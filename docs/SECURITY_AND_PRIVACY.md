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

## Updating this public extraction

The private source tree is not a safe publishable archive. Newer modules, tests, runtime configuration and production evidence require **individual review and sanitization** before publication. A private CI pass does not certify public disclosure safety, and the public workflow tests only the code committed here. Demo media must use invented episodes and examples; production review screenshots, real transcript spans and unredacted operations receipts are intentionally out of scope.

## October 2026 extraction safeguards

The public snapshot was refreshed against private source commit
`72677eee1e9e8059490d367edb41a9ffebe7b11f` on 31 October 2026.
Defaults for cloud resource identity, review URL and allowed review email
were replaced with examples. The private production configuration and
CI runner tooling were not copied. The public extraction preserves only
fictional fixtures, demonstrations and hand-reviewed code/tests.

`python scripts/check_public_release.py` scans tracked files for known
credential/token formats, real deployment identifiers, Mac user paths,
personal email domains and forbidden private-development/evidence paths.
It is a fail-closed **backstop**, not a guarantee that every kind of secret
has been detected. Review every future upstream patch for disclosure risk.

The October 31 preparation also runs Gitleaks 8.30.1 with its default rules.
`.gitleaks.toml` allows only one reviewed, sequential synthetic episode key in
two specific test files; it does not allow credentials or whole test directories.
Scan reports contain redacted findings only.

A fresh root commit has no private-history parents. Replacing the public branch
does not purge GitHub's read-only pull-request refs, cached commit views, forks
or existing clones. Those require separate reconciliation; publication must not
be described as complete removal of every server-side historical object.
