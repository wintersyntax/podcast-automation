# Slack ACK Cloud Function

`podcast-slack-ack` is a deliberately small, public Gen2 HTTP function for
Slack interactive-component callbacks. It authenticates Slack's signed raw
request body and immediately returns an empty HTTP `200`. It does not parse or
act on the interaction payload, call Google APIs, or import the Human Review
application. Keeping this source independent protects the verified cold-start
behaviour from the review service's dependency graph.

## Contract

- Entry point: `slack_interactions` in `main.py`
- Request: an HTTP request with `X-Slack-Request-Timestamp` and
  `X-Slack-Signature`
- Secret environment variable:
  `PODCAST_SLACK_SIGNING_SECRET`
- Accepted requests: `200` with an empty body
- Missing secret: `503` with an empty body
- Missing, invalid, or older-than-five-minutes signature: `401` with an empty
  body

The function verifies the exact raw request bytes. It must not read form or
JSON data before verification, because reparsing can change the signed body.

## Controlled release

Use the repository-owned script as the normal release path. It deploys the
existing Gen2 function from this source directory, binds the dedicated build
and runtime identities, and updates only the signing-secret mapping. It omits
runtime, minimum-instance, and public-invoker flags so the verified live
settings are retained.

First perform its local preflight:

```bash
scripts/deploy_slack_ack.sh --dry-run
```

`--dry-run` prints the exact intended command and does not contact or mutate
Google Cloud. Run `scripts/deploy_slack_ack.sh` only after a separate release
authorisation. The script never changes IAM or Slack settings.

Before any release, run:

```bash
PYTHONDONTWRITEBYTECODE=1 ../venv/bin/python -m unittest \
  tests.test_slack_ack_function -v
```

After an authorised release, validate only with a locally generated signed
request or an approved Slack test interaction. Never put the signing secret in
the command line, logs, test fixtures, or source control.
