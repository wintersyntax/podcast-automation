"""Authenticate and immediately acknowledge Slack interaction callbacks."""

from __future__ import annotations

import hashlib
import hmac
import os
import time

import functions_framework


MAX_REQUEST_AGE_SECONDS = 300


def _valid_slack_signature(
    raw_body: bytes,
    timestamp: str | None,
    signature: str | None,
    signing_secret: str,
    *,
    now: int | None = None,
) -> bool:
    """Return whether the request carries a fresh valid Slack v0 signature."""

    if not timestamp or not signature or not signing_secret:
        return False

    try:
        request_time = int(timestamp)
    except (TypeError, ValueError):
        return False

    current_time = int(time.time()) if now is None else int(now)
    if abs(current_time - request_time) > MAX_REQUEST_AGE_SECONDS:
        return False

    base = b"v0:" + timestamp.encode("utf-8") + b":" + raw_body
    digest = hmac.new(
        signing_secret.encode("utf-8"),
        base,
        hashlib.sha256,
    ).hexdigest()

    return hmac.compare_digest(f"v0={digest}", signature)


@functions_framework.http
def slack_interactions(request):
    """Authenticate a Slack interaction and return its required empty ACK."""

    signing_secret = os.environ.get(
        "PODCAST_SLACK_SIGNING_SECRET", ""
    ).strip()

    if not signing_secret:
        return ("", 503)

    raw_body = request.get_data(cache=True)

    if not _valid_slack_signature(
        raw_body,
        request.headers.get("X-Slack-Request-Timestamp"),
        request.headers.get("X-Slack-Signature"),
        signing_secret,
    ):
        return ("", 401)

    return ("", 200)
