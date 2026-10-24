"""Safe, platform-neutral inspection of an Apple bearer JWT.

This module only decodes the unverified JWT payload to learn its ``exp``
timestamp.  It never logs, returns, or validates the bearer token itself.
"""

from __future__ import annotations

import base64
import binascii
import json
import time
from pathlib import Path


def jwt_expiry(token: str) -> float | None:
    """Return an unverified JWT ``exp`` timestamp, if one is present."""

    try:
        payload = token.split(".")[1]
        padded = payload + "=" * (-len(payload) % 4)
        decoded = base64.urlsafe_b64decode(padded)
        return float(json.loads(decoded)["exp"])
    except (
        IndexError,
        KeyError,
        TypeError,
        ValueError,
        binascii.Error,
        json.JSONDecodeError,
    ):
        return None


def inspect_bearer_token(
    token: str | None,
    *,
    now: float | None = None,
) -> dict:
    """Describe token usability without including the token in the result."""

    current_time = time.time() if now is None else now

    if not token:
        return {"status": "missing"}

    expires_at = jwt_expiry(token)

    if expires_at is None:
        return {"status": "invalid"}

    remaining_seconds = expires_at - current_time

    if remaining_seconds <= 0:
        return {
            "status": "expired",
            "expires_at": expires_at,
            "remaining_seconds": remaining_seconds,
        }

    return {
        "status": "valid",
        "expires_at": expires_at,
        "remaining_seconds": remaining_seconds,
    }


def inspect_bearer_token_file(
    path: Path,
    *,
    now: float | None = None,
) -> tuple[str | None, dict]:
    """Read a token file and return the token plus a secret-free status."""

    try:
        token = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None, {"status": "missing"}

    return token, inspect_bearer_token(token, now=now)
