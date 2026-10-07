#!/usr/bin/env python3
"""Maintain the Cloud Run Apple bearer token from macOS only.

This is deliberately independent of the RSS and transcript watchers.  It
checks the local token cache, refreshes it with the macOS-only FetchTranscript
helper when five days or less remain, probes Apple's catalogue API without any
GCS write, then adds a Secret Manager version after that probe succeeds.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from dotenv import load_dotenv

from podcast_engine.apple_token_status import inspect_bearer_token_file
from podcast_engine.apple_transcript import DEFAULT_STOREFRONT, resolve_episode_id


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
ENV_PATH = REPO_ROOT / ".env"
DEFAULT_FETCHER = SCRIPT_DIR / "bin" / "FetchTranscript"
RUNTIME_DIR = Path(
    os.environ.get(
        "PODCAST_PIPELINE_APPLE_RUNTIME_DIR",
        Path.home()
        / "Library"
        / "Application Support"
        / "podcast-pipeline"
        / "apple-api",
    )
)
FETCHER_PATH = Path(
    os.environ.get(
        "PODCAST_PIPELINE_APPLE_FETCHER",
        DEFAULT_FETCHER,
    )
)
TOKEN_PATH = RUNTIME_DIR / "bearer_token.txt"
TOKEN_SEED_EPISODE_ID = os.environ.get(
    "PODCAST_PIPELINE_APPLE_TOKEN_SEED_EPISODE_ID",
    "1000784615533",
)
PROBE_SHOW_ID = os.environ.get(
    "PODCAST_PIPELINE_APPLE_PROBE_SHOW_ID",
    "1452114380",
)
PROBE_STOREFRONT = os.environ.get(
    "PODCAST_PIPELINE_APPLE_PROBE_STOREFRONT",
    DEFAULT_STOREFRONT,
)

GCP_PROJECT = "YOUR_GCP_PROJECT_ID"
SECRET_NAME = "apple-podcasts-bearer-token"
REFRESH_THRESHOLD_SECONDS = 5 * 24 * 60 * 60
HELPER_TIMEOUT_SECONDS = 90
HEARTBEAT_TIMEOUT_SECONDS = 15


def _log(
    event: str,
    message: str,
    *,
    severity: str = "INFO",
    error: bool = False,
    **fields: object,
) -> None:
    """Emit a secret-free structured event for local logs and monitors."""

    payload = {
        "severity": severity,
        "event": event,
        "message": message,
        **fields,
    }
    print(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        file=sys.stderr if error else sys.stdout,
        flush=True,
    )


def _read_local_token(*, now: float | None = None) -> tuple[str | None, dict]:
    """Read the existing macOS token path without exposing its contents."""

    return inspect_bearer_token_file(TOKEN_PATH, now=now)


def load_repo_env() -> None:
    """Load uncommitted local configuration without replacing launchd values."""

    load_dotenv(ENV_PATH, override=False)


def token_needs_refresh(
    status: dict,
    *,
    threshold_seconds: int = REFRESH_THRESHOLD_SECONDS,
) -> bool:
    """Return whether a missing, invalid, expired, or near-expiry token refreshes."""

    return (
        status["status"] != "valid"
        or status["remaining_seconds"] <= threshold_seconds
    )


def _refresh_with_helper() -> tuple[bool, str | None]:
    """Obtain a genuinely new token from the helper and install it atomically.

    The upstream helper's ``--cache-bearer-token`` mode reuses any
    ``bearer_token.txt`` in its working directory whose *file* is younger than
    30 days, whatever the JWT's own expiry says. Running it next to the
    existing cache therefore returned the same near-expiry (and later
    already-expired) token on every attempt (TASK-116). The helper now always
    runs in a new, empty directory, so it must fetch a new credential. Its
    later transcript download may still fail; a new token it already wrote is
    accepted on its merits, because ``maintain_token`` probes Apple before any
    Secret Manager update. The installed token is never replaced by one that
    expires sooner, and it is left untouched on every failure path.
    """
    if not FETCHER_PATH.is_file() or not os.access(FETCHER_PATH, os.X_OK):
        return False, "FetchTranscript helper is missing or not executable."
    try:
        RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        work_dir = Path(tempfile.mkdtemp(prefix=".refresh-", dir=RUNTIME_DIR))
    except OSError:
        return False, "Could not prepare an isolated helper directory."
    try:
        try:
            result = subprocess.run(
                [
                    str(FETCHER_PATH),
                    TOKEN_SEED_EPISODE_ID,
                    "--cache-bearer-token",
                ],
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=HELPER_TIMEOUT_SECONDS,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False, "FetchTranscript helper could not be run."
        candidate_path = work_dir / TOKEN_PATH.name
        candidate, candidate_status = inspect_bearer_token_file(candidate_path)
        if not candidate or candidate_status["status"] != "valid":
            if result.returncode != 0:
                return False, f"FetchTranscript helper failed with exit {result.returncode}."
            return False, "FetchTranscript helper did not write a usable token."
        _, current_status = inspect_bearer_token_file(TOKEN_PATH)
        if (
            current_status["status"] == "valid"
            and candidate_status["expires_at"] <= current_status["expires_at"]
        ):
            return False, "FetchTranscript helper did not return a newer token."
        try:
            candidate_path.chmod(0o600)
            os.replace(candidate_path, TOKEN_PATH)
        except OSError:
            return False, "Could not install the refreshed token."
        if result.returncode != 0:
            _log(
                "apple_token_helper_partial",
                "Helper wrote a new token but its seed transcript step failed.",
                severity="WARNING",
                exit_code=result.returncode,
            )
        return True, None
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def probe_apple_api(token: str) -> tuple[bool, str]:
    """Perform one authenticated catalogue request and never write to GCS."""

    result = resolve_episode_id(
        token,
        PROBE_SHOW_ID,
        rss_guid=None,
        title=None,
        storefront=PROBE_STOREFRONT,
    )

    if result.status == "API_ERROR":
        return False, "Apple API probe failed."

    # No GUID or title was supplied, so a successful catalogue response is
    # intentionally reported as EPISODE_NOT_READY rather than READY.
    return True, "Apple API probe succeeded."


def upload_secret_version(token: str) -> tuple[bool, str | None]:
    """Add, but never disable or destroy, one Secret Manager version."""

    try:
        result = subprocess.run(
            [
                "gcloud",
                "secrets",
                "versions",
                "add",
                SECRET_NAME,
                f"--project={GCP_PROJECT}",
                "--data-file=-",
            ],
            input=token,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False, "gcloud could not add the Secret Manager version."

    if result.returncode != 0:
        return False, f"gcloud secret update failed with exit {result.returncode}."

    return True, None


def send_success_heartbeat() -> bool:
    """Ping the configured monitor only after a complete successful run."""

    url = os.environ.get("PODCAST_TOKEN_MAINTENANCE_HEARTBEAT_URL", "").strip()

    if not url:
        return True

    try:
        result = subprocess.run(
            ["curl", "--fail", "--silent", "--show-error", "--max-time", str(HEARTBEAT_TIMEOUT_SECONDS), url],
            capture_output=True,
            text=True,
            timeout=HEARTBEAT_TIMEOUT_SECONDS + 5,
        )
    except (OSError, subprocess.TimeoutExpired):
        _log(
            "apple_token_heartbeat_failed",
            "Token maintenance succeeded but its heartbeat could not be sent.",
            severity="WARNING",
            error=True,
        )
        return False

    if result.returncode != 0:
        _log(
            "apple_token_heartbeat_failed",
            "Token maintenance succeeded but its heartbeat was rejected.",
            severity="WARNING",
            error=True,
            exit_code=result.returncode,
        )
        return False

    return True


def _finish_success(event: str, message: str, **fields: object) -> int:
    _log(event, message, **fields)
    send_success_heartbeat()
    return 0


def maintain_token(
    *,
    now: float | None = None,
    force: bool = False,
) -> int:
    """Refresh, probe and publish only within the five-day window or on request."""

    current_time = time.time() if now is None else now
    load_repo_env()
    token, status = _read_local_token(now=current_time)

    needs_refresh = force or token_needs_refresh(status)

    if not needs_refresh:
        days = status["remaining_seconds"] / 86_400
        return _finish_success(
            "apple_token_valid",
            "Local Apple token remains valid; no update needed.",
            remaining_days=round(days, 1),
        )

    _log(
        "apple_token_expiring" if status["status"] == "valid" else "apple_token_invalid",
        "Refreshing local Apple token before its cloud use.",
        severity="WARNING" if status["status"] == "valid" else "ERROR",
        status=status["status"],
        forced=force,
    )
    refreshed, error = _refresh_with_helper()

    if not refreshed:
        _log(
            "apple_token_invalid",
            error or "Apple token refresh failed.",
            severity="ERROR",
            error=True,
        )
        return 1

    token, status = _read_local_token(now=current_time)

    if not token or token_needs_refresh(status):
        _log(
            "apple_token_invalid",
            "Helper did not produce a token valid beyond the five-day refresh window.",
            severity="ERROR",
            error=True,
        )
        return 1

    probe_ok, probe_message = probe_apple_api(token)

    if not probe_ok:
        _log(
            "apple_acquisition_failed",
            probe_message,
            severity="ERROR",
            error=True,
        )
        return 1

    _log("apple_token_probe_succeeded", probe_message)
    uploaded, error = upload_secret_version(token)

    if not uploaded:
        _log(
            "apple_token_secret_update_failed",
            error or "Secret Manager update failed.",
            severity="ERROR",
            error=True,
        )
        return 1

    return _finish_success(
        "apple_token_secret_updated",
        "Added a validated Apple token version to Secret Manager.",
        secret=SECRET_NAME,
        project=GCP_PROJECT,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run the controlled refresh/probe/publish path even before expiry.",
    )
    args = parser.parse_args(argv)
    return maintain_token(force=args.force)


if __name__ == "__main__":
    sys.exit(main())
