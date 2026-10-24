"""Shared, bounded Cloud Run worker invocation."""

from __future__ import annotations

import os
import time
from collections.abc import Iterable, Mapping

import requests
from google.auth import default as google_auth_default
from google.auth.exceptions import TransportError
from google.auth.transport.requests import AuthorizedSession


GOOGLE_CLOUD_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_TIMEOUT_SECONDS = 10
RETRY_BASE_DELAY_SECONDS = 0.5
RECOMPILE_REQUEST_ID_ENV = "PODCAST_RECOMPILE_REQUEST_ID"
REVIEW_GENERATION_ENV = "PODCAST_REVIEW_GENERATION"
RECOMPILE_EPISODE_KEY_ENV = "PODCAST_RECOMPILE_EPISODE_KEY"
CORRELATION_ENV_NAMES = (
    RECOMPILE_REQUEST_ID_ENV,
    REVIEW_GENERATION_ENV,
    RECOMPILE_EPISODE_KEY_ENV,
)


class WorkerRunRequestError(RuntimeError):
    """Raised when the worker request did not produce a known execution."""

    def __init__(self, message: str, *, definitely_not_accepted: bool):
        super().__init__(message)
        self.definitely_not_accepted = definitely_not_accepted


class WorkerExecutionInspectionError(RuntimeError):
    """Raised when Cloud Run executions cannot be inspected safely."""


def _retry_delay(attempt: int) -> float:
    return RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1))


def _worker_resource_name() -> str:
    project = os.environ.get("WORKER_PROJECT", "YOUR_GCP_PROJECT_ID")
    region = os.environ.get("WORKER_REGION", "europe-west1")
    job_name = os.environ.get("WORKER_JOB_NAME", "podcast-worker")
    return f"projects/{project}/locations/{region}/jobs/{job_name}"


def _authenticated_session() -> AuthorizedSession:
    try:
        credentials, _ = google_auth_default(scopes=[GOOGLE_CLOUD_SCOPE])
        return AuthorizedSession(credentials)
    except Exception as exc:
        raise WorkerExecutionInspectionError(
            "Could not initialize authenticated Cloud Run client"
        ) from exc


def _override_body(correlation: Mapping[str, str] | None) -> dict:
    """Build the narrow, non-secret per-execution override body."""

    if correlation is None:
        return {}
    expected = {"request_id", "review_generation", "episode_key"}
    if set(correlation) != expected or not all(
        isinstance(value, str) and value for value in correlation.values()
    ):
        raise ValueError("Worker correlation must contain non-empty exact identifiers")
    return {
        "overrides": {
            "containerOverrides": [
                {
                    "env": [
                        {"name": RECOMPILE_REQUEST_ID_ENV, "value": correlation["request_id"]},
                        {"name": REVIEW_GENERATION_ENV, "value": correlation["review_generation"]},
                        {"name": RECOMPILE_EPISODE_KEY_ENV, "value": correlation["episode_key"]},
                    ]
                }
            ]
        }
    }


def request_worker_run(
    *,
    requested_by: str,
    episode_key: str | None = None,
    correlation: Mapping[str, str] | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict:
    """Request one normal Cloud Run Job execution.

    A successful 2xx response is the only accepted execution.  HTTP 4xx
    responses definitely reject the request (429 may be retried); transport
    failures and 5xx responses are ambiguous because Cloud Run could have
    accepted the request before the response was lost.
    """

    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    url = f"https://run.googleapis.com/v2/{_worker_resource_name()}:run"
    body = _override_body(correlation)

    try:
        session = _authenticated_session()
    except WorkerExecutionInspectionError as exc:
        raise WorkerRunRequestError(
            "Could not initialize authenticated Cloud Run client",
            definitely_not_accepted=True,
        ) from exc

    for attempt in range(1, max_attempts + 1):
        try:
            response = session.post(url, json=body, timeout=timeout_seconds)
        except (requests.RequestException, TransportError) as exc:
            raise WorkerRunRequestError(
                "Cloud Run worker request outcome is unknown after transport failure",
                definitely_not_accepted=False,
            ) from exc

        status_code = int(getattr(response, "status_code", 0) or 0)
        if 200 <= status_code <= 299:
            try:
                payload = response.json()
            except Exception:
                payload = {}
            operation = payload.get("name") if isinstance(payload, dict) else None
            return {
                "operation": operation,
                "attempts": attempt,
                "requested_by": requested_by,
                "episode_key": episode_key,
            }

        if 400 <= status_code <= 499:
            if status_code == 429 and attempt < max_attempts:
                time.sleep(_retry_delay(attempt))
                continue
            raise WorkerRunRequestError(
                f"Cloud Run worker request was rejected with HTTP {status_code}",
                definitely_not_accepted=True,
            )

        raise WorkerRunRequestError(
            f"Cloud Run worker request outcome is unknown after HTTP {status_code}",
            definitely_not_accepted=False,
        )

    raise WorkerRunRequestError(
        "Cloud Run worker request exhausted retry policy",
        definitely_not_accepted=True,
    )


def list_worker_executions(*, timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS) -> list[dict]:
    """List actual executions for the configured worker Job, read-only."""

    session = _authenticated_session()
    url = f"https://run.googleapis.com/v2/{_worker_resource_name()}/executions"
    executions: list[dict] = []
    page_token: str | None = None
    while True:
        params: dict[str, str | int] = {"pageSize": 100}
        if page_token:
            params["pageToken"] = page_token
        try:
            response = session.get(url, params=params, timeout=timeout_seconds)
        except (requests.RequestException, TransportError) as exc:
            raise WorkerExecutionInspectionError(
                "Could not inspect Cloud Run executions"
            ) from exc
        status_code = int(getattr(response, "status_code", 0) or 0)
        if not 200 <= status_code <= 299:
            raise WorkerExecutionInspectionError(
                f"Cloud Run execution inspection failed with HTTP {status_code}"
            )
        try:
            payload = response.json()
        except Exception as exc:
            raise WorkerExecutionInspectionError(
                "Cloud Run execution inspection returned invalid JSON"
            ) from exc
        if not isinstance(payload, dict) or not isinstance(
            payload.get("executions", []), list
        ):
            raise WorkerExecutionInspectionError(
                "Cloud Run execution inspection returned an invalid response"
            )
        executions.extend(
            execution
            for execution in payload.get("executions", [])
            if isinstance(execution, dict)
        )
        next_token = payload.get("nextPageToken")
        if not isinstance(next_token, str) or not next_token:
            return executions
        page_token = next_token


def execution_matches_correlation(
    execution: Mapping[str, object], *, request_id: str, review_generation: str, episode_key: str
) -> bool:
    """Return whether one execution contains precisely the reserved identifiers.

    Only the execution's container environment is considered.  In particular,
    timestamps, creator metadata, and job name are deliberately ignored.
    """

    template = execution.get("template")
    if not isinstance(template, Mapping):
        return False
    containers = template.get("containers")
    if not isinstance(containers, list):
        spec = template.get("spec")
        containers = spec.get("containers") if isinstance(spec, Mapping) else None
    if not isinstance(containers, list):
        return False
    expected = {
        RECOMPILE_REQUEST_ID_ENV: request_id,
        REVIEW_GENERATION_ENV: review_generation,
        RECOMPILE_EPISODE_KEY_ENV: episode_key,
    }
    found: dict[str, list[str]] = {name: [] for name in expected}
    for container in containers:
        if not isinstance(container, Mapping):
            continue
        env = container.get("env")
        if not isinstance(env, list):
            continue
        for item in env:
            if not isinstance(item, Mapping):
                continue
            name, value = item.get("name"), item.get("value")
            if name in expected and isinstance(value, str):
                found[name].append(value)
    return all(found[name] == [value] for name, value in expected.items())


def reconciled_execution_status(execution: Mapping[str, object]) -> str:
    """Map Cloud Run execution status to the durable recompile state machine."""

    conditions = execution.get("conditions")
    if isinstance(conditions, Iterable) and not isinstance(conditions, (str, bytes)):
        for condition in conditions:
            if not isinstance(condition, Mapping) or condition.get("type") != "Completed":
                continue
            state = str(condition.get("state", "")).upper()
            if state in {"CONDITION_SUCCEEDED", "SUCCEEDED", "SUCCESS"}:
                return "completed"
            if state in {"CONDITION_CANCELLED", "CANCELLED", "CANCELED"}:
                return "cancelled"
            if state in {"CONDITION_FAILED", "FAILED"}:
                if str(condition.get("executionReason", "")).upper() in {
                    "CANCELLED",
                    "CANCELED",
                }:
                    return "cancelled"
                return "execution_failed"
    return "started"
