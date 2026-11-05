"""warn before an OpenRouter key the Worker uses expires.

Each scheduled run asks OpenRouter (``GET /api/v1/key``, free metadata)
when each configured key expires:

* ``openrouter_key_expiring`` (WARNING) while ``KEY_WARNING_DAYS`` or fewer
  days remain;
* ``openrouter_key_invalid`` (ERROR) when OpenRouter rejects the key (expired,
  disabled or deleted);
* ``openrouter_key_check_unavailable`` (INFO) when the check itself cannot run.

A key with ``KEY_ALERT_DAYS`` or fewer days left, or a rejected key, makes the
scheduled run end failed after every episode was processed, so the existing
execution-failed alert fires (as for long provider deferrals). The check never
stops or changes episode work and never logs a key; only the environment
variable name and OpenRouter's own key label are reported.
"""

from __future__ import annotations

from datetime import datetime, timezone
import os
from typing import Callable, Mapping

import requests

from .observability import emit_event

OPENROUTER_KEY_URL = "https://openrouter.ai/api/v1/key"
KEY_ENVS = (
    "PODCAST_WHISPER_API_KEY",
    "PODCAST_KNOWLEDGE_API_KEY",
    "PODCAST_TRANSCRIPT_REVIEW_API_KEY",
    "PODCAST_REVIEW_ASR_API_KEY",
)
KEY_WARNING_DAYS = 14
KEY_ALERT_DAYS = 7
REQUEST_TIMEOUT_SECONDS = 10


class OpenRouterKeysExpiring(RuntimeError):
    """The run finished, but an OpenRouter key expires soon or is rejected."""


def _parse_expiry(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _key_status(key: str, *, now: datetime, get: Callable[..., requests.Response]) -> dict:
    response = get(
        OPENROUTER_KEY_URL,
        headers={"Authorization": f"Bearer {key}"},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    if response.status_code in (401, 403):
        return {"status": "invalid", "http_status": response.status_code}
    response.raise_for_status()
    data = (response.json() or {}).get("data") or {}
    label = data.get("label") if isinstance(data.get("label"), str) else None
    expires_at = _parse_expiry(data.get("expires_at"))
    if expires_at is None:
        return {"status": "no_expiry", "label": label}
    remaining = (expires_at - now).total_seconds() / 86400
    status = "expired" if remaining <= 0 else "expiring" if remaining <= KEY_WARNING_DAYS else "ok"
    return {
        "status": status,
        "label": label,
        "expires_at": expires_at.isoformat().replace("+00:00", "Z"),
        "remaining_days": round(remaining, 1),
    }


def check_openrouter_keys(
    *,
    environ: Mapping[str, str] | None = None,
    now: datetime | None = None,
    get: Callable[..., requests.Response] = requests.get,
) -> list[dict]:
    """Check every configured OpenRouter key; emit events; never raise."""

    environ = os.environ if environ is None else environ
    now = now or datetime.now(timezone.utc)
    results = []
    for env in KEY_ENVS:
        key = (environ.get(env) or "").strip()
        if not key:
            continue
        try:
            result = {"env": env, **_key_status(key, now=now, get=get)}
        except Exception as error:  # noqa: BLE001 - a warning check must never stop the run
            result = {"env": env, "status": "unavailable"}
            emit_event(
                "openrouter_key_check_unavailable",
                "Could not read an OpenRouter key's expiry",
                severity="INFO",
                env=env,
                error_type=type(error).__name__,
            )
            results.append(result)
            continue
        if result["status"] in ("invalid", "expired"):
            emit_event(
                "openrouter_key_invalid",
                "OpenRouter rejects or has expired a key the Worker uses; replace it",
                severity="ERROR",
                **{k: v for k, v in result.items() if k != "status"},
            )
        elif result["status"] == "expiring":
            emit_event(
                "openrouter_key_expiring",
                "An OpenRouter key the Worker uses expires soon; replace it before then",
                severity="WARNING",
                **{k: v for k, v in result.items() if k != "status"},
            )
        results.append(result)
    return results


def keys_needing_alert(results: list[dict]) -> list[dict]:
    """Keys that should end the scheduled run failed: rejected, or due within KEY_ALERT_DAYS."""

    return [
        result
        for result in results
        if result["status"] in ("invalid", "expired")
        or (result["status"] == "expiring" and result["remaining_days"] <= KEY_ALERT_DAYS)
    ]
