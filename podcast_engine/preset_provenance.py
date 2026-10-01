"""Verified OpenRouter preset provenance for transcript resolver inference.

The resolver must never infer through a mutable ``@preset/<slug>`` reference.
Instead, this module verifies the exact version committed in the local lock and
returns that version's configuration and prompt for direct request composition.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import requests


OPENROUTER_PRESETS_URL = "https://openrouter.ai/api/v1/presets"
PRESET_TIMEOUT_SECONDS = 15
LOCK_SCHEMA_VERSION = 1
TRANSCRIPT_REVIEWER_LOCK_NAME = "transcript_reviewer"
LOCK_PATH = Path(__file__).resolve().parent.parent / "config" / "openrouter-preset-lock.json"


def canonical_json_bytes(value: Any) -> bytes:
    """Encode JSON deterministically for config identity.

    The entire OpenRouter version ``config`` object is hashed. Administrative
    version metadata is intentionally outside that object and never enters the
    config hash.
    """

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def config_sha256(config: dict) -> str:
    if not isinstance(config, dict):
        raise ValueError("OpenRouter preset config must be an object")
    return sha256(canonical_json_bytes(config))


def system_prompt_sha256(system_prompt: str) -> str:
    if not isinstance(system_prompt, str):
        raise ValueError("OpenRouter preset system_prompt must be a string")
    return sha256(system_prompt.encode("utf-8"))


def _safe_reason(value: object) -> str:
    """Return a compact error reason without remote bodies or prompt content."""

    return str(value).replace("\n", " ")[:160]


@dataclass(frozen=True)
class PresetProvenance:
    """A structured result which is safe to persist in resolver records."""

    status: str
    reason: str | None = None
    slug: str | None = None
    preset_id: str | None = None
    version_id: str | None = None
    version: int | None = None
    config: dict | None = None
    system_prompt: str | None = None
    config_digest: str | None = None
    system_prompt_digest: str | None = None
    verified_at: str | None = None

    @property
    def verified(self) -> bool:
        return self.status == "verified"

    def cache_identity(self) -> dict:
        """Return the exact identity which must participate in AI cache keys."""

        return {
            "status": self.status,
            "reason": self.reason,
            "preset_slug": self.slug,
            "preset_id": self.preset_id,
            "version_id": self.version_id,
            "version": self.version,
            "config_sha256": self.config_digest,
            "system_prompt_sha256": self.system_prompt_digest,
        }

    def record(self) -> dict:
        """Return durable provenance without exposing the system prompt."""

        payload = self.cache_identity()
        if self.verified_at is not None:
            payload["verified_at"] = self.verified_at
        return payload


def _result(status: str, reason: str, *, lock: dict | None = None) -> PresetProvenance:
    return PresetProvenance(
        status=status,
        reason=reason,
        slug=lock.get("slug") if isinstance(lock, dict) else None,
        preset_id=lock.get("preset_id") if isinstance(lock, dict) else None,
        version_id=lock.get("designated_version_id") if isinstance(lock, dict) else None,
        version=lock.get("version") if isinstance(lock, dict) else None,
        config_digest=lock.get("config_sha256") if isinstance(lock, dict) else None,
        system_prompt_digest=lock.get("system_prompt_sha256") if isinstance(lock, dict) else None,
    )


def load_transcript_reviewer_lock(path: Path = LOCK_PATH, *, local_policy_version: str) -> dict:
    """Load and validate the single authoritative committed reviewer lock."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("OpenRouter preset lock is unreadable") from error

    if not isinstance(payload, dict) or payload.get("schema_version") != LOCK_SCHEMA_VERSION:
        raise ValueError("Unsupported OpenRouter preset lock schema")
    lock = payload.get("presets", {}).get(TRANSCRIPT_REVIEWER_LOCK_NAME)
    if not isinstance(lock, dict):
        raise ValueError("Transcript reviewer lock is missing")

    required_strings = (
        "slug",
        "local_policy_version",
        "preset_id",
        "designated_version_id",
        "config_sha256",
        "system_prompt_sha256",
    )
    if any(not isinstance(lock.get(field), str) or not lock[field] for field in required_strings):
        raise ValueError("Transcript reviewer lock has invalid identity fields")
    if not isinstance(lock.get("version"), int) or lock["version"] < 1:
        raise ValueError("Transcript reviewer lock has invalid version")
    if not isinstance(lock.get("config"), dict):
        raise ValueError("Transcript reviewer lock has invalid config")
    if lock["local_policy_version"] != local_policy_version:
        raise ValueError("Transcript reviewer lock policy version does not match code")
    if config_sha256(lock["config"]) != lock["config_sha256"]:
        raise ValueError("Transcript reviewer lock config hash is inconsistent")
    return copy.deepcopy(lock)


def _envelope_data(response: requests.Response) -> dict:
    try:
        payload = response.json()
    except ValueError as error:
        raise ValueError("OpenRouter preset response is not JSON") from error
    data = payload.get("data", payload) if isinstance(payload, dict) else None
    if not isinstance(data, dict):
        raise ValueError("OpenRouter preset response is malformed")
    return data


def _get(url: str, api_key: str) -> dict:
    response = requests.get(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=PRESET_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return _envelope_data(response)


def _http_result(error: requests.HTTPError, lock: dict) -> PresetProvenance:
    status_code = error.response.status_code if error.response is not None else None
    if status_code in {401, 403}:
        return _result("unavailable", f"http_{status_code}_auth", lock=lock)
    if status_code == 404:
        return _result("invalid", "locked_preset_or_version_not_found", lock=lock)
    if status_code is not None and 500 <= status_code < 600:
        return _result("unavailable", f"http_{status_code}", lock=lock)
    return _result("invalid", f"http_{status_code or 'error'}", lock=lock)


def verify_transcript_reviewer(
    *,
    local_policy_version: str,
    preset_slug: str,
    api_key: str | None = None,
    verified_at: str | None = None,
    lock_path: Path = LOCK_PATH,
) -> PresetProvenance:
    """Verify the designated preset and exact locked version fail-safely."""

    try:
        lock = load_transcript_reviewer_lock(lock_path, local_policy_version=local_policy_version)
    except ValueError as error:
        return _result("invalid", _safe_reason(error))
    if preset_slug != lock["slug"]:
        return _result("invalid", "configured_preset_does_not_match_lock", lock=lock)

    key = api_key if api_key is not None else os.getenv("PODCAST_TRANSCRIPT_REVIEW_API_KEY")
    if not key:
        return _result("unavailable", "missing_api_key", lock=lock)

    try:
        current = _get(f"{OPENROUTER_PRESETS_URL}/{lock['slug']}", key)
        designated = current.get("designated_version")
        if not isinstance(designated, dict):
            return _result("invalid", "missing_designated_version", lock=lock)
        if (
            current.get("id") != lock["preset_id"]
            or current.get("slug") != lock["slug"]
            or current.get("designated_version_id") != lock["designated_version_id"]
            or designated.get("id") != lock["designated_version_id"]
            or designated.get("version") != lock["version"]
        ):
            return _result("drift", "designated_version_identity_changed", lock=lock)

        exact = _get(f"{OPENROUTER_PRESETS_URL}/{lock['slug']}/versions/{lock['version']}", key)
    except requests.HTTPError as error:
        return _http_result(error, lock)
    except (requests.Timeout, requests.ConnectionError):
        return _result("unavailable", "network_or_timeout", lock=lock)
    except ValueError as error:
        return _result("invalid", _safe_reason(type(error).__name__), lock=lock)
    except requests.RequestException as error:
        return _result("unavailable", _safe_reason(type(error).__name__), lock=lock)

    remote_config = exact.get("config")
    remote_prompt = exact.get("system_prompt")
    if (
        exact.get("id") != lock["designated_version_id"]
        or exact.get("preset_id") != lock["preset_id"]
        or exact.get("version") != lock["version"]
        or not isinstance(remote_config, dict)
        or not isinstance(remote_prompt, str)
    ):
        return _result("drift", "exact_version_identity_or_shape_changed", lock=lock)
    if config_sha256(remote_config) != lock["config_sha256"]:
        return _result("drift", "config_hash_changed", lock=lock)
    if system_prompt_sha256(remote_prompt) != lock["system_prompt_sha256"]:
        return _result("drift", "system_prompt_hash_changed", lock=lock)

    return PresetProvenance(
        status="verified",
        slug=lock["slug"],
        preset_id=lock["preset_id"],
        version_id=lock["designated_version_id"],
        version=lock["version"],
        config=copy.deepcopy(remote_config),
        system_prompt=remote_prompt,
        config_digest=lock["config_sha256"],
        system_prompt_digest=lock["system_prompt_sha256"],
        verified_at=verified_at,
    )


def fetch_current_designated_version(slug: str, api_key: str) -> tuple[dict, dict]:
    """Fetch and validate one preset's current mutable designated version.

    This is the single shared fetch primitive reused by production-safe
    preset resolution (fetch_current_designated_preset), the docs snapshot
    sync tool (scripts/sync_openrouter_presets.py), and the transcript
    reviewer lock-update tool (fetch_current_transcript_reviewer) — so all
    three keep identical fetch/validation semantics instead of drifting.
    """

    current = _get(f"{OPENROUTER_PRESETS_URL}/{slug}", api_key)
    designated = current.get("designated_version")
    version = designated.get("version") if isinstance(designated, dict) else None
    if not isinstance(version, int):
        raise ValueError(f"OpenRouter preset {slug!r} has no valid designated version")
    exact = _get(f"{OPENROUTER_PRESETS_URL}/{slug}/versions/{version}", api_key)
    if (
        current.get("slug") != slug
        or current.get("id") != exact.get("preset_id")
        or current.get("designated_version_id") != exact.get("id")
        or exact.get("version") != version
        or not isinstance(exact.get("config"), dict)
        or not isinstance(exact.get("system_prompt"), str)
    ):
        raise ValueError(f"OpenRouter designated version for {slug!r} is inconsistent")
    return current, exact


def fetch_current_designated_preset(
    slug: str,
    *,
    api_key: str,
    verified_at: str | None = None,
) -> PresetProvenance:
    """Resolve the current mutable designated preset, safe for production use.

    Unlike verify_transcript_reviewer, there is no committed lock to check
    against: this simply returns whatever OpenRouter currently designates
    for ``slug``, wrapped as a PresetProvenance so callers (summary,
    metadata) get the same structured, cache-safe, fail-closed contract.
    The transcript reviewer must keep using the stricter
    verify_transcript_reviewer exact-version check instead of this.
    """

    try:
        current, exact = fetch_current_designated_version(slug, api_key)
    except requests.HTTPError as error:
        return _http_result(error, {"slug": slug})
    except (requests.Timeout, requests.ConnectionError):
        return _result("unavailable", "network_or_timeout", lock={"slug": slug})
    except ValueError as error:
        return _result("invalid", _safe_reason(error), lock={"slug": slug})
    except requests.RequestException as error:
        return _result("unavailable", _safe_reason(type(error).__name__), lock={"slug": slug})

    remote_config = exact.get("config")
    remote_prompt = exact.get("system_prompt")
    if not isinstance(remote_config, dict) or not isinstance(remote_prompt, str):
        return _result("invalid", "designated_version_shape_is_invalid", lock={"slug": slug})

    return PresetProvenance(
        status="verified",
        slug=slug,
        preset_id=current["id"],
        version_id=exact["id"],
        version=exact["version"],
        config=copy.deepcopy(remote_config),
        system_prompt=remote_prompt,
        config_digest=config_sha256(remote_config),
        system_prompt_digest=system_prompt_sha256(remote_prompt),
        verified_at=verified_at,
    )


def fetch_current_transcript_reviewer(
    *,
    preset_slug: str,
    api_key: str | None = None,
) -> tuple[dict, dict]:
    """Fetch the current designated version for explicit lock update tooling."""

    key = api_key if api_key is not None else os.getenv("PODCAST_TRANSCRIPT_REVIEW_API_KEY")
    if not key:
        raise RuntimeError("Missing PODCAST_TRANSCRIPT_REVIEW_API_KEY")
    return fetch_current_designated_version(preset_slug, key)
