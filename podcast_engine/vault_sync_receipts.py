"""Versioned, Mac-produced GCS receipts for canonical summary copied into a local Vault.

Receipts prove a particular summary byte digest was locally written or was already
byte-identical at one moment. They do not prove ongoing local presence or iCloud
remote upload, and cannot authorize canonical cloud episode-state changes.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import re

from google.api_core.exceptions import PreconditionFailed

from .storage import get_bucket

VAULT_SYNC_RECEIPT_SCHEMA_VERSION = 1
VAULT_SYNC_RECEIPT_PATH = "knowledge/agents/macos-vault-sync-v1.json"
MAX_RECEIPTS = 2000
MAX_RETRIES = 3
_KEY = re.compile(r"^[a-f0-9]{24}$")
_DIGEST = re.compile(r"^sha256:[a-f0-9]{64}$")


def summary_digest(content: bytes) -> str:
    if not isinstance(content, bytes):
        raise ValueError("Summary content must be bytes")
    return "sha256:" + hashlib.sha256(content).hexdigest()


def _validate_receipts(value: object) -> dict:
    if not isinstance(value, dict) or value.get("schema_version") != VAULT_SYNC_RECEIPT_SCHEMA_VERSION:
        raise ValueError("Vault Sync receipt schema is missing or unsupported")
    receipts = value.get("receipts")
    if not isinstance(receipts, dict) or len(receipts) > MAX_RECEIPTS:
        raise ValueError("Vault Sync receipt map is malformed or oversized")
    for key, entry in receipts.items():
        if not isinstance(key, str) or not _KEY.fullmatch(key):
            raise ValueError("Vault Sync receipt key is invalid")
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("digest"), str)
            or not _DIGEST.fullmatch(entry["digest"])
            or not isinstance(entry.get("confirmed_at"), str)
            or not 1 <= len(entry["confirmed_at"]) <= 64
        ):
            raise ValueError("Vault Sync receipt content is malformed")
    return value


def load_vault_sync_receipts(*, bucket=None) -> dict | None:
    """Read-only: absent object means no proof, malformed object raises."""
    blob = (bucket or get_bucket()).blob(VAULT_SYNC_RECEIPT_PATH)
    if not blob.exists():
        return None
    try:
        value = json.loads(blob.download_as_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Vault Sync receipts are not valid JSON") from exc
    return _validate_receipts(value)


def record_vault_sync_receipts(proofs: dict[str, str], *, bucket=None, clock=None) -> str:
    """CAS-upsert one successful local sync batch, no-op for identical proofs.

    A retry after a successful but unacknowledged write does not alter the
    stored confirmation timestamp. Concurrent writers re-read and retry.
    """
    if not isinstance(proofs, dict):
        raise ValueError("Vault Sync proofs must be an object")
    for key, digest in proofs.items():
        if not isinstance(key, str) or not _KEY.fullmatch(key):
            raise ValueError("Vault Sync proof key is invalid")
        if not isinstance(digest, str) or not _DIGEST.fullmatch(digest):
            raise ValueError("Vault Sync proof digest is invalid")
    if not proofs:
        return "unchanged"
    target = bucket or get_bucket()
    blob = target.blob(VAULT_SYNC_RECEIPT_PATH)
    for _ in range(MAX_RETRIES):
        if blob.exists():
            blob.reload()
            generation = int(blob.generation)
            try:
                existing = json.loads(
                    blob.download_as_text(encoding="utf-8", if_generation_match=generation)
                )
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("Vault Sync receipts are not valid JSON") from exc
            _validate_receipts(existing)
        else:
            generation = 0
            existing = {"schema_version": VAULT_SYNC_RECEIPT_SCHEMA_VERSION, "receipts": {}}
        updated = deepcopy(existing)
        changed = False
        for key, digest in sorted(proofs.items()):
            previous = updated["receipts"].get(key)
            if isinstance(previous, dict) and previous.get("digest") == digest:
                continue
            instant = clock() if clock is not None else datetime.now(timezone.utc).isoformat()
            if not isinstance(instant, str) or not instant or len(instant) > 64:
                raise ValueError("Receipt confirmation time is invalid")
            updated["receipts"][key] = {"digest": digest, "confirmed_at": instant}
            changed = True
        if not changed:
            return "unchanged"
        _validate_receipts(updated)
        try:
            blob.upload_from_string(
                json.dumps(updated, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
                content_type="application/json",
                if_generation_match=generation,
            )
            return "recorded"
        except PreconditionFailed:
            continue
    raise RuntimeError("Vault Sync receipts changed repeatedly; cannot confirm this sync")


def receipt_status(
    *,
    summary_state: str,
    summary_bytes: bytes | None,
    receipt: dict | None,
    receipt_error: bool = False,
) -> dict:
    """Compare evidence for this exact canonical summary generation."""
    result = {"status": "not_ready", "confirmed_at": None}
    if summary_state not in {"ready", "completed"}:
        return result
    if receipt_error or summary_bytes is None:
        return {"status": "unavailable", "confirmed_at": None}
    if receipt is None:
        return {"status": "unconfirmed", "confirmed_at": None}
    try:
        _validate_receipts({"schema_version": VAULT_SYNC_RECEIPT_SCHEMA_VERSION, "receipts": {"0" * 24: receipt}})
    except ValueError:
        return {"status": "unavailable", "confirmed_at": None}
    if receipt["digest"] != summary_digest(summary_bytes):
        return {"status": "outdated", "confirmed_at": receipt["confirmed_at"]}
    return {"status": "confirmed", "confirmed_at": receipt["confirmed_at"]}
