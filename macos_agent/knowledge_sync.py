#!/usr/bin/env python3
"""Periodically index configured local knowledge notes and execute approved work.

This command is deliberately finite: launchd (or a manual invocation) starts
one run, it scans only configured roots, reports a small GCS heartbeat, then
exits.  Markdown bodies never leave this Mac.
"""
from __future__ import annotations

import fcntl
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import google.auth
import yaml
from dotenv import load_dotenv
from google.auth import impersonated_credentials
from google.cloud import storage

from macos_agent.knowledge_index import (
    apply_adoption,
    apply_current_markdown,
    pending_adoptions,
    pending_tag_backfill,
    report_adoption,
    report_tag_backfill,
)
from podcast_engine.episode_contract import now_iso
from podcast_engine.storage import BUCKET_NAME
from podcast_engine.knowledge.tags import (
    LOCAL_SOURCE_TYPES,
    KnowledgeAgentStatus,
    TagRegistry,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = REPO_ROOT / ".env"
ROOTS_ENV = "PODCAST_KNOWLEDGE_ROOTS"
LOCK_ENV = "PODCAST_KNOWLEDGE_SYNC_LOCK_PATH"
IMPERSONATE_ENV = "PODCAST_KNOWLEDGE_SYNC_IMPERSONATE_SERVICE_ACCOUNT"
CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"


@dataclass(frozen=True)
class KnowledgeRoot:
    vault: str
    path: Path


class SingleRunLock:
    """Advisory file lock released by the OS if this process exits unexpectedly."""
    def __init__(self, path: Path):
        self.path, self.handle = path, None

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+")
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.handle.close()
            self.handle = None
            return False
        return True

    def release(self) -> None:
        if self.handle is not None:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()
            self.handle = None


def load_repo_env() -> None:
    load_dotenv(dotenv_path=ENV_PATH, override=False)


def knowledge_bucket_from_env(env: dict[str, str] | None = None):
    """Return an impersonated GCS bucket when the dedicated identity is configured.

    The user's normal ADC is used only as the source credential for service-account
    impersonation. If impersonation is requested and cannot be created, the error
    propagates; the agent never silently falls back to the user's identity.
    """
    source = os.environ if env is None else env
    target = source.get(IMPERSONATE_ENV, "").strip()
    if not target:
        return None

    source_credentials, detected_project = google.auth.default(
        scopes=[CLOUD_PLATFORM_SCOPE]
    )
    credentials = impersonated_credentials.Credentials(
        source_credentials=source_credentials,
        target_principal=target,
        target_scopes=[CLOUD_PLATFORM_SCOPE],
    )
    client = storage.Client(
        project=source.get("GOOGLE_CLOUD_PROJECT") or detected_project,
        credentials=credentials,
    )
    return client.bucket(BUCKET_NAME)


def knowledge_roots_from_env(env: dict[str, str] | None = None) -> list[KnowledgeRoot]:
    """Parse ``vault=/absolute/path`` roots separated by the platform path separator."""
    source = os.environ if env is None else env
    raw = source.get(ROOTS_ENV, "").strip()
    if not raw:
        raise ValueError(f"{ROOTS_ENV} is not set; configure one or more vault=/absolute/path roots")
    roots: list[KnowledgeRoot] = []
    names: set[str] = set()
    for value in raw.split(os.pathsep):
        vault, separator, path_text = value.partition("=")
        path = Path(path_text).expanduser() if separator else Path(value).expanduser()
        vault = vault.strip() if separator else path.name
        if not vault or not path.is_absolute():
            raise ValueError(f"{ROOTS_ENV} entries must use vault=/absolute/path")
        resolved = path.resolve()
        if vault in names:
            raise ValueError(f"{ROOTS_ENV} contains duplicate vault identifier: {vault}")
        names.add(vault)
        roots.append(KnowledgeRoot(vault=vault, path=resolved))
    return roots


def lock_path_from_env(env: dict[str, str] | None = None) -> Path:
    source = os.environ if env is None else env
    configured = source.get(LOCK_ENV, "").strip()
    return Path(configured).expanduser() if configured else Path(tempfile.gettempdir()) / "podcast-worker-knowledge-sync.lock"


def _frontmatter(markdown: str) -> dict:
    if not markdown.startswith("---\n"):
        raise ValueError("Knowledge note has no YAML frontmatter")
    end = markdown.find("\n---\n", 4)
    if end < 0:
        raise ValueError("Knowledge note frontmatter is not closed")
    try:
        parsed = yaml.safe_load(markdown[4:end + 1]) or {}
    except yaml.YAMLError as error:
        raise ValueError("Knowledge note frontmatter is invalid YAML") from error
    if not isinstance(parsed, dict):
        raise ValueError("Knowledge note frontmatter must be a mapping")
    return parsed


def _relative_note_path(root: KnowledgeRoot, path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.path.resolve()).as_posix()
    except ValueError as error:
        raise ValueError("Knowledge note is outside its configured root") from error


def _note_record(root: KnowledgeRoot, path: Path, markdown: str) -> dict | None:
    if not markdown.startswith("---\n"):
        return None
    data = _frontmatter(markdown)
    if not any(key in data for key in ("type", "tags", "knowledge_id")):
        return None
    source_type = data.get("type") if data.get("type") in LOCAL_SOURCE_TYPES else "other"
    knowledge_id = data.get("knowledge_id")
    if knowledge_id is not None and (not isinstance(knowledge_id, str) or not knowledge_id.startswith("note:")):
        raise ValueError("knowledge_id must use the note: prefix")
    tags = data.get("tags", [])
    if tags is not None and not isinstance(tags, list):
        raise ValueError("tags must be a YAML list")
    return {
        "knowledge_id": knowledge_id,
        "source_type": source_type,
        "title": data.get("title") if isinstance(data.get("title"), str) else path.stem,
        "vault": root.vault,
        "relative_path": _relative_note_path(root, path),
        "tags": tags or [],
        "topics": data.get("topics") if isinstance(data.get("topics"), list) else [],
        "last_seen_at": now_iso(),
    }


def atomic_write(path: Path, content: str) -> None:
    """Replace one Markdown file atomically after a complete same-directory write."""
    encoded = content.encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        try:
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except OSError:
            pass
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _safe_error(error: Exception) -> str:
    message = str(error).replace("\n", " ")[:140]
    return f"{type(error).__name__}: {message}" if message else type(error).__name__


def _log(event: str, **fields: object) -> None:
    print(json.dumps({"severity": "INFO", "event": event, **fields}, ensure_ascii=False, sort_keys=True))


class KnowledgeSyncAgent:
    def __init__(self, *, roots: list[KnowledgeRoot], registry: TagRegistry | None = None, status: KnowledgeAgentStatus | None = None, lock_path: Path | None = None, clock: Callable[[], float] = time.monotonic):
        self.roots = roots
        self.registry = registry or TagRegistry()
        self.status = status or KnowledgeAgentStatus(bucket=self.registry.bucket)
        self.lock_path = lock_path or lock_path_from_env()
        self.clock = clock

    def _root_for(self, vault: object) -> KnowledgeRoot:
        root = next((root for root in self.roots if root.vault == vault), None)
        if root is None:
            raise FileNotFoundError("Configured vault for local work item is unavailable")
        return root

    def _current_path(self, item: dict, *, use_index: bool) -> tuple[KnowledgeRoot, Path]:
        current = self.registry.artifact_index.get(item.get("artifact_id")) if use_index else None
        value = current if current else item
        root = self._root_for(value.get("vault"))
        relative = value.get("relative_path")
        if not isinstance(relative, str) or not relative:
            raise ValueError("Local work item has no relative path")
        path = root.path / relative
        _relative_note_path(root, path)
        if not path.is_file():
            raise FileNotFoundError("Current Markdown file is unavailable")
        return root, path

    def _scan(self, stats: dict, unresolved: set[str]) -> None:
        for root in self.roots:
            if not root.path.is_dir():
                stats["errors"] += 1
                _log("knowledge_sync_root_failed", vault=root.vault, error="Configured root is unavailable")
                continue
            try:
                paths = list(root.path.rglob("*.md"))
            except OSError:
                stats["errors"] += 1
                _log("knowledge_sync_root_failed", vault=root.vault, error="Configured root could not be scanned")
                continue
            for path in paths:
                if not path.is_file():
                    continue
                try:
                    markdown = path.read_text(encoding="utf-8")
                    record = _note_record(root, path, markdown)
                    if record is None:
                        continue
                    stats["scanned"] += 1
                    if record["knowledge_id"]:
                        result = self.registry.record_local_artifact(record)
                        stats["indexed"] += 1
                        unresolved.update(result.get("artifact", {}).get("unresolved_tags", []))
                    else:
                        self.registry.artifact_index.record_candidate(record)
                        stats["waiting_for_adoption"] += 1
                except (OSError, UnicodeError, ValueError) as error:
                    stats["errors"] += 1
                    _log("knowledge_sync_note_failed", path=str(path), error=_safe_error(error))

    def _adoption(self, work: dict, stats: dict) -> None:
        try:
            root, path = self._current_path(work, use_index=False)
            markdown = path.read_text(encoding="utf-8")  # mandatory fresh read
            record = _note_record(root, path, markdown)
            if record is None:
                raise ValueError("Adoption target is no longer a knowledge candidate")
            existing = record.get("knowledge_id")
            if existing and existing != work.get("knowledge_id"):
                raise ValueError("Adoption target has a different knowledge_id")
            patched, changed = apply_adoption(markdown, work)
            if changed:
                atomic_write(path, patched)
                stats["changed"] += 1
            else:
                stats["no_op"] += 1
            fresh = _note_record(root, path, patched if changed else markdown)
            self.registry.record_local_artifact(fresh or {})
            report_adoption(work, "completed", registry=self.registry)
            stats["backfilled"] += 1
            _log("knowledge_sync_adoption_completed", artifact_id=work.get("artifact_id"), status="changed" if changed else "no_op")
        except (OSError, UnicodeError, ValueError) as error:
            stats["errors"] += 1
            report_adoption(work, "failed", error=_safe_error(error), registry=self.registry)
            _log("knowledge_sync_adoption_failed", artifact_id=work.get("artifact_id"), error=_safe_error(error))

    def _tag_backfill(self, work: dict, stats: dict) -> None:
        try:
            root, path = self._current_path(work, use_index=True)
            markdown = path.read_text(encoding="utf-8")  # mandatory fresh read
            record = _note_record(root, path, markdown)
            if record is None or record.get("knowledge_id") != work.get("artifact_id"):
                raise ValueError("Backfill target no longer matches its artifact")
            patched, changed = apply_current_markdown(markdown, work)
            if changed:
                atomic_write(path, patched)
                stats["changed"] += 1
            else:
                stats["no_op"] += 1
            fresh = _note_record(root, path, patched if changed else markdown)
            self.registry.record_local_artifact(fresh or {})
            report_tag_backfill(work, "completed", registry=self.registry)
            stats["backfilled"] += 1
            _log("knowledge_sync_backfill_completed", artifact_id=work.get("artifact_id"), status="changed" if changed else "no_op")
        except (OSError, UnicodeError, ValueError) as error:
            stats["errors"] += 1
            report_tag_backfill(work, "failed", error=_safe_error(error), registry=self.registry)
            _log("knowledge_sync_backfill_failed", artifact_id=work.get("artifact_id"), error=_safe_error(error))

    def run(self) -> dict:
        lock = SingleRunLock(self.lock_path)
        if not lock.acquire():
            _log("knowledge_sync_already_running", status="already_running")
            return {"status": "already_running"}
        started_at, started = now_iso(), self.clock()
        stats = {"scanned": 0, "indexed": 0, "changed": 0, "backfilled": 0, "no_op": 0, "errors": 0, "waiting_for_adoption": 0}
        try:
            self.status.report({"status": "Running", "last_started_at": started_at})
            unresolved: set[str] = set()
            self._scan(stats, unresolved)
            for work in pending_adoptions(registry=self.registry):
                self._adoption(work, stats)
            for work in pending_tag_backfill(registry=self.registry):
                self._tag_backfill(work, stats)
            index = self.registry.artifact_index.load()
            indexed_notes = sum(1 for artifact in index.get("artifacts", {}).values() if isinstance(artifact, dict) and artifact.get("onboarding_status") == "adopted")
            unresolved = {tag for artifact in index.get("artifacts", {}).values() if isinstance(artifact, dict) for tag in artifact.get("unresolved_tags", [])}
            waiting_adoption = len(self.registry.artifact_index.local_candidates())
            pending = len(pending_tag_backfill(registry=self.registry))
            finished_at = now_iso()
            healthy = stats["errors"] == 0
            result = {"status": "Healthy" if healthy else "Error", "last_started_at": started_at, "last_finished_at": finished_at, "last_successful_sync_at": finished_at if healthy else self.status.load().get("last_successful_sync_at"), "duration_seconds": round(self.clock() - started, 3), "notes_indexed": indexed_notes, "waiting_for_adoption": waiting_adoption, "unresolved_tags": len(unresolved), "pending_backfills": pending, "last_run": {key: stats[key] for key in ("scanned", "indexed", "changed", "backfilled", "no_op", "errors")}}
            self.status.report(result)
            _log("knowledge_sync_finished", status=result["status"], **result["last_run"])
            return result
        finally:
            lock.release()


def main(argv: list[str] | None = None) -> int:
    del argv
    load_repo_env()
    try:
        roots = knowledge_roots_from_env()
        bucket = knowledge_bucket_from_env()
        if bucket is None:
            agent = KnowledgeSyncAgent(roots=roots)
        else:
            registry = TagRegistry(bucket=bucket)
            status = KnowledgeAgentStatus(bucket=bucket)
            agent = KnowledgeSyncAgent(
                roots=roots,
                registry=registry,
                status=status,
            )
        result = agent.run()
    except (OSError, ValueError, RuntimeError) as error:
        _log("knowledge_sync_fatal", status="Error", error=_safe_error(error))
        return 2
    return 0 if result["status"] in {"Healthy", "already_running"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
