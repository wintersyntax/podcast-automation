"""Future macOS knowledge-agent boundary; intentionally no watcher or scanner."""
from __future__ import annotations

from podcast_engine.knowledge.tags import TagRegistry, patch_local_markdown


def pending_tag_backfill(*, registry: TagRegistry | None = None) -> list[dict]:
    """Return waiting local tag work from GCS; this does not touch the vault."""
    return (registry or TagRegistry()).pending_local_work()


def pending_adoptions(*, registry: TagRegistry | None = None) -> list[dict]:
    """Return explicit adoption actions; this does not read local files."""
    return (registry or TagRegistry()).pending_adoption_work()


def apply_current_markdown(markdown: str, work_item: dict) -> tuple[str, bool]:
    """Apply freshly re-read Markdown; never patch a cloud snapshot."""
    if work_item.get("kind") != "tag_backfill":
        raise ValueError("Expected a tag_backfill work item")
    return patch_local_markdown(markdown, target_tag=work_item.get("target_tag"), obsolete_tags=work_item.get("obsolete_tags", []))


def apply_adoption(markdown: str, work_item: dict) -> tuple[str, bool]:
    """Insert the assigned ID only after the executor has re-read the file."""
    if work_item.get("kind") != "adopt":
        raise ValueError("Expected an adopt work item")
    return patch_local_markdown(markdown, target_tag=None, obsolete_tags=[], knowledge_id=work_item.get("knowledge_id"))


def report_tag_backfill(work_item: dict, status: str, *, error: str | None = None, registry: TagRegistry | None = None) -> dict:
    return (registry or TagRegistry()).report_local_work(work_item.get("candidate"), work_item.get("artifact_id"), status, error=error)


def report_adoption(work_item: dict, status: str, *, error: str | None = None, registry: TagRegistry | None = None) -> dict:
    return (registry or TagRegistry()).report_adoption_work(work_item.get("artifact_id"), status, error=error)
