"""Read-only, evidence-bounded status projection for PodcastOps (TASK-142).

A canonical episode's compiler state is NOT a Worker execution status.
Everything here is derived from stored episode/review audits, never Cloud Run
or inferred success. Missing audit evidence remains unknown, not zero.
"""

from __future__ import annotations

from .episode_duplicates import is_duplicate
from .human_review import pending_review_items
from .materiality_queue import GROUPS, materiality_groups
from .vault_sync_receipts import receipt_status


KNOWN_RECOMPILE_STATUSES = frozenset({
    "starting", "started", "unknown", "completed", "failed",
    "execution_failed",
})


def _state(episode: dict, stage: str) -> str:
    status = episode.get("status")
    details = status.get(stage) if isinstance(status, dict) else None
    state = details.get("state") if isinstance(details, dict) else None
    return state if isinstance(state, str) and state and len(state) <= 64 else "unknown"


def _unavailable_review(state: str) -> dict:
    return {
        "state": state,
        "decided": None,
        "pending": None,
        "groups": None,
    }


def _unavailable_recompile(state: str) -> dict:
    return {"status": state, "request_id": None, "updated_at": None, "completed_at": None}


def _bounded(value: object, limit: int = 300) -> str | None:
    return value if isinstance(value, str) and value and len(value) <= limit else None


def _duplicate_of(episode: dict) -> dict | None:
    """The earlier episode a recorded duplicate repeats (TASK-135), or ``None``."""

    if not is_duplicate(episode):
        return None
    check = episode["duplicate_check"]
    return {
        "episode_key": _bounded(check.get("original_episode_key"), 64),
        "podcast": _bounded(check.get("original_podcast")),
        "title": _bounded(check.get("original_title")),
    }


def episode_status(episode: dict, record: dict | None, *, audit_error: str | None = None, summary_bytes: bytes | None = None, sync_receipt: dict | None = None, sync_error: bool = False) -> dict:
    """Produce one minimal JSON-safe status row from existing persisted evidence.

    The caller owns GCS reads. Never load or return source text, chosen text,
    notes, tokens, provider output, or full resolver records to this dashboard.
    """
    key = episode.get("episode_key")
    if not isinstance(key, str) or not key:
        raise ValueError("Canonical episode has no episode_key")
    result = {
        "episode_key": key,
        "podcast": episode.get("podcast") if isinstance(episode.get("podcast"), str) else "",
        "title": episode.get("title") if isinstance(episode.get("title"), str) else key,
        "published": episode.get("published") if isinstance(episode.get("published"), str) else "",
        "apple": _state(episode, "apple_transcript"),
        "whisper": _state(episode, "whisper"),
        "compiler": _state(episode, "compiler"),
        "summary": _state(episode, "summary"),
        "review": _unavailable_review("not_recorded"),
        "recompile": _unavailable_recompile("not_recorded"),
    }
    # Mac evidence is independent of resolver-card/audit availability.
    result["vault_sync"] = receipt_status(
        summary_state=result["summary"],
        summary_bytes=summary_bytes,
        receipt=sync_receipt,
        receipt_error=sync_error,
    )
    # A recorded duplicate is deliberately never processed further, so its
    # pending stages are not work in progress (TASK-156).
    result["duplicate_of"] = _duplicate_of(episode)
    if result["duplicate_of"] is not None:
        result["vault_sync"] = {**result["vault_sync"], "status": "not_applicable"}
    stage_status = episode.get("status") if isinstance(episode.get("status"), dict) else {}
    summary_status = stage_status.get("summary") if isinstance(stage_status.get("summary"), dict) else {}
    compiler_status = stage_status.get("compiler") if isinstance(stage_status.get("compiler"), dict) else {}
    result["compiler_completed_at"] = (
        compiler_status.get("updated_at")
        if result["compiler"] == "completed"
        and isinstance(compiler_status.get("updated_at"), str)
        and len(compiler_status["updated_at"]) <= 64
        else None
    )
    result["summary_ready_at"] = (
        summary_status.get("updated_at")
        if result["summary"] in {"ready", "completed"}
        and isinstance(summary_status.get("updated_at"), str)
        and len(summary_status["updated_at"]) <= 64
        else None
    )
    if result["duplicate_of"] is not None:
        result["delivery"] = "duplicate"
    elif result["summary"] in {"ready", "completed"}:
        result["delivery"] = (
            "synced_to_mac" if result["vault_sync"]["status"] == "confirmed"
            else "awaiting_vault_sync" if result["vault_sync"]["status"] in {"unconfirmed", "outdated"}
            else "unverified"
        )
    elif result["compiler"] == "completed":
        result["delivery"] = "awaiting_summary"
    else:
        result["delivery"] = "processing"
    result["finished_at"] = (
        result["vault_sync"]["confirmed_at"]
        if result["delivery"] == "synced_to_mac"
        else None
    )
    if audit_error:
        result["review"] = _unavailable_review("unavailable")
        result["recompile"] = _unavailable_recompile("unavailable")
        return result
    if record is None:
        if result["compiler"] == "review_required":
            result["review"] = _unavailable_review("unavailable")
        return result

    if not isinstance(record, dict) or record.get("episode_key") != key:
        result["review"] = _unavailable_review("unavailable")
        result["recompile"] = _unavailable_recompile("unavailable")
        return result

    cards, decisions, requests = (
        record.get("human_review"),
        record.get("human_decisions"),
        record.get("recompile_requests", []),
    )
    if not isinstance(cards, list) or not isinstance(decisions, list) or not isinstance(requests, list):
        result["review"] = _unavailable_review("unavailable")
        result["recompile"] = _unavailable_recompile("unavailable")
        return result
    if any(not isinstance(card, dict) or type(card.get("id")) is not int for card in cards):
        result["review"] = _unavailable_review("unavailable")
        result["recompile"] = _unavailable_recompile("unavailable")
        return result
    if any(not isinstance(decision, dict) or type(decision.get("id")) is not int for decision in decisions):
        result["review"] = _unavailable_review("unavailable")
        result["recompile"] = _unavailable_recompile("unavailable")
        return result
    card_ids = [item["id"] for item in cards]
    decision_ids = [item["id"] for item in decisions]
    if len(set(card_ids)) != len(card_ids) or len(set(decision_ids)) != len(decision_ids) or set(card_ids) & set(decision_ids):
        result["review"] = _unavailable_review("unavailable")
        result["recompile"] = _unavailable_recompile("unavailable")
        return result

    try:
        projected = pending_review_items(record)
        grouped = materiality_groups(record, key, projected)
        if len(projected) != len(cards) or len(grouped) != len(cards):
            raise ValueError("Invalid card grouping evidence")
        counts = {group: 0 for group in GROUPS}
        for card in projected:
            group = grouped[card["id"]]["group"]
            if group not in counts:
                raise ValueError("Unknown materiality group")
            counts[group] += 1
    except (KeyError, TypeError, ValueError):
        result["review"] = _unavailable_review("unavailable")
        result["recompile"] = _unavailable_recompile("unavailable")
        return result

    result["review"] = {
        "state": "pending" if cards else ("reviewed" if decisions else "no_cards"),
        "decided": len(decisions),
        "pending": len(cards),
        "groups": counts,
    }

    if not requests:
        return result
    last = requests[-1]
    if not isinstance(last, dict) or last.get("status") not in KNOWN_RECOMPILE_STATUSES:
        result["recompile"] = _unavailable_recompile("unavailable")
        return result
    request_id = last.get("request_id")
    updated_at = last.get("updated_at") or last.get("completed_at") or last.get("requested_at")
    if not isinstance(request_id, str) or not request_id or len(request_id) > 80:
        result["recompile"] = _unavailable_recompile("unavailable")
        return result
    result["recompile"] = {
        "status": last["status"],
        "request_id": request_id,
        "updated_at": updated_at if isinstance(updated_at, str) and len(updated_at) <= 64 else None,
        "completed_at": (
            last["completed_at"] if last["status"] == "completed"
            and isinstance(last.get("completed_at"), str)
            and len(last["completed_at"]) <= 64 else None
        ),
    }
    return result


def sorted_episode_statuses(rows: list[dict]) -> list[dict]:
    """Latest published first, deterministic fallback when times match."""
    return sorted(
        rows,
        key=lambda row: (row["published"], row["episode_key"]),
        reverse=True,
    )
