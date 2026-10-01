"""User-facing notifications for podcast workflow events."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from html import escape
from zoneinfo import ZoneInfo

from google.api_core.exceptions import PreconditionFailed
import requests


SLACK_TIMEOUT_SECONDS = 10
EMAIL_TIMEOUT_SECONDS = 10

CLOUDFLARE_EMAIL_ENDPOINT = (
    "https://api.cloudflare.com/client/v4/accounts/"
    "{account_id}/email/sending/send"
)

_OSLO_TIMEZONE = ZoneInfo("Europe/Oslo")
_ENGLISH_MONTH_ABBREVIATIONS = (
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
)


def build_slack_review_payload(
    episode: dict,
    pending_count: int,
    review_url: str,
    *,
    triage_unavailable: int = 0,
) -> dict:
    """Build the Slack Block Kit message for a human-review queue."""

    count = int(pending_count)
    difference_label = "difference" if count == 1 else "differences"

    podcast = str(
        episode.get("podcast")
        or "Podcast"
    )

    title = str(
        episode.get("title")
        or episode.get("episode_key")
        or "Untitled episode"
    )

    payload = {
        "text": (
            f"Podcast ready for review: "
            f"{podcast} — {title}"
        ),
        "blocks": [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "🎙️ Podcast ready for review",
                },
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*{podcast}*\n{title}",
                },
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": (
                            f"*Needs review*\n"
                            f"{count} transcript {difference_label}"
                        ),
                    },
                    {
                        "type": "mrkdwn",
                        "text": (
                            "*Status*\n"
                            "Human review required"
                        ),
                    },
                ],
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "action_id": "open_human_review",
                        "style": "primary",
                        "text": {
                            "type": "plain_text",
                            "text": "Open Human Review",
                        },
                        "url": review_url,
                    },
                ],
            },
        ],
    }
    unavailable = int(triage_unavailable)
    if unavailable > 0:
        # TASK-123: make a silent advisory-triage outage visible where the
        # reviewer first looks, instead of only in Cloud Logging.
        payload["blocks"].insert(
            3,
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": (
                            f"⚠️ Advisory triage unavailable for {unavailable} "
                            f"of {count} {difference_label}; batch approval "
                            "is disabled for those cards."
                        ),
                    }
                ],
            },
        )
    return payload


def build_slack_review_refreshed_payload(
    episode: dict,
    previous_pending_count: int,
    pending_count: int,
    review_url: str,
) -> dict:
    """Build the Slack Block Kit message for a refreshed review queue."""

    previous = int(previous_pending_count)
    current = int(pending_count)
    difference_label = "difference" if current == 1 else "differences"
    podcast = str(episode.get("podcast") or "Podcast")
    title = str(
        episode.get("title")
        or episode.get("episode_key")
        or "Untitled episode"
    )

    return {
        "text": f"Podcast review updated: {podcast} — {title}",
        "blocks": [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "🎙️ Podcast review updated",
                },
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*{podcast}*\n{title}",
                },
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": (
                            "*Review queue refreshed*\n"
                            f"{previous} → {current} transcript {difference_label}"
                        ),
                    },
                    {
                        "type": "mrkdwn",
                        "text": "*Status*\nHuman review required",
                    },
                ],
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "action_id": "open_human_review",
                        "style": "primary",
                        "text": {
                            "type": "plain_text",
                            "text": "Open Human Review",
                        },
                        "url": review_url,
                    },
                ],
            },
        ],
    }


def _send_slack_payload(
    episode: dict,
    payload: dict,
    *,
    webhook_url: str | None,
    failure_event: str = "human_review_slack_failed",
    failure_message: str = "Slack human review notification failed",
) -> bool:
    """Deliver a Slack payload without allowing Slack to block the worker."""

    configured_url = (
        os.environ.get("PODCAST_SLACK_WEBHOOK_URL", "")
        if webhook_url is None
        else webhook_url
    ).strip()

    if not configured_url:
        return False

    try:
        response = requests.post(
            configured_url,
            json=payload,
            timeout=SLACK_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except Exception as error:
        warning = {
            "severity": "WARNING",
            "message": failure_message,
            "event": failure_event,
            "episode_key": episode.get("episode_key"),
            "error_type": type(error).__name__,
        }
        print(json.dumps(warning, ensure_ascii=False, separators=(",", ":")))
        return False

    return True


def send_slack_review_notification(
    episode: dict,
    pending_count: int,
    review_url: str,
    *,
    webhook_url: str | None = None,
    triage_unavailable: int = 0,
) -> bool:
    """Send one Slack review notification.

    Missing configuration and delivery failures are non-fatal: Slack is a
    convenience notification channel and must never block the podcast worker.
    """

    payload = build_slack_review_payload(
        episode,
        pending_count,
        review_url,
        triage_unavailable=triage_unavailable,
    )

    return _send_slack_payload(
        episode,
        payload,
        webhook_url=webhook_url,
    )


def send_slack_review_refreshed_notification(
    episode: dict,
    previous_pending_count: int,
    pending_count: int,
    review_url: str,
    *,
    webhook_url: str | None = None,
) -> bool:
    """Send a non-fatal Slack notification for a changed refreshed queue."""

    return _send_slack_payload(
        episode,
        build_slack_review_refreshed_payload(
            episode,
            previous_pending_count,
            pending_count,
            review_url,
        ),
        webhook_url=webhook_url,
    )


def build_slack_summary_ready_payload(
    episode: dict,
    generated_at: str,
) -> dict:
    """Build the content-minimal Slack message for a ready canonical summary."""

    podcast, title = _email_episode_identity(episode)
    ready_text = "Ready — will sync to Obsidian on the next Vault Sync."

    return {
        "text": f"Summary ready: {podcast} — {title}",
        "blocks": [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "✅ Summary ready",
                },
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*{podcast}*\n{title}",
                },
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": (
                            "*Generated*\n"
                            f"{_format_summary_ready_generated_at(generated_at)}"
                        ),
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Status*\n{ready_text}",
                    },
                ],
            },
        ],
    }


def _format_summary_ready_generated_at(generated_at: object) -> str:
    """Render the canonical instant for Slack without changing its stored value."""

    if not isinstance(generated_at, str) or not generated_at.strip():
        return "Unavailable"

    try:
        instant = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
    except (TypeError, ValueError, OverflowError):
        return "Unavailable"

    if instant.tzinfo is None:
        return "Unavailable"

    local_time = instant.astimezone(_OSLO_TIMEZONE)
    return (
        f"{local_time.day:02d} "
        f"{_ENGLISH_MONTH_ABBREVIATIONS[local_time.month - 1]} "
        f"{local_time.year:04d}, "
        f"{local_time.hour:02d}:{local_time.minute:02d}"
    )


def send_slack_summary_ready_notification(
    episode: dict,
    generated_at: str,
    *,
    webhook_url: str | None = None,
) -> bool:
    """Best-effort Slack signal after durable Vault-Sync-ready persistence."""

    return _send_slack_payload(
        episode,
        build_slack_summary_ready_payload(episode, generated_at),
        webhook_url=webhook_url,
        failure_event="summary_ready_slack_failed",
        failure_message="Slack summary-ready notification failed",
    )

def _email_episode_identity(
    episode: dict,
) -> tuple[str, str]:
    podcast = str(
        episode.get("podcast")
        or "Podcast"
    )
    title = str(
        episode.get("title")
        or episode.get("episode_key")
        or "Untitled episode"
    )
    return podcast, title


def _build_review_email_html(
    *,
    podcast: str,
    title: str,
    heading: str,
    detail: str,
    review_url: str,
) -> str:
    """Build conservative email-safe HTML with inline styling."""
    podcast = escape(podcast)
    title = escape(title)
    heading = escape(heading)
    detail = escape(detail)
    review_url = escape(
        review_url,
        quote=True,
    )

    return f"""<!doctype html>
<html>
<head>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
</head>

<body style="margin:0;padding:0;background:#f3f4f6;font-family:Arial,Helvetica,sans-serif;color:#111827;">

<table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0"
       style="background:#f3f4f6;padding:36px 14px;">
<tr>
<td align="center">

<table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0"
       style="max-width:600px;background:#ffffff;border:1px solid #e5e7eb;border-radius:16px;">

<tr>
<td style="padding:36px;">

<div style="
    font-size:12px;
    line-height:18px;
    letter-spacing:1.6px;
    font-weight:700;
    color:#6b7280;
    margin-bottom:22px;">
PODCAST OPS
</div>

<h1 style="
    margin:0 0 24px;
    font-size:26px;
    line-height:34px;
    color:#111827;">
{heading}
</h1>

<div style="
    font-size:14px;
    line-height:22px;
    font-weight:700;
    color:#374151;
    margin-bottom:4px;">
{podcast}
</div>

<div style="
    font-size:17px;
    line-height:26px;
    color:#111827;
    margin-bottom:26px;">
{title}
</div>

<table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0"
       style="background:#f9fafb;border:1px solid #f0f1f3;border-radius:10px;margin-bottom:28px;">
<tr>
<td style="
    padding:18px 20px;
    font-size:14px;
    line-height:22px;
    color:#374151;">
{detail}
</td>
</tr>
</table>

<table role="presentation" cellspacing="0" cellpadding="0" border="0">
<tr>
<td style="background:#111827;border-radius:9px;">
<a href="{review_url}"
   style="
       display:inline-block;
       padding:13px 21px;
       font-size:14px;
       line-height:20px;
       font-weight:700;
       color:#ffffff;
       text-decoration:none;">
Open Human Review
</a>
</td>
</tr>
</table>

<div style="
    border-top:1px solid #eeeeee;
    margin-top:32px;
    padding-top:20px;
    font-size:12px;
    line-height:19px;
    color:#9ca3af;">
Automated notification from Podcast Ops
</div>

</td>
</tr>
</table>

</td>
</tr>
</table>

</body>
</html>"""


def build_email_review_payload(
    episode: dict,
    pending_count: int,
    review_url: str,
) -> dict:
    """Build initial Human Review email."""
    count = int(pending_count)
    label = (
        "difference"
        if count == 1
        else "differences"
    )
    podcast, title = _email_episode_identity(
        episode
    )

    detail = (
        f"{count} transcript {label} "
        "need human review."
    )

    return {
        "subject": (
            f"Review needed: {podcast} — {title}"
        ),
        "text": (
            "PODCAST OPS\n\n"
            "Human review required\n\n"
            f"{podcast}\n"
            f"{title}\n\n"
            f"{detail}\n\n"
            f"Open Human Review: {review_url}\n"
        ),
        "html": _build_review_email_html(
            podcast=podcast,
            title=title,
            heading="Human review required",
            detail=detail,
            review_url=review_url,
        ),
    }


def build_email_review_refreshed_payload(
    episode: dict,
    previous_pending_count: int,
    pending_count: int,
    review_url: str,
) -> dict:
    """Build email for a changed Human Review queue."""
    previous = int(previous_pending_count)
    current = int(pending_count)

    label = (
        "difference"
        if current == 1
        else "differences"
    )

    podcast, title = _email_episode_identity(
        episode
    )

    detail = (
        "Review queue changed from "
        f"{previous} to {current} transcript "
        f"{label}."
    )

    return {
        "subject": (
            f"Review updated: {podcast} — {title}"
        ),
        "text": (
            "PODCAST OPS\n\n"
            "Human Review queue updated\n\n"
            f"{podcast}\n"
            f"{title}\n\n"
            f"{detail}\n\n"
            f"Open Human Review: {review_url}\n"
        ),
        "html": _build_review_email_html(
            podcast=podcast,
            title=title,
            heading="Review queue updated",
            detail=detail,
            review_url=review_url,
        ),
    }


_STALLED_REVIEW_IDENTITY_FIELDS = (
    "episode_key",
    "transcript_sha256",
    "draft_sha256",
    "review_policy_version",
    "review_preset",
)


def summary_review_stalled_email_notification_key(
    review_input_identity: dict,
    diagnostics: dict,
) -> str | None:
    """Return one dedupe key for the current stalled-review streak."""

    if not isinstance(review_input_identity, dict):
        raise ValueError("Summary-review email identity must be an object")

    identity = {}
    for field in _STALLED_REVIEW_IDENTITY_FIELDS:
        value = review_input_identity.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(
                f"Summary-review email identity {field} is invalid"
            )
        identity[field] = value

    if not isinstance(diagnostics, dict):
        raise ValueError("Summary-review diagnostics must be an object")

    state = diagnostics.get("state")
    failure_count = diagnostics.get("failure_count")
    if (
        state != "stalled"
        or not isinstance(failure_count, int)
        or isinstance(failure_count, bool)
        or failure_count < 2
    ):
        return None

    first_failure_at = diagnostics.get("first_failure_at")
    if not isinstance(first_failure_at, str) or not first_failure_at:
        raise ValueError(
            "Stalled summary-review diagnostics require first_failure_at"
        )

    encoded = json.dumps(
        {
            "identity": identity,
            "first_failure_at": first_failure_at,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _build_summary_review_stalled_email_html(
    *,
    podcast: str,
    title: str,
    failure_count: int,
    stage_label: str,
    explanation: str,
    podcastops_url: str,
) -> str:
    podcast = escape(podcast)
    title = escape(title)
    stage_label = escape(stage_label)
    explanation = escape(explanation)
    podcastops_url = escape(podcastops_url, quote=True)

    return f"""<!doctype html>
<html>
<head>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
</head>
<body style="margin:0;padding:0;background:#f3f4f6;font-family:Arial,Helvetica,sans-serif;color:#111827;">
<table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0"
       style="background:#f3f4f6;padding:36px 14px;">
<tr><td align="center">
<table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0"
       style="max-width:600px;background:#ffffff;border:1px solid #e5e7eb;border-radius:16px;">
<tr><td style="padding:36px;">
<div style="font-size:12px;line-height:18px;letter-spacing:1.6px;font-weight:700;color:#6b7280;margin-bottom:22px;">
PODCAST OPS
</div>
<h1 style="margin:0 0 24px;font-size:26px;line-height:34px;color:#111827;">
Summary review stalled
</h1>
<div style="font-size:14px;line-height:22px;font-weight:700;color:#374151;margin-bottom:4px;">
{podcast}
</div>
<div style="font-size:17px;line-height:26px;color:#111827;margin-bottom:26px;">
{title}
</div>
<div style="font-size:14px;line-height:22px;color:#374151;margin-bottom:24px;">
<strong>Failure count:</strong> {failure_count}<br>
<strong>Stage:</strong> {stage_label}<br>
{explanation}<br>
Canonical note preserved.
</div>
<table role="presentation" cellspacing="0" cellpadding="0" border="0">
<tr><td style="background:#111827;border-radius:9px;">
<a href="{podcastops_url}"
   style="display:inline-block;padding:13px 21px;font-size:14px;line-height:20px;font-weight:700;color:#ffffff;text-decoration:none;">
Open in PodcastOps
</a>
</td></tr>
</table>
<div style="border-top:1px solid #eeeeee;margin-top:32px;padding-top:20px;font-size:12px;line-height:19px;color:#9ca3af;">
Automated notification from Podcast Ops
</div>
</td></tr>
</table>
</td></tr>
</table>
</body>
</html>"""


def build_email_summary_review_stalled_payload(
    episode: dict,
    diagnostics: dict,
    podcastops_url: str,
) -> dict:
    """Build the bounded operator email for a stalled summary review."""

    if not isinstance(diagnostics, dict):
        raise ValueError("Summary-review diagnostics must be an object")
    if diagnostics.get("state") != "stalled":
        raise ValueError("Summary-review stalled email requires stalled state")

    failure_count = diagnostics.get("failure_count")
    if (
        not isinstance(failure_count, int)
        or isinstance(failure_count, bool)
        or failure_count < 2
    ):
        raise ValueError(
            "Summary-review stalled email requires at least two failures"
        )

    latest_failure = diagnostics.get("latest_failure")
    if not isinstance(latest_failure, dict):
        raise ValueError(
            "Summary-review stalled email requires latest failure diagnostics"
        )

    stage = latest_failure.get("stage")
    explanation = latest_failure.get("explanation")
    if not isinstance(stage, str) or not stage:
        raise ValueError("Summary-review stalled email stage is invalid")
    if not isinstance(explanation, str) or not explanation:
        raise ValueError(
            "Summary-review stalled email explanation is invalid"
        )

    if diagnostics.get("canonical_note_preserved") is not True:
        raise ValueError(
            "Summary-review stalled email requires preserved canonical note"
        )

    if not isinstance(podcastops_url, str) or not podcastops_url.strip():
        raise ValueError("PodcastOps URL is required")

    podcast, title = _email_episode_identity(episode)
    stage_label = stage.replace("_", " ").title()
    podcastops_url = podcastops_url.strip()

    return {
        "subject": f"Summary review stalled: {podcast} — {title}",
        "text": (
            "PODCAST OPS\n\n"
            "Summary review stalled\n\n"
            f"{podcast}\n"
            f"{title}\n\n"
            f"Failure count: {failure_count}\n"
            f"Stage: {stage_label}\n"
            f"{explanation}\n"
            "Canonical note preserved.\n\n"
            f"Open in PodcastOps: {podcastops_url}\n"
        ),
        "html": _build_summary_review_stalled_email_html(
            podcast=podcast,
            title=title,
            failure_count=failure_count,
            stage_label=stage_label,
            explanation=explanation,
            podcastops_url=podcastops_url,
        ),
    }


def send_email_summary_review_stalled_notification(
    *,
    bucket,
    episode: dict,
    review_input_identity: dict,
    diagnostics: dict,
    podcastops_url: str,
) -> dict:
    """Send at most one stalled-review email per durable pre-send claim."""

    notification_key = summary_review_stalled_email_notification_key(
        review_input_identity,
        diagnostics,
    )
    if notification_key is None:
        return {
            "status": "not_eligible",
            "notification_key": None,
            "marker_path": None,
        }

    identity = {
        field: review_input_identity[field]
        for field in _STALLED_REVIEW_IDENTITY_FIELDS
    }
    episode_key = identity["episode_key"]
    if not isinstance(episode, dict) or episode.get("episode_key") != episode_key:
        raise ValueError(
            "Summary-review stalled email episode identity does not match"
        )

    digest = notification_key.split(":", 1)[1]
    marker_path = (
        f"episodes/{episode_key}/summary/review_notifications/"
        f"stalled_email/{digest}.json"
    )
    marker = {
        "episode_key": episode_key,
        "first_failure_at": diagnostics["first_failure_at"],
        "notification_key": notification_key,
        "notification_type": "summary_review_stalled_email",
        "review_input_identity": identity,
    }
    marker_text = (
        json.dumps(
            marker,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    marker_bytes = marker_text.encode("utf-8")
    blob = bucket.blob(marker_path)

    if blob.exists():
        if blob.download_as_bytes() != marker_bytes:
            raise RuntimeError(
                f"summary-review stalled email marker conflict: {marker_path}"
            )
        return {
            "status": "already_claimed",
            "notification_key": notification_key,
            "marker_path": marker_path,
        }

    payload = build_email_summary_review_stalled_payload(
        episode,
        diagnostics,
        podcastops_url,
    )

    try:
        blob.upload_from_string(
            marker_text,
            content_type="application/json",
            if_generation_match=0,
        )
    except PreconditionFailed as error:
        if not blob.exists():
            raise RuntimeError(
                "summary-review stalled email claim failed without durable marker"
            ) from error
        if blob.download_as_bytes() != marker_bytes:
            raise RuntimeError(
                f"summary-review stalled email marker conflict: {marker_path}"
            ) from error
        return {
            "status": "already_claimed",
            "notification_key": notification_key,
            "marker_path": marker_path,
        }

    if not _send_email_payload(episode, payload):
        blob.delete()
        return {
            "status": "delivery_failed",
            "notification_key": notification_key,
            "marker_path": marker_path,
        }

    return {
        "status": "sent",
        "notification_key": notification_key,
        "marker_path": marker_path,
    }


def _send_email_payload(
    episode: dict,
    payload: dict,
) -> bool:
    """Deliver email without allowing it to block the worker."""
    account_id = os.environ.get(
        "CLOUDFLARE_EMAIL_ACCOUNT_ID",
        "",
    ).strip()

    api_token = os.environ.get(
        "CLOUDFLARE_EMAIL_API_TOKEN",
        "",
    ).strip()

    recipient = os.environ.get(
        "PODCAST_REVIEW_EMAIL_TO",
        "",
    ).strip()

    from_address = os.environ.get(
        "PODCAST_REVIEW_EMAIL_FROM",
        "notifications@example.com",
    ).strip()

    from_name = (
        os.environ.get(
            "PODCAST_REVIEW_EMAIL_FROM_NAME",
            "Podcast Ops",
        ).strip()
        or "Podcast Ops"
    )

    if not (
        account_id
        and api_token
        and recipient
        and from_address
    ):
        return False

    endpoint = CLOUDFLARE_EMAIL_ENDPOINT.format(
        account_id=account_id,
    )

    request_payload = {
        "to": recipient,
        "from": {
            "address": from_address,
            "name": from_name,
        },
        **payload,
    }

    try:
        response = requests.post(
            endpoint,
            headers={
                "Authorization": (
                    f"Bearer {api_token}"
                ),
                "Content-Type": "application/json",
            },
            json=request_payload,
            timeout=EMAIL_TIMEOUT_SECONDS,
        )

        response.raise_for_status()

        try:
            result = response.json()
        except ValueError:
            result = {}

        if (
            isinstance(result, dict)
            and result.get("success") is False
        ):
            raise requests.HTTPError(
                "Cloudflare Email API reported success=false",
                response=response,
            )

    except requests.RequestException as error:
        warning = {
            "severity": "WARNING",
            "message": (
                "Email human review notification failed"
            ),
            "event": "human_review_email_failed",
            "episode_key": (
                episode.get("episode_key")
            ),
            "error_type": (
                type(error).__name__
            ),
        }

        print(
            json.dumps(
                warning,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
        return False

    return True


def send_email_review_notification(
    episode: dict,
    pending_count: int,
    review_url: str,
) -> bool:
    """Send non-fatal email for a new review queue."""
    return _send_email_payload(
        episode,
        build_email_review_payload(
            episode,
            pending_count,
            review_url,
        ),
    )


def send_email_review_refreshed_notification(
    episode: dict,
    previous_pending_count: int,
    pending_count: int,
    review_url: str,
) -> bool:
    """Send non-fatal email for a refreshed review queue."""
    return _send_email_payload(
        episode,
        build_email_review_refreshed_payload(
            episode,
            previous_pending_count,
            pending_count,
            review_url,
        ),
    )
