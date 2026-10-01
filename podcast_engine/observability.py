"""Secret-safe structured events for Google Cloud Logging and Monitoring."""

from __future__ import annotations

import json


def emit_event(
    event: str,
    message: str,
    *,
    severity: str = "ERROR",
    **fields: object,
) -> None:
    """Write a stable JSON event without serializing credentials or exceptions."""

    print(
        json.dumps(
            {
                "severity": severity,
                "event": event,
                "message": message,
                **fields,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )
