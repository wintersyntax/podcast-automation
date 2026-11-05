"""Knowledge package with lazy access to storage-backed orchestration."""

from __future__ import annotations

from importlib import import_module
from typing import Any


__all__ = ["build_knowledge_note", "canonical_summary_generated_at"]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    orchestration = import_module(".orchestration", __name__)
    value = getattr(orchestration, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
