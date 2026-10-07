"""TASK-133: the materiality filter's evidence projected into review groups.

User decisions 2026-10-07. A pending card with current ``materiality_shadow``
evidence (same policy generation and identity as the card's present inputs)
falls into one group:

``settled``   immaterial, and the evidence names the reading to keep: the
              reviewer accepts all of them with one explicit click;
``sample``    a settled card drawn into the random control sample (10 %, at
              least two per episode, never cards that only spell the same
              letters or numbers differently): confirmed one by one;
``click_one`` immaterial, but nothing decided which reading to keep;
``proposal``  material, with a preselected reading the reviewer confirms or
              changes card by card (never in bulk);
``full``      everything else -- numbers that differ, material cards without
              a proposal, cards without current evidence: ordinary review.

No source is ever preferred by default. For a card whose readings differ
only in how the same letters or numbers are written (``either``) the
compiler's own source suggestion is used; without one the card is
``click_one``. Every decision is still an explicit, audited human decision:
this module only groups cards and names the filter's reading.
"""

from __future__ import annotations

import hashlib
import math
import os

from compiler.materiality import MATERIALITY_POLICY_VERSION, materiality_inputs

from .materiality_shadow import EVIDENCE_KEY, evidence_identity

QUEUE_ENV = "PODCAST_REVIEW_MATERIALITY_QUEUE"
GROUPS = ("settled", "sample", "click_one", "proposal", "full")
SAMPLE_SHARE = 0.10
SAMPLE_MIN = 2
# Same letters or the same numbers written differently: nothing to check.
NO_SAMPLE_REASONS = frozenset({"same_letters", "number_format"})
SOURCES = ("apple", "whisper")


def queue_enabled() -> bool:
    return os.getenv(QUEUE_ENV, "on").strip().lower() not in {"0", "false", "off", "no"}


def current_evidence(item: dict) -> dict | None:
    """The card's materiality evidence, only if it still describes this card."""

    evidence = item.get(EVIDENCE_KEY) if isinstance(item, dict) else None
    if not isinstance(evidence, dict) or evidence.get("policy_version") != MATERIALITY_POLICY_VERSION:
        return None
    inputs = materiality_inputs(item)
    if inputs is None or evidence.get("identity") != evidence_identity(inputs):
        return None
    return evidence


def base_group(item: dict) -> dict:
    """Group a card on its own evidence, before the control sample is drawn."""

    evidence = current_evidence(item)
    if evidence is None:
        return {"group": "full", "source": None, "step": None, "reason": None}
    reason = evidence.get("reason")
    if evidence.get("immaterial") is True:
        source, step = evidence.get("use"), evidence.get("use_step")
        if step == "either":
            suggestion = item.get("suggestion") if isinstance(item.get("suggestion"), dict) else {}
            source = suggestion.get("source")
            step = "compiler_suggestion"
        if source in SOURCES:
            return {"group": "settled", "source": source, "step": step, "reason": reason}
        return {"group": "click_one", "source": None, "step": "click_one", "reason": reason}
    if evidence.get("proposal") in SOURCES:
        return {"group": "proposal", "source": evidence["proposal"], "step": evidence.get("proposal_step"),
                "reason": reason}
    return {"group": "full", "source": None, "step": None, "reason": reason}


def _sample_rank(episode_key: str, difference_id: object) -> float:
    digest = hashlib.sha256(f"{episode_key}:{difference_id}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16 ** 12)


def control_sample_ids(record: dict, episode_key: str) -> set:
    """Stable control sample over every settled card of the review generation.

    Drawn over pending *and* already decided cards, so confirming a sampled
    card never pulls another card into the sample.
    """

    items = [item for item in record.get("human_review", []) if isinstance(item, dict)]
    items += [
        decision["review_item"]
        for decision in record.get("human_decisions", [])
        if isinstance(decision, dict) and isinstance(decision.get("review_item"), dict)
    ]
    eligible = {}
    for item in items:
        group = base_group(item)
        if group["group"] == "settled" and group["reason"] not in NO_SAMPLE_REASONS:
            eligible[item.get("id")] = _sample_rank(episode_key, item.get("id"))
    if not eligible:
        return set()
    ranked = sorted(eligible, key=lambda difference_id: (eligible[difference_id], str(difference_id)))
    size = min(len(ranked), max(SAMPLE_MIN, math.ceil(len(ranked) * SAMPLE_SHARE)))
    return set(ranked[:size])


def materiality_groups(record: dict, episode_key: str, items: list[dict]) -> dict:
    """``{card id: group}`` for the given pending cards; all ``full`` when off."""

    if not queue_enabled():
        return {item.get("id"): {"group": "full", "source": None, "step": None, "reason": None} for item in items}
    sample = control_sample_ids(record, episode_key)
    groups = {}
    for item in items:
        group = base_group(item)
        if group["group"] == "settled" and item.get("id") in sample:
            group = {**group, "group": "sample"}
        groups[item.get("id")] = group
    return groups
