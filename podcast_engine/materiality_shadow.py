"""TASK-133: shadow-mode materiality filter for Human Review cards.

For every pending review card the Worker records -- without changing what
the reviewer sees -- whether the card would have been settled automatically
because choosing either reading cannot change the knowledge note:

1. ``compiler.materiality.rule_verdict`` (deterministic, free) settles the
   obvious cases -- including a sponsor read only one source carries and the
   same numbers written differently -- and sends every other card with a
   number, or with a long stretch only one source has, to the reviewer;
2. the rest goes to the materiality judge: a cheap model behind the
   verified OpenRouter preset ``PODCAST_MATERIALITY_JUDGE_PRESET`` whose
   system prompt must equal ``prompts/review/materiality-judge-v3.md``. It
   is asked three times, with the readings in both orders; each answer also
   names the more plausible reading. The card counts as immaterial only when
   every answer is ``same_content``. Any other answer, error or
   unavailability leaves the card with the reviewer;
3. ``compiler.materiality_choice`` records, from evidence and never from the
   source, the reading to keep for a settled card (``use``/``use_step``) or a
   preselected proposal for a card that stays with the reviewer
   (``proposal``/``proposal_step``);
4. the outcome is stored on the card as ``materiality_shadow`` evidence in
   the review record (generation CAS, identity-checked, idempotent). The
   review page, tiers, notifications and decisions do not read it.

Every judge call reserves and settles its own episode AI-budget attempt
(stage ``materiality``) through the transcript-review OpenRouter path.
The run is best-effort: it never blocks the review notification.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from google.api_core.exceptions import PreconditionFailed

from compiler.materiality import MATERIALITY_POLICY_VERSION, materiality_inputs, rule_verdict
from compiler.materiality_choice import choose_reading, proposal

from .ai_budget import STAGE_MATERIALITY, BudgetLedgerError
from .ai_pricing import PricingResolutionError
from .episode_contract import now_iso
from .human_review import load_review_record_with_generation, pending_review_items, save_review_record
from .preset_provenance import PresetProvenance, fetch_current_designated_preset

EVIDENCE_KEY = "materiality_shadow"
PRESET_ENV = "PODCAST_MATERIALITY_JUDGE_PRESET"
DEFAULT_PRESET = "podcast-materiality-judge"
SHADOW_ENV = "PODCAST_WORKER_MATERIALITY_SHADOW"
API_KEY_ENV = "PODCAST_TRANSCRIPT_REVIEW_API_KEY"
PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "review" / "materiality-judge-v3.md"

# Readings in both orders; the evaluated policy asks three times.
JUDGE_ORDERS = (("apple", "whisper"), ("whisper", "apple"), ("apple", "whisper"))
VERDICTS = ("same_content", "changes_content")
BETTER_READINGS = ("reading_1", "reading_2", "unclear")
MAX_COMPLETION_TOKENS = 600
TEMPERATURE = 0.7
REASON_MAX_LENGTH = 200
# One card at a time: parallel judges contended on the episode budget ledger
# and the review record (2026-10-06 backfill: BudgetConcurrencyError and
# failed evidence writes). Sequential is fast enough for one episode.
SHADOW_WORKERS = 1
SHADOW_MAX_ITEMS = 120
SHADOW_MAX_CONSECUTIVE_FAILURES = 5
SHADOW_WALL_CLOCK_SECONDS = 900
MAX_CAS_ATTEMPTS = 5


def prompt_text() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def prompt_sha256() -> str:
    return "sha256:" + hashlib.sha256(prompt_text().encode("utf-8")).hexdigest()


def judge_preset() -> str:
    return os.getenv(PRESET_ENV, "").strip() or DEFAULT_PRESET


def shadow_enabled() -> bool:
    if os.getenv(SHADOW_ENV, "1").strip().lower() in {"0", "false", "off", "no"}:
        return False
    return bool(os.getenv(API_KEY_ENV, "").strip())


def evidence_identity(inputs: dict) -> str:
    """Identity of one card's materiality question under the current policy."""

    payload = json.dumps(
        {"policy": MATERIALITY_POLICY_VERSION, "prompt": prompt_sha256(), "inputs": inputs},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def user_message(inputs: dict, first: str, second: str) -> str:
    def show(text: str) -> str:
        return text if text and text.strip() else "(nothing)"

    return (
        f"Context before: ...{inputs['left']}\n"
        f"Context after: {inputs['right']}...\n\n"
        f"Reading 1 of the disputed span: {show(inputs[first])}\n"
        f"Reading 2 of the disputed span: {show(inputs[second])}"
    )


def response_format() -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "materiality",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "verdict": {"type": "string", "enum": list(VERDICTS)},
                    "better_reading": {"type": "string", "enum": list(BETTER_READINGS)},
                    "reason": {"type": "string"},
                },
                "required": ["verdict", "better_reading", "reason"],
                "additionalProperties": False,
            },
        },
    }


def judge_payload(inputs: dict, first: str, second: str, provenance: PresetProvenance) -> dict:
    """One request under verified preset authority whose prompt is the repo prompt."""

    if (
        not provenance.verified
        or not isinstance(provenance.config, dict)
        or not isinstance(provenance.system_prompt, str)
    ):
        raise RuntimeError("Verified materiality-judge preset provenance is required")
    # The OpenRouter preset editor trims surrounding whitespace.
    if provenance.system_prompt.strip() != prompt_text().strip():
        raise RuntimeError("Materiality-judge preset system prompt differs from the repository prompt")
    payload = copy.deepcopy(provenance.config)
    payload.update({
        "temperature": TEMPERATURE,
        "max_tokens": MAX_COMPLETION_TOKENS,
        "reasoning": {"effort": "low", "exclude": True},
        "response_format": response_format(),
        "messages": [
            {"role": "system", "content": provenance.system_prompt},
            {"role": "user", "content": user_message(inputs, first, second)},
        ],
        "usage": {"include": True},
    })
    return payload


def parse_verdict(response_payload: object) -> tuple[str, str, str]:
    """Strictly read ``(verdict, better_reading, reason)``; anything else raises ValueError."""

    try:
        content = response_payload["choices"][0]["message"]["content"]  # type: ignore[index]
    except (TypeError, KeyError, IndexError) as error:
        raise ValueError("Judge response has no message content") from error
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except json.JSONDecodeError as error:
            raise ValueError("Judge answer is not JSON") from error
    if not isinstance(content, dict) or content.get("verdict") not in VERDICTS:
        raise ValueError("Judge answer has no valid verdict")
    if content.get("better_reading") not in BETTER_READINGS:
        raise ValueError("Judge answer has no valid better_reading")
    reason = content.get("reason")
    return content["verdict"], content["better_reading"], (reason if isinstance(reason, str) else "")[:REASON_MAX_LENGTH]


def _default_post(payload: dict, **kwargs):
    from .review import _post_openrouter

    return _post_openrouter(payload, **kwargs)


def judge_card(
    inputs: dict,
    provenance: PresetProvenance,
    *,
    episode_key: str,
    source_fingerprint: str,
    post: Callable[..., object] = _default_post,
) -> dict:
    """Ask the judge three times in both reading orders.

    There is no early stop: a card that stays with the reviewer also needs
    three ``better_reading`` votes for a preselected proposal. ``votes`` maps
    each answer's better reading to its source (``apple``, ``whisper`` or
    ``unclear``).
    """

    verdicts: list[str] = []
    votes: list[str] = []
    reasons: list[str] = []
    for first, second in JUDGE_ORDERS:
        response = post(
            judge_payload(inputs, first, second, provenance),
            episode_key=episode_key,
            source_fingerprint=source_fingerprint,
            stage=STAGE_MATERIALITY,
            provenance=provenance,
        )
        body = response.json() if hasattr(response, "json") else response
        verdict, better, reason = parse_verdict(body)
        verdicts.append(verdict)
        votes.append(first if better == "reading_1" else second if better == "reading_2" else "unclear")
        reasons.append(reason)
    immaterial = len(verdicts) == len(JUDGE_ORDERS) and all(v == "same_content" for v in verdicts)
    return {"immaterial": immaterial, "verdicts": verdicts, "votes": votes, "judge_reasons": reasons}


def _choice(inputs: dict, reason: str, immaterial: bool, votes: list[str] | None = None) -> dict:
    """The reading to keep (settled card) or the preselected proposal (kept card)."""

    if immaterial:
        use, step = choose_reading(inputs, reason=reason, votes=votes)
        return {"use": use, "use_step": step}
    source, step = proposal(inputs, reason=reason, votes=votes)
    return {"proposal": source, "proposal_step": step}


def _evidence(identity: str, route: str, reason: str, immaterial: bool, *, now: Callable[[], str], **extra) -> dict:
    return {
        "policy_version": MATERIALITY_POLICY_VERSION,
        "identity": identity,
        "route": route,
        "reason": reason,
        "immaterial": immaterial,
        "recorded_at": now(),
        **extra,
    }


def finalize_evidence(
    episode_key: str,
    difference_id: int,
    evidence: dict,
    *,
    load_record: Callable[[str], tuple[dict, int]] = load_review_record_with_generation,
    save_record: Callable[..., dict] = save_review_record,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    """Generation-verified write onto the current card; idempotent.

    Returns ``"recorded"``, ``"unchanged"`` (same identity already stored),
    or ``"stale"`` (the card disappeared or its inputs changed meanwhile).
    """

    for attempt in range(MAX_CAS_ATTEMPTS):
        if attempt:
            sleep(0.25 * attempt + random.random() * 0.25)
        record, generation = load_record(episode_key)
        updated = copy.deepcopy(record)
        for index, candidate in enumerate(updated.get("human_review", [])):
            if not isinstance(candidate, dict) or candidate.get("id") != difference_id:
                continue
            inputs = materiality_inputs(candidate)
            if inputs is None or evidence_identity(inputs) != evidence["identity"]:
                return "stale"
            existing = candidate.get(EVIDENCE_KEY)
            if isinstance(existing, dict) and existing.get("identity") == evidence["identity"]:
                return "unchanged"
            updated["human_review"][index] = {**candidate, EVIDENCE_KEY: evidence}
            break
        else:
            return "stale"
        try:
            save_record(episode_key, updated, if_generation_match=generation)
        except PreconditionFailed:
            continue
        return "recorded"
    raise RuntimeError("Review record changed repeatedly while recording materiality evidence")


def run_materiality_shadow(
    episode: dict,
    *,
    load_record: Callable[[str], tuple[dict, int]] = load_review_record_with_generation,
    save_record: Callable[..., dict] = save_review_record,
    fetch_preset: Callable[..., PresetProvenance] = fetch_current_designated_preset,
    post: Callable[..., object] = _default_post,
    workers: int = SHADOW_WORKERS,
    monotonic: Callable[[], float] = time.monotonic,
    now: Callable[[], str] = now_iso,
) -> dict:
    """Record shadow materiality evidence for pending cards; return a summary."""

    summary: dict = {
        "status": "skipped",
        "cards": 0,
        "already_recorded": 0,
        "rule_immaterial": 0,
        "judge_immaterial": 0,
        "reviewer": 0,
        "judge_unavailable": 0,
        "failed": 0,
        "failure_reasons": {},
        "stopped_reason": None,
    }
    if not shadow_enabled():
        summary["status"] = "disabled"
        return summary
    episode_key = episode["episode_key"]
    try:
        record, _ = load_record(episode_key)
    except Exception as error:  # noqa: BLE001 - notification must still go out
        summary["status"] = "record_unavailable"
        summary["stopped_reason"] = type(error).__name__
        return summary
    source_fingerprint = record.get("source_fingerprint")

    def write(difference_id: int, evidence: dict) -> None:
        finalize_evidence(episode_key, difference_id, evidence, load_record=load_record, save_record=save_record)

    judge_queue: list[tuple[int, dict, str]] = []
    for item in pending_review_items(record)[:SHADOW_MAX_ITEMS]:
        difference_id = item.get("id")
        inputs = materiality_inputs(item)
        if not isinstance(difference_id, int) or inputs is None:
            continue
        summary["cards"] += 1
        identity = evidence_identity(inputs)
        existing = item.get(EVIDENCE_KEY)
        if isinstance(existing, dict) and existing.get("identity") == identity:
            summary["already_recorded"] += 1
            continue
        route, reason = rule_verdict(inputs)
        if route == "advertisement":
            # Settled by keeping the reading without the sponsor read.
            write(difference_id, _evidence(identity, "rule", "advertisement", True, now=now,
                                           **_choice(inputs, reason, True)))
            summary["rule_immaterial"] += 1
        elif route == "immaterial":
            write(difference_id, _evidence(identity, "rule", reason, True, now=now, **_choice(inputs, reason, True)))
            summary["rule_immaterial"] += 1
        elif route == "reviewer":
            write(difference_id, _evidence(identity, "reviewer", reason, False, now=now,
                                           **_choice(inputs, reason, False)))
            summary["reviewer"] += 1
        else:
            judge_queue.append((difference_id, inputs, identity))

    if not judge_queue:
        summary["status"] = "completed"
        return summary

    provenance = None
    api_key = os.getenv(API_KEY_ENV, "").strip()
    try:
        provenance = fetch_preset(judge_preset(), api_key=api_key, verified_at=now())
    except Exception:  # noqa: BLE001
        provenance = None
    if (
        not isinstance(provenance, PresetProvenance)
        or not provenance.verified
        or not isinstance(provenance.system_prompt, str)
        or provenance.system_prompt.strip() != prompt_text().strip()
        or not isinstance(source_fingerprint, str)
    ):
        # Without a verified judge the cards simply stay with the reviewer;
        # nothing is recorded, so a later run can still judge them.
        summary["judge_unavailable"] = len(judge_queue)
        summary["status"] = "judge_unavailable"
        return summary

    deadline = monotonic() + SHADOW_WALL_CLOCK_SECONDS
    state = {"consecutive_failures": 0, "stop": None}
    failures: Counter = Counter()
    preset_identity = {
        "slug": provenance.slug,
        "version_id": provenance.version_id,
        "config_digest": provenance.config_digest,
        "model": (provenance.config or {}).get("model"),
    }

    def run_one(entry: tuple[int, dict, str]) -> str:
        difference_id, inputs, identity = entry
        if state["stop"] is not None:
            return "skipped"
        if monotonic() > deadline:
            state["stop"] = "wall_clock"
            return "skipped"
        try:
            result = judge_card(
                inputs, provenance, episode_key=episode_key, source_fingerprint=source_fingerprint, post=post
            )
        except (BudgetLedgerError, PricingResolutionError) as error:
            state["stop"] = type(error).__name__
            return "stopped"
        except Exception as error:  # noqa: BLE001 - one card must not stop the queue
            failures[f"judge:{type(error).__name__}"] += 1
            state["consecutive_failures"] += 1
            if state["consecutive_failures"] >= SHADOW_MAX_CONSECUTIVE_FAILURES:
                state["stop"] = "consecutive_failures"
            return "failed"
        state["consecutive_failures"] = 0
        immaterial = result["immaterial"]
        try:
            write(
                difference_id,
                _evidence(
                    identity,
                    "judge" if immaterial else "reviewer",
                    "judge_unanimous" if immaterial else "judge_not_unanimous",
                    immaterial,
                    now=now,
                    verdicts=result["verdicts"],
                    votes=result["votes"],
                    judge_reasons=result["judge_reasons"],
                    preset=preset_identity,
                    **_choice(inputs, "judge", immaterial, result["votes"]),
                ),
            )
        except Exception as error:  # noqa: BLE001
            failures[f"write:{type(error).__name__}"] += 1
            return "failed"
        return "judge_immaterial" if immaterial else "reviewer"

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        outcomes = Counter(pool.map(run_one, judge_queue))
    summary["judge_immaterial"] = outcomes.get("judge_immaterial", 0)
    summary["reviewer"] += outcomes.get("reviewer", 0)
    summary["failed"] = outcomes.get("failed", 0)
    summary["failure_reasons"] = dict(failures)
    summary["stopped_reason"] = state["stop"]
    summary["status"] = "stopped" if state["stop"] else "completed"
    return summary
