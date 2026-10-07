#!/usr/bin/env python3
"""Create a strictly read-only Human Review telemetry report from GCS.

The tool reads only ``episodes.json`` and canonical per-episode JSON objects.
It never imports podcast_engine's I/O-performing resolver/review/compilation
modules, and it never calls a Cloud Storage mutation method.  Output files
are local derived artifacts.

TASK-076 Task 10: assisted-review routing/state distributions reuse
``compiler.assisted_review.derive_assisted_state`` -- the same pure, I/O-free
policy function the review workflow's own read-time projection uses (see
``podcast_engine.human_review.project_assisted_review_items``) -- so this
report can never drift from the routing/eligibility/state semantics actually
applied to reviewers, without this script importing any podcast_engine
module (matching the existing precedent of
``scripts/evaluate_human_review_policy.py`` importing ``compiler.review_policy``
directly). Budget/session figures are read from each episode generation's
own ledger object at its known GCS path (derived from the review record's
``source_fingerprint`` -- the same identity podcast_engine/ai_budget.py's
``budget_ledger_path`` uses; this is distinct from ``input_fingerprint``,
the broader review-reconciliation identity), never by listing
prepare-session objects. Assisted machine-support counts are always
reported separately from
``human_decisions`` and never folded into it: machine support is advisory
evidence, never a human decision.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from compiler.assisted_review import derive_assisted_state


DEFAULT_BUCKET = "podcast-worker-data-506417"
REPORT_SCHEMA_VERSION = 1
SOURCE_CHOICES = ("apple", "whisper", "third", "custom", "<unknown>")
UNKNOWN = "<unknown>"


def _known_source(value: object) -> str:
    if isinstance(value, str) and value.strip().lower() in SOURCE_CHOICES[:-1]:
        return value.strip().lower()
    return UNKNOWN


def _label(value: object) -> str:
    return value if isinstance(value, str) and value else UNKNOWN


def _objects(value: object) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _count(value: object) -> int | None:
    return len(value) if isinstance(value, list) else None


def _read_json(bucket: Any, object_name: str) -> dict[str, Any] | list[Any] | None:
    """Read one JSON object, returning None when the canonical object is absent.

    This deliberately uses only ``exists`` and ``download_as_bytes``.  In
    particular, it does not list, upload, patch, delete, or rewrite GCS data.
    """

    blob = bucket.blob(object_name)
    if not blob.exists():
        return None
    payload = json.loads(blob.download_as_bytes().decode("utf-8"))
    return payload if isinstance(payload, (dict, list)) else None


def _source_counts(values: list[dict[str, Any]], key: str = "chosen_source") -> Counter[str]:
    return Counter(_known_source(item.get(key)) for item in values)


def _source_summary(counts: Counter[str], denominator: int) -> dict[str, dict[str, int | float]]:
    return {
        source: {
            "count": counts.get(source, 0),
            "denominator": denominator,
            "percentage": round(100 * counts.get(source, 0) / denominator, 2) if denominator else 0.0,
        }
        for source in SOURCE_CHOICES
    }


def _breakdown_summary(values: dict[str, Counter[str]]) -> dict[str, dict[str, dict[str, int | float]]]:
    output: dict[str, dict[str, dict[str, int | float]]] = {}
    for label in sorted(values):
        counts = values[label]
        output[label] = _source_summary(counts, sum(counts.values()))
    return output


def _metric(value: int | None) -> dict[str, int | str | None]:
    return {
        "availability": "available" if value is not None else "unavailable",
        "count": value,
    }


def _resolver_episode(record: dict[str, Any]) -> dict[str, Any]:
    """Describe only resolver fields actually present in this durable record."""

    outcomes = record.get("resolver_outcomes")
    outcome_counts = Counter(
        _label(item.get("status")) for item in _objects(outcomes)
    ) if isinstance(outcomes, list) else None
    return {
        "schema_version": record.get("schema_version", UNKNOWN),
        "policy_version": _label(record.get("policy_version")),
        "reviewed_at": record.get("reviewed_at", UNKNOWN),
        "metrics": {
            "resolver_candidates": _metric(_count(record.get("resolver_item_ids"))),
            "accepted": _metric(_count(record.get("accepted"))),
            "human_review_fallback": _metric(_count(record.get("resolver_review"))),
            "structurally_bypassed": _metric(_count(record.get("resolver_bypassed"))),
            "deferred": _metric(_count(record.get("resolver_deferred"))),
            "resolver_outcomes": {
                "availability": "available" if outcome_counts is not None else "unavailable",
                "counts": dict(sorted(outcome_counts.items())) if outcome_counts is not None else None,
            },
        },
    }


_ASSISTED_PREPARED_STATES = (
    "machine_supported_apple",
    "machine_supported_whisper",
    "evidence_conflict",
    "ambiguous_audio",
    "neither_source_supported",
    "audio_unavailable",
)
_ASSISTED_MACHINE_SUPPORTED_STATES = ("machine_supported_apple", "machine_supported_whisper")
_ASSISTED_UNRESOLVED_STATES = ("ambiguous_audio", "neither_source_supported", "audio_unavailable")


def _assisted_review_episode(items: list[Any]) -> dict[str, Any]:
    """Routing-reason and assisted-state distributions for one episode's
    pending Human Review cards.

    Recomputed read-time from the same persisted card fields the review
    workflow itself projects from (never a separately persisted field --
    routing/assisted state is never written to the review record). A card
    that fails to evaluate (malformed/missing required fields) is counted
    under ``malformed`` rather than silently dropped or raising, matching
    this script's descriptive-evidence, never-hard-fail posture.

    ``candidate`` counts every routing-eligible card; ``prepared`` is the
    subset whose Third-ASR evidence preparation has completed one way or
    another (i.e. left ``audio_pending``); ``machine_supported``,
    ``conflict``, and ``unresolved`` are a mutually exclusive, exhaustive
    partition of ``prepared`` by ``derive_assisted_state``'s own state
    machine. This is advisory machine-support evidence, never a human
    decision, and must never be added into the ``human_decisions`` totals
    reported elsewhere.
    """

    reason_codes: Counter[str] = Counter()
    states: Counter[str] = Counter()
    malformed = 0
    candidate = 0
    for item in items:
        if not isinstance(item, dict):
            malformed += 1
            continue
        try:
            derived = derive_assisted_state(item)
        except Exception:
            malformed += 1
            continue
        routing = derived.get("routing")
        if not isinstance(routing, dict):
            malformed += 1
            continue
        reason_codes.update(_label(code) for code in routing.get("reason_codes") or [])
        if not routing.get("eligible"):
            continue
        candidate += 1
        state = derived.get("state")
        states[_label(state)] += 1

    prepared = sum(states[state] for state in _ASSISTED_PREPARED_STATES)
    return {
        "candidate": candidate,
        "prepared": prepared,
        "audio_pending": states.get("audio_pending", 0),
        "machine_supported": sum(states[state] for state in _ASSISTED_MACHINE_SUPPORTED_STATES),
        "conflict": states.get("evidence_conflict", 0),
        "unresolved": sum(states[state] for state in _ASSISTED_UNRESOLVED_STATES),
        "malformed": malformed,
        "state_distribution": dict(sorted(states.items())),
        "routing_reason_distribution": dict(sorted(reason_codes.items())),
    }


def _aggregate_assisted_review(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    entries = [episode["assisted_review"] for episode in episodes if episode.get("assisted_review")]
    reason_codes: Counter[str] = Counter()
    states: Counter[str] = Counter()
    totals = {name: 0 for name in ("candidate", "prepared", "audio_pending", "machine_supported", "conflict", "unresolved", "malformed")}
    for entry in entries:
        for name in totals:
            totals[name] += entry[name]
        states.update(entry["state_distribution"])
        reason_codes.update(entry["routing_reason_distribution"])
    return {
        **totals,
        "state_distribution": dict(sorted(states.items())),
        "routing_reason_distribution": dict(sorted(reason_codes.items())),
    }


_LEDGER_COMMITTED_STATES = ("reserved", "settled", "uncertain")


def _ledger_summary(ledger: dict[str, Any] | None) -> dict[str, Any] | None:
    """Summarize one episode generation's AI budget ledger, read verbatim.

    ``committed_usd`` sums, per attempt still counting against the cap
    (``reserved``/``settled``/``uncertain`` -- never ``released``, which is
    provably freed), the attempt's settled amount when known, else its
    conservative reservation -- never a guess. Third-ASR figures are the
    same breakdown restricted to ``third_asr: true`` attempts, i.e. every
    whole-selection prepare-session reservation this episode generation has
    made; there is no separate enumeration of prepare-session objects.
    """

    if not isinstance(ledger, dict):
        return None
    attempts = ledger.get("attempts")
    if not isinstance(attempts, dict):
        return None
    by_state: Counter[str] = Counter()
    by_state_third_asr: Counter[str] = Counter()
    committed_usd = Decimal("0")
    committed_usd_third_asr = Decimal("0")
    for attempt in attempts.values():
        if not isinstance(attempt, dict):
            continue
        state = _label(attempt.get("state"))
        by_state[state] += 1
        is_third_asr = bool(attempt.get("third_asr"))
        if is_third_asr:
            by_state_third_asr[state] += 1
        if state in _LEDGER_COMMITTED_STATES:
            amount = attempt.get("settled_usd", attempt.get("reserved_usd"))
            try:
                value = Decimal(str(amount))
            except (InvalidOperation, TypeError):
                value = Decimal("0")
            committed_usd += value
            if is_third_asr:
                committed_usd_third_asr += value
    return {
        "hard_cap_usd": ledger.get("hard_cap_usd"),
        "third_asr_subcap_usd": ledger.get("third_asr_subcap_usd"),
        "attempts_by_state": dict(sorted(by_state.items())),
        "third_asr_attempts_by_state": dict(sorted(by_state_third_asr.items())),
        "committed_usd": str(committed_usd),
        "third_asr_committed_usd": str(committed_usd_third_asr),
        "integrity_failure": bool(ledger.get("integrity_failure")),
    }


def _aggregate_budget(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    summaries = [episode["budget"] for episode in episodes if episode.get("budget")]
    by_state: Counter[str] = Counter()
    by_state_third_asr: Counter[str] = Counter()
    committed_usd = Decimal("0")
    committed_usd_third_asr = Decimal("0")
    integrity_failures = 0
    for summary in summaries:
        by_state.update(summary["attempts_by_state"])
        by_state_third_asr.update(summary["third_asr_attempts_by_state"])
        try:
            committed_usd += Decimal(summary["committed_usd"])
            committed_usd_third_asr += Decimal(summary["third_asr_committed_usd"])
        except InvalidOperation:
            pass
        integrity_failures += 1 if summary["integrity_failure"] else 0
    return {
        "ledgers_available": len(summaries),
        "attempts_by_state": dict(sorted(by_state.items())),
        "third_asr_attempts_by_state": dict(sorted(by_state_third_asr.items())),
        "committed_usd": str(committed_usd),
        "third_asr_committed_usd": str(committed_usd_third_asr),
        "integrity_failures": integrity_failures,
    }


def _compiler_episode(report: dict[str, Any]) -> dict[str, Any]:
    differences = _objects(report.get("differences"))
    source_only = report.get("source_only_summary")
    return {
        "difference_count": len(differences),
        "review_required": report.get("review_required", UNKNOWN),
        "severity": dict(sorted(Counter(_label(item.get("severity")) for item in differences).items())),
        "kind": dict(sorted(Counter(_label(item.get("kind")) for item in differences).items())),
        "resolver_category": dict(sorted(Counter(_label(item.get("resolver_category")) for item in differences).items())),
        "preservation_class": dict(sorted(Counter(_label(item.get("preservation_class")) for item in differences).items())),
        "source_only_summary": source_only if isinstance(source_only, dict) else None,
    }


def _aggregate_metric(episodes: list[dict[str, Any]], name: str) -> dict[str, Any]:
    available = [episode["resolver"]["metrics"][name]["count"] for episode in episodes if episode.get("resolver") and episode["resolver"]["metrics"][name]["count"] is not None]
    total = len(episodes)
    return {
        "availability": "available" if len(available) == total and total else "mixed" if available else "unavailable",
        "available_episodes": len(available),
        "unavailable_episodes": total - len(available),
        "count": sum(available) if available else None,
    }


def _aggregate_resolver(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    resolver_episodes = [episode for episode in episodes if episode.get("resolver")]
    metrics = {
        name: _aggregate_metric(resolver_episodes, name)
        for name in (
            "resolver_candidates",
            "accepted",
            "human_review_fallback",
            "structurally_bypassed",
            "deferred",
        )
    }
    outcome_values = [
        episode["resolver"]["metrics"]["resolver_outcomes"]["counts"]
        for episode in resolver_episodes
        if episode["resolver"]["metrics"]["resolver_outcomes"]["counts"] is not None
    ]
    outcome_counts: Counter[str] = Counter()
    for values in outcome_values:
        outcome_counts.update(values)
    metrics["resolver_outcomes"] = {
        "availability": "available" if len(outcome_values) == len(resolver_episodes) and resolver_episodes else "mixed" if outcome_values else "unavailable",
        "available_episodes": len(outcome_values),
        "unavailable_episodes": len(resolver_episodes) - len(outcome_values),
        "counts": dict(sorted(outcome_counts.items())) if outcome_values else None,
    }
    return {
        "resolver_records": len(resolver_episodes),
        "schema_versions": dict(sorted(Counter(str(item["resolver"]["schema_version"]) for item in resolver_episodes).items())),
        "metrics": metrics,
    }


def _aggregate_compiler(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    compiler_episodes = [episode["compiler"] for episode in episodes if episode.get("compiler")]
    distributions: dict[str, Counter[str]] = {
        "severity": Counter(), "kind": Counter(), "resolver_category": Counter(), "preservation_class": Counter(),
    }
    source_only: Counter[str] = Counter()
    source_only_available = 0
    review_required: Counter[str] = Counter()
    for report in compiler_episodes:
        for name, counts in distributions.items():
            counts.update(report[name])
        if isinstance(report["source_only_summary"], dict):
            source_only_available += 1
            for key, value in report["source_only_summary"].items():
                if isinstance(value, int):
                    source_only[str(key)] += value
        review_required[str(report["review_required"])] += 1
    return {
        "compiler_reports": len(compiler_episodes),
        "total_differences": sum(report["difference_count"] for report in compiler_episodes),
        **{name: dict(sorted(counts.items())) for name, counts in distributions.items()},
        "source_only_summary": {
            "availability": "available" if source_only_available == len(compiler_episodes) and compiler_episodes else "mixed" if source_only_available else "unavailable",
            "available_episodes": source_only_available,
            "counts": dict(sorted(source_only.items())) if source_only_available else None,
        },
        "current_review_required": dict(sorted(review_required.items())),
    }


def build_telemetry(bucket: Any, *, episode_key: str | None = None, bucket_name: str = DEFAULT_BUCKET) -> dict[str, Any]:
    """Read canonical review/compiler objects and return a deterministic report."""

    index = _read_json(bucket, "episodes.json")
    if not isinstance(index, list):
        raise ValueError("episodes.json must contain a JSON array")
    episode_records = [item for item in index if isinstance(item, dict)]
    if episode_key is not None:
        episode_records = [item for item in episode_records if item.get("episode_key") == episode_key]

    decisions: list[dict[str, Any]] = []
    by_podcast: dict[str, Counter[str]] = defaultdict(Counter)
    by_kind: dict[str, Counter[str]] = defaultdict(Counter)
    by_category: dict[str, Counter[str]] = defaultdict(Counter)
    by_severity: dict[str, Counter[str]] = defaultdict(Counter)
    by_tier: Counter[str] = Counter()
    episodes: list[dict[str, Any]] = []

    for episode in sorted(episode_records, key=lambda item: str(item.get("episode_key", ""))):
        key = episode.get("episode_key")
        if not isinstance(key, str) or not key:
            continue
        resolver = _read_json(bucket, f"episodes/{key}/review/resolver.json")
        compiler = _read_json(bucket, f"episodes/{key}/compiled/report.json")
        if not isinstance(resolver, dict) and not isinstance(compiler, dict):
            continue
        podcast = _label(episode.get("podcast"))
        episode_decisions = _objects(resolver.get("human_decisions")) if isinstance(resolver, dict) else []
        for decision in episode_decisions:
            source = _known_source(decision.get("chosen_source"))
            item = decision.get("review_item")
            review_item = item if isinstance(item, dict) else {}
            decisions.append(decision)
            by_podcast[podcast][source] += 1
            by_kind[_label(review_item.get("kind"))][source] += 1
            by_category[_label(review_item.get("category"))][source] += 1
            by_severity[_label(review_item.get("severity"))][source] += 1
            by_tier[_decision_tier(decision)] += 1
        raw_pending_items = resolver.get("human_review") if isinstance(resolver, dict) else None
        fingerprint = resolver.get("source_fingerprint") if isinstance(resolver, dict) else None
        ledger = None
        if isinstance(fingerprint, str) and fingerprint.startswith("sha256:") and len(fingerprint) == 71:
            ledger = _read_json(bucket, f"episodes/{key}/ai/budgets/{fingerprint.split(':', 1)[1]}.json")
        entry: dict[str, Any] = {
            "episode_key": key,
            "podcast": podcast,
            "title": _label(episode.get("title")),
            "human_decisions": len(episode_decisions),
            "decisions_by_tier": dict(sorted(Counter(_decision_tier(d) for d in episode_decisions).items())),
            "source_choice": _source_summary(_source_counts(episode_decisions), len(episode_decisions)),
            "resolver": _resolver_episode(resolver) if isinstance(resolver, dict) else None,
            "compiler": _compiler_episode(compiler) if isinstance(compiler, dict) else None,
            # Deliberately NOT pre-filtered through _objects(): a malformed
            # (non-dict) pending-card entry must be visible in the
            # "malformed" count below, never silently dropped.
            "assisted_review": _assisted_review_episode(raw_pending_items) if isinstance(raw_pending_items, list) else None,
            "budget": _ledger_summary(ledger),
        }
        episodes.append(entry)

    source_counts = _source_counts(decisions)
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "scope": {
            "bucket": bucket_name,
            "episode_key": episode_key,
            "warning": "Telemetry is descriptive evidence, not a compiler source prior.",
        },
        "human_decisions": {
            "reviewed_episodes": sum(episode["human_decisions"] > 0 for episode in episodes),
            "total": len(decisions),
            "source_choice": _source_summary(source_counts, len(decisions)),
            "by_podcast": _breakdown_summary(by_podcast),
            "by_kind": _breakdown_summary(by_kind),
            "by_category": _breakdown_summary(by_category),
            "by_severity": _breakdown_summary(by_severity),
            "by_tier": dict(sorted(by_tier.items())),
        },
        "resolver": _aggregate_resolver(episodes),
        "compiler": _aggregate_compiler(episodes),
        "assisted_review": _aggregate_assisted_review(episodes),
        "budget": _aggregate_budget(episodes),
        "episodes": episodes,
    }


def _markdown_source_table(summary: dict[str, dict[str, int | float]]) -> list[str]:
    lines = ["| Source | Decisions |", "| --- | ---: |"]
    for source, value in summary.items():
        lines.append(f"| {source} | {value['count']}/{value['denominator']} ({value['percentage']:.2f}%) |")
    return lines


def _markdown_breakdown(title: str, values: dict[str, dict[str, dict[str, int | float]]]) -> list[str]:
    lines = [f"### {title}", ""]
    if not values:
        return lines + ["No human decisions with this breakdown.", ""]
    lines += ["| Value | Apple | Whisper | Third | Custom | Unknown |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for label, sources in values.items():
        cells = [f"{sources[source]['count']}/{sources[source]['denominator']} ({sources[source]['percentage']:.2f}%)" for source in SOURCE_CHOICES]
        lines.append("| " + str(label).replace("|", "\\|") + " | " + " | ".join(cells) + " |")
    return lines + [""]


def _decision_tier(decision: dict[str, Any]) -> str:
    """Tier recorded with a decision (TASK-127); older decisions have none."""

    tier = decision.get("review_tier")
    value = tier.get("tier") if isinstance(tier, dict) else None
    return value if value in {"A", "B", "C"} else "unrecorded"


def render_markdown(report: dict[str, Any]) -> str:
    """Render a readable, deterministic summary without raw transcript evidence."""

    human = report["human_decisions"]
    lines = [
        "# Podcast Worker Human Review telemetry",
        "",
        "> **Telemetry is descriptive evidence, not a compiler source prior.**",
        "",
        "Historical percentages must not be used to derive source-preference rules.",
        "",
        "## Executive summary",
        "",
        f"- Reviewed episodes: **{human['reviewed_episodes']}**",
        f"- Human decisions: **{human['total']}**",
        f"- Resolver records: **{report['resolver']['resolver_records']}**",
        f"- Compiler reports: **{report['compiler']['compiler_reports']}**",
    ]
    if human["reviewed_episodes"] < 10:
        lines += ["", "> **Small-sample warning:** fewer than 10 reviewed episodes are represented; percentages are descriptive only."]
    lines += ["", "## Source-choice table", ""] + _markdown_source_table(human["source_choice"])
    lines += [""]
    lines += _markdown_breakdown("By podcast", human["by_podcast"])
    lines += _markdown_breakdown("By review-item kind", human["by_kind"])
    lines += _markdown_breakdown("By review-item category", human["by_category"])
    lines += _markdown_breakdown("By review-item severity", human["by_severity"])
    if human.get("by_tier"):
        lines += ["## Decisions by review tier (TASK-127)", "", "| Tier | Decisions |", "| --- | ---: |"]
        lines += [f"| {tier} | {count} |" for tier, count in human["by_tier"].items()]
        lines += [""]
    resolver = report["resolver"]
    lines += ["## Resolver outcomes", "", "| Metric | Availability | Count |", "| --- | --- | ---: |"]
    for name, value in resolver["metrics"].items():
        if name == "resolver_outcomes":
            counts = value["counts"]
            rendered = ", ".join(f"{key}: {count}" for key, count in counts.items()) if counts else "unavailable"
            lines.append(f"| {name} | {value['availability']} | {rendered} |")
        else:
            lines.append(f"| {name} | {value['availability']} | {value['count'] if value['count'] is not None else 'unavailable'} |")
    compiler = report["compiler"]
    lines += ["", "## Compiler statistics", "", f"- Total differences: **{compiler['total_differences']}**", f"- Current `review_required` distribution: `{json.dumps(compiler['current_review_required'], sort_keys=True)}`", ""]
    for name in ("severity", "kind", "resolver_category", "preservation_class"):
        lines.append(f"- {name}: `{json.dumps(compiler[name], sort_keys=True)}`")
    assisted = report["assisted_review"]
    lines += [
        "",
        "## Assisted review (advisory machine support, never a human decision)",
        "",
        "> Counts below describe routing/eligibility and derived Third-ASR-evidence state on pending cards. They are never included in the Human decisions totals above.",
        "",
        f"- Candidates (routing-eligible): **{assisted['candidate']}**",
        f"- Prepared (evidence preparation completed): **{assisted['prepared']}** (awaiting preparation: {assisted['audio_pending']})",
        f"- Machine-supported (clear recommendation): **{assisted['machine_supported']}**",
        f"- Evidence conflict (flagged for human attention): **{assisted['conflict']}**",
        f"- Unresolved (prepared, no actionable recommendation): **{assisted['unresolved']}**",
    ]
    if assisted["malformed"]:
        lines.append(f"- Malformed/unevaluable cards: **{assisted['malformed']}**")
    lines += [
        "",
        f"- State distribution: `{json.dumps(assisted['state_distribution'], sort_keys=True)}`",
        f"- Routing-reason distribution: `{json.dumps(assisted['routing_reason_distribution'], sort_keys=True)}`",
    ]
    budget = report["budget"]
    lines += [
        "",
        "## Budget and prepare-session summary",
        "",
        f"- Episode generations with a ledger: **{budget['ledgers_available']}**",
        f"- Committed USD (reserved + settled + uncertain, across all stages): **{budget['committed_usd']}**",
        f"- Committed USD attributable to assisted Third-ASR: **{budget['third_asr_committed_usd']}**",
        f"- Attempts by state: `{json.dumps(budget['attempts_by_state'], sort_keys=True)}`",
        f"- Third-ASR attempts by state (i.e. prepare-session reservations): `{json.dumps(budget['third_asr_attempts_by_state'], sort_keys=True)}`",
    ]
    if budget["integrity_failures"]:
        lines.append(f"- **Ledgers with a recorded integrity failure: {budget['integrity_failures']}**")
    lines += ["", "## Per-episode summary", "", "| Episode | Podcast | Human decisions | Resolver schema | Policy | Compiler differences |", "| --- | --- | ---: | --- | --- | ---: |"]
    for episode in report["episodes"]:
        resolver_episode = episode["resolver"] or {}
        compiler_episode = episode["compiler"] or {}
        lines.append(
            "| {key} | {podcast} | {human} | {schema} | {policy} | {differences} |".format(
                key=episode["episode_key"], podcast=episode["podcast"], human=episode["human_decisions"],
                schema=resolver_episode.get("schema_version", UNKNOWN), policy=resolver_episode.get("policy_version", UNKNOWN),
                differences=compiler_episode.get("difference_count", UNKNOWN),
            )
        )
    for episode in report["episodes"]:
        resolver_episode = episode["resolver"] or {}
        compiler_episode = episode["compiler"] or {}
        source_choice = ", ".join(
            f"{source} {value['count']}/{value['denominator']} ({value['percentage']:.2f}%)"
            for source, value in episode["source_choice"].items()
        )
        resolver_metrics = resolver_episode.get("metrics", {})
        available_metrics = ", ".join(
            f"{name}: {value['count'] if value.get('count') is not None else 'unavailable'}"
            for name, value in resolver_metrics.items()
            if name != "resolver_outcomes"
        ) or "unavailable"
        outcomes = resolver_metrics.get("resolver_outcomes", {})
        outcome_counts = outcomes.get("counts")
        outcome_text = json.dumps(outcome_counts, sort_keys=True) if outcome_counts is not None else "unavailable"
        lines += [
            "",
            f"### {episode['episode_key']} — {episode['title']}",
            "",
            f"- Podcast: `{episode['podcast']}`",
            f"- Resolver reviewed at: `{resolver_episode.get('reviewed_at', UNKNOWN)}`",
            f"- Source choice: {source_choice}",
            f"- Resolver metrics: {available_metrics}; resolver outcomes: {outcome_text}",
            f"- Compiler difference count: {compiler_episode.get('difference_count', UNKNOWN)}",
        ]
    lines += ["", "## Interpretation warning", "", "Telemetry is descriptive evidence, not a compiler source prior. Do not infer rules such as ‘prefer Apple’ from these historical percentages.", ""]
    return "\n".join(lines)


def write_report(report: dict[str, Any], output_dir: str | Path) -> tuple[Path, Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "podcast-review-telemetry.json"
    markdown_path = destination / "podcast-review-telemetry.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a read-only Human Review telemetry report.")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET, help="GCS bucket to read (default: production bucket)")
    parser.add_argument("--output-dir", default="/tmp", help="Local output directory (default: /tmp)")
    parser.add_argument("--episode-key", help="Limit the report to one episode key")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    from google.cloud import storage

    bucket = storage.Client().bucket(args.bucket.removeprefix("gs://").strip("/"))
    report = build_telemetry(bucket, episode_key=args.episode_key, bucket_name=args.bucket)
    json_path, markdown_path = write_report(report, args.output_dir)
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
