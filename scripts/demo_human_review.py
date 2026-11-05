#!/usr/bin/env python3
"""Run the real Human Review frontend against a synthetic in-memory demo backend.

No cloud credentials, external podcast feeds, API keys, or production data are used.
Restart the process to reset the demo state.
"""

from __future__ import annotations

import io
import math
import struct
import sys
import wave
from copy import deepcopy
from pathlib import Path

# Allow direct execution via `python scripts/demo_human_review.py` from the
# repository root without requiring callers to set PYTHONPATH manually.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flask import Flask, Response, jsonify, request, send_file

from podcast_engine.review_web import PAGE


EPISODE_KEY = "demo-exercise-nutrition"
EPISODE = {
    "episode_key": EPISODE_KEY,
    "podcast": "Example Strength & Nutrition Podcast",
    "title": "Protein Timing, Hypertrophy Research & Exercise Selection",
}

_INITIAL_CARDS = [
    {
        "id": 1,
        "apple_text": "Creatine monohydrate is well studied for strength.",
        "whisper_text": "Creatine monohydrate's well studied for strength.",
        "apple_context": (
            "The guest is summarizing common evidence-based supplements. "
            "Creatine monohydrate is well studied for strength, and the discussion "
            "then moves to practical dosing."
        ),
        "whisper_context": (
            "The guest is summarizing common evidence-based supplements. "
            "Creatine monohydrate's well studied for strength, and the discussion "
            "then moves to practical dosing."
        ),
        "source_choices": {
            "apple": "Creatine monohydrate is well studied for strength.",
            "whisper": "Creatine monohydrate's well studied for strength.",
        },
        "materiality": {
            "group": "sample",
            "source": "apple",
            "step": "judge",
            "reason": "same_content",
        },
        "materiality_context": {
            "left": "The guest is summarizing common evidence-based supplements.",
            "right": "and the discussion then moves to practical dosing.",
        },
        "audio_window": {"start": 72.0, "end": 87.0, "duration": 15.0},
        "third_asr": None,
        "third_available": False,
        "third_window": None,
        "focus": {"scope": "full"},
        "suggestion": None,
        "triage": None,
        "anomaly": None,
        "display": {
            "category": "Control sample",
            "severity": "Low-impact wording",
            "reason": "Verify one filter-settled card",
        },
        "assisted_review": {
            "routing": {"eligible": False},
            "state": "audio_unavailable",
            "policy_version": "synthetic-demo-v2",
        },
    },
    {
        "id": 2,
        "apple_text": "short rest periods can make the set feel harder",
        "whisper_text": "shorter rest periods can make the set feel harder",
        "apple_context": (
            "The coach explains that short rest periods can make the set feel harder "
            "without changing the main training recommendation."
        ),
        "whisper_context": (
            "The coach explains that shorter rest periods can make the set feel harder "
            "without changing the main training recommendation."
        ),
        "source_choices": {
            "apple": "short rest periods can make the set feel harder",
            "whisper": "shorter rest periods can make the set feel harder",
        },
        "materiality": {
            "group": "click_one",
            "source": None,
            "step": "fuller",
            "reason": "same_content_no_decisive_source",
        },
        "materiality_context": {
            "left": "The coach explains that",
            "right": "without changing the main training recommendation.",
        },
        "audio_window": {"start": 118.0, "end": 133.0, "duration": 15.0},
        "third_asr": None,
        "third_available": False,
        "third_window": None,
        "focus": {"scope": "full"},
        "suggestion": None,
        "triage": None,
        "anomaly": None,
        "display": {
            "category": "Harmless rewording",
            "severity": "Immaterial",
            "reason": "Pick either reading",
        },
        "assisted_review": {
            "routing": {"eligible": False},
            "state": "audio_unavailable",
            "policy_version": "synthetic-demo-v2",
        },
    },
    {
        "id": 3,
        "apple_text": "rate of perceived exertion",
        "whisper_text": "rate of perceived exhaustion",
        "apple_context": (
            "For autoregulation, the guest uses rate of perceived exertion to describe "
            "how close a set feels to failure."
        ),
        "whisper_context": (
            "For autoregulation, the guest uses rate of perceived exhaustion to describe "
            "how close a set feels to failure."
        ),
        "source_choices": {
            "apple": "rate of perceived exertion",
            "whisper": "rate of perceived exhaustion",
        },
        "materiality": {
            "group": "proposal",
            "source": "apple",
            "step": "third_asr",
            "reason": "changes_content",
        },
        "materiality_context": {
            "left": "For autoregulation, the guest uses",
            "right": "to describe how close a set feels to failure.",
        },
        "audio_window": {"start": 184.0, "end": 199.0, "duration": 15.0},
        "third_asr": {
            "text": "The guest calls it rate of perceived exertion.",
            "model": "synthetic-third-asr",
            "duration": 15,
        },
        "third_available": True,
        "third_window": "rate of perceived exertion",
        "focus": {"scope": "full"},
        "suggestion": None,
        "triage": None,
        "anomaly": None,
        "display": {
            "category": "Domain-term difference",
            "severity": "Material proposal",
            "reason": "Third ASR supports Apple",
        },
        "assisted_review": {
            "routing": {"eligible": True},
            "state": "machine_supported_apple",
            "policy_version": "synthetic-demo-v2",
        },
    },
    {
        "id": 4,
        "apple_text": "that is probably fine",
        "whisper_text": "that's probably fine",
        "apple_context": (
            "The host says that is probably fine before moving to the next listener question."
        ),
        "whisper_context": (
            "The host says that's probably fine before moving to the next listener question."
        ),
        "source_choices": {
            "apple": "that is probably fine",
            "whisper": "that's probably fine",
        },
        "materiality": {
            "group": "settled",
            "source": "whisper",
            "step": "judge",
            "reason": "same_content",
        },
        "materiality_context": {
            "left": "The host says",
            "right": "before moving to the next listener question.",
        },
        "audio_window": {"start": 232.0, "end": 247.0, "duration": 15.0},
        "third_asr": None,
        "third_available": False,
        "third_window": None,
        "focus": {"scope": "full"},
        "suggestion": None,
        "triage": None,
        "anomaly": None,
        "display": {
            "category": "Contraction",
            "severity": "Immaterial",
            "reason": "Filter-settled",
        },
        "assisted_review": {
            "routing": {"eligible": False},
            "state": "audio_unavailable",
            "policy_version": "synthetic-demo-v2",
        },
    },
    {
        "id": 5,
        "apple_text": "we are going to talk about protein timing",
        "whisper_text": "we're gonna talk about protein timing",
        "apple_context": (
            "After the break, we are going to talk about protein timing and meal frequency."
        ),
        "whisper_context": (
            "After the break, we're gonna talk about protein timing and meal frequency."
        ),
        "source_choices": {
            "apple": "we are going to talk about protein timing",
            "whisper": "we're gonna talk about protein timing",
        },
        "materiality": {
            "group": "settled",
            "source": "apple",
            "step": "fuller",
            "reason": "same_content",
        },
        "materiality_context": {
            "left": "After the break,",
            "right": "and meal frequency.",
        },
        "audio_window": {"start": 268.0, "end": 283.0, "duration": 15.0},
        "third_asr": None,
        "third_available": False,
        "third_window": None,
        "focus": {"scope": "full"},
        "suggestion": None,
        "triage": None,
        "anomaly": None,
        "display": {
            "category": "Conversational wording",
            "severity": "Immaterial",
            "reason": "Filter-settled",
        },
        "assisted_review": {
            "routing": {"eligible": False},
            "state": "audio_unavailable",
            "policy_version": "synthetic-demo-v2",
        },
    },
    {
        "id": 6,
        "apple_text": "The trial included 42 participants and lasted twelve weeks.",
        "whisper_text": "The trial included 40 participants and lasted twelve weeks.",
        "apple_context": (
            "The guest described a resistance-training trial. "
            "The trial included 42 participants and lasted twelve weeks, "
            "with strength and hypertrophy outcomes measured at the end."
        ),
        "whisper_context": (
            "The guest described a resistance-training trial. "
            "The trial included 40 participants and lasted twelve weeks, "
            "with strength and hypertrophy outcomes measured at the end."
        ),
        "source_choices": {
            "apple": "The trial included 42 participants and lasted twelve weeks.",
            "whisper": "The trial included 40 participants and lasted twelve weeks.",
        },
        "materiality": {
            "group": "full",
            "source": None,
            "step": "protected_number",
            "reason": "different_values",
        },
        "materiality_context": {
            "left": "The guest described a resistance-training trial.",
            "right": "with strength and hypertrophy outcomes measured at the end.",
        },
        "audio_window": {"start": 318.0, "end": 333.0, "duration": 15.0},
        "third_asr": {
            "text": "The trial included forty-two participants and lasted twelve weeks.",
            "model": "synthetic-third-asr",
            "duration": 15,
        },
        "third_available": True,
        "third_window": "42 participants",
        "focus": {"scope": "full"},
        "suggestion": None,
        "triage": None,
        "anomaly": None,
        "display": {
            "category": "Study sample-size difference",
            "severity": "Protected value conflict",
            "reason": "Different values require human review",
        },
        "assisted_review": {
            "routing": {"eligible": True},
            "state": "machine_supported_apple",
            "policy_version": "synthetic-demo-v2",
        },
    },
]


def _silent_wav_bytes(duration_seconds: float = 1.0, sample_rate: int = 8000) -> bytes:
    """Return a tiny valid mono WAV so the browser audio control remains functional."""

    frames = max(1, int(duration_seconds * sample_rate))
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        for i in range(frames):
            # Very quiet tone rather than true silence, so playback is visibly real.
            value = int(220 * math.sin(2 * math.pi * 220 * i / sample_rate))
            wav.writeframesraw(struct.pack("<h", value))
    return buffer.getvalue()


def create_demo_app() -> Flask:
    app = Flask(__name__)
    cards = deepcopy(_INITIAL_CARDS)
    decisions: list[dict] = []
    materiality_review_log: list[dict] = []
    recompile_started = False

    def _record() -> dict:
        return {
            "episode_key": EPISODE_KEY,
            "human_review": deepcopy(cards),
            "human_decisions": deepcopy(decisions),
            "materiality_review_log": deepcopy(materiality_review_log),
            "human_review_generation_fingerprint": "sha256:" + "d" * 64,
            "source_fingerprint": "sha256:" + "e" * 64,
        }

    def _progress() -> dict:
        total = len(cards) + len(decisions)
        return {
            "total": total,
            "reviewed": len(decisions),
            "remaining": len(cards),
            "assisted_unprepared": 0,
            "triage_unavailable": 0,
        }

    @app.get("/")
    def home():
        return Response(PAGE, mimetype="text/html")

    @app.get("/health")
    def health():
        return jsonify({"status": "healthy", "service": "synthetic-human-review-demo"})

    @app.get("/tags")
    def tags():
        return Response(
            "<main style='font:16px system-ui;padding:2rem'>"
            "<p><a href='/'>← Transcript review</a></p>"
            "<h1>Synthetic demo</h1>"
            "<p>The portfolio demo focuses on transcript adjudication; "
            "the production project also contains the controlled tag registry.</p>"
            "</main>",
            mimetype="text/html",
        )

    @app.get("/api/review/episodes")
    def episodes():
        if not cards:
            return jsonify({"episodes": []})
        return jsonify({
            "episodes": [{
                **EPISODE,
                "pending_count": len(cards),
            }]
        })

    @app.get("/api/review/episodes/<episode_key>")
    def review(episode_key: str):
        if episode_key != EPISODE_KEY:
            return jsonify({"error": "Unknown synthetic episode"}), 404
        return jsonify({
            "record": _record(),
            "cards": deepcopy(cards),
            "progress": _progress(),
            "recompile_status": "started" if recompile_started else None,
        })

    @app.get("/api/review/episodes/<episode_key>/items/<int:difference_id>/audio")
    def audio(episode_key: str, difference_id: int):
        if episode_key != EPISODE_KEY or not any(card["id"] == difference_id for card in cards):
            return jsonify({"error": "Review item is not pending"}), 404
        return send_file(
            io.BytesIO(_silent_wav_bytes()),
            mimetype="audio/wav",
            download_name="synthetic-review.wav",
        )

    @app.post("/api/review/episodes/<episode_key>/items/<int:difference_id>/third-asr")
    def third_asr(episode_key: str, difference_id: int):
        if episode_key != EPISODE_KEY:
            return jsonify({"error": "Unknown synthetic episode"}), 404
        card = next((item for item in cards if item["id"] == difference_id), None)
        if card is None:
            return jsonify({"error": "Review item is not pending"}), 404
        if difference_id in {3, 6}:
            text = (
                "The guest says rate of perceived exertion."
                if difference_id == 3
                else "The trial included forty-two participants and lasted twelve weeks."
            )
            window = (
                "rate of perceived exertion"
                if difference_id == 3
                else "42 participants"
            )
        else:
            text = "Synthetic third-ASR evidence agrees with the fuller reading."
            window = card["source_choices"]["apple"]
        card["third_asr"] = {
            "text": text,
            "model": "synthetic-third-asr",
            "duration": 15,
        }
        card["third_available"] = True
        card["third_window"] = window
        return jsonify({"evidence": deepcopy(card["third_asr"])})

    @app.post("/api/review/episodes/<episode_key>/items/<int:difference_id>/preview")
    def preview(episode_key: str, difference_id: int):
        body = request.get_json(silent=True) or {}
        if episode_key != EPISODE_KEY:
            return jsonify({"error": "Unknown synthetic episode"}), 404
        return jsonify({
            "replacement_text": body.get("text", ""),
            "replaced_text": next(
                (card["apple_text"] for card in cards if card["id"] == difference_id),
                "",
            ),
            "expand_left_words": int(body.get("expand_left_words", 0) or 0),
            "expand_right_words": int(body.get("expand_right_words", 0) or 0),
        })

    @app.post("/api/review/episodes/<episode_key>/items/<int:difference_id>/decision")
    def decision(episode_key: str, difference_id: int):
        if episode_key != EPISODE_KEY:
            return jsonify({"error": "Unknown synthetic episode"}), 404
        body = request.get_json(silent=True) or {}
        index = next((i for i, card in enumerate(cards) if card["id"] == difference_id), None)
        if index is None:
            return jsonify({"error": "Review item is not pending"}), 404
        card = cards.pop(index)
        source = body.get("source")
        chosen_text = body.get("text")
        if not isinstance(chosen_text, str) or not chosen_text:
            chosen_text = card.get("source_choices", {}).get(source)
        if not isinstance(chosen_text, str) or not chosen_text:
            return jsonify({"error": "A replacement is required"}), 400
        entry = {
            "id": difference_id,
            "chosen_source": source,
            "chosen_text": chosen_text,
            "reviewed_by": "synthetic-demo-human",
            "note": body.get("note") or None,
        }
        decisions.append(entry)
        return jsonify({
            "decision": deepcopy(entry),
            "ready_to_recompile": not cards,
        })

    @app.post("/api/review/episodes/<episode_key>/materiality-decision")
    def materiality_decision(episode_key: str):
        if episode_key != EPISODE_KEY:
            return jsonify({"error": "Unknown synthetic episode"}), 404

        body = request.get_json(silent=True) or {}
        expected_generation = body.get("expected_generation_fingerprint")
        current_generation = "sha256:" + "d" * 64
        if expected_generation != current_generation:
            return jsonify({"error": "Synthetic review generation changed"}), 409

        requested = body.get("decisions")
        if not isinstance(requested, list) or not requested:
            return jsonify({"error": "Materiality decisions are required"}), 400

        plan: list[tuple[int, dict, str, str, int]] = []
        seen: set[int] = set()
        for requested_entry in requested:
            if not isinstance(requested_entry, dict):
                return jsonify({"error": "Materiality decision entries must be objects"}), 400
            difference_id = requested_entry.get("id")
            source = requested_entry.get("source")
            if not isinstance(difference_id, int) or difference_id in seen:
                return jsonify({"error": "Materiality decision ids must be unique integers"}), 400
            if source not in {"apple", "whisper"}:
                return jsonify({"error": "Materiality source must be apple or whisper"}), 400
            index = next((i for i, card in enumerate(cards) if card["id"] == difference_id), None)
            if index is None:
                return jsonify({"error": f"Review item {difference_id} is not pending"}), 409
            card = cards[index]
            materiality = card.get("materiality") or {}
            group = materiality.get("group")
            if group == "full":
                return jsonify({"error": "Full-review cards require an individual decision"}), 409
            if group == "settled" and source != materiality.get("source"):
                return jsonify({"error": "Settled cards must keep the filter reading"}), 409
            chosen_text = card.get("source_choices", {}).get(source)
            if not isinstance(chosen_text, str) or not chosen_text:
                return jsonify({"error": "Selected source has no replacement text"}), 400
            seconds = requested_entry.get("seconds", 0)
            seconds = int(seconds) if isinstance(seconds, (int, float)) and seconds >= 0 else 0
            plan.append((difference_id, card, source, chosen_text, seconds))
            seen.add(difference_id)

        accepted_ids = {difference_id for difference_id, *_ in plan}
        cards[:] = [card for card in cards if card["id"] not in accepted_ids]

        for difference_id, card, source, chosen_text, seconds in plan:
            decisions.append({
                "id": difference_id,
                "chosen_source": source,
                "chosen_text": chosen_text,
                "reviewed_by": "synthetic-demo-human",
                "materiality": deepcopy(card.get("materiality")),
                "seconds": seconds,
            })

        session = body.get("session")
        if isinstance(session, dict):
            materiality_review_log.append({
                "accepted_count": len(plan),
                "settled_list_opened": bool(session.get("settled_list_opened")),
                "note": session.get("note") or None,
            })

        return jsonify({
            "accepted_count": len(plan),
            "ready_to_recompile": not cards,
        })

    @app.post("/api/review/episodes/<episode_key>/batch-decision")
    def batch_decision(episode_key: str):
        if episode_key != EPISODE_KEY:
            return jsonify({"error": "Unknown synthetic episode"}), 404
        body = request.get_json(silent=True) or {}
        requested = body.get("decisions")
        if not isinstance(requested, list) or not requested:
            return jsonify({"error": "Batch decisions are required"}), 400
        accepted = 0
        for entry in list(requested):
            difference_id = entry.get("id")
            index = next((i for i, card in enumerate(cards) if card["id"] == difference_id), None)
            if index is None:
                continue
            card = cards.pop(index)
            source = entry.get("source")
            decisions.append({
                "id": difference_id,
                "chosen_source": source,
                "chosen_text": card.get("source_choices", {}).get(source),
                "reviewed_by": "synthetic-demo-human",
            })
            accepted += 1
        return jsonify({"accepted_count": accepted, "ready_to_recompile": not cards})

    @app.post("/api/review/episodes/<episode_key>/recompile")
    def recompile(episode_key: str):
        nonlocal recompile_started
        if episode_key != EPISODE_KEY:
            return jsonify({"error": "Unknown synthetic episode"}), 404
        if cards:
            return jsonify({"error": "Synthetic review still has unresolved cards"}), 409
        recompile_started = True
        return jsonify({
            "recompile": {
                "status": "accepted",
                "operation": "synthetic-demo-recompile",
                "episode_key": EPISODE_KEY,
            }
        }), 202

    return app


def main() -> None:
    app = create_demo_app()
    print("Synthetic Human Review demo: http://127.0.0.1:8765")
    print("No cloud credentials or production data are used. Restart to reset.")
    app.run(host="127.0.0.1", port=8765, debug=False)


if __name__ == "__main__":
    main()
