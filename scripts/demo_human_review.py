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
        "audio_window": {"start": 93.0, "end": 108.0, "duration": 15.0},
        "third_asr": None,
        "third_available": False,
        "third_window": None,
        "focus": {"scope": "full"},
        "suggestion": None,
        "triage": None,
        "anomaly": None,
        "display": {
            "category": "Study sample-size difference",
            "severity": "High-risk difference",
            "reason": "Compiler requires human review",
        },
        "assisted_review": {
            "routing": {"eligible": False},
            "state": "audio_unavailable",
            "policy_version": "synthetic-demo-v1",
        },
    },
    {
        "id": 2,
        "apple_text": "Romanian deadlift",
        "whisper_text": "Roman deadlift",
        "apple_context": (
            "For the hip-hinge example, the speaker specifically used the "
            "Romanian deadlift when discussing hamstring loading."
        ),
        "whisper_context": (
            "For the hip-hinge example, the speaker specifically used the "
            "Roman deadlift when discussing hamstring loading."
        ),
        "source_choices": {
            "apple": "Romanian deadlift",
            "whisper": "Roman deadlift",
        },
        "audio_window": {"start": 301.5, "end": 316.5, "duration": 15.0},
        "third_asr": None,
        "third_available": False,
        "third_window": None,
        "focus": {"scope": "full"},
        "suggestion": None,
        "triage": None,
        "anomaly": None,
        "display": {
            "category": "Exercise-name difference",
            "severity": "Needs review",
            "reason": "Needs human review",
        },
        "assisted_review": {
            "routing": {"eligible": False},
            "state": "audio_unavailable",
            "policy_version": "synthetic-demo-v1",
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
    recompile_started = False

    def _record() -> dict:
        return {
            "episode_key": EPISODE_KEY,
            "human_review": deepcopy(cards),
            "human_decisions": deepcopy(decisions),
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
        if difference_id == 1:
            text = "The trial included forty-two participants and lasted twelve weeks."
            window = "42 participants"
        else:
            text = "The speaker used the Romanian deadlift as the hip-hinge example."
            window = "Romanian deadlift"
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
