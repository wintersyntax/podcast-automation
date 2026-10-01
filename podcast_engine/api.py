"""Flask application exposed by the HTTP service entry point."""

from flask import Flask, jsonify

from .pipeline import run_pipeline
from .storage import load_episodes


app = Flask(__name__)


@app.route("/")
def home():
    return "Podcast worker is alive!"


@app.route("/health")
def health():
    return jsonify(
        {
            "status": "healthy",
            "service": "podcast-worker",
            "version": "1.3",
        }
    )


@app.route("/process")
def process():
    return jsonify(run_pipeline())


@app.route("/episodes")
def episodes():
    return jsonify(load_episodes())
