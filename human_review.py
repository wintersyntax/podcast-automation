"""Run the local-only human transcript review browser UI."""

from __future__ import annotations

import os

from podcast_engine.review_web import create_review_app


if __name__ == "__main__":
    create_review_app().run(
        host="127.0.0.1",
        port=int(os.environ.get("HUMAN_REVIEW_PORT", "8765")),
        debug=False,
    )
