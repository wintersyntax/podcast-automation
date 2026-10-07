"""Cloud Run entry point for the authenticated Human Review UI."""

from __future__ import annotations

import os

from podcast_engine.review_web import create_review_app


app = create_review_app()


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8080")),
        debug=False,
    )
