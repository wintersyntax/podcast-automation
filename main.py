"""HTTP API plus the Cloud Storage Apple-transcript ingest entry point."""

import functions_framework

from podcast_engine.api import app
from podcast_engine.apple_ingest import gcs_apple_transcript_finalize


@functions_framework.cloud_event
def apple_transcript_ingest(cloud_event):
    """Receive a finalized cloud-owned Apple source completion object."""

    return gcs_apple_transcript_finalize(cloud_event)



if __name__ == "__main__":

    app.run(host="0.0.0.0", port=8080)
