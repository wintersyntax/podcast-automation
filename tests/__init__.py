"""Automated tests for Podcast Worker."""

import os

# TASK-148: tests never run on Google Compute Engine. Without this,
# every test that constructs a real google-cloud client without
# credentials waits for google-auth's metadata-server probe (about 3 s
# each), which made up about two thirds of the suite's run time.
# google-auth reads the variable once, when its metadata module is first
# imported, so it must be set before any test module imports google.
os.environ.setdefault("NO_GCE_CHECK", "true")
