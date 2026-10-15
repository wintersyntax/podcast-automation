from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import tempfile
from typing import Iterator

import requests

def download_audio(
    audio_url: str,
    destination: str | Path,
) -> Path:
    """Download an RSS enclosure to a caller-owned local path.

    The caller owns the destination lifetime. This function intentionally
    never uploads source audio to Cloud Storage.
    """

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading audio: {destination.name}")
    response = requests.get(audio_url, stream=True)
    response.raise_for_status()

    with destination.open("wb") as output:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                output.write(chunk)

    print(f"Audio downloaded to ephemeral storage: {destination}")
    return destination


@contextmanager
def downloaded_audio(
    audio_url: str,
    filename: str,
) -> Iterator[Path]:
    """Yield one downloaded enclosure and remove it when processing ends."""

    safe_name = Path(filename).name or "episode-audio"
    with tempfile.TemporaryDirectory(prefix="podcast-audio-") as directory:
        yield download_audio(audio_url, Path(directory) / safe_name)
