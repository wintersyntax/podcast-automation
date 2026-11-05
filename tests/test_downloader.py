import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from podcast_engine.downloader import downloaded_audio


class DownloaderTests(unittest.TestCase):
    def test_downloaded_audio_is_local_and_removed_after_processing(self):
        response = Mock()
        response.iter_content.return_value = [b"first", b"", b"second"]
        response.raise_for_status.return_value = None

        with patch("podcast_engine.downloader.requests.get", return_value=response) as get:
            with downloaded_audio("https://cdn.example.test/episode.mp3", "episode.mp3") as audio:
                captured = audio
                self.assertTrue(audio.is_file())
                self.assertEqual(audio.read_bytes(), b"firstsecond")
                self.assertNotIn(Path.cwd(), audio.parents)

        self.assertFalse(captured.exists())
        get.assert_called_once_with("https://cdn.example.test/episode.mp3", stream=True)


if __name__ == "__main__":
    unittest.main()
