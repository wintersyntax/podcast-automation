"""Checks for the public extraction's fail-closed publication guard."""
import unittest

from scripts.check_public_release import path_problem, text_problems


class PublicReleaseTests(unittest.TestCase):
    def test_private_paths_fail_closed(self):
        self.assertIsNotNone(path_problem(".clinerules/shell-policy"))
        self.assertIsNotNone(path_problem("docs/knowledge-eval/real-evidence.md"))
        self.assertIsNotNone(path_problem("AGENTS.md"))
        self.assertIsNone(path_problem("assets/demo-podcastops-dashboard.png"))
        self.assertIsNone(path_problem("config/podcasts.example.json"))

    def test_credentials_and_real_identity_rejected(self):
        secret = "sk-" + "or-v1-" + "x" * 32
        self.assertTrue(text_problems(secret))
        self.assertTrue(text_problems("review." + "podcastops.link"))
        self.assertTrue(text_problems("/Users/" + "real-user/projects/code"))
        self.assertFalse(text_problems("/Users/YOUR_USER/project"))
        self.assertFalse(text_problems("REVIEW_ALLOWED_EMAIL=reviewer@example.com"))


if __name__ == "__main__":
    unittest.main()
