"""Synthetic PodcastOps browser demo has no production access or write routes."""

import unittest

from scripts.demo_ops_dashboard import DEMO_DAY, demo_html, synthetic_episodes



class PodcastOpsDemoTests(unittest.TestCase):
    def test_synthetic_ops_episode_projection(self):
        rows = synthetic_episodes()
        assert len(rows) == 12
        assert len({row["episode_key"] for row in rows}) == len(rows)
        assert rows[0]["compiler"] == "review_required"
        assert rows[0]["review"]["state"] == "pending"
        assert rows[0]["delivery"] == "processing"
        assert rows[1]["summary"] == "ready"
        assert rows[1]["vault_sync"]["status"] == "unconfirmed"
        assert rows[1]["delivery"] == "awaiting_vault_sync"
        assert any(row["vault_sync"]["status"] == "confirmed" for row in rows)
        assert any(row["vault_sync"]["status"] == "unavailable" for row in rows)
        assert any(row["recompile"]["status"] == "completed" for row in rows)
        assert all(row["episode_key"].startswith("demo-episode-") for row in rows)
        assert all(row["published"] <= DEMO_DAY.isoformat() for row in rows)
        for row in rows:
            if row["delivery"] == "synced_to_mac":
                assert row["summary"] == "ready"
                assert row["vault_sync"]["confirmed_at"]
                assert row["finished_at"] == row["vault_sync"]["confirmed_at"]
            if row["review"]["state"] == "pending":
                assert sum(row["review"]["groups"].values()) == row["review"]["pending"]


    def test_ops_demo_uses_canonical_read_only_frontend(self):
        body = demo_html()
        assert "PodcastOps — episode status" in body
        assert "SYNTHETIC DEMO" in body
        assert "/api/ops/episodes" in body
        assert "Awaiting Mac Vault Sync" in body
        assert "Mac copy confirmed" in body
        assert "not that iCloud uploaded it" in body
        assert "Synthetic demo index" in body
