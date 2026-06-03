# -*- coding: utf-8 -*-
"""Social API helpers and delete-post route."""
from __future__ import annotations

import os
import tempfile
import unittest

os.environ.setdefault("SECRET_KEY", "test-secret-key-for-unit-tests")
os.environ.setdefault("JWT_SECRET_KEY", "test-jwt-secret-key-for-unit-tests")

from app.routers import social as social_mod


class TestNetworkSuggestionExclusions(unittest.TestCase):
    def test_peer_ids_from_invites_excludes_connected_and_pending(self):
        invites = [
            (1, 2, "accepted"),   # user 1 connected to 2
            (1, 3, "pending"),    # user 1 sent pending to 3
            (4, 1, "pending"),    # user 4 sent pending to 1
            (1, 5, "declined"),   # should not exclude
        ]
        excluded = social_mod._peer_ids_from_invites(invites, user_id=1)
        self.assertEqual(excluded, {2, 3, 4})
        self.assertNotIn(5, excluded)

    def test_peer_ids_from_invites_bidirectional_accepted(self):
        invites = [(7, 9, "accepted")]
        self.assertEqual(social_mod._peer_ids_from_invites(invites, user_id=7), {9})
        self.assertEqual(social_mod._peer_ids_from_invites(invites, user_id=9), {7})


class TestSocialStaticHelpers(unittest.TestCase):
    def test_static_path_from_public_url(self):
        url = "https://srv1723974.hstgr.cloud/static/profile_avatars/1_abc.jpg"
        path = social_mod._static_path_from_public_url(url)
        self.assertIsNotNone(path)
        assert path is not None
        self.assertTrue(path.replace("\\", "/").endswith("static/profile_avatars/1_abc.jpg"))

    def test_delete_local_static_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            rel_dir = os.path.join(tmp, "static", "post_images")
            os.makedirs(rel_dir, exist_ok=True)
            fpath = os.path.join(rel_dir, "x.jpg")
            with open(fpath, "wb") as f:
                f.write(b"test")
            url = f"https://example.com/static/post_images/x.jpg"
            cwd = os.getcwd()
            try:
                os.chdir(tmp)
                social_mod._delete_local_static_file(url)
                self.assertFalse(os.path.isfile(fpath))
            finally:
                os.chdir(cwd)


if __name__ == "__main__":
    unittest.main()
