# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import tempfile
import unittest
import unittest.mock
from pathlib import Path


class TestListChunkJsonlFiles(unittest.TestCase):
    def test_neutral_wins_when_both_exist(self) -> None:
        from app.chunks_loader import list_chunk_jsonl_files

        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "labor_law_chunks.cleaned.jsonl").write_text("{}\n", encoding="utf-8")
            (d / "labor14_2025_chunks.cleaned.jsonl").write_text("{}\n", encoding="utf-8")
            files = list_chunk_jsonl_files(d)
            self.assertEqual(len(files), 1)
            self.assertEqual(files[0].name, "labor_law_chunks.cleaned.jsonl")

    def test_legacy_when_neutral_missing(self) -> None:
        from app.chunks_loader import list_chunk_jsonl_files

        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "labor14_2025_chunks.cleaned.jsonl").write_text("{}\n", encoding="utf-8")
            files = list_chunk_jsonl_files(d)
            self.assertEqual(len(files), 1)
            self.assertEqual(files[0].name, "labor14_2025_chunks.cleaned.jsonl")

    def test_glob_when_no_preferred(self) -> None:
        from app.chunks_loader import list_chunk_jsonl_files

        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "other_chunks.jsonl").write_text("{}\n", encoding="utf-8")
            files = list_chunk_jsonl_files(d)
            self.assertEqual(len(files), 1)
            self.assertEqual(files[0].name, "other_chunks.jsonl")


class TestReindexSuccessfulMemoryOnly(unittest.TestCase):
    def test_memory_only_ignores_chroma_skip(self) -> None:
        from app.law_update_service import reindex_is_successful

        with unittest.mock.patch.dict(os.environ, {"LEGAL_RAG_QUERY_BACKEND": "memory_only"}, clear=False):
            ok, reason = reindex_is_successful(
                {
                    "chroma": {"skipped": True, "reason": "test"},
                    "retriever": {"ok": True, "docs": 5},
                }
            )
            self.assertTrue(ok)
            self.assertEqual(reason, "ok")

    def test_chroma_first_fails_on_skip(self) -> None:
        from app.law_update_service import reindex_is_successful

        with unittest.mock.patch.dict(os.environ, {"LEGAL_RAG_QUERY_BACKEND": "chroma_first"}, clear=False):
            ok, reason = reindex_is_successful(
                {
                    "chroma": {"skipped": True},
                    "retriever": {"ok": True},
                }
            )
            self.assertFalse(ok)
            self.assertIn("chroma", reason.lower())


if __name__ == "__main__":
    unittest.main()
