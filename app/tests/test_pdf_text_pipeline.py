# -*- coding: utf-8 -*-
"""Tests for shared PDF extraction and law PDF wiring."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class TestPdfTextPipeline(unittest.TestCase):
    def test_non_pdf_bytes_returns_empty(self) -> None:
        from app.pdf_text_pipeline import extract_pdf_pages_from_bytes

        pages, used = extract_pdf_pages_from_bytes(b"not a pdf header")
        self.assertEqual(pages, [])
        self.assertFalse(used)

    def test_native_pdf_skips_heavy_path(self) -> None:
        try:
            import fitz
        except Exception:
            self.skipTest("PyMuPDF not available")

        doc = fitz.open()
        doc.new_page()
        doc[0].insert_text((72, 72), "A" * 200)
        data = doc.tobytes()
        doc.close()

        from app.pdf_text_pipeline import extract_full_text_from_pdf_bytes, extract_pdf_pages_from_bytes

        pages, used_heavy = extract_pdf_pages_from_bytes(data)
        self.assertEqual(len(pages), 1)
        self.assertFalse(used_heavy)
        self.assertIn("AAA", pages[0]["text"])

        full = extract_full_text_from_pdf_bytes(data)
        self.assertIn("AAA", full)


class TestLawPdfExtract(unittest.TestCase):
    def test_build_articles_from_arabic_markers(self) -> None:
        from app.law_pdf_extract import build_articles_from_text, nfkc_norm

        body1 = "نص طويل كافي لاجتياز الحد الأدنى للطول دون قص المادة التالية. " * 3
        body2 = "نص المادة الثانية بنفس الطول الكافي للاختبار. " * 3
        raw = f"المادة 1:\n{body1}\nالمادة 2:\n{body2}"
        text = nfkc_norm(raw)
        arts = build_articles_from_text(text, source_path="/tmp/x.pdf")
        self.assertGreaterEqual(len(arts), 1)
        self.assertEqual(arts[0].get("article"), "1")

    def test_extract_pdf_text_auto_uses_pipeline(self) -> None:
        try:
            import fitz
        except Exception:
            self.skipTest("PyMuPDF not available")

        doc = fitz.open()
        doc.new_page()
        doc[0].insert_text((72, 72), "B" * 200)
        buf = doc.tobytes()
        doc.close()

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(buf)
            path = Path(tmp.name)

        try:
            with patch.dict(os.environ, {"LAW_PDF_EXTRACTION_MODE": "auto"}):
                with patch(
                    "app.pdf_text_pipeline.extract_full_text_from_pdf_bytes",
                    return_value="pipeline_was_here",
                ):
                    from app.law_pdf_extract import extract_pdf_text

                    self.assertEqual(extract_pdf_text(path), "pipeline_was_here")
        finally:
            path.unlink(missing_ok=True)

    def test_extract_pdf_text_pymupdf_only(self) -> None:
        try:
            import fitz
        except Exception:
            self.skipTest("PyMuPDF not available")

        doc = fitz.open()
        doc.new_page()
        doc[0].insert_text((72, 72), "C" * 200)
        buf = doc.tobytes()
        doc.close()

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(buf)
            path = Path(tmp.name)

        try:
            with patch.dict(os.environ, {"LAW_PDF_EXTRACTION_MODE": "pymupdf_only"}):
                from app.law_pdf_extract import extract_pdf_text

                text = extract_pdf_text(path)
                self.assertIn("CCC", text)
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
