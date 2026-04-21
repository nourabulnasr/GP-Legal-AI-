# -*- coding: utf-8 -*-
"""Language detection, chunk annotation, glossary loading, and BCP-47 override validation."""

from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import app.utils_text as utils_text_mod
from app.translation_service import enrich_ocr_chunks_with_arabic, load_glossary_pairs_for_source
from app.labor_scope import egyptian_labor_content_related
from app.utils_text import (
    annotate_ocr_chunks_language,
    detect_language_detailed,
    detect_language_for_document,
    is_valid_bcp47_primary_override,
    llm_locale_from_detection,
)


class TestBcp47Override(unittest.TestCase):
    def test_valid_tags(self):
        for raw in ("en", "ar-EG", "fr", "de", "cmn", "de-DE", "zh-Hans", "en_US"):
            with self.subTest(raw=raw):
                self.assertTrue(is_valid_bcp47_primary_override(raw))

    def test_invalid_tags(self):
        for raw in ("bad!", "e", "", "123"):
            with self.subTest(raw=raw):
                self.assertFalse(is_valid_bcp47_primary_override(raw))


class TestLlmLocaleFromDetection(unittest.TestCase):
    def test_ar_en_und(self):
        self.assertEqual(llm_locale_from_detection("ar"), "ar")
        self.assertEqual(llm_locale_from_detection("ar-EG"), "ar")
        self.assertEqual(llm_locale_from_detection("en"), "en")
        self.assertEqual(llm_locale_from_detection("en-US"), "en")
        self.assertEqual(llm_locale_from_detection("und"), "en")
        self.assertEqual(llm_locale_from_detection(""), "en")

    def test_other_iso_uses_arabic_prompts(self):
        self.assertEqual(llm_locale_from_detection("fr"), "ar")
        self.assertEqual(llm_locale_from_detection("de"), "ar")
        self.assertEqual(llm_locale_from_detection("es"), "ar")


class TestDetectLanguageDetailed(unittest.TestCase):
    def test_pure_arabic(self):
        r = detect_language_detailed("عقد عمل لمدة سنة كاملة")
        self.assertEqual(r.language_code, "ar")
        self.assertGreaterEqual(r.confidence, 0.5)

    def test_cyrillic_russian_hint(self):
        s = "Настоящий трудовой договор заключён между сторонами на срок один год. " * 3
        r = detect_language_detailed(s)
        self.assertEqual(r.language_code, "ru")

    def test_french_contract_snippet(self):
        s = (
            "Le présent contrat de travail est conclu pour une durée indéterminée. "
            "La période d'essai est fixée conformément au code du travail. "
        ) * 2
        r = detect_language_detailed(s)
        self.assertEqual(r.language_code, "fr")

    def test_ocr_like_spacing_still_french(self):
        s = "Le  présent\n\ncontrat   de\ntravail  définit  la  durée  du  préavis  pour  le  salarié. " * 2
        r = detect_language_detailed(s)
        self.assertEqual(r.language_code, "fr")

    def test_german_snippet(self):
        s = (
            "Dieser Arbeitsvertrag regelt die Kündigungsfrist und die Probezeit des Arbeitnehmers. "
            "Die Vergütung wird monatlich auf das angegebene Konto überwiesen. "
        ) * 2
        r = detect_language_detailed(s)
        self.assertEqual(r.language_code, "de")


class TestAnnotateChunksAndDocument(unittest.TestCase):
    def test_annotate_sets_detected_lang(self):
        chunks = [
            {"normalized_text": "Hello employment agreement clause one." * 5},
            {"normalized_text": "عقد عمل يحدد الأجر والمواعيد لصالح العامل." * 2},
        ]
        annotate_ocr_chunks_language(chunks)
        self.assertIn("detected_lang", chunks[0])
        self.assertIn("detected_lang_confidence", chunks[0])
        self.assertEqual(chunks[1]["detected_lang"], "ar")

    def test_detect_language_for_document_mixed(self):
        chunks = [
            {"normalized_text": "This is an English employment contract header. " * 4},
            {"normalized_text": "البند العربي يحدد الأجر والمواعيد لصالح العامل في العقد." * 3},
        ]
        annotate_ocr_chunks_language(chunks)
        full = "\n\n".join(c["normalized_text"] for c in chunks)
        doc = detect_language_for_document(full, chunks)
        self.assertTrue(doc.is_mixed)


class TestEnrichTranslationMeta(unittest.TestCase):
    def test_not_requested_returns_disabled(self):
        chunks = [{"normalized_text": "hello", "text": "hello"}]
        meta = enrich_ocr_chunks_with_arabic(chunks, "en", requested=False)
        self.assertEqual(meta.get("translation_status"), "disabled")
        self.assertNotIn("translated_ar_text", chunks[0])


class TestGlossaryBySource(unittest.TestCase):
    def test_merges_global_and_fr_pairs(self):
        pairs = load_glossary_pairs_for_source("fr")
        terms = [a for a, _ in pairs]
        self.assertTrue(any("probation period" in t for t in terms))
        self.assertTrue(any("période" in t for t in terms))


class TestFasttextFusionPath(unittest.TestCase):
    def test_fusion_prefers_fasttext_when_configured(self):
        fake_model = Mock()

        def predict(text, k=1):
            return ("__label__de",), (0.91,)

        fake_model.predict = predict
        with patch.object(utils_text_mod, "_get_fasttext_lid_model", return_value=fake_model):
            compact = "Employment terms and salary rules. " * 8
            ar_l = 0
            lat_l = sum(1 for ch in compact if ch.isascii() and ch.isalpha())
            code, prob = utils_text_mod._fuse_fasttext_with_candidate(
                compact,
                ld_code="en",
                ld_prob=0.62,
                ar_l=ar_l,
                lat_l=lat_l,
                is_mixed=False,
            )
        self.assertEqual(code, "de")
        self.assertGreater(prob, 0.62)


class TestEgyptianLaborContentRelated(unittest.TestCase):
    """Gating: apply Egyptian labor analysis only when translated/source text relates to Egyptian labor scope."""

    def test_explicit_law_true(self):
        self.assertTrue(egyptian_labor_content_related("يخضع هذا العقد لقانون العمل المصري.", "ar"))

    def test_arabic_employment_no_foreign_true(self):
        self.assertTrue(egyptian_labor_content_related("عقد عمل بين الموظف وصاحب العمل لمدة سنة.", "ar"))

    def test_german_employment_translated_false(self):
        s = "عقد عمل بين الموظف وصاحب العمل وفق القانون في ألمانيا."
        self.assertFalse(egyptian_labor_content_related(s, "de"))

    def test_non_employment_arabic_false(self):
        self.assertFalse(egyptian_labor_content_related("عقد بيع سيارة بين الطرفين.", "ar"))

    def test_english_employment_no_egypt_signals_false(self):
        self.assertFalse(
            egyptian_labor_content_related(
                "Employment contract between employer and employee. Salary monthly.", "en"
            )
        )


if __name__ == "__main__":
    unittest.main()
