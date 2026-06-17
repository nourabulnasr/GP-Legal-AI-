# -*- coding: utf-8 -*-
"""Document chat helpers — token budget and low-quality detection (no live LFM)."""
from __future__ import annotations

import os
import unittest
from unittest.mock import MagicMock

os.environ.setdefault("SECRET_KEY", "test-secret-key-for-unit-tests")
os.environ.setdefault("JWT_SECRET_KEY", "test-jwt-secret-key-for-unit-tests")

from app.routers import chat as chat_mod
from app.local_llm import _THINK_CLOSE, _THINK_OPEN, strip_thinking_output


class TestDocumentChatHelpers(unittest.TestCase):
    def test_is_general_contract_question_ar(self):
        self.assertTrue(chat_mod._is_general_contract_question("اشرح العقد"))

    def test_violation_question_not_general(self):
        self.assertTrue(chat_mod._is_violation_focus_question("اشرح المشاكل التي وردت في العقد"))
        self.assertFalse(chat_mod._is_general_contract_question("اشرح المشاكل التي وردت في العقد"))

    def test_build_detected_violations_reply(self):
        hits = [
            {
                "rule_id": "LABOR25_WORKING_HOURS",
                "severity": "error",
                "description": "ساعات العمل تتجاوز الحد القانوني",
                "matched_text": "10 ساعات يومياً",
            }
        ]
        out = chat_mod.build_detected_violations_reply_ar(hits)
        self.assertIn("LABOR25_WORKING_HOURS", out)
        self.assertIn("10 ساعات", out)
        self.assertNotIn("ضرائب", out)

    def test_lfm_document_low_quality_detects_garbage(self):
        self.assertTrue(chat_mod._lfm_document_low_quality(").\n\n\nالرجوع: [لا 182]", "اشرح العقد"))

    def test_lfm_document_low_quality_detects_thinking_tags(self):
        self.assertTrue(
            chat_mod._lfm_document_low_quality(
                f"some text {_THINK_CLOSE}\n### broken", "اشرح المشاكل"
            )
        )

    def test_strip_thinking_output(self):
        raw = f"garbage {_THINK_CLOSE}\n\n### القانون المطبق:\nنص صحيح"
        out = strip_thinking_output(raw)
        self.assertNotIn(_THINK_CLOSE, out)
        self.assertIn("القانون", out)

    def test_lfm_document_low_quality_accepts_real_answer(self):
        text = (
            "هذا عقد عمل محدد المدة بين صاحب عمل وعامل. "
            "يحدد الأجر والواجبات ومدة العقد وفقاً للنص."
        )
        self.assertFalse(chat_mod._lfm_document_low_quality(text, "اشرح العقد"))

    def test_fit_prompt_preserves_suffix(self):
        from app.local_llm import fit_prompt_to_token_budget

        tok = MagicMock()

        def encode(text, add_special_tokens=False):
            return list(range(len(text) // 4))

        def decode(ids, skip_special_tokens=True):
            return "x" * (len(ids) * 4)

        tok.encode = encode
        tok.decode = decode
        prompt = ("A" * 4000) + "\nسؤال المستخدم: اشرح\n\nالشرح:"
        out = fit_prompt_to_token_budget(prompt, tok, max_tokens=200)
        self.assertIn("سؤال المستخدم:", out)
        self.assertIn("الشرح:", out)


if __name__ == "__main__":
    unittest.main()
