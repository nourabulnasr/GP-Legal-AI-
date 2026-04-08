# -*- coding: utf-8 -*-
"""
Local LFM2.5-1.2B-Instruct for explanation-only (no violation detection, no invented law).
Loads from disk (LOCAL_LLM_PATH or project LFM2.5-1.2B-Instruct). No HuggingFace API.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

_BASE = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = _BASE / "LFM2.5-1.2B-Instruct"
MAX_PROMPT_CHARS = 6000

_tokenizer = None
_model = None
_loaded_path: Optional[str] = None


def _model_path() -> Optional[Path]:
    path_env = os.environ.get("LOCAL_LLM_PATH", "").strip()
    if path_env:
        p = Path(path_env).expanduser().resolve()
        if p.exists():
            return p
    if DEFAULT_MODEL_PATH.exists():
        return DEFAULT_MODEL_PATH
    return None


def is_available() -> bool:
    """True if local model path exists and can be loaded."""
    return _model_path() is not None


def load_model():
    """Load tokenizer and model from local path. Idempotent."""
    global _tokenizer, _model, _loaded_path
    path = _model_path()
    if not path:
        raise FileNotFoundError("Local LLM path not set or missing. Set LOCAL_LLM_PATH or add LFM2.5-1.2B-Instruct/.")
    path_str = str(path)
    if _model is not None and _loaded_path == path_str:
        return _tokenizer, _model

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    _tokenizer = AutoTokenizer.from_pretrained(path_str, local_files_only=True)
    _model = AutoModelForCausalLM.from_pretrained(
        path_str,
        local_files_only=True,
        device_map="auto" if torch.cuda.is_available() else None,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    if _model.device.type == "cpu":
        _model = _model.to("cpu")
    _model.eval()
    _loaded_path = path_str
    return _tokenizer, _model


def generate(
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.1,
    do_sample: bool = False,
) -> str:
    """
    Generate text from prompt. Used for explanation only (violation + matched text + RAG).
    """
    if not prompt or not prompt.strip():
        return ""
    prompt = (prompt[:MAX_PROMPT_CHARS] + "...") if len(prompt) > MAX_PROMPT_CHARS else prompt

    try:
        tokenizer, model = load_model()
    except Exception as e:
        return f"[LLM load error: {e!r}]"

    import torch

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    if hasattr(model, "device") and next(model.parameters()).device.type != "cpu":
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": do_sample, "pad_token_id": tokenizer.eos_token_id}
    if do_sample:
        gen_kwargs["temperature"] = temperature
    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    # Return only the generated part (reasoning/explanation), not the prompt
    prompt_clean = prompt.strip()
    if prompt_clean in text:
        text = text.split(prompt_clean)[-1].strip()
    # Fallback: strip by last instruction line so we don't show prompt
    for sentinel in (
        "الشرح والتصحيح المقترح (بناءً على النصوص أعلاه فقط):",
        "explanation and suggested correction (based only on the texts above):",
        "الشرح والتصحيح",
        "explanation and suggested correction",
    ):
        if sentinel in text:
            parts = text.split(sentinel, 1)
            if len(parts) > 1 and parts[-1].strip():
                text = parts[-1].strip()
                break
    return text.strip()


# Prompt template: violation explanation from rule + matched text + RAG articles only
EXPLANATION_SYSTEM = """أنت مساعد قانوني. مهمتك فقط شرح سبب مخالفة البند للقانون واستنتاج تصحيح بناءً على النصوص المقدمة.
- استخدم فقط المواد القانونية المقدمة أدناه. لا تخترع أرقام مواد أو نصوصاً.
- إذا لم يكن السياق كافياً، قل: "لا يوجد نص قانوني كافٍ في المستند."
- اذكر رقم المادة عند الاقتباس.
- قدم تصحيحاً مقترحاً للبند بناءً على النص القانوني فقط."""

EXPLANATION_SYSTEM_EN = """You are a legal assistant. Your task is only to explain why the clause violates the law and suggest a correction based strictly on the provided texts.
- Use ONLY the law articles provided below. Do not invent article numbers or text.
- If the context is insufficient, say: "Insufficient legal text in the provided document."
- Cite article numbers when quoting.
- Provide a suggested correction for the clause based only on the law text."""


def build_explanation_prompt(
    rule_id: str,
    description: str,
    matched_text: str,
    law_articles: List[Dict[str, Any]],
    language: str = "ar",
) -> str:
    """
    Build prompt for LLM: violation metadata + matched contract text + retrieved law.
    law_articles: list of {"text": ..., "metadata": {"article": ..., "law": ...}}
    """
    law_block = []
    for a in law_articles[:6]:
        text = (a.get("text") or "").strip()
        meta = a.get("metadata") or {}
        art = meta.get("article", "")
        law = meta.get("law", "")
        if text:
            law_block.append(f"المادة {art} - {law}:\n{text[:1500]}")
    law_str = "\n\n---\n\n".join(law_block) if law_block else "لم تُقدّم مواد قانونية."

    sys_prompt = EXPLANATION_SYSTEM if language == "ar" else EXPLANATION_SYSTEM_EN
    prompt = f"""{sys_prompt}

المخالفة: {rule_id}
الوصف: {description}

نص العقد المعني:
{matched_text[:1500]}

النصوص القانونية المقدمة:
{law_str}

الشرح والتصحيح المقترح (بناءً على النصوص أعلاه فقط):"""
    return prompt


def explain_violation(
    rule_id: str,
    description: str,
    matched_text: str,
    law_articles: List[Dict[str, Any]],
    max_new_tokens: int = 400,
) -> str:
    """
    One-shot: build prompt and generate explanation. Returns LLM text only.
    """
    prompt = build_explanation_prompt(rule_id, description, matched_text, law_articles)
    return generate(prompt, max_new_tokens=max_new_tokens, do_sample=False)
