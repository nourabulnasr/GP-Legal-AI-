# -*- coding: utf-8 -*-
"""
Local LFM2.5-1.2B-Instruct for explanation-only (no violation detection, no invented law).
Loads from disk (LOCAL_LLM_PATH or project LFM2.5-1.2B-Instruct). No HuggingFace API.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

_BASE = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = _BASE / "LFM2.5-1.2B-Instruct"
MODEL_UNDER_MODELS = _BASE / "models" / "LFM2.5-1.2B-Instruct"
MAX_PROMPT_CHARS = 6000

_tokenizer = None
_model = None
_loaded_path: Optional[str] = None

# Single-thread executor: Lfm2ForCausalLM has thread-local state — load and
# generate must both execute in the same thread to avoid the "Tensor on device
# cpu is not on the expected device meta!" error when called cross-thread.
_model_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lfm-worker")


def _model_path() -> Optional[Path]:
    path_env = os.environ.get("LOCAL_LLM_PATH", "").strip()
    if path_env:
        p = Path(path_env).expanduser().resolve()
        if p.exists():
            return p
    for candidate in (DEFAULT_MODEL_PATH, MODEL_UNDER_MODELS):
        if candidate.exists():
            return candidate
    return None


def is_available() -> bool:
    """True if local model path exists and can be loaded."""
    return _model_path() is not None


def _verify_local_hf_snapshot(model_dir: Path) -> None:
    """
    Fail fast with a clear message if the folder is not a complete HuggingFace model dir.
    Common mistake: Git LFS pointer file, wrong subdirectory, or transformers too old for model_type.
    """
    cfg = model_dir / "config.json"
    if not cfg.is_file():
        raise FileNotFoundError(
            f"Missing config.json under {model_dir}. "
            "Copy the full HuggingFace snapshot of LiquidAI/LFM2.5-1.2B-Instruct (all files), not only weights."
        )
    raw = cfg.read_text(encoding="utf-8", errors="replace").strip()
    if raw.startswith("version https://git-lfs") or raw.startswith("oid "):
        raise ValueError(
            f"{cfg} is a Git LFS pointer, not the real JSON. "
            "Run: git lfs pull  OR  huggingface-cli download LiquidAI/LFM2.5-1.2B-Instruct --local-dir ./LFM2.5-1.2B-Instruct"
        )
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {cfg}: {e}") from e
    mt = obj.get("model_type")
    arch = (obj.get("architectures") or ["?"])[0]
    if not mt:
        raise ValueError(
            f"{cfg} has no 'model_type'. Expected LFM2.5 config (e.g. model_type 'lfm2'). "
            f"You may have pointed LOCAL_LLM_PATH at the wrong folder. Files here: {sorted(p.name for p in model_dir.iterdir())[:25]}"
        )
    try:
        import transformers
        tv = getattr(transformers, "__version__", "?")
    except Exception:
        tv = "?"
    if str(mt).lower() == "lfm2" and tv != "?":
        parts = tv.split(".")
        ok = False
        if len(parts) >= 2:
            try:
                major, minor = int(parts[0]), int(parts[1])
                ok = major > 4 or (major == 4 and minor >= 57)
            except ValueError:
                ok = False
        if not ok:
            raise ValueError(
                f"model_type is {mt!r} but installed transformers=={tv}. "
                "Upgrade to transformers>=4.57.2 (see requirements.txt) and rebuild the Docker image: "
                "docker compose build --no-cache backend"
            )


def load_model():
    """Load tokenizer and model from local path. Idempotent."""
    global _tokenizer, _model, _loaded_path
    path = _model_path()
    if not path:
        raise FileNotFoundError("Local LLM path not set or missing. Set LOCAL_LLM_PATH or add LFM2.5-1.2B-Instruct/.")
    path_str = str(path)
    if _model is not None and _loaded_path == path_str:
        return _tokenizer, _model

    _verify_local_hf_snapshot(path)

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    _tokenizer = AutoTokenizer.from_pretrained(path_str, local_files_only=True, trust_remote_code=True)
    _model = AutoModelForCausalLM.from_pretrained(
        path_str,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
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

    def _run_in_model_thread():
        import gc
        import torch

        try:
            tokenizer, model = load_model()
        except Exception as e:
            return f"[LLM load error: {e!r}]"

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.eos_token_id,
            "repetition_penalty": 1.3,
            "no_repeat_ngram_size": 4,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
        try:
            with torch.no_grad():
                out = model.generate(**inputs, **gen_kwargs)
        except RuntimeError as e:
            # Lfm2Cache conv_state can get corrupted between calls (mark_static_address
            # interaction). Reset model so next call reloads fresh.
            global _tokenizer, _model, _loaded_path
            _tokenizer = None
            _model = None
            _loaded_path = None
            gc.collect()
            return f"[LLM generate error: {e!r}]"
        input_len = int(inputs["input_ids"].shape[1])
        gen_ids = out[0][input_len:]
        raw = tokenizer.decode(gen_ids, skip_special_tokens=True)
        gc.collect()  # Release Lfm2Cache before next call to prevent conv_state address reuse
        return raw

    try:
        text = _model_executor.submit(_run_in_model_thread).result(timeout=600)
    except Exception as e:
        return f"[LLM generate error: {e!r}]"
    # Fallback: strip prompt if tokenizer left overlap
    prompt_clean = prompt.strip()
    if prompt_clean and prompt_clean in text:
        text = text.split(prompt_clean)[-1].strip()
    # Fallback: strip by last instruction line so we don't show prompt
    for sentinel in (
        "الجواب:",
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
            law_block.append(f"المادة {art} - {law}:\n{text[:200]}")
    law_str = "\n\n---\n\n".join(law_block) if law_block else "لم تُقدّم مواد قانونية."

    sys_prompt = EXPLANATION_SYSTEM if language == "ar" else EXPLANATION_SYSTEM_EN
    prompt = f"""{sys_prompt}

المخالفة: {rule_id}
الوصف: {description}

نص العقد المعني:
{matched_text[:300]}

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
    language: str = "ar",
) -> str:
    """
    One-shot: build prompt and generate explanation. Returns LLM text only.
    language: ar | en (drives system prompt in build_explanation_prompt).
    Callers map ISO detection via ``llm_locale_from_detection``: non-English, non-Arabic
    languages use ``ar`` prompts so explanations stay Arabic-first while clause text may be FR/DE/… .
    """
    prompt = build_explanation_prompt(
        rule_id, description, matched_text, law_articles, language=language
    )
    return generate(prompt, max_new_tokens=max_new_tokens, do_sample=False)
