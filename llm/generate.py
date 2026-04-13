"""LFM inference entrypoints; load weights in ``lfm_model.py`` — attach LoRA/adapter training here."""
from typing import Any, Dict, List

from .lfm_model import load_model, is_available as _model_available
from .prompt import SYSTEM_PROMPT_AR

MAX_PROMPT_CHARS = 12000
"""Match ``app.lfm_runtime`` / long compare prompts; tokenizer still truncates by tokens."""


def _model_device(model):
    return next(model.parameters()).device


def generate_answer(context, question):
    tokenizer, model = load_model()

    prompt = f"""
{SYSTEM_PROMPT_AR}

Context:
{context}

Question:
{question}

Answer:
    """.strip()

    max_ctx = 4096
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_ctx)
    device = _model_device(model)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    gen_kwargs = {
        "max_new_tokens": 300,
        "do_sample": False,
        "pad_token_id": tokenizer.eos_token_id,
    }
    with torch.no_grad():
        output = model.generate(**inputs, **gen_kwargs)

    input_len = inputs["input_ids"].shape[1]
    gen_ids = output[0][input_len:]
    if gen_ids.numel() == 0:
        return ""
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def generate(
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.1,
    do_sample: bool = False,
) -> str:
    """Generate text from prompt. Returns **new** tokens only (no prompt echo)."""
    if not prompt or not prompt.strip():
        return ""
    raw = prompt.strip()
    prompt_trunc = (raw[:MAX_PROMPT_CHARS] + "...") if len(raw) > MAX_PROMPT_CHARS else raw

    try:
        tokenizer, model = load_model()
    except Exception as e:
        return f"[LLM load error: {e!r}]"

    import torch

    max_ctx = 4096
    inputs = tokenizer(prompt_trunc, return_tensors="pt", truncation=True, max_length=max_ctx)
    device = _model_device(model)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": do_sample, "pad_token_id": tokenizer.eos_token_id}
    if do_sample:
        gen_kwargs["temperature"] = temperature
    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    input_len = inputs["input_ids"].shape[1]
    gen_ids = out[0][input_len:]
    if gen_ids.numel() == 0:
        return ""
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


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
    """Build prompt for LLM: violation metadata + matched contract text + retrieved law."""
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
    """One-shot: build prompt and generate explanation. Returns LLM text only."""
    prompt = build_explanation_prompt(rule_id, description, matched_text, law_articles)
    return generate(prompt, max_new_tokens=max_new_tokens, do_sample=False)


def is_available() -> bool:
    """True if local LLM can be loaded (for explanation-only use)."""
    return _model_available()
