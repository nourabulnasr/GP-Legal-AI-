"""
Central LFM entry points for legal/contract work (explain, summarize, negotiation, analysis chat).
Uses ``llm.generate`` (primary — adapter/LoRA fine-tuning attaches here) with ``app.local_llm`` fallback.

Gemini must not be used from this module — only local LFM stacks.
"""
from __future__ import annotations

MAX_PROMPT_CHARS = 12000
_MAX_CONTEXT_FOR_QA = 8000
_MAX_QUESTION = 4000


def _build_fallback_qa_prompt(context: str, message: str) -> str:
    """When ``generate_answer`` is unavailable; same idea as chat router document prompt."""
    return f"""أنت مساعد قانوني. أجب على سؤال المستخدم بناءً على النص التالي فقط. إذا لم يكن الجواب في النص فقل ذلك.

النص:
{context[:6000]}

سؤال المستخدم:
{message[:2000]}

الجواب:"""


def lfm_full_prompt(prompt: str, max_new_tokens: int = 512) -> str:
    """
    Single full prompt → completion. Used for structured tasks (explain, summarize, negotiation, compare summary).

    Fine-tuning: extend ``llm.generate.generate`` / weights under ``llm/lfm_model.py``.
    """
    p = (prompt or "").strip()
    if not p:
        return "[Empty prompt]"
    p = p[:MAX_PROMPT_CHARS]
    err_first: Exception | None = None
    try:
        from llm.generate import is_available, generate

        if is_available():
            out = generate(p, max_new_tokens=max_new_tokens, do_sample=False)
            if out and not str(out).startswith("[LLM load error"):
                return str(out).strip()
    except Exception as e:
        err_first = e

    try:
        from app.local_llm import is_available, generate as generate_local

        if is_available():
            return str(generate_local(p, max_new_tokens=max_new_tokens, do_sample=False)).strip()
    except Exception as e2:
        err_first = err_first or e2

    return (
        "[LFM unavailable] Configure the LFM checkpoint (see llm/lfm_model.py or LOCAL_LLM_PATH). "
        f"Details: {err_first!r}"
    )


def lfm_context_question(context: str, question: str, max_new_tokens: int = 384) -> str:
    """
    Context + question (RAG/analysis style). Prefer ``llm.generate.generate_answer`` for fine-tune parity.

    Used for: /chat/message, /chat/document.
    """
    ctx = (context or "").strip()
    q = (question or "").strip()
    if not q:
        return "[No question]"
    ctx = ctx[:_MAX_CONTEXT_FOR_QA]
    q = q[:_MAX_QUESTION]

    try:
        from llm.generate import is_available, generate_answer

        if is_available():
            out = generate_answer(ctx, q)
            if out and not str(out).startswith("[LLM load error"):
                return str(out).strip()
    except Exception:
        pass

    try:
        from app.local_llm import is_available, generate as generate_local

        if is_available():
            prompt = _build_fallback_qa_prompt(ctx, q)
            return str(generate_local(prompt, max_new_tokens=max_new_tokens, do_sample=False)).strip()
    except Exception:
        pass

    return lfm_full_prompt(
        f"Context:\n{ctx}\n\nQuestion:\n{q}\n\nAnswer:",
        max_new_tokens=max_new_tokens,
    )


def lfm_failed_for_http(text: str) -> bool:
    """True if the client should receive HTTP 503 (model missing / hard error)."""
    t = (text or "").strip()
    if not t:
        return True
    if t.startswith("[LFM unavailable]"):
        return True
    if t.startswith("[LLM load error"):
        return True
    if t.startswith("[Local LLM not available"):
        return True
    if t.startswith("[Local LLM error"):
        return True
    if t.startswith("[Empty prompt]") or t.startswith("[No question]"):
        return True
    return False
