"""
Chat router: Gemini-powered chatbot scoped to analysis_id.
Context: contract OCR text + analysis results.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.core.deps import get_current_user
from app.db.session import get_db
from app.db.models import Analysis, User
from sqlalchemy.orm import Session

router = APIRouter(prefix="/chat", tags=["chat"])

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
_gemini_client = None
# Use a model supported by the current Gemini API (gemini-1.5-flash deprecated in v1beta)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")


def _get_gemini_client():
    """Return google.genai Client (new SDK); None if key missing or init fails."""
    global _gemini_client
    if _gemini_client is not None:
        return _gemini_client
    key = (os.getenv("GEMINI_API_KEY") or "").strip() or (GEMINI_API_KEY or "").strip()
    if not key:
        return None
    try:
        from google import genai
        _gemini_client = genai.Client(api_key=key)
        return _gemini_client
    except ImportError:
        try:
            # Fallback: legacy SDK if google-genai not installed
            import google.generativeai as genai_legacy
            genai_legacy.configure(api_key=key)
            return ("legacy", genai_legacy.GenerativeModel("gemini-1.5-flash"))
        except Exception as e:
            print(f"[Chat] Gemini init failed: {e}")
            return None
    except Exception as e:
        print(f"[Chat] Gemini init failed: {e}")
        return None


def _gemini_generate(client_or_legacy, prompt: str, max_retries: int = 3, delay: float = 2.0) -> str:
    """Call Gemini with exponential-backoff retry on 429 / quota errors. Returns response text."""
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            if isinstance(client_or_legacy, tuple) and client_or_legacy[0] == "legacy":
                _, model = client_or_legacy
                response = model.generate_content(prompt)
                return response.text if hasattr(response, "text") else str(response)
            else:
                response = client_or_legacy.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt,
                )
                content = getattr(response, "text", None)
                if not content and getattr(response, "candidates", None) and len(response.candidates):
                    c = response.candidates[0]
                    if getattr(c, "content", None) and getattr(c.content, "parts", None) and len(c.content.parts):
                        content = getattr(c.content.parts[0], "text", None)
                return content or str(response)
        except Exception as e:
            last_exc = e
            err_str = str(e).lower()
            is_quota = "429" in err_str or "resource_exhausted" in err_str or "quota" in err_str or "rate limit" in err_str
            if is_quota and attempt < max_retries - 1:
                time.sleep(delay * (attempt + 1))
                continue
            raise
    raise last_exc  # type: ignore[misc]


class ChatMessage(BaseModel):
    role: str = Field(..., description="user or assistant")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    analysis_id: int = Field(..., description="Analysis ID to scope chat context")
    message: str = Field(..., description="User message")
    history: Optional[List[ChatMessage]] = Field(default=None, description="Conversation history")


class ChatResponse(BaseModel):
    content: str
    analysis_id: int


class AssistantChatRequest(BaseModel):
    """General AI assistant (no contract context)."""
    message: str = Field(..., description="User message")
    history: Optional[List[ChatMessage]] = Field(default=None, description="Conversation history")


class AssistantChatResponse(BaseModel):
    content: str


class DocumentChatRequest(BaseModel):
    """Request for document chat using local LFM. Provide either analysis_id or document_context."""
    message: str = Field(..., description="User message")
    document_context: Optional[str] = Field(default=None, description="Full document text (used when no analysis_id)")
    analysis_id: Optional[int] = Field(default=None, description="Saved analysis ID to use as context (owner only)")
    history: Optional[List[ChatMessage]] = Field(default=None, description="Conversation history (optional)")


class DocumentChatResponse(BaseModel):
    content: str
    used_fallback: bool = False


def _build_context(result: Dict[str, Any]) -> str:
    """Build readable context for LFM — plain text only, no raw JSON blobs."""
    parts: List[str] = []

    ocr_chunks = result.get("ocr_chunks") or []
    if ocr_chunks:
        ocr_text = "\n\n".join(
            str(c.get("normalized_text") or c.get("text") or "").strip()
            for c in ocr_chunks
            if c.get("normalized_text") or c.get("text")
        )
        if ocr_text:
            parts.append("## نص العقد\n" + ocr_text[:10000])

    rule_hits = result.get("rule_hits") or []
    if rule_hits:
        hits_lines = []
        for h in rule_hits[:20]:
            rid = h.get("rule_id") or h.get("id") or "?"
            sev = h.get("severity") or "?"
            desc = h.get("description") or ""
            hits_lines.append(f"- [{sev}] {rid}: {desc}")
        parts.append("## المخالفات المكتشفة\n" + "\n".join(hits_lines))

    labor = result.get("labor_summary") or {}
    if isinstance(labor, dict):
        labor_lines: List[str] = []
        violations_detected = labor.get("violations_detected")
        if violations_detected is not None:
            labor_lines.append(f"مخالفات مكتشفة: {'نعم' if violations_detected else 'لا'}")
        risk = labor.get("risk_level") or labor.get("risk") or ""
        if risk:
            labor_lines.append(f"مستوى المخاطرة: {risk}")
        if labor_lines:
            parts.append("## ملخص قانون العمل\n" + "\n".join(labor_lines))

    cb = result.get("cross_border_summary") or {}
    if isinstance(cb, dict) and cb.get("is_cross_border"):
        jur = cb.get("jurisdiction") or cb.get("country") or ""
        parts.append(f"## ملاحظة: عقد عابر للحدود\nالاختصاص القضائي: {jur}")

    return "\n\n".join(parts) if parts else "لا يوجد سياق متاح."


@router.post("/assistant", response_model=AssistantChatResponse)
def chat_assistant(
    payload: AssistantChatRequest,
    current_user: User = Depends(get_current_user),
):
    """General AI assistant chat (no contract context). Uses Gemini with a legal-assistant system prompt."""
    client_or_legacy = _get_gemini_client()
    if not client_or_legacy:
        return AssistantChatResponse(
            content="Chat is not configured. Set GEMINI_API_KEY in the environment to enable the assistant.",
        )
    system_prompt = (
        "You are a helpful legal assistant for Legato. Answer general questions about contracts, "
        "labor law, compliance, and legal terminology concisely. If the user asks about a specific contract, "
        "suggest they use the contract-specific chat with an analysis selected."
    )
    history_str = ""
    if payload.history:
        for m in payload.history[-10:]:
            role = "User" if (m.role or "").lower() == "user" else "Assistant"
            history_str += f"{role}: {m.content}\n"
    full_prompt = f"{system_prompt}\n\n{history_str}User: {payload.message}\n\nAssistant:"
    try:
        content = _gemini_generate(client_or_legacy, full_prompt)
    except Exception as e:
        err_str = str(e).lower()
        if "429" in err_str or "resource_exhausted" in err_str or "quota" in err_str or "rate limit" in err_str:
            content = "The AI chat has reached its usage limit for now. Please try again in a few minutes, or check your API plan and billing."
        else:
            content = "Sorry, the chat request failed. Please try again."
    return AssistantChatResponse(content=content)


@router.post("/message", response_model=ChatResponse)
def chat_message(
    payload: ChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Chat with Gemini about a specific analysis. Context: contract OCR + analysis results."""
    row = db.query(Analysis).filter(Analysis.id == payload.analysis_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Analysis not found")
    is_admin = getattr(current_user, "role", "user") == "admin"
    if row.user_id != current_user.id and not is_admin:
        raise HTTPException(status_code=403, detail="Forbidden")

    try:
        result = json.loads(row.result_json) if isinstance(row.result_json, str) else row.result_json
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid analysis data")

    context = _build_context(result)

    client_or_legacy = _get_gemini_client()
    if not client_or_legacy:
        return ChatResponse(
            content="Chat is not configured. Set GEMINI_API_KEY in the environment to enable contract-aware chat.",
            analysis_id=payload.analysis_id,
        )

    system_prompt = f"""You are a legal contract assistant. Answer questions about the contract based ONLY on the context below.
Context (contract OCR text and analysis results):
{context}

Be concise. Cite sections or rule IDs when relevant. If the answer is not in the context, say so."""

    full_prompt = f"{system_prompt}\n\nUser: {payload.message}\n\nAssistant:"

    try:
        content = _gemini_generate(client_or_legacy, full_prompt)
    except Exception as e:
        err_str = str(e).lower()
        if "429" in err_str or "resource_exhausted" in err_str or "quota" in err_str or "rate limit" in err_str:
            content = (
                "The AI chat has reached its usage limit for now. "
                "Please try again in a few minutes, or check your API plan and billing."
            )
        else:
            content = "Sorry, the chat request failed. Please try again."
    return ChatResponse(content=content, analysis_id=payload.analysis_id)


# ----- Document chat with local LFM -----
_MAX_DOCUMENT_CONTEXT_CHARS = 6000


def _local_model_hint() -> str:
    path_env = (os.getenv("LOCAL_LLM_PATH") or "").strip()
    try:
        from pathlib import Path
        default_path = Path(__file__).resolve().parents[2] / "LFM2.5-1.2B-Instruct"
    except Exception:
        default_path = None
    parts = []
    if path_env:
        parts.append(f"LOCAL_LLM_PATH={path_env}")
    if default_path is not None:
        parts.append(f"default={default_path}")
    return ", ".join(parts) if parts else "no model path configured"


def _format_document_chat_history(history: Optional[List[ChatMessage]]) -> str:
    if not history:
        return ""
    lines: List[str] = []
    for m in history[-10:]:
        role = (m.role or "").strip().lower()
        label = "المستخدم" if role == "user" else "المساعد"
        content = (m.content or "").strip()
        if not content:
            continue
        lines.append(f"{label}: {content[:1200]}")
    if not lines:
        return ""
    return "محادثة سابقة:\n" + "\n".join(lines) + "\n\n"


def _build_document_chat_prompt(context: str, message: str, history: Optional[List[ChatMessage]] = None) -> str:
    """Build prompt for document Q&A; includes optional multi-turn history."""
    hist = _format_document_chat_history(history)
    return f"""أنت مساعد قانوني. أجب على سؤال المستخدم بناءً على النص التالي فقط. إذا لم يكن الجواب في النص فقل ذلك.
استخدم المحادثة السابقة فقط لربط الأسئلة دون إضافة حقائق من خارج النص.

{hist}النص:
{context[: _MAX_DOCUMENT_CONTEXT_CHARS]}

سؤال المستخدم:
{message}

الجواب:"""


def _get_lfm_document_reply(
    document_context: str,
    message: str,
    *,
    history: Optional[List[ChatMessage]] = None,
    max_new_tokens: int = 256,
) -> str:
    """Call local LFM (app/local_llm first, then llm/generate) for document Q&A."""
    context = (document_context or "").strip()
    if len(context) > _MAX_DOCUMENT_CONTEXT_CHARS:
        context = context[:_MAX_DOCUMENT_CONTEXT_CHARS] + "..."
    if not context:
        return "[No document context provided.]"
    try:
        from app.local_llm import is_available as app_avail, generate as app_generate
        if app_avail():
            prompt = _build_document_chat_prompt(context, message, history=history)
            return app_generate(prompt, max_new_tokens=max_new_tokens, do_sample=False)
    except Exception as e_app:
        try:
            from llm.generate import is_available as gen_avail, generate_answer
            if gen_avail():
                ctx_in = context
                if history:
                    ctx_in = _format_document_chat_history(history) + "\nDocument:\n" + context
                return generate_answer(ctx_in, message)
            raise RuntimeError("llm.generate not available") from e_app
        except Exception as e1:
            return f"[Local LLM error: app.local_llm={e_app!r}; llm.generate={e1!r}]"
    try:
        from llm.generate import is_available, generate_answer
        if not is_available():
            return (
                "[Local LLM not available. Check model folder and path "
                f"({_local_model_hint()}).]"
            )
        ctx_in = context
        if history:
            ctx_in = _format_document_chat_history(history) + "\nDocument:\n" + context
        return generate_answer(ctx_in, message)
    except Exception as e2:
        return (
            "[Local LLM not available. Check model folder and path "
            f"({_local_model_hint()}).] Details: {e2!r}"
        )


# Gemini can use more context than the local LFM path
_GEMINI_DOCUMENT_CONTEXT_CHARS = 120000


def _get_gemini_document_reply(
    context: str,
    message: str,
    history: Optional[List[ChatMessage]] = None,
) -> Optional[str]:
    """Answer document Q&A with Gemini when local LFM is not installed. Returns None if not configured."""
    client_or_legacy = _get_gemini_client()
    if not client_or_legacy:
        return None
    ctx = (context or "").strip()
    if len(ctx) > _GEMINI_DOCUMENT_CONTEXT_CHARS:
        ctx = ctx[:_GEMINI_DOCUMENT_CONTEXT_CHARS] + "\n\n[Context truncated for API size limits.]"
    prior = ""
    if history:
        parts: List[str] = []
        for m in history[-10:]:
            role = "User" if (m.role or "").lower() == "user" else "Assistant"
            c = (m.content or "").strip()
            if c:
                parts.append(f"{role}: {c[:4000]}")
        if parts:
            prior = "Prior conversation:\n" + "\n".join(parts) + "\n\n"
    system_prompt = f"""You are a legal contract assistant. Answer the user's question using ONLY the document context below (OCR/analysis text).
If the answer is not supported by the context, say so clearly.
Be concise. Mention rule IDs or sections when they appear in the context.
Use prior conversation only to resolve follow-up questions; do not invent facts not in the document.

{prior}Document context:
{ctx}"""
    full_prompt = f"{system_prompt}\n\nUser: {message}\n\nAssistant:"
    try:
        out = _gemini_generate(client_or_legacy, full_prompt)
        return (out or "").strip()
    except Exception as e:
        err_str = str(e).lower()
        if "429" in err_str or "resource_exhausted" in err_str or "quota" in err_str or "rate limit" in err_str:
            return (
                "[Gemini API] Usage limit or quota reached. "
                "This reply used the cloud fallback, not your local LFM. "
                "Fix LOCAL_LLM_PATH/mount or set DOCUMENT_CHAT_GEMINI_FALLBACK=0 to see the real local error."
            )
        print(f"[Chat] Gemini document reply failed: {e}")
        return None


def _local_document_failed(content: str) -> bool:
    return bool(
        content.startswith("[Local LLM not available")
        or content.startswith("[Local LLM error")
        or content.startswith("[LLM load error")
    )


@router.get("/document")
def chat_document_get():
    """GET is not supported. Use POST with JSON body: document_context, message (and optional history)."""
    raise HTTPException(
        status_code=405,
        detail="Method not allowed. Use POST to send a chat message with document_context and message.",
    )


@router.post("/document", response_model=DocumentChatResponse)
def chat_document(
    payload: DocumentChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Chat with the document using the local LFM model. Provide document_context or analysis_id."""
    context: Optional[str] = None
    if payload.analysis_id is not None:
        row = db.query(Analysis).filter(Analysis.id == payload.analysis_id).first()
        if not row:
            raise HTTPException(status_code=404, detail="Analysis not found")
        is_admin = getattr(current_user, "role", "user") == "admin"
        if row.user_id != current_user.id and not is_admin:
            raise HTTPException(status_code=403, detail="Forbidden")
        try:
            result = json.loads(row.result_json) if isinstance(row.result_json, str) else row.result_json
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid analysis data")
        context = _build_context(result)
    else:
        context = payload.document_context

    if not context or not (context or "").strip():
        raise HTTPException(status_code=400, detail="Provide document_context or analysis_id")

    used_fallback = False
    content = _get_lfm_document_reply(context, payload.message, history=payload.history)
    if _local_document_failed(content):
        allow_gemini = os.getenv("DOCUMENT_CHAT_GEMINI_FALLBACK", "1").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if allow_gemini:
            gemini_reply = _get_gemini_document_reply(context, payload.message, history=payload.history)
            if gemini_reply:
                content = gemini_reply
                used_fallback = True
            else:
                detail = (
                    content.strip("[]")
                    + " Cloud fallback failed or GEMINI_API_KEY missing."
                )
                raise HTTPException(status_code=503, detail=detail)
        else:
            raise HTTPException(status_code=503, detail=content.strip("[]"))
    return DocumentChatResponse(content=content, used_fallback=used_fallback)
