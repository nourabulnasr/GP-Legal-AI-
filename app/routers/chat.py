"""
Chat router: Gemini-powered chatbot scoped to analysis_id.
Context: contract OCR text + analysis results.
"""
from __future__ import annotations

import json
import os
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


def _build_context(result: Dict[str, Any]) -> str:
    """Build context string from analysis result_json."""
    parts: List[str] = []

    ocr_chunks = result.get("ocr_chunks") or []
    if ocr_chunks:
        ocr_text = "\n\n".join(
            str(c.get("normalized_text") or c.get("text") or "").strip()
            for c in ocr_chunks
            if c.get("normalized_text") or c.get("text")
        )
        if ocr_text:
            parts.append("## Contract OCR Text\n" + ocr_text[:15000])

    rule_hits = result.get("rule_hits") or []
    if rule_hits:
        hits_sum = []
        for h in rule_hits[:50]:
            rid = h.get("rule_id") or h.get("id")
            sev = h.get("severity")
            desc = h.get("description")
            hits_sum.append(f"- [{sev}] {rid}: {desc}")
        parts.append("## Analysis Violations / Rule Hits\n" + "\n".join(hits_sum))

    labor = result.get("labor_summary") or {}
    if isinstance(labor, dict):
        labor_str = json.dumps(labor, ensure_ascii=False)[:2000]
        parts.append("## Labor Summary\n" + labor_str)

    cb = result.get("cross_border_summary") or {}
    if isinstance(cb, dict):
        cb_str = json.dumps(cb, ensure_ascii=False)[:1000]
        parts.append("## Cross-Border Summary\n" + cb_str)

    return "\n\n".join(parts) if parts else "No analysis context available."


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
        if isinstance(client_or_legacy, tuple) and client_or_legacy[0] == "legacy":
            _, model = client_or_legacy
            response = model.generate_content(full_prompt)
            content = response.text if hasattr(response, "text") else str(response)
        else:
            response = client_or_legacy.models.generate_content(
                model=GEMINI_MODEL,
                contents=full_prompt,
            )
            content = getattr(response, "text", None)
            if not content and getattr(response, "candidates", None) and len(response.candidates):
                c = response.candidates[0]
                if getattr(c, "content", None) and getattr(c.content, "parts", None) and len(c.content.parts):
                    content = getattr(c.content.parts[0], "text", None)
            content = content or str(response)
    except Exception as e:
        err_str = str(e).lower()
        if "429" in err_str or "resource_exhausted" in err_str or ("resource" in err_str and "exhausted" in err_str) or "quota" in err_str or "rate limit" in err_str:
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
        if isinstance(client_or_legacy, tuple) and client_or_legacy[0] == "legacy":
            _, model = client_or_legacy
            response = model.generate_content(full_prompt)
            content = response.text if hasattr(response, "text") else str(response)
        else:
            response = client_or_legacy.models.generate_content(
                model=GEMINI_MODEL,
                contents=full_prompt,
            )
            content = getattr(response, "text", None)
            if not content and getattr(response, "candidates", None) and len(response.candidates):
                c = response.candidates[0]
                if getattr(c, "content", None) and getattr(c.content, "parts", None) and len(c.content.parts):
                    content = getattr(c.content.parts[0], "text", None)
            content = content or str(response)
    except Exception as e:
        err_str = str(e).lower()
        if "429" in err_str or "resource_exhausted" in err_str or ("resource" in err_str and "exhausted" in err_str) or "quota" in err_str or "rate limit" in err_str:
            content = (
                "The AI chat has reached its usage limit for now. "
                "Please try again in a few minutes, or check your API plan and billing."
            )
        else:
            content = f"Sorry, the chat request failed. Please try again."
    return ChatResponse(content=content, analysis_id=payload.analysis_id)


# ----- Document chat with local LFM -----
_MAX_DOCUMENT_CONTEXT_CHARS = 6000


def _build_document_chat_prompt(context: str, message: str) -> str:
    """Build a single prompt for document Q&A (used when generate_answer is not available)."""
    return f"""أنت مساعد قانوني. أجب على سؤال المستخدم بناءً على النص التالي فقط. إذا لم يكن الجواب في النص فقل ذلك.

النص:
{context[: _MAX_DOCUMENT_CONTEXT_CHARS]}

سؤال المستخدم:
{message}

الجواب:"""


def _get_lfm_document_reply(document_context: str, message: str, max_new_tokens: int = 256) -> str:
    """Call local LFM (llm/generate or app/local_llm) for document Q&A. Returns reply or error message."""
    context = (document_context or "").strip()
    if len(context) > _MAX_DOCUMENT_CONTEXT_CHARS:
        context = context[:_MAX_DOCUMENT_CONTEXT_CHARS] + "..."
    if not context:
        return "[No document context provided.]"
    try:
        from llm.generate import is_available, generate_answer
        if not is_available():
            raise RuntimeError("Local LLM not available")
        return generate_answer(context, message)
    except Exception as e1:
        try:
            from app.local_llm import is_available, generate
            if not is_available():
                return "[Local LLM not available. Check LOCAL_LLM_PATH or LFM2.5-1.2B-Instruct folder.]"
            prompt = _build_document_chat_prompt(context, message)
            return generate(prompt, max_new_tokens=max_new_tokens, do_sample=False)
        except Exception:
            return f"[Local LLM error: {e1!r}]"


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

    content = _get_lfm_document_reply(context, payload.message)
    if content.startswith("[Local LLM not available") or content.startswith("[Local LLM error") or content.startswith("[LLM load error"):
        raise HTTPException(status_code=503, detail=content)
    return DocumentChatResponse(content=content)
