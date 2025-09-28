from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
from typing import Any, List, Dict
import io

import fitz  # PyMuPDF
from PIL import Image
import pytesseract

from .schema import HealthResponse, OCRResponse, ClauseCheckRequest, ClauseCheckResponse
from .utils_text import detect_language, norm_ar
from .rules import RuleEngine

# --- app instance ---
api = FastAPI(title="legalai")
app = api  # alias

# --- paths ---
BASE_DIR = Path(__file__).resolve().parent.parent
RULES_DIR = BASE_DIR / "rules"
LAWS_DIR = BASE_DIR / "laws"

# --- rule engine ---
rule_engine = RuleEngine(RULES_DIR, LAWS_DIR)

# -------------------- health --------------------
@api.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()

# -------------------- ocr -----------------------
def _ocr_image_bytes(img_bytes: bytes) -> str:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return pytesseract.image_to_string(img, lang="ara+eng")

@api.post("/ocr", response_model=OCRResponse)
async def ocr(file: UploadFile = File(...)) -> OCRResponse:
    data = await file.read()
    text = ""
    is_pdf = file.filename.lower().endswith(".pdf") or (file.content_type or "").lower() == "application/pdf"
    if is_pdf:
        doc = fitz.open(stream=data, filetype="pdf")
        parts: List[str] = []
        for page in doc:
            txt = page.get_text("text") or ""
            if len(txt.strip()) < 5:
                pix = page.get_pixmap(dpi=300)
                png = pix.tobytes("png")
                parts.append(_ocr_image_bytes(png))
            else:
                parts.append(txt)
        text = "\n".join(parts)
    else:
        text = _ocr_image_bytes(data)
    return OCRResponse(text=text)

# -------------------- helpers -------------------
def _coerce_hits(raw: Any) -> List[Dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [x if isinstance(x, dict) else {"description": str(x)} for x in raw]
    if isinstance(raw, dict):
        if "matches" in raw and isinstance(raw["matches"], list):
            return [h if isinstance(h, dict) else {"description": str(h)} for h in raw["matches"]]
        if "hits" in raw:
            h = raw["hits"]
            if isinstance(h, list):
                return [x if isinstance(x, dict) else {"description": str(x)} for x in h]
            if isinstance(h, dict):
                return [v if isinstance(v, dict) else {"description": str(v)} for v in h.values()]
        return [raw]
    return [{"description": str(raw)}]

# -------------------- check_clause --------------
@api.post("/check_clause", response_model=ClauseCheckResponse)
def check_clause(req: ClauseCheckRequest):
    try:
        text = req.clause_text or ""
        lang = (req.language or detect_language(text)).lower()
        text_norm = norm_ar(text) if lang == "ar" else text.strip()

        raw_hits = rule_engine.check_text(text_norm, law_scope=req.law_scope) or []
        hits = _coerce_hits(raw_hits)

        resp = ClauseCheckResponse(clause_text=text, language=lang, matches=hits)
        # Force UTF-8 JSON so Arabic shows correctly in files/clients
        return JSONResponse(content=resp.model_dump(), media_type="application/json; charset=utf-8")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Rule check failed: {e.__class__.__name__}: {e}")
