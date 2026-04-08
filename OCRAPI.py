# ocr_api.py
import os
import io
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import requests

OCR_SPACE_URL = "https://api.ocr.space/parse/image"

app = FastAPI()

OCR_SPACE_API_KEY = os.environ.get("OCR_SPACE_API_KEY", "")  # set this in env or docker-compose

@app.post("/ocr/ocrspace")
async def ocr_with_ocrspace(file: UploadFile = File(...), language: str = "ara", isOverlayRequired: bool = False):
    """
    Upload an image/pdf and get OCR.space response (supports Arabic).
    Returns the parsed text and raw API JSON.
    """
    if not OCR_SPACE_API_KEY:
        raise HTTPException(status_code=500, detail="OCR_SPACE_API_KEY not set in environment")

    contents = await file.read()
    files = {
        "file": (file.filename, io.BytesIO(contents))
    }
    data = {
        "apikey": OCR_SPACE_API_KEY,
        "language": language,
        "isOverlayRequired": str(isOverlayRequired).lower()  # "true" or "false"
    }
    # For PDFs or multi-page images, OCR.space handles it.
    resp = requests.post(OCR_SPACE_URL, files=files, data=data, timeout=60)
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"OCR provider error: {resp.status_code}")

    result = resp.json()
    # Basic parsing: aggregate parsed text
    parsed_texts = []
    try:
        for parsed_result in result.get("ParsedResults", []):
            parsed_texts.append(parsed_result.get("ParsedText", ""))
    except Exception:
        pass

    return JSONResponse({"ok": True, "raw": result, "text": "\n\n".join(parsed_texts)})
