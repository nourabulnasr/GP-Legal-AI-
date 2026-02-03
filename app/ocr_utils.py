# ocr_utils.py
import io
import os
import requests
from PIL import Image, ImageOps, ImageFilter

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except Exception:
    TESSERACT_AVAILABLE = False


def _preprocess_image(img: Image.Image) -> Image.Image:
    """
    Improve OCR accuracy for Arabic:
    - grayscale
    - autocontrast
    - slight sharpening
    """
    img = ImageOps.grayscale(img)
    img = ImageOps.autocontrast(img)
    img = img.filter(ImageFilter.SHARPEN)
    return img


def _pytesseract_ocr(img: Image.Image):
    """
    Use pytesseract to produce line-level chunks.
    """
    config = (
        "--oem 3 "
        "--psm 6 "
        "-c preserve_interword_spaces=1"
    )

    data = pytesseract.image_to_data(
        img,
        lang="ara+eng",
        config=config,
        output_type=pytesseract.Output.DICT
    )

    chunks = []
    current_line = None

    for i in range(len(data["level"])):
        text = (data.get("text") or [""])[i]
        if not text or not text.strip():
            continue

        conf_raw = data.get("conf", ["-1"])[i]
        try:
            conf = float(conf_raw)
        except Exception:
            conf = None

        line_num = (data.get("line_num") or [None])[i]
        left = data.get("left")[i]
        top = data.get("top")[i]
        width = data.get("width")[i]
        height = data.get("height")[i]

        key = f"line_{line_num}"

        if current_line != key:
            chunks.append({
                "id": f"ocr_line_{i}",
                "text": text.strip(),
                "left": left,
                "top": top,
                "width": width,
                "height": height,
                "conf": conf,
            })
            current_line = key
        else:
            chunks[-1]["text"] += " " + text.strip()
            chunks[-1]["conf"] = max(chunks[-1]["conf"] or -1, conf or -1)

    return chunks


def _ocr_space_api(img_bytes: bytes, api_key: str, language="ara"):
    """
    Fallback OCR using OCR.space API
    """
    url = "https://api.ocr.space/parse/image"
    files = {"file": ("image.png", img_bytes)}
    payload = {
        "language": language,
        "isOverlayRequired": True,
        "apikey": api_key or "helloworld",
    }

    r = requests.post(url, files=files, data=payload, timeout=60)
    j = r.json()

    if j.get("IsErroredOnProcessing"):
        raise RuntimeError(j.get("ErrorMessage"))

    chunks = []
    for i, parsed in enumerate(j.get("ParsedResults") or []):
        text = parsed.get("ParsedText", "")
        if text.strip():
            chunks.append({
                "id": f"ocr_space_{i}",
                "text": text.strip(),
                "left": None,
                "top": None,
                "width": None,
                "height": None,
                "conf": None,
            })

    return chunks


def ocr_image_bytes(img_bytes: bytes, ocr_api_key: str = None):
    """
    Main OCR wrapper
    """
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = _preprocess_image(img)

    if TESSERACT_AVAILABLE:
        try:
            return _pytesseract_ocr(img)
        except Exception:
            pass

    return _ocr_space_api(img_bytes, api_key=ocr_api_key)
