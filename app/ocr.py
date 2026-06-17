from typing import List, Literal
from PIL import Image
import io
import fitz  # PyMuPDF
import pytesseract

def _pdf_to_images(pdf_bytes: bytes, dpi: int = 220) -> List[Image.Image]:
    images = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for page in doc:
        mat = fitz.Matrix(dpi/72, dpi/72)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        images.append(img)
    doc.close()
    return images

def _tesseract_ocr_image(img: Image.Image, lang: str = "ara+eng") -> str:
    return pytesseract.image_to_string(img, lang=lang)

def ocr_any(file_bytes: bytes, filename: str, prefer: Literal["tesseract","auto"]="tesseract") -> dict:
    ext = (filename.split(".")[-1] or "").lower()
    pages_text = []
    if ext == "pdf":
        imgs = _pdf_to_images(file_bytes)
    elif ext in ("png","jpg","jpeg","tif","tiff","bmp","webp"):
        imgs = [Image.open(io.BytesIO(file_bytes)).convert("RGB")]
    else:
        raise ValueError("Unsupported file type. Upload PDF or image.")

    for i, im in enumerate(imgs, start=1):
        text = _tesseract_ocr_image(im)
        pages_text.append({"page": i, "text": text})

    return {"engine": "tesseract", "pages": pages_text}

