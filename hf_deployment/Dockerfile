FROM python:3.11-slim

# System deps for OCR + OpenCV
RUN apt-get update && apt-get install -y \
    tesseract-ocr tesseract-ocr-ara \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    curl wget git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app/ ./app/
COPY rules/ ./rules/
COPY laws/ ./laws/
COPY chunks/ ./chunks/
COPY model_ML/ ./model_ML/
COPY "Legal Rag/" ./"Legal Rag"/
COPY llm/ ./llm/
COPY scripts/ ./scripts/

# Large model files (uploaded via Git LFS)
COPY LFM2.5-1.2B-Instruct/ ./LFM2.5-1.2B-Instruct/

# Pre-seeded DB and RAG
COPY legalai.db ./legalai.db
COPY chroma_legal/ ./chroma_legal/

ENV PYTHONPATH=/app
ENV PORT=7860
ENV ENABLE_STARTUP_RAG=1
ENV WARMUP_LFM_AT_STARTUP=1
ENV DOCUMENT_CHAT_GEMINI_FALLBACK=0
ENV ACCESS_TOKEN_EXPIRE_MINUTES=720

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
