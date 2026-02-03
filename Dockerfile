FROM python:3.11-slim

# System deps: SQLite, Tesseract OCR (Arabic + English for contract OCR)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        sqlite3 \
        tesseract-ocr \
        tesseract-ocr-ara \
        tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "app.main:api", "--host", "0.0.0.0", "--port", "8000"]
