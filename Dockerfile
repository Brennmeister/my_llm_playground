# FastAPI order extractor (the LLM runs in the separate "llm" service, see docker-compose.yml)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Tesseract OCR with German, English, Polish and Italian language packs
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      tesseract-ocr tesseract-ocr-deu tesseract-ocr-eng tesseract-ocr-pol tesseract-ocr-ita \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /srv
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app

RUN useradd --create-home --uid 1000 appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
