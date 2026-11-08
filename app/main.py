"""FastAPI app: upload a purchase order PDF, get ERP-ready JSON back."""
from __future__ import annotations

import logging

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import RedirectResponse

from . import llm
from .config import settings
from .extractor import extract_order
from .pdf_extract import OcrMode, PdfError, extract_text
from .schemas import ErpOrder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = FastAPI(
    title="Order PDF Extractor",
    description="Extracts purchase order data (DE/EN/PL/IT) from PDFs using Qwen3.5-0.8B on llama-server.",
    version="1.0.0",
)


async def _read_pdf(file: UploadFile) -> bytes:
    data = await file.read()
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"File larger than {settings.max_upload_mb} MB.")
    if not data.startswith(b"%PDF"):
        raise HTTPException(415, "Uploaded file is not a PDF.")
    return data


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse("/docs")


@app.get("/health")
async def health():
    llm_status = await llm.health()
    return {"api": "ok", "llm": llm_status, "llm_url": settings.llm_base_url}


@app.post("/extract", response_model=ErpOrder)
async def extract(
    file: UploadFile = File(..., description="Purchase order PDF"),
    ocr: OcrMode = Query("auto", description="auto = OCR only pages without text layer, always = OCR every page, never = text layer only"),
):
    """Extract order data from a PDF and return it as ERP import JSON."""
    data = await _read_pdf(file)
    try:
        return await extract_order(data, file.filename, ocr)
    except PdfError as exc:
        raise HTTPException(422, str(exc)) from exc
    except llm.LlmError as exc:
        raise HTTPException(503, str(exc)) from exc


@app.post("/text")
async def text_only(file: UploadFile = File(...), ocr: OcrMode = Query("auto")):
    """Debug helper: returns the text the LLM would see (no LLM call)."""
    data = await _read_pdf(file)
    try:
        pdf = extract_text(data, ocr)
    except PdfError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"pages": pdf.pages, "ocr_pages": pdf.ocr_pages, "warnings": pdf.warnings}
