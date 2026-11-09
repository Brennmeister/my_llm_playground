"""FastAPI app: upload a purchase order PDF, get ERP-ready JSON back."""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import llm
from .config import settings
from .extractor import extract_order
from .pdf_extract import OcrMode, PdfError, extract_text
from .review_store import approve_review, create_review, get_review, match_sources, update_review, validate_order
from .schemas import ErpOrder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = FastAPI(
    title="Order PDF Extractor",
    description="Extracts purchase order data (DE/EN/PL/IT) from PDFs using Qwen3.5-0.8B on llama-server.",
    version="1.0.0",
)
_STATIC = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=_STATIC), name="static")


class ReviewUpdate(BaseModel):
    order: ErpOrder


class ReviewApproval(BaseModel):
    acknowledge_warnings: bool = False


def _review_payload(review: dict) -> dict:
    unmatched = [path for path, source in review["sources"].items() if not source["locations"]]
    return {key: value for key, value in review.items() if key != "pdf"} | {"unmatched_fields": unmatched}


async def _read_pdf(file: UploadFile) -> bytes:
    data = await file.read()
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"File larger than {settings.max_upload_mb} MB.")
    if not data.startswith(b"%PDF"):
        raise HTTPException(415, "Uploaded file is not a PDF.")
    return data


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse("/review")


@app.get("/review", include_in_schema=False)
@app.get("/review/{review_id}", include_in_schema=False)
async def review_page(review_id: str | None = None):
    return FileResponse(_STATIC / "review.html")


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


@app.post("/api/reviews", status_code=201)
async def create_order_review(
    file: UploadFile = File(..., description="Purchase order PDF"),
    ocr: OcrMode = Form("auto", description="OCR strategy"),
):
    data = await _read_pdf(file)
    try:
        pdf = extract_text(data, ocr)
        order = await extract_order(data, file.filename, ocr, pdf_text=pdf)
    except PdfError as exc:
        raise HTTPException(422, str(exc)) from exc
    except llm.LlmError as exc:
        raise HTTPException(503, str(exc)) from exc
    sources = match_sources(order, pdf)
    review_id = create_review(settings.review_db_path, data, order, pdf, sources)
    saved = get_review(settings.review_db_path, review_id)
    return _review_payload(saved)


@app.get("/api/reviews/{review_id}")
async def read_order_review(review_id: str):
    review = get_review(settings.review_db_path, review_id)
    if review is None:
        raise HTTPException(404, "Review not found.")
    return _review_payload(review)


@app.put("/api/reviews/{review_id}")
async def save_order_review(review_id: str, update: ReviewUpdate):
    try:
        review = update_review(settings.review_db_path, review_id, validate_order(update.order))
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    if review is None:
        raise HTTPException(404, "Review not found.")
    return _review_payload(review)


@app.post("/api/reviews/{review_id}/approve")
async def approve_order_review(review_id: str, approval: ReviewApproval):
    review = get_review(settings.review_db_path, review_id)
    if review is None:
        raise HTTPException(404, "Review not found.")
    order = validate_order(ErpOrder.model_validate(review["order"]))
    if order.validation.missing_required_fields:
        raise HTTPException(409, {
            "message": "Complete the required fields before approval.",
            "missing_required_fields": order.validation.missing_required_fields,
        })
    unmatched = [path for path, source in review["sources"].items() if not source["locations"]]
    if (order.validation.warnings or unmatched) and not approval.acknowledge_warnings:
        raise HTTPException(409, "Acknowledge the remaining warnings before approval.")
    approved = approve_review(settings.review_db_path, review_id)
    return _review_payload(approved)


@app.get("/api/reviews/{review_id}/export")
async def export_approved_order(review_id: str):
    review = get_review(settings.review_db_path, review_id)
    if review is None:
        raise HTTPException(404, "Review not found.")
    if review["status"] != "approved":
        raise HTTPException(409, "Approve the review before exporting it.")
    return JSONResponse(
        review["order"],
        headers={"Content-Disposition": f'attachment; filename="purchase-order-{review_id[:8]}-erp.json"'},
    )


@app.get("/api/reviews/{review_id}/pdf")
async def review_pdf(review_id: str):
    review = get_review(settings.review_db_path, review_id)
    if review is None:
        raise HTTPException(404, "Review not found.")
    return Response(review["pdf"], media_type="application/pdf", headers={"Cache-Control": "private, max-age=3600"})
