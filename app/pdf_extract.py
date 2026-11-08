"""PDF -> text, with optional OCR (Tesseract) for scanned pages."""
from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Literal

import pymupdf as fitz

from .config import settings

log = logging.getLogger(__name__)

OcrMode = Literal["auto", "always", "never"]


class PdfError(ValueError):
    pass


@dataclass
class PdfText:
    pages: list[str] = field(default_factory=list)
    ocr_pages: list[int] = field(default_factory=list)  # 1-based page numbers
    warnings: list[str] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return "\n\n".join(f"--- Page {i + 1} ---\n{t}" for i, t in enumerate(self.pages))


@lru_cache(maxsize=1)
def _available_ocr_langs() -> tuple[str, ...]:
    import pytesseract

    try:
        return tuple(pytesseract.get_languages(config=""))
    except Exception as exc:  # tesseract binary missing
        log.warning("Tesseract not available: %s", exc)
        return ()


def _ocr_lang_string(warnings: list[str]) -> str | None:
    available = set(_available_ocr_langs())
    if not available:
        warnings.append("OCR requested but Tesseract is not installed (run install.sh).")
        return None
    wanted = [l for l in settings.ocr_languages.split("+") if l]
    usable = [l for l in wanted if l in available]
    missing = [l for l in wanted if l not in available]
    if missing:
        warnings.append(f"Tesseract language packs missing: {', '.join(missing)}")
    return "+".join(usable or ["eng"])


def _ocr_page(page: fitz.Page, lang: str) -> str:
    import pytesseract
    from PIL import Image

    pix = page.get_pixmap(dpi=settings.ocr_dpi, colorspace=fitz.csGRAY)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    # psm 6 = "uniform block of text" keeps table rows on one line, which is what
    # we want for order positions.
    return pytesseract.image_to_string(img, lang=lang, config="--psm 6")


def extract_text(pdf_bytes: bytes, ocr: OcrMode = "auto") -> PdfText:
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise PdfError(f"Could not open PDF: {exc}") from exc

    if doc.needs_pass:
        raise PdfError("PDF is password protected.")
    if doc.page_count == 0:
        raise PdfError("PDF has no pages.")

    result = PdfText()
    if doc.page_count > settings.max_pages:
        result.warnings.append(f"PDF has {doc.page_count} pages, only the first {settings.max_pages} are processed.")

    ocr_lang: str | None = None
    for idx, page in enumerate(doc):
        if idx >= settings.max_pages:
            break
        text = page.get_text("text", sort=True).strip()
        needs_ocr = ocr == "always" or (ocr == "auto" and len(text) < settings.ocr_min_chars_per_page)
        if needs_ocr:
            if ocr_lang is None:
                ocr_lang = _ocr_lang_string(result.warnings) or ""
            if ocr_lang:
                try:
                    ocr_text = _ocr_page(page, ocr_lang).strip()
                    # In "auto" mode keep whichever text is richer
                    if ocr == "always" or len(ocr_text) > len(text):
                        text = ocr_text
                    result.ocr_pages.append(idx + 1)
                except Exception as exc:
                    result.warnings.append(f"OCR failed on page {idx + 1}: {exc}")
        result.pages.append(text)

    doc.close()
    if not any(p.strip() for p in result.pages):
        result.warnings.append("No text found in PDF (scanned document without OCR?).")
    return result


def chunk_pages(pages: list[str], max_chars: int) -> list[str]:
    """Group pages into chunks of at most ``max_chars`` (pages longer than that are split)."""
    chunks: list[str] = []
    current = ""
    for i, page in enumerate(pages):
        block = f"--- Page {i + 1} ---\n{page}\n"
        while len(block) > max_chars:  # oversize page: split on line boundaries
            cut = block.rfind("\n", 0, max_chars)
            cut = cut if cut > max_chars // 2 else max_chars
            if current:
                chunks.append(current)
                current = ""
            chunks.append(block[:cut])
            block = block[cut:]
        if len(current) + len(block) > max_chars and current:
            chunks.append(current)
            current = ""
        current += block
    if current.strip():
        chunks.append(current)
    return chunks
