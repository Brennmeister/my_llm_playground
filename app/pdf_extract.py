"""PDF -> text, with optional OCR (Tesseract) for scanned pages."""
from __future__ import annotations

import io
import time
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
    # Non-empty text lines per page; line numbers are global (1-based, across pages) and
    # every PdfWord knows its line, so the LLM can cite lines and we can map them to the page.
    page_lines: list[list[str]] = field(default_factory=list)
    page_words: list[list["PdfWord"]] = field(default_factory=list)
    page_sizes: list[tuple[float, float]] = field(default_factory=list)
    ocr_pages: list[int] = field(default_factory=list)  # 1-based page numbers
    warnings: list[str] = field(default_factory=list)
    seconds: float = 0.0  # time for text layer + OCR

    @property
    def full_text(self) -> str:
        return "\n\n".join(f"--- Page {i + 1} ---\n{t}" for i, t in enumerate(self.pages))

    @property
    def lines(self) -> dict[int, str]:
        """Global line number -> text."""
        numbered, number = {}, 0
        for page in self.page_lines:
            for line in page:
                number += 1
                numbered[number] = line
        return numbered

    @property
    def numbered_pages(self) -> list[str]:
        """Pages with "L<n>: " line prefixes - the form the LLM sees, so it can cite source lines."""
        pages, number = [], 0
        for page in self.page_lines:
            out = []
            for line in page:
                number += 1
                out.append(f"L{number}: {line}")
            pages.append("\n".join(out))
        return pages


@dataclass
class PdfWord:
    id: str
    page: int
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    line: int = 0  # global line number (see PdfText.lines)


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


def _ocr_page(page: fitz.Page, lang: str, page_number: int) -> list[list[PdfWord]]:
    import pytesseract
    from PIL import Image

    pix = page.get_pixmap(dpi=settings.ocr_dpi, colorspace=fitz.csGRAY)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    data = pytesseract.image_to_data(img, lang=lang, config="--psm 6", output_type=pytesseract.Output.DICT)
    scale_x = page.rect.width / pix.width
    scale_y = page.rect.height / pix.height
    lines: dict[tuple[int, int, int], list[PdfWord]] = {}
    count = 0
    for i, text in enumerate(data["text"]):
        text = text.strip()
        if not text:
            continue
        x0, y0 = data["left"][i], data["top"][i]
        x1, y1 = x0 + data["width"][i], y0 + data["height"][i]
        word = PdfWord(
            id=f"p{page_number}-w{count}", page=page_number, text=text,
            x0=x0 * scale_x, y0=y0 * scale_y, x1=x1 * scale_x, y1=y1 * scale_y,
        )
        count += 1
        lines.setdefault((data["block_num"][i], data["par_num"][i], data["line_num"][i]), []).append(word)
    return list(lines.values())


def _native_page_lines(page: fitz.Page, page_number: int) -> list[list[PdfWord]]:
    """Words of the text layer grouped into visual lines (top to bottom, left to right).

    Grouping by position instead of by PDF text block keeps table rows on one line even when
    every cell is its own block - the same reading order the OCR produces for scans.
    """
    words = []
    for item in page.get_text("words"):
        text = item[4].strip()
        if not text:
            continue
        bounds = fitz.Rect(item[:4]) * page.rotation_matrix
        words.append((bounds, text))
    words.sort(key=lambda w: ((w[0].y0 + w[0].y1) / 2, w[0].x0))
    lines: list[list[tuple[fitz.Rect, str]]] = []
    for bounds, text in words:
        center = (bounds.y0 + bounds.y1) / 2
        if lines:
            ref = lines[-1]
            top = sum(r.y0 for r, _ in ref) / len(ref)
            bottom = sum(r.y1 for r, _ in ref) / len(ref)
            if top <= center <= bottom:
                ref.append((bounds, text))
                continue
        lines.append([(bounds, text)])
    result, count = [], 0
    for line in lines:
        line.sort(key=lambda w: w[0].x0)
        row = []
        for bounds, text in line:
            row.append(PdfWord(id=f"p{page_number}-w{count}", page=page_number, text=text,
                               x0=bounds.x0, y0=bounds.y0, x1=bounds.x1, y1=bounds.y1))
            count += 1
        result.append(row)
    return result


def _line_text(words: list[PdfWord]) -> str:
    """Join words; a wide horizontal gap (column break) becomes several spaces."""
    parts = []
    for previous, word in zip([None, *words], words):
        if previous is not None:
            gap = word.x0 - previous.x1
            height = max(1.0, word.y1 - word.y0)
            parts.append("   " if gap > 2.5 * height else " ")
        parts.append(word.text)
    return "".join(parts)


def extract_text(pdf_bytes: bytes, ocr: OcrMode = "auto") -> PdfText:
    started = time.perf_counter()
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
    line_number = 0
    for idx, page in enumerate(doc):
        if idx >= settings.max_pages:
            break
        lines = _native_page_lines(page, idx + 1)
        text_chars = sum(len(word.text) for line in lines for word in line)
        needs_ocr = ocr == "always" or (ocr == "auto" and text_chars < settings.ocr_min_chars_per_page)
        if needs_ocr:
            if ocr_lang is None:
                ocr_lang = _ocr_lang_string(result.warnings) or ""
            if ocr_lang:
                try:
                    ocr_lines = _ocr_page(page, ocr_lang, idx + 1)
                    ocr_chars = sum(len(word.text) for line in ocr_lines for word in line)
                    # In "auto" mode keep whichever text is richer
                    if ocr == "always" or ocr_chars > text_chars:
                        lines = ocr_lines
                    result.ocr_pages.append(idx + 1)
                except Exception as exc:
                    result.warnings.append(f"OCR failed on page {idx + 1}: {exc}")
        page_lines = []
        for line in lines:
            line_number += 1
            for word in line:
                word.line = line_number
            page_lines.append(_line_text(line))
        result.page_lines.append(page_lines)
        result.pages.append("\n".join(page_lines))
        result.page_words.append([word for line in lines for word in line])
        result.page_sizes.append((page.rect.width, page.rect.height))

    doc.close()
    if not any(p.strip() for p in result.pages):
        result.warnings.append("No text found in PDF (scanned document without OCR?).")
    result.seconds = round(time.perf_counter() - started, 2)
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
