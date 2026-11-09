"""Page and row images for the vision part of the model.

The text (text layer or OCR) gives exact characters and word positions; the image gives what text
loses: faint/gray print, stamps, crossed-out values and handwriting. Every LLM call gets both:
the header pass sees the pages of its chunk, each position call sees the strip of its own rows.
Image size is budgeted in tokens so a call stays inside the context window.
"""
from __future__ import annotations

import logging
import math
from typing import Optional

import pymupdf as fitz

from .config import settings
from .pdf_extract import PdfText

log = logging.getLogger(__name__)

# Qwen-VL style encoders: 16 px patches merged 2x2 -> one token per 32 x 32 px
_PIXELS_PER_TOKEN = 32 * 32
MIN_DPI = 60


def image_tokens(width_pt: float, height_pt: float, dpi: float) -> int:
    scale = dpi / 72
    return math.ceil(width_pt * scale * height_pt * scale / _PIXELS_PER_TOKEN)


class PageImages:
    """Renders images of one PDF. Pages are numbered 1-based like PdfWord.page."""

    def __init__(self, pdf_bytes: bytes, pdf: PdfText):
        self.mode = settings.llm_images  # all | scans | off
        self.pdf = pdf
        self.doc: Optional[fitz.Document] = None
        self.sent = 0
        if self.mode != "off" and pdf_bytes:
            try:
                self.doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            except Exception as exc:  # noqa: BLE001 - images are an addition, text still works
                log.warning("Cannot render page images: %s", exc)
        # pages whose content is (partly) an image: scans, stamps, ink annotations - only there
        # handwriting can exist
        self.visual_pages: set[int] = set(pdf.ocr_pages)
        if self.doc is not None:
            for index, page in enumerate(self.doc):
                if index >= len(pdf.page_sizes):
                    break
                if page.get_images() or any(annot.type[0] == fitz.PDF_ANNOT_INK for annot in page.annots()):
                    self.visual_pages.add(index + 1)

    def enabled_for(self, page_number: int) -> bool:
        if self.doc is None or self.mode == "off":
            return False
        return self.mode == "all" or page_number in self.pdf.ocr_pages

    def pages(self, page_numbers: list[int], token_budget: Optional[int] = None) -> list[bytes]:
        """Whole pages, scaled so that all of them together fit the token budget."""
        page_numbers = [n for n in page_numbers if self.enabled_for(n)]
        if not page_numbers:
            return []
        budget = token_budget or settings.llm_page_image_tokens
        area = sum(self.pdf.page_sizes[n - 1][0] * self.pdf.page_sizes[n - 1][1] for n in page_numbers)
        dpi = 72 * math.sqrt(budget * _PIXELS_PER_TOKEN / area)
        dpi = max(MIN_DPI, min(settings.llm_page_image_dpi, dpi))
        return [self._render(n, None, dpi) for n in page_numbers]

    def rows(self, first: int, last: int, stop_before: Optional[int] = None) -> list[bytes]:
        """Strips of the page(s) holding lines first..last, full page width, with the space below the
        last line up to the next position (handwritten corrections are often written there)."""
        words = [w for page in self.pdf.page_words for w in page if first <= w.line <= last]
        if not words:
            return []
        next_top = {}
        if stop_before is not None:
            for word in (w for page in self.pdf.page_words for w in page if w.line == stop_before):
                next_top[word.page] = min(next_top.get(word.page, word.y0), word.y0)
        images = []
        for page_number in sorted({w.page for w in words}):
            if not self.enabled_for(page_number):
                continue
            on_page = [w for w in words if w.page == page_number]
            height = max(w.y1 - w.y0 for w in on_page)
            top = min(w.y0 for w in on_page) - 0.6 * height
            bottom = max(w.y1 for w in on_page) + 2.5 * height
            if page_number in next_top:
                bottom = min(bottom, next_top[page_number] - 0.2 * height)
            width, page_height = self.pdf.page_sizes[page_number - 1]
            clip = fitz.Rect(0, max(0, top), width, min(page_height, max(bottom, top + 2 * height)))
            images.append(self._render(page_number, clip, settings.llm_row_image_dpi))
        return images

    def _render(self, page_number: int, clip: Optional[fitz.Rect], dpi: float) -> bytes:
        page = self.doc[page_number - 1]
        self.sent += 1
        return page.get_pixmap(clip=clip, dpi=round(dpi)).tobytes("png")

    def close(self) -> None:
        if self.doc is not None:
            self.doc.close()
