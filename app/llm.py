"""Client for llama.cpp's llama-server (OpenAI compatible API).

Extraction runs in two focused passes per document chunk (header/parties/totals, then
line items). A 0.8B model is far more accurate with a small schema and a single task.
The document text comes FIRST in the prompt and the task instruction LAST, so the second
pass reuses the KV cache of the first one (prompt caching) and costs almost nothing extra.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import time
from typing import Any, Literal

import httpx

from .config import settings
from .schemas import llm_json_schema

log = logging.getLogger(__name__)

Task = Literal["header", "items", "row"]

# A notebook should only run one generation at a time; parallel requests would just
# compete for the same CPU and make every request slower.
_llm_lock = asyncio.Semaphore(1)

SYSTEM_PROMPT = """You extract purchase order data from business documents for ERP import.
Documents are in German, English, Polish or Italian. Reply ONLY with JSON matching the schema.
Copy values EXACTLY as written in the document (numbers, dates, units, codes). Do not translate or calculate.
If a value is not written in the document, use null. NEVER invent or guess values.
Lines are prefixed with their number ("L12: "); never copy these prefixes into values.
You may also get IMAGES of the document. The text was read from the same document (text layer or OCR) and
gives the exact characters; OCR text can contain errors - then read the value from the image.
Faint or gray print that is missing in the text can be read from the image.
HANDWRITING: if a printed value is crossed out or overwritten by hand, or a value is added by hand, the
HANDWRITTEN value is the valid one: put it into the field and list it in "handwritten" (field, printed value,
handwritten value). Only list what you clearly see written by hand in the image; else "handwritten" is []."""

HANDWRITING_HEADER = """- handwritten: ONLY header values that are crossed out / overwritten / added BY HAND in the image.
  Printed values are never listed there. Usually []."""

HANDWRITING_ROW = """- handwritten: ONLY values of THIS position changed BY HAND in the image (a crossed-out price with a new
  one written next to it, a corrected quantity, ...); printed values are never listed there. Usually [].
- discount: only if a discount is printed for this position, else null."""

TASKS: dict[str, str] = {
    "header": """TASK: Extract the order header, the parties and the totals. Ignore the article lines.

- buyer = company that ISSUES/SENDS the order (Besteller, Auftraggeber, Kunde, Buyer, Customer, Zamawiający, Nabywca, Committente, Acquirente). Its address is usually the letterhead.
- supplier = company the order is SENT TO, which delivers the goods (An, To, Lieferant, Supplier, Vendor, Dostawca, Sprzedawca, Fornitore, Spett.le).
- Each party: company_name, street, postal_code (only the code), city (only the city), country, vat_id (USt-IdNr, VAT, NIP, P.IVA), contact_person, email, phone - only what is written in THAT party's block.
- delivery_address: only if a SEPARATE delivery address is written (Lieferadresse, Lieferanschrift, Ship to, Deliver to, Adres dostawy, Luogo di consegna, Destinazione). Else all null.
- invoice_address: only if a SEPARATE invoice address is written (Rechnungsadresse, Bill to, Invoice to, Adres do faktury, Fatturare a). Else all null.
- order_number: Bestellnummer, Bestell-Nr., Auftrags-Nr., PO Number, Order No., Numer zamówienia, Nr zamówienia, Numero ordine, N. ordine
- order_date: Bestelldatum, Datum, Order date, Date, Data zamówienia, Data ordine
- requested_delivery_date: Liefertermin, Lieferdatum, Wunschtermin, Delivery date, Termin dostawy, Data di consegna
- customer_number: Kundennummer, Kunden-Nr., Customer No., Account No., Nr klienta, Codice cliente
- payment_terms: text after Zahlungsbedingungen, Payment terms, Warunki płatności, Pagamento
- incoterms: Incoterms / Lieferbedingung / Resa, e.g. "DAP München", "FCA Leeds", "EXW"
- net_total: Summe netto, Nettobetrag, Net total, Subtotal, Wartość netto, Imponibile
- tax_rate / tax_amount: MwSt., USt., VAT, Podatek VAT, IVA
- gross_total: Gesamtbetrag, Bruttobetrag, Total, Wartość brutto, Totale
""" + HANDWRITING_HEADER,
    "items": """TASK: Extract every ordered position (article lines of the order table) as line_items. Ignore everything else.

Every document line starts with its number, e.g. "L12: ". These numbers are NOT part of any value.
A position can span SEVERAL lines, e.g. quantity/price/total on one line and the article number and
description on the line(s) below or above. Keep all values of ONE position together; the next position
starts with the next position number. Never take a value from a neighbouring position.

For each position:
- first_line / last_line: numbers of the first and last document line of THIS position
- position: Pos., Item, Lp., Pos. number as written
- article_number: supplier article / material / part number (Artikel-Nr., Art.-Nr., Material, Part No., Item No., Indeks, Kod, Codice). Never a quantity or unit.
- customer_article_number: ONLY if the position has a SECOND, different article number of the buyer (Ihre Art.-Nr., Your ref.). Else null.
- description: the article name / text (Bezeichnung, Beschreibung, Description, Nazwa, Opis, Descrizione).
- quantity: Menge, Bestellmenge, Qty, Quantity, Ilość, Quantità
- unit: Einheit, ME, Unit, UoM, J.m., U.M. (e.g. Stk, Stück, pcs, szt., pz, kg, m, l)
- unit_price: Einzelpreis, Preis, Preis pro Einheit, Unit price, Cena jedn., Prezzo unitario
- discount: Rabatt, Discount, Rabat, Sconto (null if none)
- line_total: Gesamt, Gesamtpreis, Nettowert, Betrag, Amount, Wartość, Importo
- delivery_date: only if the POSITION has its own delivery date, else null.
Do NOT include header rows, notes, subtotal, tax, shipping or total rows.
""" + HANDWRITING_ROW,
    "row": """TASK: The excerpt above contains the table header and the lines of ONE order position (maybe with a
neighbouring line). Extract exactly the position whose quantity / price are on the line given below, with all
of its values, and set first_line / last_line to the lines that belong to it.
Every line starts with its number ("L12: "); these numbers are NOT part of any value. The position can span
several lines (values on one line, article number / description on the next). Use the table header to know
which column is which. Copy values exactly as written; null if a value is not written for THIS position.
- article_number: the supplier's article / material number from the article or material column. NOT a
  reference (Referenz, Ref., Ihre Bestellung, Kontrakt), NOT the position number, NOT a quantity.
- description: the article name / text; quantity, unit, unit_price, line_total: from this position only.
The image shows the strip of the page with this position: use it to read values the text got wrong.
If a printed value of this position is crossed out and a new value is written by hand (below or next to
it), the handwritten value goes into the field and into "handwritten".
""" + HANDWRITING_ROW,
}


class LlmError(RuntimeError):
    pass


def _extract_json(content: str) -> dict[str, Any]:
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.S).strip()
    content = re.sub(r"^```(?:json)?|```$", "", content.strip(), flags=re.M).strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start, end = content.find("{"), content.rfind("}")
        if start != -1 and end > start:
            return json.loads(content[start : end + 1])
        raise


async def health() -> dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{settings.llm_base_url}/health")
            return {"ok": r.status_code == 200, "status": r.json() if r.content else r.status_code}
    except Exception as exc:
        return {"ok": False, "status": str(exc)}


async def extract(document_text: str, task: Task, chunk_info: str = "", instruction: str = "",
                  images: list[bytes] | None = None) -> dict[str, Any]:
    doc_part = f"<document{(' ' + chunk_info) if chunk_info else ''}>\n{document_text}\n</document>\n\n"
    text = doc_part + TASKS[task] + (f"\n{instruction}" if instruction else "")
    user: Any = text
    if images:
        intro = ("Images of the pages of this document part:" if task != "row"
                 else "Image of the page strip with this position:")
        user = [{"type": "text", "text": intro}]
        user += [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}}
                 for png in images]
        user.append({"type": "text", "text": text})
    payload = {
        "model": settings.llm_model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        "temperature": settings.llm_temperature,
        "top_p": 1.0,
        # Line items need far more output tokens than the header (~80 tokens per row)
        "max_tokens": {"items": settings.llm_max_tokens_items, "row": 512}.get(task, settings.llm_max_tokens),
        # Grammar-constrained decoding: the server can only emit schema-valid JSON.
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": f"order_{task}", "schema": llm_json_schema(task)},
        },
        # Qwen3.5 thinking mode costs many tokens/time and does not help extraction
        "chat_template_kwargs": {"enable_thinking": False},
        "cache_prompt": True,
    }

    async with _llm_lock:
        started = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=settings.llm_timeout_s) as client:
                r = await client.post(f"{settings.llm_base_url}/v1/chat/completions", json=payload)
            elapsed = time.perf_counter() - started
        except httpx.HTTPError as exc:
            raise LlmError(f"llama-server not reachable at {settings.llm_base_url}: {exc!r}") from exc

    if r.status_code != 200:
        raise LlmError(f"llama-server returned {r.status_code}: {r.text[:500]}")

    body = r.json()
    choice = body["choices"][0]
    content = choice["message"].get("content") or ""
    truncated = choice.get("finish_reason") == "length"
    if truncated:
        log.warning("LLM output for %s hit max_tokens; result is truncated", task)
    try:
        data = _extract_json(content)
    except json.JSONDecodeError as exc:
        if not truncated:
            raise LlmError(f"LLM did not return valid JSON: {content[:300]}") from exc
        data = _salvage_truncated(content)
    data["_truncated"] = truncated
    data["_stats"] = _call_stats(body, elapsed)
    return data


_CROSSED_SCHEMA = {"type": "object", "properties": {"crossed_out": {"type": "boolean"}},
                   "required": ["crossed_out"], "additionalProperties": False}


async def crossed_out(image: bytes, value: str) -> tuple[bool, dict[str, float]]:
    """One yes/no question about one printed value in a row image: is it struck through by hand?

    A small model that read a row is not reliable at telling a handwritten correction from a note or
    a check mark next to a value; asked about one value, it is."""
    payload = {
        "model": settings.llm_model_name,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(image).decode()}},
            {"type": "text", "text": f'Look only at the printed value "{value}" in this image. Is it CROSSED OUT (a pen '
                                     "line drawn through it)? A check mark, a note or a value written next to it is NOT "
                                     "crossing out."},
        ]}],
        "temperature": 0.0, "max_tokens": 20,
        "response_format": {"type": "json_schema", "json_schema": {"name": "crossed_out", "schema": _CROSSED_SCHEMA}},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    async with _llm_lock:
        started = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=settings.llm_timeout_s) as client:
                r = await client.post(f"{settings.llm_base_url}/v1/chat/completions", json=payload)
        except httpx.HTTPError as exc:
            raise LlmError(f"llama-server not reachable at {settings.llm_base_url}: {exc!r}") from exc
        elapsed = time.perf_counter() - started
    if r.status_code != 200:
        raise LlmError(f"llama-server returned {r.status_code}: {r.text[:300]}")
    body = r.json()
    try:
        answer = bool(_extract_json(body["choices"][0]["message"].get("content") or "").get("crossed_out"))
    except (json.JSONDecodeError, AttributeError):
        answer = False
    return answer, _call_stats(body, elapsed)


def _call_stats(body: dict[str, Any], elapsed: float) -> dict[str, float]:
    timings = body.get("timings") or {}
    usage = body.get("usage") or {}
    return {
        "seconds": elapsed,
        "prompt_tokens": timings.get("prompt_n", usage.get("prompt_tokens", 0)) or 0,
        "cached_prompt_tokens": timings.get("cache_n", 0) or 0,
        "prompt_ms": timings.get("prompt_ms", 0.0) or 0.0,
        "completion_tokens": timings.get("predicted_n", usage.get("completion_tokens", 0)) or 0,
        "completion_ms": timings.get("predicted_ms", 0.0) or 0.0,
    }


_model_info: dict[str, Any] = {}


async def model_info(refresh: bool = False) -> dict[str, Any]:
    """File name, parameter count, size and context of the model llama-server has loaded (cached)."""
    if _model_info and not refresh:
        return _model_info
    info: dict[str, Any] = {}
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            props = (await client.get(f"{settings.llm_base_url}/props")).json()
            models = (await client.get(f"{settings.llm_base_url}/v1/models")).json()
        path = props.get("model_path") or ""
        file_name = path.replace("\\", "/").rsplit("/", 1)[-1]
        # llama-server's download cache prefixes the file with "<owner>_<repo>_"
        info["model_file"] = re.sub(r"^[^_/]+_[^_/]+-GGUF_", "", file_name, flags=re.I) or None
        info["context_size"] = (props.get("default_generation_settings") or {}).get("n_ctx")
        meta = ((models.get("data") or [{}])[0].get("meta")) or {}
        if meta.get("n_params"):
            info["model_params"] = f"{meta['n_params'] / 1e9:.1f}B"
        if meta.get("size"):
            info["model_size_mb"] = round(meta["size"] / 1e6, 1)
    except Exception as exc:  # informational only
        log.info("Could not read model info from llama-server: %s", exc)
        return info
    _model_info.clear()
    _model_info.update(info)
    return info


def _salvage_truncated(content: str) -> dict[str, Any]:
    """Keep the complete line items of a JSON answer that was cut off mid-way."""
    end = content.rfind("},")
    while end > 0:
        try:
            return json.loads(content[: end + 1] + "]}")
        except json.JSONDecodeError:
            end = content.rfind("},", 0, end)
    raise LlmError("LLM output was cut off before the first complete line item.")
