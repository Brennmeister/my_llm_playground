"""Client for llama.cpp's llama-server (OpenAI compatible API).

Extraction runs in two focused passes per document chunk (header/parties/totals, then
line items). A 0.8B model is far more accurate with a small schema and a single task.
The document text comes FIRST in the prompt and the task instruction LAST, so the second
pass reuses the KV cache of the first one (prompt caching) and costs almost nothing extra.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Literal

import httpx

from .config import settings
from .schemas import llm_json_schema

log = logging.getLogger(__name__)

Task = Literal["header", "items"]

# A notebook should only run one generation at a time; parallel requests would just
# compete for the same CPU and make every request slower.
_llm_lock = asyncio.Semaphore(1)

SYSTEM_PROMPT = """You extract purchase order data from business documents for ERP import.
Documents are in German, English, Polish or Italian. Reply ONLY with JSON matching the schema.
Copy values EXACTLY as written in the document (numbers, dates, units, codes). Do not translate or calculate.
If a value is not written in the document, use null. NEVER invent or guess values."""

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
- gross_total: Gesamtbetrag, Bruttobetrag, Total, Wartość brutto, Totale""",
    "items": """TASK: Extract every ordered article line (table rows) as line_items. Ignore everything else.

For each row:
- position: Pos., Item, Lp., Pos. number as written
- article_number: supplier article / part number (Artikel-Nr., Art.-Nr., Part No., Item No., Indeks, Kod, Codice)
- customer_article_number: ONLY if the row has a SECOND, different article number of the buyer (Ihre Art.-Nr., Your ref.). Else null.
- description: the article name / text (Bezeichnung, Beschreibung, Description, Nazwa, Opis, Descrizione). Always fill it.
- quantity: Menge, Qty, Quantity, Ilość, Quantità
- unit: Einheit, ME, Unit, UoM, J.m., U.M. (e.g. Stk, pcs, szt., pz, kg, m, l)
- unit_price: Einzelpreis, Preis, Unit price, Cena jedn., Prezzo unitario
- discount: Rabatt, Discount, Rabat, Sconto (null if none)
- line_total: Gesamt, Gesamtpreis, Betrag, Amount, Wartość, Importo
- delivery_date: only if the ROW has its own delivery date column, else null.
Do NOT include header rows, subtotal, tax, shipping or total rows.""",
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


async def extract(document_text: str, task: Task, chunk_info: str = "") -> dict[str, Any]:
    doc_part = f"<document{(' ' + chunk_info) if chunk_info else ''}>\n{document_text}\n</document>\n\n"
    payload = {
        "model": settings.llm_model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": doc_part + TASKS[task]},
        ],
        "temperature": settings.llm_temperature,
        "top_p": 1.0,
        "max_tokens": settings.llm_max_tokens,
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
        try:
            async with httpx.AsyncClient(timeout=settings.llm_timeout_s) as client:
                r = await client.post(f"{settings.llm_base_url}/v1/chat/completions", json=payload)
        except httpx.HTTPError as exc:
            raise LlmError(f"llama-server not reachable at {settings.llm_base_url}: {exc!r}") from exc

    if r.status_code != 200:
        raise LlmError(f"llama-server returned {r.status_code}: {r.text[:500]}")

    choice = r.json()["choices"][0]
    content = choice["message"].get("content") or ""
    if choice.get("finish_reason") == "length":
        log.warning("LLM output hit max_tokens (%s); result may be truncated", settings.llm_max_tokens)
    try:
        return _extract_json(content)
    except json.JSONDecodeError as exc:
        raise LlmError(f"LLM did not return valid JSON: {content[:300]}") from exc
