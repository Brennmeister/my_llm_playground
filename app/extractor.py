"""Orchestration: PDF -> text -> LLM (2 passes per chunk) -> merge -> normalize -> validate."""
from __future__ import annotations

import logging
import re
import time
from typing import Optional

from pydantic import ValidationError

from . import llm
from .config import settings
from .normalize import (
    Grounder,
    clean_str,
    detect_language,
    normalize_country,
    normalize_currency,
    normalize_unit,
    parse_date,
    parse_number,
    split_postal_city,
    valid_email,
)
from .pdf_extract import OcrMode, chunk_pages, extract_text
from .schemas import (
    Contact,
    DocumentHeader,
    ErpOrder,
    LineItem,
    Meta,
    Party,
    RawHeader,
    RawItems,
    RawLineItem,
    RawOrder,
    RawParty,
    Totals,
    Validation,
)

log = logging.getLogger(__name__)

_PARTY_FIELDS = ("buyer", "supplier", "delivery_address", "invoice_address")
_SCALAR_FIELDS = ("order_number", "order_date", "requested_delivery_date", "customer_number", "currency",
                  "payment_terms", "incoterms")
_TOTAL_FIELDS = ("net_total", "tax_rate", "tax_amount", "gross_total")

# Labels that typically appear right before the supplier / buyer address block
_SUPPLIER_LABELS = re.compile(r"\b(an|to|lieferant|supplier|vendor|dostawca|sprzedawca|fornitore|spett\.?\s*le|destinatario)\b", re.I)
_BUYER_LABELS = re.compile(r"\b(besteller|auftraggeber|buyer|customer|bill to|invoice to|zamawiający|nabywca|committente|acquirente|cliente)\b", re.I)


# ---------------------------------------------------------------------------
# Merging chunk results
# ---------------------------------------------------------------------------


def _item_key(it: RawLineItem) -> tuple:
    return (clean_str(it.position), clean_str(it.article_number), clean_str(it.description), clean_str(it.quantity))


def merge_raw(results: list[RawOrder]) -> RawOrder:
    """Header fields: first non-null wins. Totals: last non-null wins (they are on the
    last page). Line items: concatenated, exact duplicates removed."""
    merged = RawOrder()
    for r in results:
        for f in _SCALAR_FIELDS:
            if getattr(merged, f) is None and clean_str(getattr(r, f)):
                setattr(merged, f, getattr(r, f))
        for f in _PARTY_FIELDS:
            target, src = getattr(merged, f), getattr(r, f)
            for pf in RawParty.model_fields:
                if getattr(target, pf) is None and clean_str(getattr(src, pf)):
                    setattr(target, pf, getattr(src, pf))
        for f in _TOTAL_FIELDS:
            if clean_str(getattr(r, f)):
                setattr(merged, f, getattr(r, f))
    seen: set[tuple] = set()
    for r in results:
        for it in r.line_items:
            key = _item_key(it)
            if key in seen or not any(key):
                continue
            seen.add(key)
            merged.line_items.append(it)
    return merged


# ---------------------------------------------------------------------------
# Hallucination guards
# ---------------------------------------------------------------------------


def _ground_party(p: RawParty, g: Grounder, label: str, warnings: list[str]) -> RawParty:
    p = p.model_copy()
    for f in ("company_name", "street", "postal_code", "city", "vat_id", "contact_person"):
        v = clean_str(getattr(p, f))
        if v and not g.has(v):
            warnings.append(f"{label}.{f} '{v}' not found in document - removed.")
            setattr(p, f, None)
    if p.phone and not g.has_digits(p.phone):
        p.phone = None
    p.email = valid_email(p.email)
    if p.email and not g.has(p.email):
        p.email = None
    return p


def _same_party(a: RawParty, b: RawParty) -> bool:
    na, nb = clean_str(a.company_name), clean_str(b.company_name)
    return bool(na and nb and na.lower() == nb.lower())


def _label_score(text: str, name: Optional[str]) -> int:
    """+1 if the name is preceded by a supplier label, -1 if by a buyer label."""
    if not name:
        return 0
    idx = text.lower().find(name.lower())
    if idx < 0:
        return 0
    before = re.sub(r"\s+", " ", text[max(0, idx - 400) : idx])[-50:]
    return int(bool(_SUPPLIER_LABELS.search(before))) - int(bool(_BUYER_LABELS.search(before)))


def sanitize(raw: RawOrder, text: str, warnings: list[str]) -> RawOrder:
    g = Grounder(text)
    raw = raw.model_copy(deep=True)

    if raw.customer_number and "@" in raw.customer_number:
        raw.customer_number = None
    for f in ("order_number", "customer_number", "incoterms"):
        v = clean_str(getattr(raw, f))
        if v and not g.has(v):
            warnings.append(f"{f} '{v}' not found in document - removed.")
            setattr(raw, f, None)

    for f in _PARTY_FIELDS:
        setattr(raw, f, _ground_party(getattr(raw, f), g, f, warnings))

    b, s = raw.buyer, raw.supplier
    # Buyer/supplier role check by the labels in front of the address blocks
    if _label_score(text, b.company_name) > 0 and _label_score(text, s.company_name) <= 0:
        raw.buyer, raw.supplier = s, b
        b, s = raw.buyer, raw.supplier
        warnings.append("Buyer and supplier swapped (address labels indicate the opposite roles).")
    # The model often copies the same VAT id / contact into both parties: keep it on the buyer
    for f in ("vat_id", "contact_person", "email", "phone"):
        if getattr(b, f) and getattr(b, f) == getattr(s, f):
            setattr(s, f, None)
    # Separate addresses that merely repeat the supplier are wrong
    for f in ("delivery_address", "invoice_address"):
        if _same_party(getattr(raw, f), s):
            setattr(raw, f, RawParty())

    for it in raw.line_items:
        if clean_str(it.customer_article_number) and clean_str(it.customer_article_number) == clean_str(it.article_number):
            it.customer_article_number = None
        for f in ("article_number", "customer_article_number"):
            v = clean_str(getattr(it, f))
            if v and not g.has(v):
                warnings.append(f"Article number '{v}' not found in document - removed.")
                setattr(it, f, None)
        if it.delivery_date and clean_str(it.delivery_date) == clean_str(raw.order_date):
            it.delivery_date = None
    return raw


# ---------------------------------------------------------------------------
# Normalization to ERP model
# ---------------------------------------------------------------------------


def _party(raw: RawParty) -> Optional[Party]:
    name = clean_str(raw.company_name)
    street = clean_str(raw.street)
    postal, city, country = split_postal_city(clean_str(raw.postal_code), clean_str(raw.city), clean_str(raw.country))
    if not any((name, street, city)):
        return None
    vat = clean_str(raw.vat_id)
    vat = re.sub(r"[\s.-]", "", vat).upper() if vat else None
    return Party(
        name=name,
        street=street,
        postal_code=postal,
        city=city,
        country_code=normalize_country(country, vat, postal),
        vat_id=vat,
        contact=Contact(name=clean_str(raw.contact_person), email=clean_str(raw.email), phone=clean_str(raw.phone)),
    )


def _is_own_company(party: Optional[Party]) -> bool:
    if not party or not party.name or not settings.own_company_names.strip():
        return False
    name = party.name.lower()
    return any(n.strip().lower() in name for n in settings.own_company_names.split(",") if n.strip())


def _close(a: Optional[float], b: Optional[float]) -> bool:
    return a is not None and b is not None and abs(a - b) <= 0.05


def _repair_totals(t: Totals, warnings: list[str]) -> None:
    """Fix the typical small-model mix-ups of net/gross using arithmetic consistency."""
    net, tax, gross, s = t.net_amount, t.tax_amount, t.gross_amount, t.line_items_sum
    if net is not None and gross is not None and net > gross:
        net, gross = gross, net
    if s is not None and not _close(net, s):
        if _close(gross, s) and (net is None or net > s):
            net, gross = s, net if net is not None and net > s else None  # model put net into gross
        elif tax is not None and _close(net, s + tax):
            net, gross = s, net  # model put gross into net
        elif net is None:
            net = s
    if gross is None and net is not None and tax is not None:
        gross = round(net + tax, 2)
    if tax is None and net is not None and gross is not None and gross >= net:
        tax = round(gross - net, 2)
    if (net, tax, gross) != (t.net_amount, t.tax_amount, t.gross_amount):
        warnings.append("Totals were corrected/completed by arithmetic check - please verify.")
    t.net_amount, t.tax_amount, t.gross_amount = net, tax, gross


def normalize_order(raw: RawOrder, lang: Optional[str], full_text: str) -> ErpOrder:
    warnings: list[str] = []
    raw = sanitize(raw, full_text, warnings) if full_text else raw

    buyer, supplier = _party(raw.buyer), _party(raw.supplier)
    if _is_own_company(buyer) and not _is_own_company(supplier):
        buyer, supplier = supplier, buyer
        warnings.append("Buyer and supplier swapped (buyer matched OWN_COMPANY_NAMES).")

    header = DocumentHeader(
        order_number=clean_str(raw.order_number),
        order_date=parse_date(raw.order_date, lang),
        requested_delivery_date=parse_date(raw.requested_delivery_date, lang),
        customer_number=clean_str(raw.customer_number),
        currency=normalize_currency(raw.currency, raw.gross_total, raw.net_total,
                                    *(i.unit_price for i in raw.line_items[:3]))
        or normalize_currency(None, full_text[:4000])
        or settings.default_currency,
        language=lang,
        payment_terms=clean_str(raw.payment_terms),
        incoterms=clean_str(raw.incoterms),
    )
    if raw.order_date and not header.order_date:
        warnings.append(f"Could not parse order date '{raw.order_date}'.")
    if raw.requested_delivery_date and not header.requested_delivery_date:
        warnings.append(f"Could not parse delivery date '{raw.requested_delivery_date}'.")

    items: list[LineItem] = []
    for idx, ri in enumerate(raw.line_items, start=1):
        qty = parse_number(ri.quantity, lang)
        price = parse_number(ri.unit_price, lang)
        disc = parse_number(ri.discount, lang) or None
        total = parse_number(ri.line_total, lang)
        pos_num = parse_number(ri.position, lang)
        position = int(pos_num) if pos_num and pos_num > 0 and pos_num == int(pos_num) else idx
        factor = 1 - (disc or 0) / 100

        if total is None and qty is not None and price is not None:
            total = round(qty * price * factor, 2)
        elif total is not None and qty and price is not None:
            expected = qty * price * factor
            tol = max(0.02, 0.01 * abs(total))
            # price per 100/1000 units ("Preis pro 100 Stk") is fine
            if abs(expected - total) > tol and not any(abs(expected / f - total) <= tol for f in (100, 1000)):
                warnings.append(f"Position {position}: quantity x unit price ({expected:.2f}) != line total ({total:.2f}).")
        if price is None and total is not None and qty and factor:
            price = round(total / qty / factor, 4)

        item = LineItem(
            position=position,
            supplier_article_number=clean_str(ri.article_number),
            customer_article_number=clean_str(ri.customer_article_number),
            description=clean_str(ri.description),
            quantity=qty,
            unit_code=normalize_unit(ri.unit) or ("H87" if qty is not None and not ri.unit else None),
            unit_raw=clean_str(ri.unit),
            unit_price=price,
            discount_percent=disc,
            line_total=total,
            requested_delivery_date=parse_date(ri.delivery_date, lang) or header.requested_delivery_date,
        )
        if item.description is None and item.supplier_article_number is None and item.customer_article_number is None:
            continue  # garbage row
        if ri.unit and not item.unit_code:
            warnings.append(f"Position {position}: unknown unit '{ri.unit}'.")
        items.append(item)

    line_sum = round(sum(i.line_total for i in items if i.line_total is not None), 2) if items else None
    totals = Totals(
        net_amount=parse_number(raw.net_total, lang),
        tax_rate_percent=parse_number(raw.tax_rate, lang),
        tax_amount=parse_number(raw.tax_amount, lang),
        gross_amount=parse_number(raw.gross_total, lang),
        line_items_sum=line_sum,
    )
    _repair_totals(totals, warnings)
    if totals.net_amount is not None and line_sum is not None and abs(totals.net_amount - line_sum) > 0.05:
        warnings.append(f"Sum of line items ({line_sum:.2f}) differs from net total ({totals.net_amount:.2f}) "
                        "- positions may be missing or totals include surcharges/discounts.")
    if None not in (totals.net_amount, totals.tax_amount, totals.gross_amount):
        if abs(totals.net_amount + totals.tax_amount - totals.gross_amount) > 0.05:
            warnings.append("net + tax != gross.")

    missing = []
    if not header.order_number:
        missing.append("document.order_number")
    if not header.order_date:
        missing.append("document.order_date")
    if not buyer or not buyer.name:
        missing.append("buyer.name")
    if not items:
        missing.append("line_items")
    for it in items:
        if it.quantity is None:
            missing.append(f"line_items[{it.position}].quantity")
        if not (it.supplier_article_number or it.customer_article_number):
            warnings.append(f"Position {it.position}: no article number - ERP matching by description needed.")

    return ErpOrder(
        document=header,
        buyer=buyer or Party(),
        supplier=supplier or Party(),
        ship_to=_party(raw.delivery_address) or buyer,
        bill_to=_party(raw.invoice_address) or buyer,
        line_items=items,
        totals=totals,
        validation=Validation(is_complete=not missing, missing_required_fields=missing, warnings=warnings),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def _extract_chunk(chunk: str, info: str, warnings: list[str]) -> RawOrder:
    parts: dict = {}
    for task, model in (("header", RawHeader), ("items", RawItems)):
        data = await llm.extract(chunk, task, info)
        try:
            parts.update(model.model_validate(data).model_dump())
        except ValidationError as exc:
            warnings.append(f"{info or 'document'} / {task}: LLM output did not match schema ({exc.error_count()} errors).")
    return RawOrder.model_validate(parts)


async def extract_order(pdf_bytes: bytes, filename: Optional[str], ocr: OcrMode = "auto") -> ErpOrder:
    t0 = time.perf_counter()
    pdf = extract_text(pdf_bytes, ocr)
    full_text = pdf.full_text
    lang = detect_language(full_text)

    chunks = chunk_pages(pdf.pages, settings.max_chars_per_chunk) if full_text.strip() else []
    warnings = list(pdf.warnings)
    raw_results = []
    for i, chunk in enumerate(chunks, start=1):
        info = f"part {i} of {len(chunks)}" if len(chunks) > 1 else ""
        raw_results.append(await _extract_chunk(chunk, info, warnings))

    if raw_results:
        order = normalize_order(merge_raw(raw_results), lang, full_text)
    else:
        order = ErpOrder()
        order.validation.missing_required_fields = ["document.order_number", "buyer.name", "line_items"]
    order.validation.warnings = warnings + order.validation.warnings
    order.meta = Meta(
        source_filename=filename,
        page_count=len(pdf.pages),
        ocr_mode=ocr,
        ocr_pages=pdf.ocr_pages,
        text_chars=len(full_text),
        llm_chunks=len(chunks),
        model=settings.llm_model_name,
        duration_s=round(time.perf_counter() - t0, 2),
    )
    return order
