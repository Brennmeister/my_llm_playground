"""Orchestration: PDF -> text -> LLM (2 passes per chunk) -> merge -> normalize -> validate."""
from __future__ import annotations

import logging
import re
import time
from typing import Optional

from pydantic import ValidationError

from . import labels, llm
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
from .pdf_extract import OcrMode, PdfText, chunk_pages, extract_text
from .table_rows import TableRow, parse_rows, reconcile_with_net_total, rows_look_reliable
from .table_rows import _repair_amounts as _repair_row_amounts
from .verify import _numbers as numbers_in_text
from .verify import anchor_segments, arithmetic_ok, extract_positions, fix_position_number, harmonize_with_line, reconcile_items_with_net_total, reread_failed_items
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
# Summary rows that small models sometimes return as line items
_SUMMARY_ROW = re.compile(
    r"^\W*(zwischen)?(summe|gesamt\w*|total\w*|subtotal|netto\w*|brutto\w*|mwst|ust|vat|iva|imponibile|"
    r"versand\w*|fracht\w*|porto|verpackung|shipping|freight|packaging|razem|suma|podatek|wysy\w*|"
    r"spedizione|trasporto|imballo)\b",
    re.I,
)


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
        article = clean_str(it.article_number)
        if article and re.fullmatch(r"[\d.,]+\s*(\S+)", article) and normalize_unit(article.split()[-1]):
            warnings.append(f"Article number '{article}' is a quantity with unit - removed.")
            it.article_number = None
        for f in ("article_number", "customer_article_number"):
            v = clean_str(getattr(it, f))
            if v and not g.has(v):
                warnings.append(f"Article number '{v}' not found in document - removed.")
                setattr(it, f, None)
        # the order date copied into the position - unless the date is printed again (in the row)
        date = clean_str(it.delivery_date)
        if date and date == clean_str(raw.order_date) and text.count(date) < 2:
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


def apply_label_fallbacks(raw: RawOrder, text: str, lang: Optional[str], warnings: list[str]) -> RawOrder:
    """Fill or fix the key header fields from their printed labels (deterministic)."""
    raw = raw.model_copy(deep=True)
    printed_dates = labels.dates_in_text(text, lang)
    for field in ("order_date", "requested_delivery_date"):
        value = clean_str(getattr(raw, field))
        iso = parse_date(value, lang) if value else None
        if value and printed_dates and iso not in printed_dates:
            warnings.append(f"{field} '{value}' does not occur in the document - removed.")
            setattr(raw, field, None)
    printed_numbers = numbers_in_text(text, lang)
    for field in ("net_total", "tax_amount", "gross_total"):
        value = parse_number(getattr(raw, field), lang)
        if value is not None and not any(abs(value - n) < 0.005 for n in printed_numbers):
            warnings.append(f"{field} '{getattr(raw, field)}' does not occur in the document - removed.")
            setattr(raw, field, None)
    for field, finder in (("net_total", labels.find_net_total), ("gross_total", labels.find_gross_total)):
        if not clean_str(getattr(raw, field)):
            found_amount = finder(text)
            if found_amount:
                log.info("%s filled from document label: %r", field, found_amount)
                setattr(raw, field, found_amount)
    order_number = labels.find_order_number(text)
    customer_number = labels.find_customer_number(text)
    # Typical small-model mix-up: the customer number is returned as the order number
    if order_number and raw.order_number and clean_str(raw.order_number) != order_number \
            and clean_str(raw.order_number) in (customer_number, clean_str(raw.customer_number)):
        warnings.append(f"order_number '{raw.order_number}' is the customer number - replaced by '{order_number}'.")
        raw.order_number = None
    found = {
        "order_number": order_number,
        "customer_number": customer_number,
        "order_date": labels.find_order_date(text, lang),
        "requested_delivery_date": labels.find_delivery_date(text, lang),
    }
    if raw.requested_delivery_date and not found["requested_delivery_date"] \
            and parse_date(raw.requested_delivery_date, lang) == parse_date(raw.order_date or found["order_date"], lang):
        warnings.append(f"requested_delivery_date '{raw.requested_delivery_date}' is the order date and has no "
                        "delivery label - removed.")
        raw.requested_delivery_date = None
    for field, value in found.items():
        if value and not clean_str(getattr(raw, field)):
            log.info("%s filled from document label: %r", field, value)
            setattr(raw, field, value)
    return raw


def _is_summary_row(item: RawLineItem) -> bool:
    description = clean_str(item.description) or ""
    return bool(
        not clean_str(item.article_number) and not clean_str(item.customer_article_number)
        and len(description) < 40 and _SUMMARY_ROW.match(description)
    )


def _grounded_text(text: str, g: Grounder) -> bool:
    """True if most words of a longer text occur in the document (allows small OCR/LLM drift)."""
    words = [w for w in re.findall(r"[^\W_]{3,}", text)]
    if not words:
        return True
    return sum(g.has(w) for w in words) / len(words) >= 0.6



def _items_from_llm(raw: RawOrder, lang: Optional[str], header: DocumentHeader, grounder: Optional[Grounder],
                    warnings: list[str]) -> tuple[list[LineItem], list[Optional[list[int]]]]:
    items: list[LineItem] = []
    spans: list[Optional[list[int]]] = []
    for idx, ri in enumerate(raw.line_items, start=1):
        if _is_summary_row(ri):
            warnings.append(f"Ignored a summary row returned as line item: '{clean_str(ri.description)}'.")
            continue
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
        if grounder and item.description and not _grounded_text(item.description, grounder):
            warnings.append(f"Position {position}: description '{item.description}' not found in document - please verify.")
        items.append(item)
        spans.append([ri.first_line, ri.last_line if ri.last_line is not None else ri.first_line]
                     if ri.first_line is not None else None)

    return items, spans


def _items_from_rows(rows: list[TableRow], raw: RawOrder, header: DocumentHeader, grounder: Optional[Grounder],
                     warnings: list[str]) -> list[LineItem]:
    """Line items from the parsed table rows; the LLM only adds what the row text lacks."""
    by_article = {clean_str(item.article_number): item for item in raw.line_items if clean_str(item.article_number)}
    items: list[LineItem] = []
    for index, row in enumerate(rows, start=1):
        llm_item = by_article.pop(row.article_number, None) if row.article_number else None
        position = row.position or index
        description = row.description or (clean_str(llm_item.description) if llm_item else None)
        customer_article = clean_str(llm_item.customer_article_number) if llm_item else None
        if customer_article and grounder and not grounder.has(customer_article):
            customer_article = None
        items.append(LineItem(
            position=position,
            supplier_article_number=row.article_number,
            customer_article_number=customer_article,
            description=description,
            quantity=row.quantity,
            unit_code=normalize_unit(row.unit),
            unit_raw=row.unit,
            unit_price=row.unit_price,
            discount_percent=row.discount_percent,
            line_total=row.line_total,
            requested_delivery_date=header.requested_delivery_date,
        ))
        warnings.extend(f"Position {position}: {note}." for note in row.notes)
    for article, llm_item in by_article.items():
        looks_like_quantity = re.fullmatch(r"[\d.,]+\s*(\S+)", article) and normalize_unit(article.split()[-1])
        if grounder and grounder.has(article) and not _is_summary_row(llm_item) and not looks_like_quantity:
            warnings.append(f"Article '{article}' was returned by the model but not found as a table row - not added, please check.")
    log.info("Line items taken from %d parsed table rows", len(rows))
    return items


def _drop_stale_mismatch_warnings(items: list[LineItem], warnings: list[str]) -> None:
    """Remove "quantity x unit price != line total" warnings of positions the net total check fixed."""
    fixed = {str(item.position) for item in items
             if arithmetic_ok(item.quantity, item.unit_price, item.line_total, item.discount_percent) is not False}
    warnings[:] = [w for w in warnings
                   if not (w.startswith("Position ") and "quantity x unit price" in w and w.split()[1].rstrip(":") in fixed)]


def _drop_repeated_customer_articles(items: list[LineItem], header: DocumentHeader, warnings: list[str]) -> None:
    """A 'customer article number' that equals the customer number or repeats on every row is a header value."""
    values = [item.customer_article_number for item in items if item.customer_article_number]
    repeated = len(items) > 1 and len(values) == len(items) and len(set(values)) == 1
    for item in items:
        if item.customer_article_number and (repeated or item.customer_article_number == header.customer_number):
            item.customer_article_number = None
    if repeated:
        warnings.append(f"Removed customer article number '{values[0]}' repeated on every line (header value).")


def normalize_order(raw: RawOrder, lang: Optional[str], full_text: str) -> ErpOrder:
    warnings: list[str] = []
    raw = sanitize(raw, full_text, warnings) if full_text else raw
    if full_text:
        raw = apply_label_fallbacks(raw, full_text, lang, warnings)
    grounder = Grounder(full_text) if full_text else None

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

    net_total = parse_number(raw.net_total, lang)
    item_lines: list[Optional[list[int]]] = []
    rows = parse_rows([full_text], lang) if full_text and not raw.line_items else []
    if raw.line_items or not rows_look_reliable(rows, net_total):
        # Main path: the LLM's positions (any layout), checked arithmetically
        items, item_lines = _items_from_llm(raw, lang, header, grounder, warnings)
        item_source = "llm"
        reconcile_items_with_net_total(items, net_total, warnings)
        _drop_stale_mismatch_warnings(items, warnings)
    else:
        # Fallback only when the model returned no positions at all: row-wise table parser
        reconcile_with_net_total(rows, net_total)
        items = _items_from_rows(rows, raw, header, grounder, warnings)
        item_lines = [None] * len(items)
        item_source = "table_rows"
        warnings.append("The model returned no line items - positions were read by the table row parser.")
    _drop_repeated_customer_articles(items, header, warnings)
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
        meta=Meta(item_source=item_source, item_lines=item_lines),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def _llm_pass(chunk: str, task: str, info: str, warnings: list[str], stats: dict[str, float]) -> dict:
    data = await llm.extract(chunk, task, info)
    for key, value in data.pop("_stats", {}).items():
        stats[key] = stats.get(key, 0) + value
    stats["calls"] = stats.get("calls", 0) + 1
    if data.pop("_truncated", False):
        warnings.append(f"{info or 'document'} / {task}: LLM output was cut off - "
                        + ("line items may be missing." if task != "header" else "header fields may be missing."))
    return data


async def _extract_header(chunk: str, info: str, warnings: list[str], stats: dict[str, float]) -> RawOrder:
    try:
        return RawOrder.model_validate(RawHeader.model_validate(await _llm_pass(chunk, "header", info, warnings, stats)).model_dump())
    except ValidationError as exc:
        warnings.append(f"{info or 'document'} / header: LLM output did not match schema ({exc.error_count()} errors).")
        return RawOrder()


def _check_against_lines(items: list[RawLineItem], segments, lines: dict[int, str], lang: Optional[str],
                         warnings: list[str]) -> None:
    for item, (first, last, printed, row) in zip(items, segments):
        fix_position_number(item, printed, lines.get(first, ""))
        harmonize_with_line(item, row, [lines[n] for n in range(first, last + 1) if n in lines], lang, warnings)


async def _extract_items_whole(chunks: list[str], warnings: list[str], stats: dict[str, float]) -> list[RawLineItem]:
    """All positions in one LLM call per chunk - for tables without recognizable position lines."""
    items: list[RawLineItem] = []
    for i, chunk in enumerate(chunks, start=1):
        info = f"part {i} of {len(chunks)}" if len(chunks) > 1 else ""
        try:
            items += RawItems.model_validate(await _llm_pass(chunk, "items", info, warnings, stats)).line_items
        except ValidationError as exc:
            warnings.append(f"{info or 'document'} / items: LLM output did not match schema ({exc.error_count()} errors).")
    return items


async def extract_order(
    pdf_bytes: bytes,
    filename: Optional[str],
    ocr: OcrMode = "auto",
    pdf_text: Optional[PdfText] = None,
) -> ErpOrder:
    t0 = time.perf_counter()
    pdf = pdf_text or extract_text(pdf_bytes, ocr)
    full_text = pdf.full_text
    lang = detect_language(full_text)

    # The model sees numbered lines ("L12: ...") so it can cite where each position comes from
    chunks = chunk_pages(pdf.numbered_pages, settings.max_chars_per_chunk) if full_text.strip() else []
    warnings = list(pdf.warnings)
    stats: dict[str, float] = {}
    raw_results = []
    for i, chunk in enumerate(chunks, start=1):
        info = f"part {i} of {len(chunks)}" if len(chunks) > 1 else ""
        raw_results.append(await _extract_header(chunk, info, warnings, stats))

    rereads = 0
    if raw_results:
        merged = merge_raw(raw_results)
        lines = pdf.lines
        segments = anchor_segments(lines, lang)
        if segments:
            # line readings of the position lines, cross-checked with the printed net total (OCR repair)
            net = parse_number(merged.net_total, lang) or parse_number(labels.find_net_total(full_text), lang)
            rows = [segment[3] for segment in segments]
            for row in rows:
                _repair_row_amounts(row)
            reconcile_with_net_total(rows, net)
            # one focused LLM call per position: values cannot slip into a neighbouring position
            merged.line_items = await extract_positions(segments, lines, stats)
            rereads = await reread_failed_items(merged.line_items, lines, lang, warnings, stats)
            _check_against_lines(merged.line_items, segments, lines, lang, warnings)
        else:
            merged.line_items = await _extract_items_whole(chunks, warnings, stats)
            if merged.line_items:
                rereads = await reread_failed_items(merged.line_items, lines, lang, warnings, stats)
                for item in merged.line_items:  # a cited line number is not a position number
                    if item.first_line:
                        fix_position_number(item, None, lines.get(item.first_line, ""))
        order = normalize_order(merged, lang, full_text)
    else:
        order = ErpOrder()
        order.validation.missing_required_fields = ["document.order_number", "buyer.name", "line_items"]
    order.validation.warnings = warnings + order.validation.warnings
    rate = lambda tokens, ms: round(tokens / ms * 1000, 1) if tokens and ms else None  # noqa: E731
    order.meta = Meta(
        source_filename=filename,
        page_count=len(pdf.pages),
        ocr_mode=ocr,
        ocr_pages=pdf.ocr_pages,
        text_chars=len(full_text),
        llm_chunks=len(chunks),
        model=settings.llm_model_name,
        duration_s=round(time.perf_counter() - t0 + (pdf.seconds if pdf_text is not None else 0), 2),
        **(await llm.model_info() if stats else {}),
        text_extraction_s=pdf.seconds,
        llm_s=round(stats.get("seconds", 0.0), 2),
        llm_calls=int(stats.get("calls", 0)),
        prompt_tokens=int(stats.get("prompt_tokens", 0)),
        cached_prompt_tokens=int(stats.get("cached_prompt_tokens", 0)),
        completion_tokens=int(stats.get("completion_tokens", 0)),
        prompt_tokens_per_s=rate(stats.get("prompt_tokens"), stats.get("prompt_ms")),
        completion_tokens_per_s=rate(stats.get("completion_tokens"), stats.get("completion_ms")),
        item_source=order.meta.item_source,
        item_lines=order.meta.item_lines,
        items_reread=rereads,
    )
    return order
