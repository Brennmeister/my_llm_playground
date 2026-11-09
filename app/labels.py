"""Deterministic label lookup for the key order header fields (DE/EN/PL/IT).

Small models sometimes skip or confuse the header values. Purchase orders almost always
print them next to a fixed label ("Bestell-Nr.: 4500012345"), so a regex lookup is a
reliable, zero-cost fallback. Values found here are grounded by construction.
"""
from __future__ import annotations

import re
from typing import Callable, Optional

from .normalize import parse_date

_SEP = r"[\s.:#°º/-]*"  # "Nr.:", "No.", "#", "n°" ...


def _labels(*labels: str) -> re.Pattern[str]:
    alternatives = "|".join(sorted(labels, key=len, reverse=True))
    return re.compile(rf"(?<![\w-])(?:{alternatives}){_SEP}", re.I)


ORDER_NUMBER = _labels(
    r"bestell(?:ung)?[\s-]*(?:nummer|nr|no)", r"auftrags?[\s-]*(?:nummer|nr)", r"bestellung",
    r"purchase\s+order\s+(?:number|no)", r"p\.?\s?o\.?\s*(?:number|no|nr)?", r"order\s+(?:number|no|nr)",
    r"numer\s+zam[oó]wienia", r"nr\s+zam[oó]wienia", r"zam[oó]wienie\s+(?:nr|numer)",
    r"numero\s+(?:d[ei]ll?'?\s*)?ordine", r"n\.?\s*ordine", r"ordine\s+(?:n|nr|numero)",
)
ORDER_DATE = _labels(
    r"bestelldatum", r"auftragsdatum", r"datum", r"order\s+date", r"date", r"data\s+zam[oó]wienia",
    r"data\s+(?:dell'?\s*)?ordine", r"data",
)
DELIVERY_DATE = _labels(
    r"liefertermin", r"lieferdatum", r"wunschtermin", r"wunsch-?liefertermin", r"anlieferung",
    r"(?:requested\s+)?delivery\s+date", r"required\s+(?:by|date)", r"due\s+date",
    r"termin\s+dostawy", r"data\s+dostawy", r"data\s+(?:di\s+)?consegna", r"consegna\s+richiesta",
)
CUSTOMER_NUMBER = _labels(
    r"kunden[\s-]*(?:nummer|nr)", r"kd\.?[\s-]*nr", r"customer\s+(?:number|no|id)", r"account\s+(?:number|no)",
    r"(?:nr|numer)\s+klienta", r"(?:codice|cod\.?)\s+cliente",
)
# Generic date labels must not steal the delivery date ("Delivery date", "Data di consegna")
_DELIVERY_BEFORE = re.compile(r"(liefer|deliver|consegna|dostaw|termin|due|required|wunsch)\w*\W*$", re.I)
_DELIVERY_AFTER = re.compile(r"^(di\s+|della\s+)?(consegna|dostawy|delivery)", re.I)
# Bare labels ("Bestellung", "PO") only count with the value on the same line; a value on
# the next line is accepted for explicit labels ("Bestell-Nr.", "Order number:")
_EXPLICIT_LABEL = re.compile(r"n[ro]\b|num|:", re.I)

_IDENTIFIER = re.compile(r"[A-Z0-9][A-Z0-9/._-]{1,30}", re.I)
_DATE = re.compile(
    r"(?<![\d.])(?:"  # never start inside a number ("4503191146 / 03.11.2026")
    r"\d{4}-\d{1,2}-\d{1,2}"
    r"|\d{1,2}\s?[./-]\s?\d{1,2}\s?[./-]\s?\d{2,4}"
    r"|\d{1,2}\.?\s+[^\W\d_]{3,12}\.?\s+\d{4}"
    r"|[^\W\d_]{3,12}\.?\s+\d{1,2},?\s+\d{4}"
    r"|(?:KW|CW|wk|week|tydz\w*|settimana)\.?\s*\d{1,2}\s*[/.-]?\s*\d{2,4})",
    re.I,
)


def _slash_columns(text: str) -> list[str]:
    """Turn column headers like  "EinkGrp / Bestellnummer / Datum"  over  "C30 / 4503191146 / 03.11.2026"
    into "Bestellnummer: 4503191146" lines (SAP and similar ERP print layouts)."""
    lines = text.splitlines()
    pairs = []
    for labels_line, values_line in zip(lines, lines[1:]):
        labels = [part.strip() for part in re.split(r"\s+/\s+", labels_line)]
        values = [part.strip() for part in re.split(r"\s+/\s+", values_line)]
        if 2 <= len(labels) == len(values) and not any(re.search(r"\d", label) for label in labels[1:]):
            pairs.extend(f"{label}: {value}" for label, value in zip(labels, values) if label and value)
    return pairs


def _candidates(text: str, label: re.Pattern[str], skip_delivery: bool, next_line: Callable[[str], bool]):
    """Yield the text right after each label: rest of the line, or the next line if empty."""
    lines = _slash_columns(text) + text.splitlines()
    for index, line in enumerate(lines):
        for match in label.finditer(line):
            rest = line[match.end():].strip()
            if skip_delivery and (_DELIVERY_BEFORE.search(line[:match.start()][-25:]) or _DELIVERY_AFTER.match(rest)):
                continue
            if not rest and index + 1 < len(lines) and next_line(match.group(0)):
                rest = lines[index + 1].strip()
            if rest:
                yield rest[:60]


def _first(text: str, label: re.Pattern[str], pick: Callable[[str], Optional[str]], *,
           skip_delivery: bool = False, next_line: Callable[[str], bool] = lambda _: True) -> Optional[str]:
    for rest in _candidates(text, label, skip_delivery, next_line):
        value = pick(rest)
        if value:
            return value
    return None


def _pick_identifier(rest: str) -> Optional[str]:
    match = _IDENTIFIER.match(rest)
    if not match:
        return None
    value = match.group(0).rstrip("._-/")
    return value if re.search(r"\d", value) else None


def _pick_date(lang: Optional[str]) -> Callable[[str], Optional[str]]:
    def pick(rest: str) -> Optional[str]:
        match = _DATE.search(rest[:30])
        if match and parse_date(match.group(0), lang):
            return match.group(0)
        return None
    return pick


def find_order_number(text: str) -> Optional[str]:
    return _first(text, ORDER_NUMBER, _pick_identifier, next_line=lambda label: bool(_EXPLICIT_LABEL.search(label)))


def find_customer_number(text: str) -> Optional[str]:
    return _first(text, CUSTOMER_NUMBER, _pick_identifier, next_line=lambda label: bool(_EXPLICIT_LABEL.search(label)))


def find_order_date(text: str, lang: Optional[str]) -> Optional[str]:
    return _first(text, ORDER_DATE, _pick_date(lang), skip_delivery=True)


def find_delivery_date(text: str, lang: Optional[str]) -> Optional[str]:
    return _first(text, DELIVERY_DATE, _pick_date(lang))


def dates_in_text(text: str, lang: Optional[str]) -> set[str]:
    """All dates printed in the document, as ISO strings (any format, incl. calendar weeks)."""
    found = set()
    for match in _DATE.finditer(text):
        iso = parse_date(match.group(0), lang)
        if iso:
            found.add(iso)
    return found
