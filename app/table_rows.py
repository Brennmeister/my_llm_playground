"""Deterministic line item parser for row-wise order tables.

Most purchase orders print one line per position: "<pos> <article> [text] <qty> <unit> <price> <total>",
optionally followed by description lines below it. A small LLM tends to mix up values of
neighbouring rows in such tables; parsing the row structure directly keeps every value in
its own row. The LLM result is only used to fill fields the parser cannot see.
"""
from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field
from typing import Optional

from .normalize import _UNIT_LOOKUP, _UNITS, normalize_unit, parse_number

_UNIT_WORDS = sorted(set(_UNIT_LOOKUP) | {code.lower() for code in _UNITS}, key=len, reverse=True)
_QTY_UNIT = re.compile(
    r"(?<![\w,.])(?P<qty>\d{1,3}(?:[.' ]\d{3})+(?:,\d+)?|\d+(?:[.,]\d+)?)\s*(?P<unit>" + "|".join(map(re.escape, _UNIT_WORDS)) + r")\.?(?![\w])",
    re.I,
)
# Money-like numbers after the unit. OCR often turns the decimal comma into ":" or ";".
_AMOUNT = re.compile(r"(?<![\w.,])-?\d{1,3}(?:[.,' ]\d{3})*(?:[,.:;]\d{2})(?![\d])|(?<![\w.,])-?\d+(?![\w.,])")
# A free-standing money amount ("525,00", "5,355.00 EUR") - description lines do not contain those
_MONEY = re.compile(r"(?:^|\s)\d{1,3}(?:[.,' ]\d{3})*[.,]\d{2}(?=\s|$|€|zł|EUR|PLN)")
_PERCENT = re.compile(r"(\d+(?:[.,]\d+)?)\s*%")
_POSITION = re.compile(r"^\d{1,5}(?:[.,]0{1,2})?$")  # "10", "00010", "1,0" / "2.00"
# "60 x 10 x 175mm": a dimension, not a quantity
_DIMENSION = re.compile(r"\d\s*[x×*]\s*$|^\s*[x×*]\s*\d", re.I)
_DECIMAL_AMOUNT = re.compile(r"\d[,.:;]\d{2}(?!\d)")
_ARTICLE = re.compile(r"^(?=.*\d)[\w./+-]{3,}$")
_STOP_LINE = re.compile(
    r"^\s*(referenz|reference|ref\.|ihre\s+ref|total|summe|zwischensumme|gesamt|übertrag|uebertrag|netto|brutto|"
    r"mwst|ust|vat|vorsteuer|endbetrag|razem|suma|totale|imponibile|post-\s+und|bankverbindung|seite|page|strona|pagina|"
    r"net\b|net\s+total|subtotal|amount|tax|wartość|podsumowanie|iva\b|importo|"
    r"bestellung:|datum:|liefer|zahlungs|payment|mit\s+freundlichen|rechnung|invoice|warenwert)",
    re.I,
)
# OCR debris at the end of a line: "|", "©", "/", stray single letters
_NOISE_TAIL = re.compile(r"(\s+([^\w\s]+|[^\W\d_]))+\s*$")


@dataclass
class TableRow:
    position: Optional[int]
    article_number: Optional[str]
    description: Optional[str]
    quantity: float
    unit: str
    unit_price: Optional[float]
    line_total: Optional[float]
    discount_percent: Optional[float] = None
    notes: list[str] = field(default_factory=list)
    total_alternatives: list[float] = field(default_factory=list)  # other readings of an OCR'd total


def _amount(token: str, lang: Optional[str]) -> Optional[float]:
    token = re.sub(r"(?<=\d)[:;](?=\d{2}$)", ",", token)
    return parse_number(token, lang)


def _clean_line(line: str) -> str:
    return _NOISE_TAIL.sub("", line.strip()).strip(" |")


def _join(parts: list[str]) -> Optional[str]:
    text = ""
    for part in parts:
        part = _clean_line(part)
        if not part:
            continue
        last_token = text.split(" ")[-1] if text else ""
        glue = "" if text and ("_" in last_token or part[0] in "_." or text.endswith(("_", "-"))) else " "
        text = f"{text}{glue}{part}" if text else part
    return text or None


def _parse_anchor(line: str, lang: Optional[str]) -> Optional[TableRow]:
    # the LAST "<number> <unit>" followed by amounts: "Stahlblech 2 mm  12,5 kg  3,40  42,50" -> 12,5 kg
    match = None
    for candidate in _QTY_UNIT.finditer(line):
        if _DIMENSION.search(line[:candidate.start()]) or _DIMENSION.search(line[candidate.end():]):
            continue
        # a price column carries decimals: "10 Stk 0,00", not "175mm / 100 Stck."
        if _DECIMAL_AMOUNT.search(_PERCENT.sub(" ", line[candidate.end():])):
            match = candidate
    if not match:
        return None
    before = line[:match.start()].split()
    if not before or not any(re.search(r"\d", token) for token in before[:3]):
        return None
    position = int(re.split(r"[.,]", before[0])[0]) if _POSITION.match(before[0]) else None
    rest_tokens = before[1:] if position is not None else before
    article = None
    if rest_tokens and _ARTICLE.match(rest_tokens[0].strip("—–-")):
        article = rest_tokens[0].strip("—–-")
        rest_tokens = rest_tokens[1:]
    if position is None and article is None:
        return None
    inline = " ".join(rest_tokens).strip(" /|-—")
    after = line[match.end():]
    discount = _PERCENT.search(after)
    tokens = [token for token in _AMOUNT.findall(_PERCENT.sub(" ", after)) if _amount(token, lang) is not None]
    amounts = [_amount(token, lang) for token in tokens]
    alternatives = []
    if tokens and re.search(r"\d\s\d", tokens[-1]):  # "7 151,00": stray digit or thousands group?
        alternatives.append(_amount(tokens[-1].split(None, 1)[1], lang))
    quantity = parse_number(match.group("qty"), lang)
    if quantity is None or not amounts:
        return None
    price = amounts[0] if len(amounts) >= 2 else None
    total = amounts[-1]
    return TableRow(
        position=position, article_number=article,
        description=inline if re.search(r"[^\W\d_]{3,}", inline) else None,
        quantity=quantity, unit=match.group("unit"), unit_price=price, line_total=total,
        discount_percent=parse_number(discount.group(1), lang) if discount else None,
        total_alternatives=[value for value in alternatives if value is not None],
    )


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= max(0.02, 0.005 * abs(b))


def _repair_amounts(row: TableRow) -> None:
    """Use quantity x price = total to repair typical OCR slips in one of the amounts."""
    factor = 1 - (row.discount_percent or 0) / 100
    qty, price, total = row.quantity, row.unit_price, row.line_total
    if not qty or total is None:
        return
    if price is not None and _close(qty * price * factor, total):
        return
    if price is not None:
        for divisor in (100, 10, 1000):  # lost decimal separator ("1545" for 15,45)
            if _close(qty * price / divisor * factor, total):
                row.unit_price = round(price / divisor, 4)
                row.notes.append(f"unit price read as '{price:g}', corrected to {row.unit_price:g} (qty x price = total)")
                return
        if any(_close(qty * price * factor / per, total) for per in (100, 1000)):
            return  # price per 100/1000 units
        row.notes.append(f"quantity x unit price ({qty * price * factor:.2f}) != line total ({total:.2f})")
        return
    row.unit_price = round(total / qty / factor, 4) if factor else None
    row.notes.append("unit price not readable - derived from line total / quantity")


def _second_line_article(line: str) -> Optional[tuple[str, Optional[str]]]:
    """Article number at the start of the line below a row: digits (>= 4) or a code with digits."""
    tokens = line.strip().split(None, 1)
    if not tokens:
        return None
    token = tokens[0].strip("—–-|")
    if not _ARTICLE.match(token) or not re.search(r"\d{4,}|[A-Za-z].*\d|\d.*[A-Za-z]", token):
        return None
    if re.fullmatch(r"\d{1,2}[./-]\d{1,2}[./-]\d{2,4}", token):  # a date, not an article
        return None
    rest = _clean_line(tokens[1]) if len(tokens) > 1 else ""
    return token, rest or None


def parse_rows(pages: list[str], lang: Optional[str]) -> list[TableRow]:
    rows: list[TableRow] = []
    current: Optional[TableRow] = None
    description_lines: list[str] = []
    collecting = False

    def finish():
        if current is None:
            return
        below = _join(description_lines)
        if below:
            current.description = f"{current.description} {below}" if current.description and not below.startswith(current.description) else below
        _repair_amounts(current)
        rows.append(current)

    for page in pages:
        for line in page.splitlines():
            if not line.strip():
                continue
            row = _parse_anchor(line, lang)
            if row:
                finish()
                current, description_lines, collecting = row, [], True
                continue
            if current is not None and collecting:
                if _STOP_LINE.match(line) or _MONEY.search(line) or len(description_lines) >= 4:
                    collecting = False
                elif current.article_number is None and not description_lines and _second_line_article(line):
                    # Two-line rows (SAP style): "00010 150 Stück 9,85 1.477,50" / "2145723 HALTEWINKEL ..."
                    # The second line holds article + description; lines after it are notes.
                    article, rest = _second_line_article(line)
                    current.article_number = article
                    if rest:
                        current.description = f"{current.description} {rest}" if current.description else rest
                    collecting = False
                else:
                    description_lines.append(line)
    finish()
    return rows


def _expected(row: TableRow) -> Optional[float]:
    if row.unit_price is None or not row.quantity:
        return None
    return round(row.quantity * row.unit_price * (1 - (row.discount_percent or 0) / 100), 2)


def _consistent(row: TableRow) -> bool:
    expected = _expected(row)
    if expected is None or row.line_total is None:
        return True
    return _close(expected, row.line_total) or any(_close(expected / per, row.line_total) for per in (100, 1000))


def reconcile_with_net_total(rows: list[TableRow], net_total: Optional[float]) -> None:
    """Use the printed net total to decide which OCR reading of each inconsistent row is right.

    For every row where quantity x price != total, the candidates are: the total as read, other
    readings of it ("7 151,00" -> 151,00) and quantity x price. The combination with the fewest
    changes whose sum equals the net total wins - but only if it is unique.
    """
    if net_total is None or len(rows) < 2 or any(row.line_total is None for row in rows):
        return
    suspects = [row for row in rows if not _consistent(row)][:8]
    if suspects and not _close(sum(row.line_total for row in rows), net_total):
        fixed_sum = sum(row.line_total for row in rows if row not in suspects)
        options = []
        for row in suspects:
            values = [row.line_total, *row.total_alternatives]
            if _expected(row) is not None:
                values.append(_expected(row))
            options.append(list(dict.fromkeys(values)))
        solutions = []
        for combo in itertools.product(*options):
            if _close(fixed_sum + sum(combo), net_total):
                changes = sum(1 for row, value in zip(suspects, combo) if value != row.line_total)
                solutions.append((changes, combo))
        if solutions:
            solutions.sort(key=lambda item: item[0])
            if len(solutions) == 1 or solutions[0][0] < solutions[1][0]:
                for row, value in zip(suspects, solutions[0][1]):
                    if value != row.line_total:
                        row.notes = [note for note in row.notes if "!=" not in note]
                        row.notes.append(f"line total read as {row.line_total:.2f}, corrected to {value:.2f} (matches net total)")
                        row.line_total = value
    if not _close(sum(row.line_total for row in rows), net_total):
        return
    # The totals add up to the net amount, so they are right: a remaining mismatch is a misread price
    for row in rows:
        factor = 1 - (row.discount_percent or 0) / 100
        if not _consistent(row) and row.quantity and factor:
            corrected = round(row.line_total / row.quantity / factor, 4)
            row.notes = [note for note in row.notes if "!=" not in note]
            row.notes.append(f"unit price read as {row.unit_price:g}, corrected to {corrected:g} (line totals match net total)")
            row.unit_price = corrected


def rows_look_reliable(rows: list[TableRow], net_total: Optional[float] = None) -> bool:
    """A parsed table is trusted if most rows are arithmetically consistent.

    A single row is trusted only if it is consistent and its total equals the printed net total.
    """
    if not rows or not all(normalize_unit(row.unit) for row in rows):
        return False
    if len(rows) == 1:
        row = rows[0]
        return bool(net_total is not None and row.line_total is not None and _close(row.line_total, net_total)
                    and _expected(row) is not None and _close(_expected(row), row.line_total))
    consistent = sum(
        1 for row in rows
        if row.unit_price is not None and row.line_total is not None and row.quantity
        and _close(row.quantity * row.unit_price * (1 - (row.discount_percent or 0) / 100), row.line_total)
    )
    return consistent / len(rows) >= 0.6 and all(normalize_unit(row.unit) for row in rows)
