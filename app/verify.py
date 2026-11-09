"""Layout-agnostic verification of the LLM's line items, and targeted re-reading.

The LLM extracts the positions (it copes with any layout) and names the document lines each
position comes from. This module checks that the values really are in those lines and that
quantity x price = total. Positions that fail get a second, focused LLM call on just their lines
(plus the table header) - a small model is far more reliable on a short excerpt.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from . import llm
from .normalize import normalize_unit, parse_number
from .schemas import LineItem, RawLineItem
from .table_rows import _MONEY, _STOP_LINE, TableRow, _join, _parse_anchor, _second_line_article

log = logging.getLogger(__name__)

MAX_SPAN = 8          # lines per position the check accepts
MAX_REREADS = 20      # bounds the extra LLM time on long orders
_NUMBER = re.compile(r"(?<![\w.,])-?\d{1,3}(?:[.,' ]\d{3})*(?:[,.:;]\d{1,4})?(?![\d])|(?<![\w.,])-?\d+(?:[.,]\d+)?(?![\w])")


def _squash(text: str) -> str:
    return re.sub(r"[\W_]+", "", text.lower())


@dataclass
class RowCheck:
    span: Optional[tuple[int, int]]
    missing: list[str] = field(default_factory=list)   # values not found in the cited lines
    arithmetic: Optional[bool] = None                  # quantity x price = total?

    @property
    def ok(self) -> bool:
        return self.span is not None and not self.missing and self.arithmetic is not False

    @property
    def score(self) -> int:
        return (0 if self.span is None else 1) - 2 * len(self.missing) + {True: 2, False: -2, None: 0}[self.arithmetic]


def item_span(item: RawLineItem, lines: dict[int, str]) -> Optional[tuple[int, int]]:
    first = item.first_line
    if first is None or first not in lines:
        return None
    last = item.last_line if item.last_line is not None and item.last_line >= first else first
    return first, min(last, first + MAX_SPAN - 1, max(lines))


def _numbers(text: str, lang: Optional[str]) -> list[float]:
    values = []
    for token in _NUMBER.findall(text):
        value = parse_number(re.sub(r"(?<=\d)[:;](?=\d{2}$)", ",", token), lang)
        if value is not None:
            values.append(value)
    return values


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= max(0.02, 0.005 * abs(b))


def arithmetic_ok(quantity: Optional[float], price: Optional[float], total: Optional[float],
                  discount: Optional[float] = None) -> Optional[bool]:
    if quantity is None or price is None or total is None or not quantity:
        return None
    expected = quantity * price * (1 - (discount or 0) / 100)
    return any(_close(expected / per, total) for per in (1, 100, 1000))  # price per 1/100/1000 units


def check_row(item: RawLineItem, lines: dict[int, str], lang: Optional[str]) -> RowCheck:
    span = item_span(item, lines)
    quantity = parse_number(item.quantity, lang)
    price = parse_number(item.unit_price, lang)
    total = parse_number(item.line_total, lang)
    check = RowCheck(span, arithmetic=arithmetic_ok(quantity, price, total, parse_number(item.discount, lang)))
    if span is None:
        return check
    text = "\n".join(lines[n] for n in range(span[0], span[1] + 1) if n in lines)
    numbers = _numbers(text, lang)
    for name, value in (("quantity", quantity), ("unit_price", price), ("line_total", total)):
        if value is not None and not any(abs(value - n) < 0.005 for n in numbers):
            check.missing.append(name)
    article = (item.article_number or "").strip()
    if article and (_suspicious_article(article, item.description) or _squash(article) not in _squash(text)):
        check.missing.append("article_number")
    if item.description and not _grounded_words(item.description, text):
        check.missing.append("description")
    return check


def _header_lines(items: list[RawLineItem], lines: dict[int, str]) -> list[int]:
    """The table header: the lines right above the first position the model cited."""
    firsts = [item.first_line for item in items if item.first_line in lines]
    if not firsts:
        return []
    start = min(firsts)
    return [n for n in range(max(1, start - 3), start) if n in lines]


def anchor_segments(lines: dict[int, str], lang: Optional[str]) -> list[tuple[int, int, Optional[str], TableRow]]:
    """Candidate position lines: "<quantity> <unit> ... <amount>" - the one pattern almost every order
    table shares, whatever the column order. Only WHERE positions are is decided here; what the
    values mean is left to the LLM. Each segment runs up to the next candidate (or a total/footer line)."""
    anchors = []
    for number, text in lines.items():
        row = _parse_anchor(text, lang)
        if row is not None:
            anchors.append((number, row))
    segments = []
    for index, (first, row) in enumerate(anchors):
        end = anchors[index + 1][0] - 1 if index + 1 < len(anchors) else max(lines)
        last = first
        for number in range(first + 1, min(end, first + MAX_SPAN - 1) + 1):
            text = lines.get(number, "")
            if _STOP_LINE.match(text) or _MONEY.search(text):
                break
            last = number
        segments.append((first, last, str(row.position) if row.position is not None else None, row))
    return segments


async def _read_row(excerpt: str, first: int, stats: dict[str, float]) -> Optional[RawLineItem]:
    try:
        data = await llm.extract(excerpt, "row", instruction=f"The position to extract starts at line L{first}.")
    except llm.LlmError as exc:
        log.warning("Reading position at L%s failed: %s", first, exc)
        return None
    for key, value in data.pop("_stats", {}).items():
        stats[key] = stats.get(key, 0) + value
    stats["calls"] = stats.get("calls", 0) + 1
    data.pop("_truncated", None)
    try:
        return RawLineItem.model_validate(data)
    except Exception:  # noqa: BLE001 - a broken answer is treated like no answer
        return None


async def extract_positions(segments: list[tuple[int, int, Optional[str], TableRow]], lines: dict[int, str],
                            stats: dict[str, float]) -> list[RawLineItem]:
    """One focused LLM call per position: table header + only this position's lines."""
    start = segments[0][0]
    header = [n for n in range(max(1, start - 3), start) if n in lines]
    items = []
    for first, last, position, _row in segments:
        item = await _read_row(_excerpt(list(range(first, last + 1)), lines, header), first, stats)
        if item is None:
            item = RawLineItem(position=position)
        # keep the model's range if it lies within the segment, else use the segment
        if not (item.first_line and first <= item.first_line <= last):
            item.first_line = first
        if not (item.last_line and item.first_line <= item.last_line <= last):
            item.last_line = last
        items.append(item)
    return items


def fix_position_number(item: RawLineItem, printed: Optional[str], line_text: str) -> None:
    """Small models sometimes return the line number ("16", "L16") as position number: the position
    must be a token of the position line, otherwise the printed one is used."""
    value = (item.position or "").strip()
    tokens = {token.lstrip("0") or "0" for token in re.findall(r"\w+", line_text)}
    if printed and (not value or value.upper().startswith("L") or (value.lstrip("0") or "0") not in tokens):
        item.position = printed


def _grounded_words(text: Optional[str], source: str) -> bool:
    words = re.findall(r"[^\W_]{3,}", text or "")
    squashed = _squash(source)
    return not words or sum(_squash(word) in squashed for word in words) / len(words) >= 0.6


def _suspicious_article(article: Optional[str], description: Optional[str]) -> bool:
    article = (article or "").strip()
    if len(re.sub(r"\W", "", article)) < 3:
        return True
    if description and _squash(article) == _squash(description):
        return True  # the description copied into the article field
    if re.fullmatch(r"[\d.,]+\s*(\S+)", article) and normalize_unit(article.split()[-1]):
        return True  # "150 Stück"
    return " " in article and not re.search(r"\d", article)  # words without any digit


def harmonize_with_line(item: RawLineItem, row: TableRow, segment: list[str], lang: Optional[str],
                        warnings: list[str]) -> None:
    """Verify the model's values against the position's own lines.

    A model value is kept when it occurs in the position lines and quantity x price = total holds.
    Otherwise the reading of the position line ("<qty> <unit> <amounts>", already cross-checked with
    the net total) replaces the values that differ. Article numbers that are empty, a quantity or a
    copy of the description, and descriptions that are not in the position lines, are taken from the
    lines below the values. Every change is reported as a warning.
    """
    text = "\n".join(segment)
    numbers = _numbers(text, lang)
    discount = parse_number(item.discount, lang)
    model = {"quantity": parse_number(item.quantity, lang), "unit_price": parse_number(item.unit_price, lang),
             "line_total": parse_number(item.line_total, lang)}
    line = {"quantity": row.quantity, "unit_price": row.unit_price, "line_total": row.line_total}
    grounded = all(v is None or any(abs(v - n) < 0.005 for n in numbers) or (line[k] is not None and abs(v - line[k]) < 0.005)
                   for k, v in model.items())
    changed = []
    if not (grounded and arithmetic_ok(model["quantity"], model["unit_price"], model["line_total"], discount)):
        if arithmetic_ok(line["quantity"], line["unit_price"], line["line_total"], row.discount_percent) is not False:
            target = line
        else:  # line reading inconsistent too: replace only what the model did not find in the lines
            target = {k: (line[k] if model[k] is None or not any(abs(model[k] - n) < 0.005 for n in numbers) else model[k])
                      for k in model}
        for name, value in target.items():
            if value is not None and (model[name] is None or abs(model[name] - value) > 0.0001):
                changed.append(f"{name} {getattr(item, name) or '-'} -> {value:g}")
                setattr(item, name, f"{value:g}".replace(".", "," if lang in ("de", "pl", "it") else "."))
    # "1545" for 15,45: per-100/1000 price or a lost decimal separator - the ERP needs the per-unit price
    qty, price, total = (parse_number(getattr(item, k), lang) for k in ("quantity", "unit_price", "line_total"))
    if qty and price and total and not _close(qty * price * (1 - (discount or 0) / 100), total):
        for per in (100, 1000):
            if _close(qty * price / per * (1 - (discount or 0) / 100), total):
                changed.append(f"unit_price {item.unit_price} -> {price / per:g} (per unit)")
                item.unit_price = f"{price / per:g}".replace(".", "," if lang in ("de", "pl", "it") else ".")
                break
    if not item.unit and row.unit:
        item.unit = row.unit
    below = segment[1:]
    if _suspicious_article(item.article_number, item.description):
        candidate = row.article_number
        if not candidate and below:
            second = _second_line_article(below[0])  # article on the line below the values (SAP style)
            candidate = second[0] if second else None
        if candidate and candidate != item.article_number:
            changed.append(f"article_number {item.article_number or '-'} -> {candidate}")
            item.article_number = candidate
    if item.description and not _grounded_words(item.description, text):
        rest = (_second_line_article(below[0]) or (None, None))[1] if below and row.article_number is None else None
        replacement = row.description or rest or _join([l for l in below if not _STOP_LINE.match(l)][:3])
        changed.append(f"description '{item.description}' not in the position lines")
        item.description = replacement
    if changed:
        warnings.append(f"Position {item.position}: checked against its lines L{item.first_line}-L{item.last_line} "
                        f"({'; '.join(changed)}).")


def _excerpt(numbers: list[int], lines: dict[int, str], header: Optional[list[int]] = None) -> str:
    position_lines = "\n".join(f"L{n}: {lines[n]}" for n in sorted(set(numbers)) if n in lines and n not in (header or []))
    if not header:
        return position_lines
    header_lines = "\n".join(f"L{n}: {lines[n]}" for n in header if n in lines)
    return (f"TABLE HEADER (column names only - never take values from here):\n{header_lines}\n\n"
            f"POSITION LINES:\n{position_lines}")


async def reread_failed_items(items: list[RawLineItem], lines: dict[int, str], lang: Optional[str],
                              warnings: list[str], stats: dict[str, float]) -> int:
    """Re-extract positions whose values are not in their cited lines or don't add up."""
    header = _header_lines(items, lines)
    rereads = 0
    for index, item in enumerate(items):
        check = check_row(item, lines, lang)
        if check.ok or rereads >= MAX_REREADS:
            continue
        span = check.span or _locate(item, lines)
        if span is None:
            continue
        excerpt = _excerpt(list(range(span[0] - 1, span[1] + 3)), lines, header)
        rereads += 1
        candidate = await _read_row(excerpt, span[0], stats)
        if candidate is None:
            continue
        if candidate.first_line not in lines:
            candidate.first_line, candidate.last_line = span
        new_check = check_row(candidate, lines, lang)
        if new_check.score > check.score:
            changed = [name for name in RawLineItem.model_fields
                       if name not in ("first_line", "last_line") and getattr(candidate, name) != getattr(item, name)]
            items[index] = candidate
            label = item.position or index + 1
            warnings.append(f"Position {label}: re-read from lines L{new_check.span[0]}-L{new_check.span[1]}"
                            + (f" ({', '.join(changed)} corrected)" if changed else "") + ".")
        log.info("Re-read position at L%s: score %s -> %s", span[0], check.score, new_check.score)
    return rereads


def _locate(item: RawLineItem, lines: dict[int, str]) -> Optional[tuple[int, int]]:
    """The model gave no usable line numbers: find the line with the article number."""
    article = _squash(item.article_number or "")
    if len(article) < 3:
        return None
    for number, text in lines.items():
        if article in _squash(text):
            return number, number
    return None


def reconcile_items_with_net_total(items: list[LineItem], net_total: Optional[float], warnings: list[str]) -> None:
    """Numeric cross-check (no layout assumptions): if exactly one way of fixing inconsistent
    positions (total := quantity x price) makes the positions add up to the printed net total,
    apply it; if the totals add up, a remaining mismatch is a misread unit price."""
    totals = [item.line_total for item in items]
    if net_total is None or not items or any(total is None for total in totals):
        return

    def inconsistent(item: LineItem) -> bool:
        return arithmetic_ok(item.quantity, item.unit_price, item.line_total, item.discount_percent) is False

    if not _close(sum(totals), net_total):
        suspects = [item for item in items if inconsistent(item)]
        fixes = []
        for item in suspects:
            expected = round(item.quantity * item.unit_price * (1 - (item.discount_percent or 0) / 100), 2)
            if _close(sum(totals) - item.line_total + expected, net_total):
                fixes.append((item, expected))
        if len(fixes) == 1:
            item, expected = fixes[0]
            warnings.append(f"Position {item.position}: line total {item.line_total:.2f} corrected to {expected:.2f} "
                            "(quantity x price, matches the net total).")
            item.line_total = expected
    if not _close(sum(item.line_total for item in items), net_total):
        return
    for item in items:
        if inconsistent(item):
            factor = 1 - (item.discount_percent or 0) / 100
            corrected = round(item.line_total / item.quantity / factor, 4)
            warnings.append(f"Position {item.position}: unit price {item.unit_price:g} corrected to {corrected:g} "
                            "(line totals match the net total).")
            item.unit_price = corrected
