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
    quantity_like = re.fullmatch(r"[\d.,]+\s*(\S+)", article) and normalize_unit(article.split()[-1])
    if article and (quantity_like or _squash(article) not in _squash(text)):
        check.missing.append("article_number")
    return check


def _header_lines(items: list[RawLineItem], lines: dict[int, str]) -> list[int]:
    """The table header: the lines right above the first position the model cited."""
    firsts = [item.first_line for item in items if item.first_line in lines]
    if not firsts:
        return []
    start = min(firsts)
    return [n for n in range(max(1, start - 3), start) if n in lines]


def _excerpt(numbers: list[int], lines: dict[int, str]) -> str:
    return "\n".join(f"L{n}: {lines[n]}" for n in sorted(set(numbers)) if n in lines)


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
        excerpt = _excerpt(header + list(range(span[0] - 1, span[1] + 3)), lines)
        rereads += 1
        try:
            data = await llm.extract(excerpt, "row", instruction=f"The position to extract starts at line L{span[0]}.")
        except llm.LlmError as exc:
            log.warning("Re-reading position at L%s failed: %s", span[0], exc)
            continue
        for key, value in data.pop("_stats", {}).items():
            stats[key] = stats.get(key, 0) + value
        stats["calls"] = stats.get("calls", 0) + 1
        data.pop("_truncated", None)
        try:
            candidate = RawLineItem.model_validate(data)
        except Exception:  # noqa: BLE001 - a broken answer just keeps the first reading
            continue
        if candidate.first_line not in lines:
            candidate.first_line, candidate.last_line = span
        new_check = check_row(candidate, lines, lang)
        if new_check.score > check.score:
            changed = [name for name in RawLineItem.model_fields
                       if name not in ("first_line", "last_line") and getattr(candidate, name) != getattr(item, name)]
            items[index] = candidate
            label = candidate.position or item.position or index + 1
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
