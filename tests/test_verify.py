import asyncio
from unittest import mock

from app import llm
from app.schemas import LineItem, RawLineItem
from app.verify import check_row, reconcile_items_with_net_total, reread_failed_items

LINES = {
    1: "Pos. Bestellmenge Einheit Preis pro Einheit Nettowert",
    2: "Material Bezeichnung",
    3: "00010 150 Stück 9,85 1.477,50",
    4: "2145723 HALTEWINKEL 1 M.SCHWEISSMUTTER",
    5: "00020 20 Stück 2,00 40,00",
    6: "2145799 SCHRAUBE M6",
}


def test_check_row_finds_values_in_cited_lines_only():
    good = RawLineItem(first_line=3, last_line=4, position="00010", article_number="2145723", quantity="150",
                       unit_price="9,85", line_total="1.477,50")
    assert check_row(good, LINES, "de").ok
    # values of the neighbouring position: the article is not in the cited lines
    shifted = good.model_copy(update={"article_number": "2145799"})
    assert check_row(shifted, LINES, "de").missing == ["article_number"]
    wrong_math = good.model_copy(update={"line_total": "1.000,00"})
    check = check_row(wrong_math, LINES, "de")
    assert check.arithmetic is False and "line_total" in check.missing


def test_failed_position_is_reread_from_its_lines():
    items = [
        RawLineItem(first_line=3, last_line=3, position="00010", article_number="150 Stück", quantity="150",
                    unit_price="9,85", line_total="1.477,50"),
        RawLineItem(first_line=5, last_line=6, position="00020", article_number="2145799", quantity="20",
                    unit_price="2,00", line_total="40,00"),
    ]
    calls = []

    async def fake_extract(text, task, info="", instruction=""):
        calls.append((task, text, instruction))
        return {"first_line": 3, "last_line": 4, "position": "00010", "article_number": "2145723",
                "description": "HALTEWINKEL 1 M.SCHWEISSMUTTER", "quantity": "150", "unit": "Stück",
                "unit_price": "9,85", "line_total": "1.477,50", "_stats": {"seconds": 1.0}}

    warnings, stats = [], {}
    with mock.patch.object(llm, "extract", fake_extract):
        rereads = asyncio.run(reread_failed_items(items, LINES, "de", warnings, stats))
    assert rereads == 1 and len(calls) == 1  # only the failing position is re-read
    task, excerpt, instruction = calls[0]
    assert task == "row" and "L1: Pos. Bestellmenge" in excerpt and "L4: 2145723" in excerpt and "L3" in instruction
    assert items[0].article_number == "2145723" and items[0].last_line == 4
    assert any("re-read" in w for w in warnings) and stats["calls"] == 1


def test_net_total_fixes_one_misread_total():
    items = [LineItem(position=10, quantity=40, unit_price=8.9, line_total=356.0),
             LineItem(position=20, quantity=50, unit_price=15.15, line_total=457.5)]   # OCR: 757,50
    warnings = []
    reconcile_items_with_net_total(items, 1113.5, warnings)
    assert items[1].line_total == 757.5 and warnings
