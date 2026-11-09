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


def test_position_lines_are_found_whatever_the_column_order():
    from app.verify import anchor_segments

    lines = {1: "Pos Material Menge Preis Wert", 2: "00010 150 Stück 9,85 1.477,50", 3: "2145723 HALTEWINKEL",
             4: "Referenz: X_1", 5: "20 4711 Schraube M8 40 Stk 0,10 4,00", 6: "Summe netto 1.481,50"}
    assert [(first, last, position) for first, last, position, _ in anchor_segments(lines, "de")] == [
        (2, 3, "10"), (5, 5, "20")]


def test_model_values_are_kept_when_they_are_in_the_lines_and_add_up():
    from app.table_rows import _parse_anchor
    from app.verify import fix_position_number, harmonize_with_line

    segment = ["50 1128035 160 Stk 14,50 / 2.320,00", "TEIL_Halterung_Logger"]
    row = _parse_anchor(segment[0], "de")
    good = RawLineItem(first_line=32, last_line=33, position="50", article_number="1128035", description="TEIL_Halterung_Logger",
                       quantity="160", unit="Stk", unit_price="14,50", line_total="2.320,00")
    warnings = []
    harmonize_with_line(good, row, segment, "de", warnings)
    assert warnings == [] and good.quantity == "160"
    # the model took the position number as quantity and invented a total, copied the line number as position
    bad = good.model_copy(update={"position": "32", "quantity": "50", "line_total": "725,00", "article_number": "TEIL_Halterung_Logger"})
    fix_position_number(bad, "50", segment[0])
    harmonize_with_line(bad, row, segment, "de", warnings)
    assert (bad.position, bad.quantity, bad.line_total, bad.article_number) == ("50", "160", "2320", "1128035")
    assert warnings and "quantity 50 -> 160" in warnings[0]


def test_decimal_position_numbers_dimensions_and_zero_prices():
    """Order without prices: "1,0" positions, dimension lines ("60 x 10 x 175mm / 100 Stck.")."""
    from app.table_rows import _parse_anchor
    from app.verify import anchor_segments, harmonize_with_line

    lines = {1: "Pos. Artikel / Bezeichnung   Termin   Menge/ME Preis EUR Rabatt Gesamt EUR",
             2: "1,0 06.1950.010_0   15.11.2026   10 Stk   0,00 0,00%", 3: "Gabel Boden S355", 4: "60 x 10 x 175mm",
             5: "2,0 06.1951.010_0   15.11.2026   10 Stk   0,00 0,00%", 6: "60 x 10 x 175mm / 100 Stck.",
             7: "Rechnungen bitte an: invoice@example.com", 8: "Warenwert   0,00 EUR"}
    segments = anchor_segments(lines, "de")
    assert [(first, last, position) for first, last, position, _ in segments] == [(2, 4, "1"), (5, 6, "2")]
    assert segments[0][3].article_number == "06.1950.010_0" and segments[0][3].quantity == 10

    segment = [lines[n] for n in range(2, 5)]
    item = RawLineItem(first_line=2, last_line=4, position="1", article_number="06.1950.010_0",
                       description="Gabel Boden S355", quantity="1,0", unit="Stk", unit_price="0,00", line_total="0,00")
    warnings = []
    harmonize_with_line(item, _parse_anchor(segment[0], "de"), segment, "de", warnings)
    assert item.quantity == "10" and "number before the unit" in warnings[0]
