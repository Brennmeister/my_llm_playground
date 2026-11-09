from app.extractor import normalize_order
from app.schemas import RawLineItem, RawOrder, RawParty
from app.table_rows import parse_rows, reconcile_with_net_total, rows_look_reliable

# Layout of a typical scanned ERP order: values on one line, description wrapped below it,
# OCR noise ("u", "/", "©"), a lost decimal comma ("1545") and a misread total.
SCANNED = """Bestellung: B-100
Pos. Artikel Menge ME Preis / PE Gesamt
10 5000001 40 Stk 8,90 356,00
TEIL_Rack_Winkel_Batterie_H
alterung_vorne_1.00_00 |
Referenz: X_1
20 5000002 50 Stk 10,50 u 525,00
TEIL_Rack_Halteblech_Switch
_1.00_00
Referenz: X_2
30 5000003 70 Stk 1545 1.081,50
TEIL_Rack_Aufnahmeblech
Referenz: X_3
40 5000004 50 Stk 15,15 / 151,90
TEIL_Rack_Halteblech_ADAS
Referenz: X_4
50 5000005 / Halterung Probe 20 Stk 1,55 7 151,00
TEIL_Rack_Halterung_Probe_1.00
Referenz: X_5
Total netto EUR 2.871,00
"""


def test_rows_keep_values_and_descriptions_in_their_row():
    rows = parse_rows([SCANNED], "de")
    assert [r.position for r in rows] == [10, 20, 30, 40, 50]
    assert [r.quantity for r in rows] == [40, 50, 70, 50, 20]
    assert rows[0].description == "TEIL_Rack_Winkel_Batterie_Halterung_vorne_1.00_00"
    assert rows[1].description == "TEIL_Rack_Halteblech_Switch_1.00_00"
    assert rows[2].unit_price == 15.45  # "1545" -> 15,45 because 70 x 15,45 = 1.081,50
    assert rows_look_reliable(rows)


def test_net_total_decides_between_ocr_readings():
    rows = parse_rows([SCANNED], "de")
    reconcile_with_net_total(rows, 2871.0)
    assert rows[3].line_total == 757.5   # "151,90" misread, qty x price fits the net total
    assert rows[4].line_total == 151.0   # "7 151,00": stray digit from the price column
    assert rows[4].unit_price == 7.55    # price "1,55" misread, derived from the confirmed total
    assert round(sum(r.line_total for r in rows), 2) == 2871.0


def test_digital_layout_with_inline_description():
    text = ("Pos Art.-Nr. Bezeichnung Menge Einheit Preis Gesamt\n"
            "10 SR-M8-40 Sechskantschraube M8x40 1.000 Stk 0,12 € 120,00 €\n"
            "30 BL-2MM Stahlblech 2 mm 12,5 kg 3,40 € 42,50 €\n"
            "Summe netto 162,50 €")
    rows = parse_rows([text], "de")
    assert [(r.article_number, r.quantity, r.unit, r.description) for r in rows] == [
        ("SR-M8-40", 1000.0, "Stk", "Sechskantschraube M8x40"), ("BL-2MM", 12.5, "kg", "Stahlblech 2 mm")]


def test_table_rows_are_only_a_fallback():
    base = dict(order_number="B-100", order_date="01.11.2026", net_total="2.871,00", buyer=RawParty(company_name="Kunde GmbH"))
    # the LLM's positions are the main source, even if the row parser would find others
    llm_items = [RawLineItem(position="10", article_number="5000001", quantity="40", unit="Stk", unit_price="8,90", line_total="356,00")]
    order = normalize_order(RawOrder(**base, line_items=llm_items), "de", SCANNED)
    assert order.meta.item_source == "llm" and len(order.line_items) == 1
    # only without any LLM positions the deterministic parser steps in
    order = normalize_order(RawOrder(**base), "de", SCANNED)
    assert order.meta.item_source == "table_rows" and len(order.line_items) == 5
    assert any("no line items" in w for w in order.validation.warnings)


# SAP-style print: two-line table header, article number on the line below the values,
# order number/date in a slash-separated column header.
TWO_LINE = """Muster AG - Musterweg 1 - 79100 Freiburg
Lieferant GmbH EinkGrp / Bestellnummer / Datum Seite
Industriestr. 4 C30 / 4500012345 / 03.11.2026 1/2
Liefertermin: Tag 15.12.2026
Pos. Bestellmenge Einheit Preis pro Einheit Nettowert
Material Bezeichnung
00010 150 Stück 9,85 v 1.477,50
2145723 HALTEWINKEL 1 M.SCHWEISSMUTTER
244 FG 2 RoHS-Konformität gefordert
Abruf zu Kontrakt 46081982 Position 00010
Gesamtnettowert ohne Mwst EUR 1.477,50
"""


def test_article_number_on_the_second_row_line():
    rows = parse_rows([TWO_LINE], "de")
    assert [(r.position, r.article_number, r.quantity, r.unit, r.unit_price, r.line_total, r.description) for r in rows] == [
        (10, "2145723", 150.0, "Stück", 9.85, 1477.5, "HALTEWINKEL 1 M.SCHWEISSMUTTER")]
    assert rows_look_reliable(rows, 1477.5)
    assert not rows_look_reliable(rows, 999.0)  # one row is only trusted if it matches the net total


def test_slash_header_and_invented_dates():
    # what a small model returned for this layout: label text as order number, an invented date,
    # and "150 Stück" as article number
    raw = RawOrder(order_number="Lieferant GmbH EinkGrp / Bestellnummer / 03.11.2026", order_date="14.11.2026",
                   net_total="1.477,50", buyer=RawParty(company_name="Muster AG"),
                   line_items=[RawLineItem(position="00010", article_number="150 Stück", description="HALTEWINKEL 1",
                                           quantity="150", unit="Stück")])
    order = normalize_order(raw, "de", TWO_LINE)
    assert order.document.order_number == "4500012345"
    assert order.document.order_date == "2026-11-03"
    assert order.document.requested_delivery_date == "2026-12-15"
    # "150 Stück" is a quantity, not an article number (the focused re-read pass fills the real one)
    assert order.line_items[0].supplier_article_number is None
    assert any("14.11.2026" in w for w in order.validation.warnings)
