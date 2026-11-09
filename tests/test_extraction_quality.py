import pytest

from app import labels
from app.extractor import normalize_order
from app.llm import _salvage_truncated
from app.schemas import RawLineItem, RawOrder, RawParty


@pytest.mark.parametrize("text,lang,expected", [
    ("Muster GmbH\nBestellung\n80331 München\nBestell-Nr.: BE-2026-0815    Datum: 03.11.2026\n"
     "Kunden-Nr. K-4711\nLiefertermin: KW 48/2026", "de",
     ("BE-2026-0815", "03.11.2026", "KW 48/2026", "K-4711")),
    ("PURCHASE ORDER\nPO Number:\n4500012345\nDelivery date: 30/11/2026\nOrder date: 12 November 2026\n"
     "Customer No: 77-12", "en", ("4500012345", "12 November 2026", "30/11/2026", "77-12")),
    ("Zamówienie nr ZAM/2026/11/07\nData zamówienia: 05.11.2026\nTermin dostawy: 20.11.2026\nNr klienta: 3321", "pl",
     ("ZAM/2026/11/07", "05.11.2026", "20.11.2026", "3321")),
    ("ORDINE N. 2026/1187\nData di consegna: 25/11/2026\nData: 04/11/2026\nCodice cliente: C0042", "it",
     ("2026/1187", "04/11/2026", "25/11/2026", "C0042")),
])
def test_label_lookup(text, lang, expected):
    found = (labels.find_order_number(text), labels.find_order_date(text, lang),
             labels.find_delivery_date(text, lang), labels.find_customer_number(text))
    assert found == expected


def test_bare_title_does_not_take_postal_code():
    assert labels.find_order_number("Bestellung\n80331 München") is None


def test_label_fallback_fills_missing_header_and_fixes_customer_number_mixup():
    text = "Beispiel AG\nBestell-Nr.: PO-77\nKunden-Nr.: K-4711\nBestelldatum: 03.11.2026\nLiefertermin: 20.11.2026"
    raw = RawOrder(order_number="K-4711", buyer=RawParty(company_name="Beispiel AG"),
                   line_items=[RawLineItem(position="1", description="Beispiel", quantity="1")])
    order = normalize_order(raw, "de", text)
    assert order.document.order_number == "PO-77"
    assert order.document.customer_number == "K-4711"
    assert order.document.order_date == "2026-11-03"
    assert order.document.requested_delivery_date == "2026-11-20"


def test_summary_rows_are_not_line_items():
    text = "A1 Schraube M8 2 Stk 10,00 Versandkosten 5,00 Summe netto 25,00"
    raw = RawOrder(line_items=[
        RawLineItem(position="1", article_number="A1", description="Schraube M8", quantity="2", unit_price="10,00"),
        RawLineItem(description="Versandkosten", line_total="5,00"),
        RawLineItem(description="Summe netto", line_total="25,00"),
    ])
    order = normalize_order(raw, "de", text)
    assert [item.description for item in order.line_items] == ["Schraube M8"]


def test_invented_description_is_flagged():
    raw = RawOrder(line_items=[RawLineItem(position="1", article_number="A1", description="Hydraulikpumpe Typ X", quantity="1")])
    order = normalize_order(raw, "de", "Pos 1 A1 Schraube M8 1 Stk")
    assert any("description" in w for w in order.validation.warnings)


def test_truncated_items_keep_complete_rows():
    content = '{"line_items": [{"position": "1", "description": "A"}, {"position": "2", "description": "B"}, {"position": "3", "desc'
    assert [row["position"] for row in _salvage_truncated(content)["line_items"]] == ["1", "2"]


def test_dates_after_numbers_are_found():
    text = "C30 / 4503191146 / 03.11.2026   1/2\nLiefertermin: Tag 15.12.2026"
    assert labels.dates_in_text(text, "de") == {"2026-11-03", "2026-12-15"}
