import pytest

from app.extractor import merge_raw, normalize_order
from app.normalize import normalize_country, normalize_currency, normalize_unit, parse_date, parse_number
from app.pdf_extract import chunk_pages
from app.schemas import RawLineItem, RawOrder, RawParty, llm_json_schema


@pytest.mark.parametrize("raw,lang,expected", [
    ("1.234,56", "de", 1234.56),
    ("1,234.56", "en", 1234.56),
    ("1 234,56 zł", "pl", 1234.56),
    ("1.000", "de", 1000.0),
    ("1.000", "it", 1000.0),
    ("1.5", "en", 1.5),
    ("12,5", "de", 12.5),
    ("EUR 5,000.00", "en", 5000.0),
    ("1,234", "en", 1234.0),
    ("0,12 €", "de", 0.12),
    ("19 %", "de", 19.0),
    ("1'250.00", "de", 1250.0),
    ("-3,50", "de", -3.5),
    (None, "de", None),
    ("n/a", "en", None),
])
def test_parse_number(raw, lang, expected):
    assert parse_number(raw, lang) == expected


@pytest.mark.parametrize("raw,lang,expected", [
    ("03.11.2026", "de", "2026-11-03"),
    ("2026-11-03", None, "2026-11-03"),
    ("5 listopada 2026", "pl", "2026-11-05"),
    ("4 novembre 2026", "it", "2026-11-04"),
    ("November 2, 2026", "en", "2026-11-02"),
    ("3. November 2026", "de", "2026-11-03"),
    ("30/11/2026", "en", "2026-11-30"),
    ("11/30/2026", "en", "2026-11-30"),
    ("KW 48/2026", "de", "2026-11-23"),
    ("15/12/26", "it", "2026-12-15"),
    ("bald", "de", None),
])
def test_parse_date(raw, lang, expected):
    assert parse_date(raw, lang) == expected


def test_units_currency_country():
    assert normalize_unit("Stk") == "H87"
    assert normalize_unit("szt.") == "H87"
    assert normalize_unit("pz") == "H87"
    assert normalize_unit("kg") == "KGM"
    assert normalize_unit("m²") == "MTK"
    assert normalize_currency("zł") == "PLN"
    assert normalize_currency(None, "Total EUR 5,355.00") == "EUR"
    assert normalize_country("Deutschland") == "DE"
    assert normalize_country("Włochy") == "IT"
    assert normalize_country(None, vat_id="ATU12345678") == "AT"
    assert normalize_country(None, postal_code="60-123") == "PL"


def test_schema_is_strict():
    h = llm_json_schema("header")
    assert "$defs" not in h and "line_items" not in h["properties"]
    assert set(h["required"]) == set(h["properties"])
    assert h["properties"]["buyer"]["additionalProperties"] is False
    s = llm_json_schema("items")
    assert s["properties"]["line_items"]["items"]["additionalProperties"] is False


def test_sanitize_removes_hallucinations():
    text = "Beispiel AG Hauptstr. 1 80331 München Deutschland Tel 089 123456 An: Lieferant GmbH Bestellnr. PO-1"
    raw = RawOrder(order_number="PO-9", buyer=RawParty(company_name="Beispiel AG", phone="0171 999999", email="Max Muster",
                                                       postal_code="80331 München", city="Deutschland"),
                   supplier=RawParty(company_name="Lieferant GmbH"),
                   delivery_address=RawParty(company_name="Lieferant GmbH"))
    order = normalize_order(raw, "de", text)
    assert order.document.order_number is None
    assert order.buyer.contact.phone is None and order.buyer.contact.email is None
    assert (order.buyer.postal_code, order.buyer.city, order.buyer.country_code) == ("80331", "München", "DE")
    assert order.ship_to.name == "Beispiel AG"


def test_label_swap():
    text = "Lieferant GmbH Weg 1\nAn / To:\nKunde AG Platz 2"
    raw = RawOrder(buyer=RawParty(company_name="Kunde AG"), supplier=RawParty(company_name="Lieferant GmbH"))
    order = normalize_order(raw, "de", text)
    assert order.buyer.name == "Lieferant GmbH" and order.supplier.name == "Kunde AG"


def test_merge_and_normalize():
    a = RawOrder(order_number="PO-1", order_date="03.11.2026", buyer=RawParty(company_name="Buyer GmbH", country="Deutschland"),
                 line_items=[RawLineItem(position="1", article_number="A1", description="Teil", quantity="2", unit="Stk", unit_price="10,00")])
    b = RawOrder(line_items=[RawLineItem(position="1", article_number="A1", description="Teil", quantity="2", unit="Stk", unit_price="10,00"),
                             RawLineItem(position="2", article_number="B2", description="Teil 2", quantity="1.000", unit="Stk", unit_price="0,05", line_total="50,00")],
                 net_total="70,00")
    order = normalize_order(merge_raw([a, b]), "de", "PO-1 Buyer GmbH A1 B2")
    assert order.document.order_number == "PO-1"
    assert order.document.order_date == "2026-11-03"
    assert order.buyer.country_code == "DE"
    assert [i.quantity for i in order.line_items] == [2.0, 1000.0]
    assert order.line_items[0].line_total == 20.0
    assert order.totals.line_items_sum == 70.0
    assert order.validation.is_complete
    assert not [w for w in order.validation.warnings if "differs" in w]


def test_chunk_pages():
    chunks = chunk_pages(["a" * 100, "b" * 100, "c" * 250], 220)
    assert all(len(c) <= 220 for c in chunks)
    assert "".join(chunks).count("a") >= 100


def test_repair_totals():
    from app.extractor import _repair_totals
    from app.schemas import Totals
    w = []
    t = Totals(net_amount=252.88, tax_amount=40.38, gross_amount=212.5, line_items_sum=212.5)
    _repair_totals(t, w)
    assert (t.net_amount, t.tax_amount, t.gross_amount) == (212.5, 40.38, 252.88)
    t = Totals(net_amount=1079.7, tax_amount=194.7, gross_amount=1079.7, line_items_sum=885.0)
    _repair_totals(t, w)
    assert (t.net_amount, t.tax_amount, t.gross_amount) == (885.0, 194.7, 1079.7)
    t = Totals(net_amount=7530.0, line_items_sum=7530.0)
    _repair_totals(t, w)
    assert t.gross_amount is None
