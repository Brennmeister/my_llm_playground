import pymupdf as fitz

from app.pdf_extract import extract_text
from app.review_store import create_review, get_review, match_sources, update_review
from app.schemas import ErpOrder, LineItem


def _sample_pdf():
    document = fitz.open()
    page = document.new_page(width=400, height=300)
    page.insert_text((40, 60), "PO-42")
    page.insert_text((40, 100), "Quantity 2.00")
    content = document.tobytes()
    document.close()
    return content


def test_native_pdf_words_and_source_matches():
    pdf = _sample_pdf()
    extracted = extract_text(pdf, "never")
    order = ErpOrder(
        document={"order_number": "PO-42", "order_date": "2026-11-08"},
        buyer={"name": "Acme GmbH"},
        line_items=[LineItem(position=1, description="Widget", quantity=2.0)],
    )

    sources = match_sources(order, extracted)

    assert extracted.page_sizes == [(400.0, 300.0)]
    assert sources["document.order_number"]["confidence"] == 1.0
    assert sources["document.order_number"]["locations"][0]["page"] == 1
    assert sources["line_items.0.quantity"]["locations"][0]["x0"] > 0
    assert not sources["buyer.name"]["locations"]


def test_review_persists_pdf_and_audits_corrections(tmp_path):
    pdf = _sample_pdf()
    extracted = extract_text(pdf, "never")
    order = ErpOrder(
        document={"order_number": "PO-42", "order_date": "2026-11-08"},
        buyer={"name": "Acme GmbH"},
        line_items=[LineItem(position=1, description="Widget", quantity=2.0)],
    )
    sources = match_sources(order, extracted)
    db_path = str(tmp_path / "reviews.sqlite3")

    review_id = create_review(db_path, pdf, order, extracted, sources)
    saved = get_review(db_path, review_id)
    assert saved["status"] == "pending"
    assert saved["pdf"] == pdf

    corrected = order.model_copy(deep=True)
    corrected.document.order_number = "PO-43"
    updated = update_review(db_path, review_id, corrected)
    assert updated["status"] == "reviewed"
    assert updated["order"]["document"]["order_number"] == "PO-43"
    assert updated["history"][0]["path"] == "document.order_number"
    assert updated["sources"]["document.order_number"]["extracted_value"] == "PO-42"


def test_corrected_values_are_matched_again(tmp_path):
    pdf = _sample_pdf()
    extracted = extract_text(pdf, "never")
    order = ErpOrder(
        document={"order_number": "PO-99", "order_date": "2026-11-08"},
        buyer={"name": "Acme GmbH"},
        line_items=[LineItem(position=1, description="Widget", quantity=2.0)],
    )
    db_path = str(tmp_path / "reviews.sqlite3")
    review_id = create_review(db_path, pdf, order, extracted, match_sources(order, extracted))
    assert not get_review(db_path, review_id)["sources"]["document.order_number"]["locations"]

    corrected = order.model_copy(deep=True)
    corrected.document.order_number = "PO-42"
    source = update_review(db_path, review_id, corrected)["sources"]["document.order_number"]
    assert source["value"] == "PO-42" and source["extracted_value"] == "PO-99"
    assert source["locations"] and source["confidence"] == 1.0


def test_approved_review_cannot_be_approved_twice(tmp_path):
    import pytest
    from app.review_store import approve_review

    pdf = _sample_pdf()
    extracted = extract_text(pdf, "never")
    order = ErpOrder(document={"order_number": "PO-42"})
    db_path = str(tmp_path / "reviews.sqlite3")
    review_id = create_review(db_path, pdf, order, extracted, match_sources(order, extracted))
    approve_review(db_path, review_id)
    with pytest.raises(ValueError):
        approve_review(db_path, review_id)


def test_repeated_values_are_resolved_by_their_row():
    document = fitz.open()
    page = document.new_page(width=500, height=300)
    page.insert_text((40, 60), "Hauptstr. 1, Stuttgart")
    page.insert_text((40, 120), "10  A-100  Bolt   1.000  Stk")
    page.insert_text((40, 140), "20  B-200  Nut    1.000  Stk")
    pdf = document.tobytes()
    document.close()
    extracted = extract_text(pdf, "never")
    order = ErpOrder(document={"language": "de"}, line_items=[
        LineItem(position=10, supplier_article_number="A-100", quantity=1000.0, unit_raw="Stk", unit_code="H87"),
        LineItem(position=20, supplier_article_number="B-200", quantity=1000.0, unit_raw="Stk", unit_code="H87"),
    ])
    sources = match_sources(order, extracted)
    rows = [sources[f"line_items.{i}.quantity"]["locations"][0]["y0"] for i in (0, 1)]
    assert rows[0] < rows[1]
    assert sources["line_items.1.unit_raw"]["locations"][0]["y0"] == sources["line_items.1.quantity"]["locations"][0]["y0"]
    assert sources["line_items.0.unit_code"]["derived"] and not sources["line_items.0.unit_code"]["locations"]


def test_wrapped_and_misread_text_gets_a_location():
    document = fitz.open()
    page = document.new_page(width=500, height=300)
    page.insert_text((40, 60), "TEIL_Rack_Winkel_Batterie_H")
    page.insert_text((40, 72), "alterung_vorne_1.00_00")
    page.insert_text((40, 120), "SUP_Diverse_Schutzbtigel_1.00_0B")
    pdf = document.tobytes()
    document.close()
    extracted = extract_text(pdf, "never")
    order = ErpOrder(line_items=[
        LineItem(position=1, description="TEIL_Rack_Winkel_Batterie_Halterung_vorne_1.00_00"),
        LineItem(position=2, description="SUP_Diverse_Schutzbügel_1.00_0B"),
    ])
    sources = match_sources(order, extracted)
    wrapped = sources["line_items.0.description"]
    assert len(wrapped["locations"]) == 2 and wrapped["match"] == "exact"
    misread = sources["line_items.1.description"]
    assert misread["locations"] and misread["approximate"] and 0.72 <= misread["confidence"] < 1


def test_confirmed_values_are_no_longer_open_issues(tmp_path):
    from app.review_store import unmatched_paths

    pdf = _sample_pdf()
    extracted = extract_text(pdf, "never")
    order = ErpOrder(document={"order_number": "PO-42"}, buyer={"name": "Acme GmbH"})
    db_path = str(tmp_path / "reviews.sqlite3")
    review_id = create_review(db_path, pdf, order, extracted, match_sources(order, extracted))
    assert unmatched_paths(get_review(db_path, review_id)["sources"]) == ["buyer.name"]
    updated = update_review(db_path, review_id, order, confirmed=["buyer.name"])
    assert unmatched_paths(updated["sources"]) == []
    assert updated["history"][-1] == {"at": updated["history"][-1]["at"], "path": "buyer.name", "confirmed": "Acme GmbH"}
    # confirmation is dropped when the next save does not confirm it again
    assert unmatched_paths(update_review(db_path, review_id, order)["sources"]) == ["buyer.name"]
