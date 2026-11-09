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
