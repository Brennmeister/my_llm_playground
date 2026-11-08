"""Generate sample purchase order PDFs (DE/EN/PL/IT) plus a scanned (image-only) variant
for testing:  .venv/bin/python scripts/make_samples.py  -> samples/*.pdf"""
from __future__ import annotations

from pathlib import Path

import pymupdf as fitz

OUT = Path(__file__).resolve().parent.parent / "samples"

SAMPLES = {
    "order_de": dict(
        title="Bestellung", supplier="Muster Lieferant GmbH<br>Industriestraße 5<br>70173 Stuttgart<br>Deutschland",
        buyer="Beispiel Maschinenbau AG<br>Hauptstraße 12<br>80331 München<br>Deutschland<br>USt-IdNr.: DE123456789",
        meta=[("Bestellnummer", "BE-2026-0815"), ("Bestelldatum", "03.11.2026"), ("Kundennummer", "K-4711"),
              ("Liefertermin", "KW 48/2026"), ("Ansprechpartner", "Anna Schmidt, a.schmidt@beispiel-ag.de")],
        cols=("Pos.", "Artikel-Nr.", "Bezeichnung", "Menge", "Einheit", "Einzelpreis", "Gesamt"),
        rows=[("10", "SR-M8-40", "Sechskantschraube M8x40 verzinkt", "1.000", "Stk", "0,12 €", "120,00 €"),
              ("20", "MU-M8", "Mutter M8 DIN 934", "1.000", "Stk", "0,05 €", "50,00 €"),
              ("30", "BL-2MM", "Stahlblech 2 mm", "12,5", "kg", "3,40 €", "42,50 €")],
        totals=[("Summe netto", "212,50 €"), ("MwSt. 19 %", "40,38 €"), ("Gesamtbetrag", "252,88 €")],
        footer="Zahlungsbedingungen: 30 Tage netto. Lieferbedingung: DAP München. Lieferadresse: Werk 2, Am Hafen 7, 80339 München",
    ),
    "order_en": dict(
        title="Purchase Order", supplier="Acme Components Ltd.<br>1 Station Road<br>Leeds LS1 4AB<br>United Kingdom",
        buyer="Northwind Trading GmbH<br>Ringstraße 3<br>1010 Wien<br>Austria<br>VAT: ATU12345678",
        meta=[("PO Number", "PO-77812"), ("Order date", "November 2, 2026"), ("Delivery date", "30/11/2026"),
              ("Customer No.", "C-9921"), ("Contact", "John Miller, +43 1 234567")],
        cols=("Item", "Part No.", "Description", "Qty", "Unit", "Unit Price", "Amount"),
        rows=[("1", "AC-5531", "Hydraulic valve 3/8\"", "4", "pcs", "EUR 1,250.00", "EUR 5,000.00"),
              ("2", "AC-1020", "Seal kit", "10", "sets", "EUR 35.50", "EUR 355.00")],
        totals=[("Net total", "EUR 5,355.00"), ("VAT 0% (reverse charge)", "EUR 0.00"), ("Total", "EUR 5,355.00")],
        footer="Payment terms: 14 days 2% discount, 30 days net. Incoterms: FCA Leeds.",
    ),
    "order_pl": dict(
        title="Zamówienie", supplier="Dostawca Sp. z o.o.<br>ul. Przemysłowa 8<br>60-123 Poznań<br>Polska",
        buyer="Fabryka Mebli Łódź S.A.<br>ul. Piotrkowska 100<br>90-001 Łódź<br>Polska<br>NIP: PL7251234567",
        meta=[("Numer zamówienia", "ZAM/2026/11/042"), ("Data zamówienia", "5 listopada 2026"),
              ("Termin dostawy", "20.11.2026"), ("Osoba kontaktowa", "Piotr Nowak, p.nowak@fabryka-lodz.pl")],
        cols=("Lp.", "Indeks", "Nazwa towaru", "Ilość", "J.m.", "Cena jedn. netto", "Wartość netto"),
        rows=[("1", "PL-1001", "Płyta wiórowa 18 mm", "150", "szt.", "45,90 zł", "6 885,00 zł"),
              ("2", "OK-22", "Okleina dębowa", "300", "m", "2,15 zł", "645,00 zł")],
        totals=[("Wartość netto", "7 530,00 zł"), ("VAT 23%", "1 731,90 zł"), ("Wartość brutto", "9 261,90 zł")],
        footer="Warunki płatności: przelew 21 dni.",
    ),
    "order_it": dict(
        title="Ordine di acquisto", supplier="Fornitore S.r.l.<br>Via Roma 15<br>20121 Milano<br>Italia",
        buyer="Officine Bianchi S.p.A.<br>Via Garibaldi 22<br>10122 Torino<br>Italia<br>P.IVA: IT01234567890",
        meta=[("Numero ordine", "OA-2026-1187"), ("Data ordine", "4 novembre 2026"),
              ("Data di consegna", "15/12/2026"), ("Codice cliente", "CL-330")],
        cols=("Pos.", "Codice", "Descrizione", "Quantità", "U.M.", "Prezzo unitario", "Importo"),
        rows=[("1", "CU-450", "Cuscinetto a sfere 6204", "200", "pz", "3,80", "760,00"),
              ("2", "OL-5L", "Olio idraulico ISO 46", "20", "l", "6,25", "125,00")],
        totals=[("Imponibile", "885,00 €"), ("IVA 22%", "194,70 €"), ("Totale", "1.079,70 €")],
        footer="Pagamento: bonifico bancario 60 gg d.f.f.m. Resa: franco destino.",
    ),
}


def build(spec: dict) -> bytes:
    meta = "".join(f"<tr><td><b>{k}:</b></td><td>{v}</td></tr>" for k, v in spec["meta"])
    head = "".join(f"<th>{c}</th>" for c in spec["cols"])
    rows = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in spec["rows"])
    totals = "".join(f"<tr><td>{k}</td><td><b>{v}</b></td></tr>" for k, v in spec["totals"])
    html = f"""
    <div style="font-family: sans-serif; font-size: 9pt">
      <p>{spec['buyer']}</p>
      <p style="margin-left: 300px">An / To:<br>{spec['supplier']}</p>
      <h2>{spec['title']}</h2>
      <table>{meta}</table><br>
      <table border="1" cellpadding="3" style="border-collapse: collapse; width: 100%"><tr>{head}</tr>{rows}</table><br>
      <table style="margin-left: 300px">{totals}</table>
      <p>{spec['footer']}</p>
    </div>"""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_htmlbox(fitz.Rect(40, 40, 555, 800), html)
    return doc.tobytes()


def to_scan(pdf_bytes: bytes) -> bytes:
    """Rasterize every page -> PDF without text layer (simulates a scanned document)."""
    src, out = fitz.open(stream=pdf_bytes, filetype="pdf"), fitz.open()
    for page in src:
        pix = page.get_pixmap(dpi=200, colorspace=fitz.csGRAY)
        new = out.new_page(width=page.rect.width, height=page.rect.height)
        new.insert_image(new.rect, pixmap=pix)
    return out.tobytes()


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for name, spec in SAMPLES.items():
        data = build(spec)
        (OUT / f"{name}.pdf").write_bytes(data)
        print("wrote", OUT / f"{name}.pdf")
    scan = to_scan((OUT / "order_de.pdf").read_bytes())
    (OUT / "order_de_scanned.pdf").write_bytes(scan)
    print("wrote", OUT / "order_de_scanned.pdf")
