# Order PDF Extractor (Qwen3.5-0.8B + llama-server + FastAPI)

A REST API that takes a **purchase order PDF** (digital, or scanned and read with OCR) in
**German, English, Polish or Italian** and returns **ERP-ready JSON**. Everything runs locally
on a notebook: a small LLM (Qwen3.5-0.8B, GGUF) on llama.cpp's `llama-server`, and Tesseract for OCR.

```
PDF ──► PyMuPDF text layer ──┐
     └► Tesseract OCR (scans)┴► Qwen3.5-0.8B on llama-server ──► normalization and checks ──► ERP JSON
                                (2 passes, JSON-schema grammar)   (numbers, dates, units, totals)
```

## Quick start

### Option A: native (one command after a one-time install)

```bash
./install.sh      # one-time: Tesseract (deu/eng/pol/ita), .venv, llama-server, model (~0.8 GB)
./start.sh        # starts llama-server AND the FastAPI app; Ctrl+C stops both
```

Then open **http://127.0.0.1:8000/** to review an order in the browser, or
**http://127.0.0.1:8000/docs** for the API.

`install.sh` also installs the pinned PDF.js browser assets locally (Node.js/npm is required for
the one-time asset install). `--no-system` skips system packages, so Node.js/npm must already be
available. Options also include `--cpu` (force the CPU build),
`--force-llama` (reinstall llama-server). Tested on Ubuntu 22.04 (WSL2); it also supports
Debian/Ubuntu (apt), Fedora (dnf) and macOS (Homebrew).

### Option B: Docker (one command, nothing else to install)

```bash
docker compose up --build        # add -d to run in the background
```

- `llm`: the official llama.cpp server image (CPU build, pinned version). It downloads the model
  into the `llm-models` volume on the first start, which takes a few minutes.
- `api`: this FastAPI app plus Tesseract with all 4 language packs. PDF.js and its worker,
  CMaps, standard fonts, and WASM assets are copied from a pinned npm package during the
  multi-stage image build; Node.js/npm are not included in the running API image.
- The review UI is available at **http://localhost:8000/** and the API at **http://localhost:8000/docs**.
  Only the API is published, on localhost only;
  see the comments in `docker-compose.yml` to change this.
- Stop with `docker compose down`. Add `-v` to also delete the downloaded model.

## Using the API

| Endpoint | Description |
|---|---|
| `GET /review` | Upload a PDF and open the human review interface |
| `POST /api/reviews` | Extract a PDF and create a persistent review |
| `GET /api/reviews/{id}` | Read extracted values, source matches, and correction history |
| `PUT /api/reviews/{id}` | Save corrected order values |
| `POST /api/reviews/{id}/approve` | Approve a complete order (explicitly acknowledge warnings) |
| `GET /api/reviews/{id}/export` | Download ERP JSON after approval |
| `POST /extract?ocr=auto` | Upload a PDF (`file` form field) and get order JSON back |
| `POST /text?ocr=auto` | Debug: the text the LLM would see (no LLM call) |
| `GET /health` | API and LLM status |

`ocr` parameter:
- `auto` (default): OCR is used only on pages without a usable text layer.
- `always`: OCR every page, for PDFs with a broken text layer.
- `never`: use the text layer only.

```bash
curl -F "file=@samples/order_de.pdf" "http://127.0.0.1:8000/extract"
curl -F "file=@scan.pdf" "http://127.0.0.1:8000/extract?ocr=always"
```

Generate test PDFs (DE/EN/PL/IT and a scanned DE PDF) with
`.venv/bin/python scripts/make_samples.py`. The output goes to `samples/`.

## Output (ERP JSON)

```jsonc
{
  "schema_version": "1.0",
  "document": {
    "document_type": "purchase_order",
    "order_number": "BE-2026-0815",
    "order_date": "2026-11-03",               // ISO 8601
    "requested_delivery_date": "2026-11-23",  // "KW 48/2026" -> Monday of that week
    "customer_number": "K-4711",
    "currency": "EUR",                        // ISO 4217
    "language": "de",
    "payment_terms": "30 Tage netto",
    "incoterms": "DAP München"
  },
  "buyer":    { "name": "Beispiel Maschinenbau AG", "street": "Hauptstraße 12", "postal_code": "80331",
                "city": "München", "country_code": "DE", "vat_id": "DE123456789",
                "contact": { "name": "Anna Schmidt", "email": "a.schmidt@beispiel-ag.de", "phone": null } },
  "supplier": { "name": "Muster Lieferant GmbH", "...": "..." },
  "ship_to":  { "...": "separate delivery address, otherwise the buyer" },
  "bill_to":  { "...": "separate invoice address, otherwise the buyer" },
  "line_items": [
    { "position": 10, "supplier_article_number": "SR-M8-40", "customer_article_number": null,
      "description": "Sechskantschraube M8x40 verzinkt", "quantity": 1000.0,
      "unit_code": "H87", "unit_raw": "Stk",    // UN/ECE Rec. 20 unit code
      "unit_price": 0.12, "discount_percent": null, "line_total": 120.0,
      "requested_delivery_date": "2026-11-23" }
  ],
  "totals": { "net_amount": 212.5, "tax_rate_percent": 19.0, "tax_amount": 40.38,
              "gross_amount": 252.88, "line_items_sum": 212.5 },
  "validation": {
    "is_complete": true,                      // all ERP-required fields present
    "missing_required_fields": [],
    "warnings": []                            // e.g. totals mismatch, removed hallucinations
  },
  "meta": { "source_filename": "order_de.pdf", "page_count": 1, "ocr_mode": "auto",
            "ocr_pages": [], "text_chars": 1477, "llm_chunks": 1,
            "model": "qwen3.5-0.8b", "duration_s": 24.7 }
}
```

**Check `validation` before you import.** Send orders with `is_complete: false` or with warnings
to a person for review.

## Human review

The browser workflow keeps the PDF beside the extracted fields. Native PDF text and OCR output
include word-level boxes; matching values get color-coded SVG highlights, and unmatched or
incomplete values are flagged for review. Click a field to locate its source, or click a highlight
to focus its field. Corrections can be saved with the button or `Ctrl+S`/`Cmd+S`; approving requires
all required values and acknowledgement of remaining extraction warnings. Only approved orders
can be downloaded from the ERP JSON endpoint.

PDF.js renders pages in the browser, with selectable text and document search; PyMuPDF is retained
for local text extraction, OCR coordination, and word-level source coordinates. No PDF pages are
sent to an external viewer or font service.

Review sessions, original PDFs, and correction history are stored locally in SQLite. Docker uses
the persistent `review-data` volume; native runs use `data/order_reviews.sqlite3` (override with
`REVIEW_DB_PATH`). The browser viewer assets are served by FastAPI from `/static/vendor/pdfjs/`.

## How it handles a 0.8B model on a notebook

- **The model only copies text.** It returns values exactly as they appear in the PDF. Python does
  every conversion: number formats such as `1.234,56`, `1,234.56`, `1 234,56 zł`; dates in 4
  languages including month names and calendar weeks; units mapped to UN/ECE codes; currencies;
  country codes.
- **Two short passes instead of one large one.** Pass 1 extracts the header, parties and totals.
  Pass 2 extracts the line items. The document text comes first in the prompt, so pass 2 reuses
  pass 1's KV cache (prompt caching).
- **Grammar-constrained JSON** (`response_format: json_schema`). The model cannot produce invalid
  JSON. All keys are required and may be null, so the model has to consider every field.
- **Thinking mode is off** (`enable_thinking: false`). This saves tokens and time.
- **Hallucination guards:**
  - Order and customer numbers, article numbers, VAT IDs, names, streets, emails and phones must
    appear in the PDF text. If they don't, they are removed and a warning is added.
  - Buyer and supplier are swapped back if the address labels (`An`, `To`, `Lieferant`,
    `Dostawca`, `Fornitore`, ...) show the opposite.
  - Net and gross totals are repaired when the arithmetic doesn't add up.
- **Resource-friendly server settings:**
  - Q8_0 quantization: about 0.8 GB RAM, the best quality for this model size.
  - `--parallel 1` and an 8k context.
  - The API sends one LLM request at a time.
  - Long documents are split into chunks, then merged.
- **Speed:** about 20–30 s per 1-page order on a notebook CPU.

## Configuration

Copy `.env.example` to `.env`. `install.sh` does this for you. Environment variables override `.env`.

| Variable | Default | Notes |
|---|---|---|
| `MODEL_FILE` | `Qwen3.5-0.8B-Q8_0.gguf` | `Qwen3.5-0.8B-Q4_K_M.gguf` for very low-RAM machines |
| `LLM_THREADS` | auto | Set to the number of **physical performance cores** on hybrid notebook CPUs |
| `LLM_CTX` | `8192` | Context window |
| `API_PORT` | `8000` | |
| `OWN_COMPANY_NAMES` | – | Your company name(s). Recommended: helps tell buyer and supplier apart |
| `MAX_CHARS_PER_CHUNK` | `9000` | PDF text per LLM call |
| `OCR_LANGUAGES` | `deu+eng+pol+ita` | Tesseract languages |

## GPU (optional)

CPU inference is enough for this model. `install.sh` installs a CUDA build of llama-server
automatically if it finds an NVIDIA driver **and** the system has glibc ≥ 2.38 (Ubuntu 24.04+).
Otherwise it uses the CPU build. For Docker, `ghcr.io/ggml-org/llama.cpp:server-cuda` and the
NVIDIA Container Toolkit can be used. The GPU setups are not tested.

## Tests

```bash
.venv/bin/python -m pytest -q
```

## Project layout

```
app/
  main.py          FastAPI endpoints
  review_store.py  SQLite review sessions, correction history, source matching
  static/          Browser review UI, SVG source highlights, and locally bundled PDF.js
  extractor.py     pipeline: chunks -> 2 LLM passes -> merge -> sanity checks -> ERP model
  llm.py           llama-server client, prompts (multilingual label hints)
  pdf_extract.py   PyMuPDF text extraction, Tesseract OCR, chunking
  normalize.py     locale-aware numbers/dates, units, currencies, countries, grounding
  schemas.py       raw LLM schema and ERP output schema
  config.py        settings (.env)
install.sh         one-time installer (native)
start.sh           starts llama-server and the API with one command
Dockerfile         multi-stage build bundles PDF.js locally; runtime stays Python-only
package.json       pinned PDF.js browser dependency
package-lock.json  reproducible PDF.js asset build
scripts/install_pdfjs.sh  installs/copies local PDF.js assets
docker-compose.yml
scripts/make_samples.py   generates multilingual test PDFs
tests/             unit tests (no LLM needed)
```
