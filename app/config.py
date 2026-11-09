"""Central configuration. All values can be overridden via environment variables or a .env file."""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- llama-server connection -------------------------------------------------
    llm_base_url: str = "http://127.0.0.1:8081"
    llm_model_name: str = "qwen3.5-0.8b"  # alias passed to llama-server (--alias)
    llm_timeout_s: float = 300.0
    llm_max_tokens: int = 2048        # header pass
    llm_max_tokens_items: int = 4096  # line item pass (~80 tokens per row)
    llm_temperature: float = 0.0
    # Context window the server is started with (keep in sync with LLM_CTX in start.sh)
    llm_ctx: int = 8192
    # The model also SEES the document (Qwen3.5 is a vision-language model; llama-server loads the
    # mmproj file with the model): all = every page, scans = OCR'd pages only, off = text only
    llm_images: str = "all"
    llm_page_image_tokens: int = 1500   # image budget of a header call (all pages of the chunk)
    llm_page_image_dpi: int = 110       # upper limit for page images
    llm_row_image_dpi: int = 150        # strip of one position (small, sharp: handwriting, faint digits)

    # --- Document handling -------------------------------------------------------
    # Max characters of PDF text sent to the LLM per call. 0.8B models degrade on long
    # inputs and a notebook should stay responsive, so long documents are chunked.
    max_chars_per_chunk: int = 9000
    max_upload_mb: int = 25
    max_pages: int = 30
    review_db_path: str = "data/order_reviews.sqlite3"

    # --- OCR ---------------------------------------------------------------------
    ocr_languages: str = "deu+eng+pol+ita"
    ocr_dpi: int = 300
    # In "auto" mode a page is OCR'd if its text layer has fewer characters than this
    ocr_min_chars_per_page: int = 40

    # --- Business context --------------------------------------------------------
    # Comma separated names of YOUR company. Used to tell buyer and supplier apart:
    # if the model puts your company as buyer, the parties are swapped.
    own_company_names: str = ""
    default_currency: str = "EUR"


settings = Settings()
