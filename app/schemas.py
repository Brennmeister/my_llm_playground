"""Data models.

Two layers:
* ``Raw*`` models  -> what the LLM fills in. Every value is a plain string copied from the
  document (or null). Small models are bad at converting "1.234,56" or "8 listopada 2026"
  reliably, so conversion happens deterministically in Python (see normalize.py).
* ``Erp*`` models  -> the normalized, typed result returned by the API for ERP import.
"""
from __future__ import annotations

import copy
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Raw LLM layer
# ---------------------------------------------------------------------------


class RawParty(BaseModel):
    company_name: Optional[str] = None
    street: Optional[str] = None
    postal_code: Optional[str] = None
    city: Optional[str] = None
    country: Optional[str] = None
    vat_id: Optional[str] = None
    contact_person: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None


class RawLineItem(BaseModel):
    position: Optional[str] = None
    article_number: Optional[str] = Field(None, description="supplier article/material number")
    customer_article_number: Optional[str] = Field(None, description="buyer's own article number")
    description: Optional[str] = None
    quantity: Optional[str] = None
    unit: Optional[str] = None
    unit_price: Optional[str] = None
    discount: Optional[str] = None
    line_total: Optional[str] = None
    delivery_date: Optional[str] = None


class RawHeader(BaseModel):
    """Pass 1: header, parties and totals (no line items -> short, focused output)."""
    order_number: Optional[str] = None
    order_date: Optional[str] = None
    requested_delivery_date: Optional[str] = None
    customer_number: Optional[str] = Field(None, description="buyer's customer number at the supplier")
    currency: Optional[str] = None
    payment_terms: Optional[str] = None
    incoterms: Optional[str] = None
    buyer: RawParty = Field(default_factory=RawParty)
    supplier: RawParty = Field(default_factory=RawParty)
    delivery_address: RawParty = Field(default_factory=RawParty)
    invoice_address: RawParty = Field(default_factory=RawParty)
    net_total: Optional[str] = None
    tax_rate: Optional[str] = None
    tax_amount: Optional[str] = None
    gross_total: Optional[str] = None


class RawItems(BaseModel):
    """Pass 2: line items only."""
    line_items: list[RawLineItem] = Field(default_factory=list)


class RawOrder(RawHeader, RawItems):
    """Merged result of both passes (and of all chunks)."""


def _inline_and_require_all(schema: dict[str, Any]) -> dict[str, Any]:
    """Inline $refs and mark every property as required.

    With grammar-constrained decoding, "required + nullable" forces the small model to
    visit each field in a fixed order (better recall) instead of skipping keys.
    """
    schema = copy.deepcopy(schema)
    defs = schema.pop("$defs", {})

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                return walk(copy.deepcopy(defs[node["$ref"].split("/")[-1]]))
            node = {
                # keep property NAMES (a field may be called "description"), drop metadata keys
                k: ({pk: walk(pv) for pk, pv in v.items()} if k == "properties" else walk(v))
                for k, v in node.items()
                if k not in ("title", "default", "description")
            }
            if node.get("type") == "object" and "properties" in node:
                node["required"] = list(node["properties"].keys())
                node["additionalProperties"] = False
            return node
        if isinstance(node, list):
            return [walk(x) for x in node]
        return node

    return walk(schema)


def llm_json_schema(task: Literal["header", "items"]) -> dict[str, Any]:
    model = RawHeader if task == "header" else RawItems
    schema = _inline_and_require_all(model.model_json_schema())
    if task == "items":
        # Hard cap on items keeps generation time bounded on a notebook
        schema["properties"]["line_items"]["maxItems"] = 60
    return schema


# ---------------------------------------------------------------------------
# ERP output layer
# ---------------------------------------------------------------------------


class Contact(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None


class Party(BaseModel):
    name: Optional[str] = None
    street: Optional[str] = None
    postal_code: Optional[str] = None
    city: Optional[str] = None
    country_code: Optional[str] = Field(None, description="ISO 3166-1 alpha-2")
    vat_id: Optional[str] = None
    contact: Contact = Field(default_factory=Contact)


class DocumentHeader(BaseModel):
    document_type: Literal["purchase_order"] = "purchase_order"
    order_number: Optional[str] = None
    order_date: Optional[str] = Field(None, description="ISO 8601 date (YYYY-MM-DD)")
    requested_delivery_date: Optional[str] = Field(None, description="ISO 8601 date (YYYY-MM-DD)")
    customer_number: Optional[str] = None
    currency: Optional[str] = Field(None, description="ISO 4217")
    language: Optional[str] = Field(None, description="ISO 639-1 (de, en, pl, it)")
    payment_terms: Optional[str] = None
    incoterms: Optional[str] = None


class LineItem(BaseModel):
    position: int
    supplier_article_number: Optional[str] = None
    customer_article_number: Optional[str] = None
    description: Optional[str] = None
    quantity: Optional[float] = None
    unit_code: Optional[str] = Field(None, description="UN/ECE Rec. 20 unit code, e.g. H87, KGM, MTR")
    unit_raw: Optional[str] = None
    unit_price: Optional[float] = None
    discount_percent: Optional[float] = None
    line_total: Optional[float] = None
    requested_delivery_date: Optional[str] = None


class Totals(BaseModel):
    net_amount: Optional[float] = None
    tax_rate_percent: Optional[float] = None
    tax_amount: Optional[float] = None
    gross_amount: Optional[float] = None
    line_items_sum: Optional[float] = None


class Validation(BaseModel):
    is_complete: bool = False
    missing_required_fields: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class Meta(BaseModel):
    source_filename: Optional[str] = None
    page_count: int = 0
    ocr_mode: str = "auto"
    ocr_pages: list[int] = Field(default_factory=list)
    text_chars: int = 0
    llm_chunks: int = 0
    model: Optional[str] = None
    duration_s: float = 0.0


class ErpOrder(BaseModel):
    schema_version: str = "1.0"
    document: DocumentHeader = Field(default_factory=DocumentHeader)
    buyer: Party = Field(default_factory=Party)
    supplier: Party = Field(default_factory=Party)
    ship_to: Optional[Party] = None
    bill_to: Optional[Party] = None
    line_items: list[LineItem] = Field(default_factory=list)
    totals: Totals = Field(default_factory=Totals)
    validation: Validation = Field(default_factory=Validation)
    meta: Meta = Field(default_factory=Meta)
