"""Local persistence and source matching for human-reviewed orders."""
from __future__ import annotations

import json
import re
import sqlite3
import unicodedata
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .normalize import parse_date, parse_number
from .pdf_extract import PdfText, PdfWord
from .schemas import ErpOrder


@contextmanager
def _connect(db_path: str) -> Iterator[sqlite3.Connection]:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("""
        CREATE TABLE IF NOT EXISTS reviews (
            id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            status TEXT NOT NULL,
            original_order TEXT NOT NULL,
            current_order TEXT NOT NULL,
            sources TEXT NOT NULL,
            page_sizes TEXT NOT NULL,
            pdf BLOB NOT NULL,
            history TEXT NOT NULL
        )
    """)
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def _word_tokens(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return re.findall(r"[^\W_]+", normalized)


def _field_leaves(value: Any, path: str = ""):
    if isinstance(value, dict):
        for key, child in value.items():
            if key not in {"meta", "validation", "schema_version", "document_type"}:
                yield from _field_leaves(child, f"{path}.{key}" if path else key)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _field_leaves(child, f"{path}.{index}")
    elif value is not None and value != "":
        yield path, value


def _all_field_leaves(value: Any, path: str = ""):
    if isinstance(value, dict):
        for key, child in value.items():
            if key not in {"meta", "validation", "schema_version", "document_type"}:
                yield from _all_field_leaves(child, f"{path}.{key}" if path else key)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _all_field_leaves(child, f"{path}.{index}")
    else:
        yield path, value


def match_sources(order: ErpOrder, pdf: PdfText) -> dict[str, dict[str, Any]]:
    """Match normalized order values to exact word sequences or equivalent dates/numbers."""
    sources: dict[str, dict[str, Any]] = {}
    language = order.document.language
    for path, value in _field_leaves(order.model_dump(mode="json")):
        target = _word_tokens(str(value))
        candidates: dict[tuple[str, ...], tuple[list[PdfWord], float]] = {}
        if isinstance(value, str):
            expected_date = parse_date(value, language)
            expected_number = None
        elif isinstance(value, (int, float)):
            expected_date = None
            expected_number = float(value)
        else:
            expected_date = expected_number = None

        for page_words in pdf.page_words:
            page_tokens = [_word_tokens(word.text) for word in page_words]
            flat_tokens = [token for tokens in page_tokens for token in tokens]
            starts: list[tuple[int, int]] = []
            offset = 0
            for tokens in page_tokens:
                starts.append((offset, len(tokens)))
                offset += len(tokens)
            if target:
                for start in range(max(0, len(flat_tokens) - len(target) + 1)):
                    if flat_tokens[start:start + len(target)] == target:
                        matched = [
                            word for word, (word_start, word_len) in zip(page_words, starts)
                            if word_len and word_start < start + len(target) and word_start + word_len > start
                        ]
                        key = tuple(word.id for word in matched)
                        candidates[key] = (matched, 1.0)

        if not candidates and (expected_date or expected_number is not None):
            for page_words in pdf.page_words:
                for start in range(len(page_words)):
                    for size in range(1, min(5, len(page_words) - start + 1) + 1):
                        words = page_words[start:start + size]
                        phrase = " ".join(word.text for word in words)
                        if expected_date and parse_date(phrase, language) == expected_date:
                            candidates[tuple(word.id for word in words)] = (words, 0.85)
                            continue
                        number = parse_number(phrase, language)
                        if expected_number is not None and number is not None and abs(number - expected_number) < 0.0001:
                            candidates[tuple(word.id for word in words)] = (words, 0.85)

        if candidates and not any(confidence == 1.0 for _, confidence in candidates.values()):
            shortest = min(len(words) for words, _ in candidates.values())
            candidates = {key: match for key, match in candidates.items() if len(match[0]) == shortest}
        locations = []
        found = next(iter(candidates.values())) if len(candidates) == 1 else None
        if found:
            words, confidence = found
            locations = [
                {"word_id": word.id, "page": word.page, "x0": word.x0, "y0": word.y0, "x1": word.x1, "y1": word.y1}
                for word in words
            ]
        else:
            confidence = 0.0
        sources[path] = {"extracted_value": value, "confidence": confidence, "locations": locations}
    return sources


def create_review(
    db_path: str, pdf_bytes: bytes, order: ErpOrder, pdf_text: PdfText, sources: dict[str, Any],
) -> str:
    review_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    order_json = json.dumps(order.model_dump(mode="json"), ensure_ascii=False)
    with _connect(db_path) as connection:
        connection.execute(
            """INSERT INTO reviews
               (id, created_at, updated_at, status, original_order, current_order, sources, page_sizes, pdf, history)
               VALUES (?, ?, ?, 'pending', ?, ?, ?, ?, ?, '[]')""",
            (
                review_id, now, now, order_json, order_json, json.dumps(sources, ensure_ascii=False),
                json.dumps(pdf_text.page_sizes), pdf_bytes,
            ),
        )
    return review_id


def get_review(db_path: str, review_id: str) -> dict[str, Any] | None:
    with _connect(db_path) as connection:
        row = connection.execute("SELECT * FROM reviews WHERE id = ?", (review_id,)).fetchone()
    if row is None:
        return None
    return {
        "id": row["id"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "status": row["status"],
        "order": json.loads(row["current_order"]),
        "sources": json.loads(row["sources"]),
        "page_sizes": json.loads(row["page_sizes"]),
        "history": json.loads(row["history"]),
        "pdf": row["pdf"],
    }


def update_review(db_path: str, review_id: str, order: ErpOrder) -> dict[str, Any] | None:
    with _connect(db_path) as connection:
        row = connection.execute("SELECT * FROM reviews WHERE id = ?", (review_id,)).fetchone()
        if row is None:
            return None
        if row["status"] == "approved":
            raise ValueError("Approved reviews cannot be edited.")
        previous = json.loads(row["current_order"])
        updated = order.model_dump(mode="json")
        changes = []
        previous_values = dict(_all_field_leaves(previous))
        updated_values = dict(_all_field_leaves(updated))
        for path in previous_values.keys() | updated_values.keys():
            before, after = previous_values.get(path), updated_values.get(path)
            if before != after:
                changes.append({"at": datetime.now(timezone.utc).isoformat(), "path": path, "before": before, "after": after})
        old_history = json.loads(row["history"])
        old_history.extend(changes)
        now = datetime.now(timezone.utc).isoformat()
        connection.execute(
            "UPDATE reviews SET updated_at = ?, status = 'reviewed', current_order = ?, history = ? WHERE id = ?",
            (now, json.dumps(updated, ensure_ascii=False), json.dumps(old_history, ensure_ascii=False), review_id),
        )
    return get_review(db_path, review_id)


def validate_order(order: ErpOrder) -> ErpOrder:
    missing = []
    if not order.document.order_number:
        missing.append("document.order_number")
    if not order.document.order_date:
        missing.append("document.order_date")
    if not order.buyer.name:
        missing.append("buyer.name")
    if not order.line_items:
        missing.append("line_items")
    for item in order.line_items:
        if item.quantity is None:
            missing.append(f"line_items[{item.position}].quantity")
    order.validation.missing_required_fields = missing
    order.validation.is_complete = not missing
    return order


def approve_review(db_path: str, review_id: str) -> dict[str, Any] | None:
    with _connect(db_path) as connection:
        row = connection.execute("SELECT * FROM reviews WHERE id = ?", (review_id,)).fetchone()
        if row is None:
            return None
        history = json.loads(row["history"])
        history.append({"at": datetime.now(timezone.utc).isoformat(), "status": "approved"})
        now = datetime.now(timezone.utc).isoformat()
        connection.execute(
            "UPDATE reviews SET updated_at = ?, status = 'approved', history = ? WHERE id = ?",
            (now, json.dumps(history, ensure_ascii=False), review_id),
        )
    return get_review(db_path, review_id)
