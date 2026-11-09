"""Local persistence and source matching for human-reviewed orders."""
from __future__ import annotations

import json
import re
import sqlite3
import unicodedata
import uuid
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterator, NamedTuple

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
            history TEXT NOT NULL,
            page_words TEXT NOT NULL DEFAULT '[]'
        )
    """)
    columns = {row["name"] for row in connection.execute("PRAGMA table_info(reviews)")}
    if "page_words" not in columns:  # databases created before word persistence
        connection.execute("ALTER TABLE reviews ADD COLUMN page_words TEXT NOT NULL DEFAULT '[]'")
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


def _dump_words(pdf: PdfText) -> str:
    return json.dumps([[asdict(word) for word in page] for page in pdf.page_words], ensure_ascii=False)


def _load_words(raw: str) -> PdfText:
    return PdfText(page_words=[[PdfWord(**word) for word in page] for page in json.loads(raw)])


# Values the pipeline derives or normalizes (unit codes, ISO codes, sums): they often do not
# appear literally in the PDF, so a missing match is expected and not a review issue.
_DERIVED = re.compile(
    r"^(document\.(language|currency)|.*\.country_code|line_items\.\d+\.unit_code|totals\.line_items_sum)$"
)


def _group(path: str) -> str:
    """Fields that sit together on the page: one party block, or one line item row."""
    parts = path.split(".")
    return ".".join(parts[:2]) if parts[0] == "line_items" else parts[0]


def _center(words: list[PdfWord]) -> tuple[int, float, float, float]:
    page = words[0].page
    x = (min(w.x0 for w in words) + max(w.x1 for w in words)) / 2
    y0, y1 = min(w.y0 for w in words), max(w.y1 for w in words)
    return page, x, (y0 + y1) / 2, y1 - y0


class Match(NamedTuple):
    words: list[PdfWord]
    confidence: float
    kind: str  # exact | equivalent (same date/number, other format) | approximate (similar text)


def _candidates(value: Any, language: str | None, pdf: PdfText, percent: bool = False) -> list[Match]:
    target = _word_tokens(str(value))
    candidates: dict[tuple[str, ...], Match] = {}
    if isinstance(value, str):
        expected_date, expected_number = parse_date(value, language), None
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        expected_date, expected_number = None, float(value)
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
                    end = start + len(target)
                    spans = [(word, word_start, word_len) for word, (word_start, word_len) in zip(page_words, starts)
                             if word_len and word_start < end and word_start + word_len > start]
                    # a number must cover whole words: position "1" is not the "1" in "1,250.00"
                    if expected_number is not None and any(ws < start or ws + wl > end for _, ws, wl in spans):
                        continue
                    matched = [word for word, _, _ in spans]
                    candidates[tuple(word.id for word in matched)] = Match(matched, 1.0, "exact")

    if not candidates and (expected_date or expected_number is not None):
        for page_words in pdf.page_words:
            for start in range(len(page_words)):
                for size in range(1, min(5, len(page_words) - start + 1) + 1):
                    words = page_words[start:start + size]
                    phrase = " ".join(word.text for word in words)
                    if expected_date and parse_date(phrase, language) == expected_date:
                        candidates[tuple(word.id for word in words)] = Match(words, 0.85, "equivalent")
                        continue
                    if not percent and phrase.rstrip().endswith("%"):
                        continue  # "0,00%" in the discount column is not an amount
                    number = parse_number(phrase, language)
                    if expected_number is not None and number is not None and abs(number - expected_number) < 0.0001:
                        candidates[tuple(word.id for word in words)] = Match(words, 0.85, "equivalent")

    if candidates and not any(match.kind == "exact" for match in candidates.values()):
        shortest = min(len(match.words) for match in candidates.values())
        candidates = {key: match for key, match in candidates.items() if len(match.words) == shortest}
    if not candidates and isinstance(value, str):
        return _approximate_candidates(value, pdf)
    return list(candidates.values())


def _squash(text: str) -> str:
    return "".join(_word_tokens(text))


APPROXIMATE_MIN_RATIO = 0.72


def _approximate_candidates(value: str, pdf: PdfText) -> list[Match]:
    """Locate text the model returned in a slightly different form than the OCR/text layer.

    Typical cases: a description wrapped over two lines ("..._Batterie_H" + "alterung_..."),
    OCR slips ("Schutzbtigel"), or changed spacing/punctuation. Windows of consecutive words
    are compared on their squashed text; confidence = similarity ratio (< 1.0 = approximate).
    """
    target = _squash(value)
    if len(target) < 4:
        return []
    grams = {target[i:i + 3] for i in range(len(target) - 2)}
    found: list[tuple[float, list[PdfWord]]] = []
    for page_words in pdf.page_words:
        squashed = [_squash(word.text) for word in page_words]
        for start, first in enumerate(squashed):
            if not first or not (first in target or any(first[i:i + 3] in grams for i in range(max(1, len(first) - 2)))):
                continue
            text = ""
            for end in range(start, min(len(page_words), start + 60)):
                text += squashed[end]
                if len(text) < len(target) * 0.7:
                    continue
                if len(text) > len(target) * 1.35:
                    break
                matcher = SequenceMatcher(None, target, text, autojunk=False)
                if matcher.real_quick_ratio() < APPROXIMATE_MIN_RATIO or matcher.quick_ratio() < APPROXIMATE_MIN_RATIO:
                    continue
                ratio = matcher.ratio()
                if ratio >= APPROXIMATE_MIN_RATIO:
                    found.append((ratio, [w for w, sq in zip(page_words[start:end + 1], squashed[start:end + 1]) if sq]))
    if not found:
        return []
    found.sort(key=lambda item: -item[0])
    best = found[0][0]
    results: list[Match] = []
    used: set[str] = set()
    for ratio, words in found:  # best non-overlapping windows; near-ties go to the context resolution
        if ratio < best - 0.03 or any(word.id in used for word in words):
            continue
        used.update(word.id for word in words)
        if ratio >= 0.995:  # identical apart from line breaks/spacing/punctuation
            results.append(Match(words, 0.95, "exact"))
        else:
            results.append(Match(words, round(ratio, 2), "approximate"))
    return results


def _pick_by_context(path: str, options: list[Match], anchors: list[list[PdfWord]]) -> Match | None:
    """Choose among several occurrences: in the line item's row band, or nearest to the party block."""
    if not anchors:
        return None
    row = path.startswith("line_items.")
    scored = []
    for match in options:
        page, x, y, height = _center(match.words)
        best = None
        for anchor in anchors:
            a_page, a_x, a_y, a_height = _center(anchor)
            if a_page != page:
                continue
            if row:
                line = max(height, a_height)
                # same line, or a description line just below the row's values
                if not (a_y - 0.8 * line <= y <= a_y + 4.5 * line):
                    continue
                distance = abs(y - a_y)
            else:
                distance = ((x - a_x) ** 2 + (y - a_y) ** 2) ** 0.5
            best = distance if best is None else min(best, distance)
        if best is not None:
            scored.append((best, match))
    if not scored:
        return None
    scored.sort(key=lambda item: item[0])
    if len(scored) > 1 and abs(scored[0][0] - scored[1][0]) < 1.0:
        if row and path.endswith(".position"):  # the position is the leftmost column of its row
            tied = [item for item in scored if abs(item[0] - scored[0][0]) < 1.0]
            return min(tied, key=lambda item: item[1].words[0].x0)[1]._replace(confidence=0.9)
        return None  # still ambiguous
    best_match = scored[0][1]
    return best_match._replace(confidence=round(best_match.confidence * 0.95, 2))


def _reading_order(match: Match) -> tuple[int, float, float]:
    first = match.words[0]
    return first.page, round(first.y0), first.x0


def match_sources(order: ErpOrder, pdf: PdfText) -> dict[str, dict[str, Any]]:
    """Match order values to the PDF words: exact text, equivalent dates/numbers, or similar text.

    A value that occurs several times is resolved by its context: line item values must sit in
    the row band of the item's uniquely matched fields, party values near the party's block.
    Header/party values that stay ambiguous point to their first occurrence (letterhead).
    """
    language = order.document.language
    leaves = list(_field_leaves(order.model_dump(mode="json")))
    options = {path: _candidates(value, language, pdf, path.endswith("_percent")) for path, value in leaves}
    spans = _item_spans(order)
    for path, found in options.items():
        span = spans.get(_group(path))
        if span and found:  # the model cited the lines of this position: prefer matches inside them
            # a position's values sit on its own lines - a match elsewhere is a coincidence (the "1"
            # of "Seite 1/1"); then the row itself is shown instead of a wrong place
            options[path] = [m for m in found if all(span[0] <= word.line <= span[1] for word in m.words)]
    chosen: dict[str, Match] = {path: found[0] for path, found in options.items() if len(found) == 1}
    for _ in range(3):  # resolved values become anchors for the rest of their group
        progress = False
        for path, found in options.items():
            if path in chosen or len(found) < 2:
                continue
            siblings = {other: match for other, match in chosen.items() if other != path and _group(other) == _group(path)}
            claimed = {word.id for match in siblings.values() for word in match.words}
            # words already used by another field of the same row/party are not this field's source
            free = [match for match in found if not claimed & {word.id for word in match.words}] or found
            if len(free) == 1 and len(found) > 1 and siblings:
                chosen[path] = free[0]._replace(confidence=round(free[0].confidence * 0.95, 2))
                progress = True
                continue
            picked = _pick_by_context(path, free, [match.words for match in siblings.values()])
            if picked:
                chosen[path] = picked
                progress = True
        if not progress:
            break
    ambiguous = set()
    for path, found in options.items():
        if path not in chosen and len(found) > 1 and not path.startswith("line_items."):
            first = min(found, key=_reading_order)
            chosen[path] = first._replace(confidence=round(first.confidence * 0.9, 2))
            ambiguous.add(path)

    sources: dict[str, dict[str, Any]] = {}
    for path, value in leaves:
        match = chosen.get(path)
        sources[path] = {
            "extracted_value": value,
            "confidence": match.confidence if match else 0.0,
            "match": match.kind if match else None,
            "approximate": bool(match and match.kind == "approximate"),
            "ambiguous": path in ambiguous,
            "occurrences": len(options[path]),
            "derived": bool(_DERIVED.match(path)),
            "lines": spans.get(_group(path)),
            "row_locations": _row_boxes(pdf, spans.get(_group(path))) if not match else [],
            "locations": [
                {"word_id": word.id, "page": word.page, "x0": word.x0, "y0": word.y0, "x1": word.x1, "y1": word.y1}
                for word in (match.words if match else [])
            ],
        }
    return sources


def _item_spans(order: ErpOrder) -> dict[str, tuple[int, int]]:
    """line_items.N -> (first, last) document line the model cited for that position."""
    spans = {}
    for index, span in enumerate(order.meta.item_lines):
        if span and span[0]:
            spans[f"line_items.{index}"] = (span[0], span[-1] or span[0])
    return spans


def _row_boxes(pdf: PdfText, span: tuple[int, int] | None) -> list[dict[str, Any]]:
    """One box per cited line - shown when a value is not printed literally (e.g. a corrected amount)."""
    if not span:
        return []
    boxes: dict[int, dict[str, Any]] = {}
    for page_words in pdf.page_words:
        for word in page_words:
            if span[0] <= word.line <= span[1]:
                box = boxes.setdefault(word.line, {"page": word.page, "x0": word.x0, "y0": word.y0, "x1": word.x1, "y1": word.y1})
                box["x0"], box["y0"] = min(box["x0"], word.x0), min(box["y0"], word.y0)
                box["x1"], box["y1"] = max(box["x1"], word.x1), max(box["y1"], word.y1)
    return [boxes[line] | {"word_id": f"line-{line}"} for line in sorted(boxes)]


def unmatched_paths(sources: dict[str, dict[str, Any]]) -> list[str]:
    """Values a reviewer should look at: no (or only an approximate) source location, not yet confirmed."""
    return [
        path for path, source in sources.items()
        if not source.get("confirmed") and not source.get("derived")
        and (not source["locations"] or source.get("approximate"))
    ]


def create_review(
    db_path: str, pdf_bytes: bytes, order: ErpOrder, pdf_text: PdfText, sources: dict[str, Any],
) -> str:
    review_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    order_json = json.dumps(order.model_dump(mode="json"), ensure_ascii=False)
    with _connect(db_path) as connection:
        connection.execute(
            """INSERT INTO reviews
               (id, created_at, updated_at, status, original_order, current_order, sources, page_sizes, pdf, history,
                page_words)
               VALUES (?, ?, ?, 'pending', ?, ?, ?, ?, ?, '[]', ?)""",
            (
                review_id, now, now, order_json, order_json, json.dumps(sources, ensure_ascii=False),
                json.dumps(pdf_text.page_sizes), pdf_bytes, _dump_words(pdf_text),
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


def update_review(
    db_path: str, review_id: str, order: ErpOrder, confirmed: list[str] | None = None,
) -> dict[str, Any] | None:
    """Save corrected values. `confirmed` lists the fields a reviewer explicitly confirmed (Enter)."""
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
        old_sources = json.loads(row["sources"])
        sources = _rematch_sources(order, old_sources, row["page_words"])
        wanted = set(confirmed or [])
        for path, source in sources.items():
            source["confirmed"] = path in wanted
            if source["confirmed"] and not old_sources.get(path, {}).get("confirmed"):
                old_history.append({"at": datetime.now(timezone.utc).isoformat(), "path": path, "confirmed": source["value"]})
        now = datetime.now(timezone.utc).isoformat()
        connection.execute(
            "UPDATE reviews SET updated_at = ?, status = 'reviewed', current_order = ?, history = ?, sources = ? "
            "WHERE id = ?",
            (
                now, json.dumps(updated, ensure_ascii=False), json.dumps(old_history, ensure_ascii=False),
                json.dumps(sources, ensure_ascii=False), review_id,
            ),
        )
    return get_review(db_path, review_id)


def _rematch_sources(order: ErpOrder, previous: dict[str, Any], raw_words: str) -> dict[str, Any]:
    """Match the corrected values against the PDF; keep what the model originally extracted."""
    words = _load_words(raw_words)
    matched = match_sources(order, words) if words.page_words else {}
    current = {}
    for path, value in _field_leaves(order.model_dump(mode="json")):
        old = previous.get(path)
        if path in matched:
            source = matched[path]
        elif old is not None and old.get("value", old["extracted_value"]) == value:
            source = dict(old)  # legacy review without stored words: unchanged value keeps its match
        else:
            source = {"confidence": 0.0, "locations": []}
        source["value"] = value
        source["derived"] = bool(_DERIVED.match(path))
        source["extracted_value"] = old["extracted_value"] if old is not None else None
        current[path] = source
    return current


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
        if row["status"] == "approved":
            raise ValueError("This review is already approved.")
        history = json.loads(row["history"])
        history.append({"at": datetime.now(timezone.utc).isoformat(), "status": "approved"})
        now = datetime.now(timezone.utc).isoformat()
        connection.execute(
            "UPDATE reviews SET updated_at = ?, status = 'approved', history = ? WHERE id = ?",
            (now, json.dumps(history, ensure_ascii=False), review_id),
        )
    return get_review(db_path, review_id)
