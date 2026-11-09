const NS = "http://www.w3.org/2000/svg";
const SKIP_KEYS = new Set(["meta", "validation", "schema_version", "document_type"]);
const WIDE_FIELDS = new Set(["name", "street", "description", "payment_terms", "incoterms", "email"]);
const NUMERIC_FIELDS = new Set(["position", "quantity", "unit_price", "discount_percent", "line_total", "net_amount", "tax_rate_percent", "tax_amount", "gross_amount", "line_items_sum"]);
const ITEM_MAIN = ["position", "supplier_article_number", "description", "quantity", "unit_raw", "unit_price", "line_total"];
const ITEM_EXTRA = ["customer_article_number", "unit_code", "discount_percent", "requested_delivery_date"];
const SECTIONS = [
  ["document", "Document"], ["buyer", "Buyer"], ["supplier", "Supplier"], ["ship_to", "Ship to"],
  ["bill_to", "Bill to"], ["line_items", "Line items"], ["totals", "Totals"],
];
const LABELS = {
  order_number: "Order number", order_date: "Order date", requested_delivery_date: "Delivery date",
  customer_number: "Customer no.", payment_terms: "Payment terms", postal_code: "Postal code",
  country_code: "Country", vat_id: "VAT ID", position: "Pos.", supplier_article_number: "Article no.",
  customer_article_number: "Customer article no.", quantity: "Qty", unit_raw: "Unit", unit_code: "Unit code",
  unit_price: "Unit price", line_total: "Total", discount_percent: "Discount %", net_amount: "Net",
  tax_rate_percent: "Tax %", tax_amount: "Tax", gross_amount: "Gross", line_items_sum: "Sum of lines",
};
const ISSUE_STATES = ["missing", "unmatched", "approx"];
const STATE_TEXT = {
  matched: "Found in the PDF", unmatched: "Not found in the PDF – please verify", missing: "Required – missing",
  empty: "Not detected", edited: "Changed – save to re-check against the PDF",
  derived: "Normalized value (code, unit or sum) – not printed literally in the PDF",
  approx: "Similar text found in the PDF – check it and press Enter to confirm",
  confirmed: "Confirmed by reviewer",
};

let reviewId;
let confirmed = new Set();
let saveChain = Promise.resolve();
let review;
let savedOrder;
let pdfjsLib;
let pdfDocument;
let pageObserver;
let zoom = 1;
let selectedPath = null;
let renderGeneration = 0;
const pageRenders = new Map();
let searchMatches = [];
let currentSearchMatch = -1;
let lastSearchQuery = "";

const $ = (selector) => document.querySelector(selector);
const isApproved = () => review?.status === "approved";
const getPath = (object, path) => path.split(".").reduce(
  (node, part) => (node == null ? undefined : node[Array.isArray(node) ? Number(part) : part]), object);
const setPath = (object, path, value) => {
  const parts = path.split(".");
  const key = parts.pop();
  const parent = parts.reduce((node, part) => node[Array.isArray(node) ? Number(part) : part], object);
  parent[Array.isArray(parent) ? Number(key) : key] = value;
};
const humanize = (key) => key.replaceAll("_", " ").replace(/^\w/, (letter) => letter.toUpperCase());
function labelFor(path) {
  const parts = path.split(".");
  const key = parts.at(-1);
  const label = LABELS[key] || humanize(key);
  return parts.at(-2) === "contact" ? `Contact ${label.toLowerCase()}` : label;
}
function describePath(path) {
  const parts = path.split(".");
  if (parts[0] === "line_items") return `Item ${Number(parts[1]) + 1}: ${labelFor(path)}`;
  const section = SECTIONS.find(([key]) => key === parts[0])?.[1] ?? humanize(parts[0]);
  return `${section}: ${labelFor(path)}`;
}
const isRequired = (path) => ["document.order_number", "document.order_date", "buyer.name"].includes(path)
  || /^line_items\.\d+\.quantity$/.test(path);
const isNumericPath = (path) => NUMERIC_FIELDS.has(path.split(".").at(-1));

function fieldState(path) {
  const value = getPath(review.order, path);
  const saved = getPath(savedOrder, path);
  if (String(value ?? "") !== String(saved ?? "")) return "edited";
  if (confirmed.has(path)) return "confirmed";
  if (value == null || value === "") return isRequired(path) ? "missing" : "empty";
  const source = review.sources[path];
  if (source?.locations?.length) return source.approximate ? "approx" : "matched";
  return source?.derived ? "derived" : "unmatched";
}

function parseNumberInput(value, language) {
  let number = value.trim().replace(/[^\d,.\-−+]/g, "").replace("−", "-");
  if (!number) return null;
  const comma = number.lastIndexOf(",");
  const dot = number.lastIndexOf(".");
  if (comma >= 0 && dot >= 0) {
    const decimal = comma > dot ? "," : ".";
    number = number.replace(decimal === "," ? /\./g : /,/g, "").replace(decimal, ".");
  } else if (comma >= 0) {
    const grouped = language === "en" && /^[-+]?\d{1,3}(,\d{3})+$/.test(number);
    number = grouped ? number.replaceAll(",", "") : number.replace(",", ".");
  } else if (dot >= 0 && ["de", "pl", "it"].includes(language) && /^[-+]?\d{1,3}(\.\d{3})+$/.test(number)) {
    number = number.replaceAll(".", "");
  }
  const parsed = Number(number);
  return Number.isFinite(parsed) ? parsed : Number.NaN;
}
function visibleFields(value, path) {
  const fields = [];
  for (const [key, child] of Object.entries(value ?? {})) {
    if (SKIP_KEYS.has(key)) continue;
    const childPath = `${path}.${key}`;
    if (child && typeof child === "object" && !Array.isArray(child)) fields.push(...visibleFields(child, childPath));
    else fields.push({ path: childPath, value: child });
  }
  return fields;
}
async function readJson(response) {
  try { return await response.json(); } catch { return {}; }
}
function errorMessage(result, fallback) {
  const detail = result?.detail;
  if (!detail) return fallback;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) return detail.map((error) => `${(error.loc ?? []).slice(2).join(".") || "value"}: ${error.msg}`).join("; ");
  if (detail.message) {
    const missing = detail.missing_required_fields?.length ? ` (${detail.missing_required_fields.join(", ")})` : "";
    return detail.message + missing;
  }
  return fallback;
}
let toastTimer;
function toast(message, kind = "info") {
  const node = $("#toast");
  node.textContent = message;
  node.dataset.kind = kind;
  node.classList.add("show");
  window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => node.classList.remove("show"), kind === "error" ? 5000 : 2600);
}
const reportError = (error) => toast(error.message, "error");

// ---------------------------------------------------------------------------
// Field panel
// ---------------------------------------------------------------------------
function makeInput(path, value) {
  const input = document.createElement("input");
  input.id = `field-${path}`;
  input.dataset.path = path;
  input.value = value ?? "";
  input.placeholder = "—";
  input.autocomplete = "off";
  input.disabled = isApproved();
  input.setAttribute("aria-label", describePath(path));
  if (isNumericPath(path)) input.inputMode = "decimal";
  input.addEventListener("focus", () => selectField(path).catch(reportError));
  input.addEventListener("input", () => {
    setPath(review.order, path, input.value === "" ? null : input.value);
    confirmed.delete(path);
    updateFieldState(path);
  });
  input.addEventListener("keydown", (event) => {
    if (event.key !== "Enter" || event.isComposing || isApproved()) return;
    event.preventDefault();
    confirmField(path, input);
  });
  return input;
}
function makeField(path, value, { label = true, wide = false } = {}) {
  const field = document.createElement("div");
  field.className = `field${wide ? " wide" : ""}`;
  field.dataset.field = path;
  const input = makeInput(path, value);
  if (label) {
    const text = document.createElement("label");
    text.htmlFor = input.id;
    const dot = document.createElement("span");
    dot.className = "dot";
    text.append(dot, labelFor(path));
    if (isRequired(path)) text.append(Object.assign(document.createElement("span"), { className: "req", textContent: "*" }));
    field.append(text);
  } else {
    field.append(Object.assign(document.createElement("span"), { className: "dot" }));
  }
  field.append(input);
  field.addEventListener("click", (event) => { if (event.target !== input) input.focus(); });
  applyFieldState(field, path);
  return field;
}
function applyFieldState(field, path) {
  const state = fieldState(path);
  const source = review.sources[path];
  field.dataset.state = state;
  field.title = ["matched", "approx"].includes(state)
    ? `${STATE_TEXT[state]} (${Math.round(source.confidence * 100)}%)` : STATE_TEXT[state];
  field.classList.toggle("active", path === selectedPath);
}
function updateFieldState(path) {
  const field = document.querySelector(`[data-field="${CSS.escape(path)}"]`);
  if (field) applyFieldState(field, path);
  renderSummary();
}
function makeSection(key, title) {
  const section = document.createElement("details");
  section.className = "section";
  section.open = true;
  section.dataset.section = key;
  const summary = document.createElement("summary");
  summary.append(Object.assign(document.createElement("span"), { textContent: title }));
  summary.append(Object.assign(document.createElement("span"), { className: "section-count" }));
  section.append(summary);
  return section;
}
function renderFields() {
  const root = $("#field-list");
  const focusedPath = document.activeElement?.dataset?.path;
  const scrollTop = root.scrollTop;
  root.replaceChildren();
  for (const [key, title] of SECTIONS) {
    const value = review.order[key];
    const section = makeSection(key, title);
    if (key === "line_items") {
      section.append(renderItems());
    } else if (value == null) {
      if (!["ship_to", "bill_to"].includes(key)) continue;
      section.open = false;
      section.append(Object.assign(document.createElement("p"), { className: "section-empty", textContent: "No separate address in the document (the buyer address applies)." }));
    } else {
      const grid = document.createElement("div");
      grid.className = "field-grid";
      for (const field of visibleFields(value, key)) {
        grid.append(makeField(field.path, field.value, { wide: WIDE_FIELDS.has(field.path.split(".").at(-1)) }));
      }
      section.append(grid);
    }
    root.append(section);
  }
  root.scrollTop = scrollTop;
  if (focusedPath) document.getElementById(`field-${focusedPath}`)?.focus({ preventScroll: true });
  renderSummary();
}
function renderItems() {
  const wrapper = document.createElement("div");
  wrapper.className = "items";
  const scroller = document.createElement("div");
  scroller.className = "items-scroll";
  const table = document.createElement("table");
  table.className = "items-table";
  const head = table.createTHead().insertRow();
  for (const key of [...ITEM_MAIN, ""]) {
    const th = document.createElement("th");
    th.textContent = key ? LABELS[key] || humanize(key) : "";
    th.className = `col-${key || "actions"}`;
    head.append(th);
  }
  const body = table.createTBody();
  review.order.line_items.forEach((item, index) => {
    const row = body.insertRow();
    row.className = "item-row";
    for (const key of ITEM_MAIN) {
      const cell = row.insertCell();
      cell.className = `col-${key}`;
      cell.append(makeField(`line_items.${index}.${key}`, item[key], { label: false }));
    }
    const actions = row.insertCell();
    actions.className = "col-actions";
    const detail = body.insertRow();
    detail.className = "item-detail";
    const extras = ITEM_EXTRA.filter((key) => item[key] != null).length;
    detail.hidden = true;
    const more = Object.assign(document.createElement("button"), {
      type: "button", className: "icon-button", textContent: extras ? `+${extras}` : "⋯", title: "More fields",
    });
    more.addEventListener("click", () => { detail.hidden = !detail.hidden; more.classList.toggle("open", !detail.hidden); });
    actions.append(more);
    if (!isApproved()) {
      const remove = Object.assign(document.createElement("button"), { type: "button", className: "icon-button danger", textContent: "×", title: "Remove line item" });
      remove.addEventListener("click", () => removeLineItem(index));
      actions.append(remove);
    }
    const detailCell = detail.insertCell();
    detailCell.colSpan = ITEM_MAIN.length + 1;
    const grid = document.createElement("div");
    grid.className = "field-grid four";
    for (const key of ITEM_EXTRA) grid.append(makeField(`line_items.${index}.${key}`, item[key]));
    detailCell.append(grid);
  });
  scroller.append(table);
  wrapper.append(scroller);
  if (!review.order.line_items.length) {
    wrapper.append(Object.assign(document.createElement("p"), { className: "section-empty", textContent: "No line items detected." }));
  }
  if (!isApproved() && review.order.line_items.length < 200) {
    const add = Object.assign(document.createElement("button"), { type: "button", className: "add-item", textContent: "+ Add line item" });
    add.addEventListener("click", addLineItem);
    wrapper.append(add);
  }
  return wrapper;
}
function addLineItem() {
  const items = review.order.line_items;
  const positions = items.map((item) => Number(item.position)).filter(Number.isFinite);
  const step = positions.length > 1 ? positions.at(-1) - positions.at(-2) : 1;
  items.push({
    position: positions.length ? positions.at(-1) + (step > 0 ? step : 1) : 1, supplier_article_number: null,
    customer_article_number: null, description: null, quantity: null, unit_code: null, unit_raw: null,
    unit_price: null, discount_percent: null, line_total: null, requested_delivery_date: null,
  });
  renderFields();
  document.getElementById(`field-line_items.${items.length - 1}.description`)?.focus();
}
function removeLineItem(index) {
  review.order.line_items.splice(index, 1);
  savedOrder.line_items.splice(index, 1);
  const sources = {};
  for (const [path, source] of Object.entries(review.sources)) {
    const match = path.match(/^line_items\.(\d+)\.(.+)$/);
    if (!match) { sources[path] = source; continue; }
    const itemIndex = Number(match[1]);
    if (itemIndex === index) continue;
    sources[itemIndex > index ? `line_items.${itemIndex - 1}.${match[2]}` : path] = source;
  }
  review.sources = sources;
  savedOrder.__dirty = true;
  selectedPath = null;
  renderFields();
  redrawAllHighlights();
}
function issuePaths() {
  const paths = [];
  document.querySelectorAll("[data-field]").forEach((field) => {
    if (ISSUE_STATES.includes(field.dataset.state)) paths.push(field.dataset.field);
  });
  return paths;
}
function renderSummary() {
  const fields = [...document.querySelectorAll("[data-field]")];
  const count = (state) => fields.filter((field) => field.dataset.state === state).length;
  const summary = $("#summary");
  summary.replaceChildren();
  const chip = (text, kind) => Object.assign(document.createElement("span"), { className: `chip ${kind}`, textContent: text });
  summary.append(chip(`${count("matched")} found`, "ok"));
  const toCheck = count("unmatched") + count("missing") + count("approx");
  if (count("confirmed")) summary.append(chip(`${count("confirmed")} confirmed`, "ok"));
  if (toCheck) summary.append(chip(`${toCheck} to check`, "warn"));
  if (count("edited") || savedOrder.__dirty) summary.append(chip("unsaved", "edit"));
  document.querySelectorAll(".section").forEach((section) => {
    const open = [...section.querySelectorAll("[data-field]")].filter((field) => ISSUE_STATES.includes(field.dataset.state)).length;
    const counter = section.querySelector(".section-count");
    counter.textContent = open ? `${open} to check` : "";
  });
  renderAttention();
}
function renderAttention() {
  const attention = $("#attention");
  attention.replaceChildren();
  const warnings = review.order.validation?.warnings ?? [];
  const issues = issuePaths();
  if (issues.length) {
    const list = document.createElement("div");
    list.className = "issue-list";
    for (const path of issues.slice(0, 12)) {
      const state = document.querySelector(`[data-field="${CSS.escape(path)}"]`)?.dataset.state;
      const button = Object.assign(document.createElement("button"), { type: "button", className: `issue ${state}`, textContent: describePath(path) });
      button.addEventListener("click", () => focusField(path));
      list.append(button);
    }
    if (issues.length > 12) list.append(Object.assign(document.createElement("span"), { className: "issue-more", textContent: `+${issues.length - 12} more` }));
    attention.append(list);
  }
  if (warnings.length) {
    const details = document.createElement("details");
    details.className = "warnings";
    details.append(Object.assign(document.createElement("summary"), { textContent: `${warnings.length} extraction warning(s)` }));
    const ul = document.createElement("ul");
    warnings.forEach((warning) => ul.append(Object.assign(document.createElement("li"), { textContent: warning })));
    details.append(ul);
    attention.append(details);
  }
  attention.hidden = !attention.childElementCount;
}
function focusField(path) {
  const input = document.getElementById(`field-${path}`);
  if (!input) return;
  const detail = input.closest(".item-detail");
  if (detail) detail.hidden = false;
  input.closest("details.section")?.setAttribute("open", "");
  input.focus({ preventScroll: true });
  input.scrollIntoView({ block: "nearest", behavior: "smooth" });
}

// ---------------------------------------------------------------------------
// Locate a field in the PDF
// ---------------------------------------------------------------------------
function setHint(text) {
  const hint = $("#locate-hint");
  hint.textContent = text ?? "";
  hint.hidden = !text;
}
async function selectField(path, { scroll = true } = {}) {
  selectedPath = path;
  document.querySelectorAll(".field.active").forEach((field) => field.classList.remove("active"));
  document.querySelector(`[data-field="${CSS.escape(path)}"]`)?.classList.add("active");
  document.querySelectorAll(".source-rect.active").forEach((rect) => rect.classList.remove("active", "pulse"));
  const state = fieldState(path);
  const source = review.sources[path];
  if (state === "edited") { setHint("Changed value — save to locate it in the PDF"); return; }
  if (!source?.locations?.length) {
    const row = rowContext(path);
    if (row) {
      setHint(`${labelFor(path)}: ${row.note ?? "value not printed like this"} — showing its row`);
      await highlightPath(row.path, row.locations[0].page - 1, true);
      return;
    }
    const reason = { missing: "not detected", empty: "not detected", derived: "normalized value, not printed literally" }[state];
    setHint(`${labelFor(path)}: ${reason ?? "no matching text in the PDF"}`);
    return;
  }
  if (source.approximate) setHint(`Similar text (${Math.round(source.confidence * 100)}%) — check and press Enter to confirm`);
  else if (source.ambiguous) setHint(`Found ${source.occurrences}× in the PDF — showing the first occurrence`);
  else setHint(null);
  await highlightPath(path, source.locations[0].page - 1, scroll);
}
async function highlightPath(path, pageIndex, scroll) {
  const requested = selectedPath;
  await renderPdfPage(pageIndex);
  if (selectedPath !== requested) return;
  const rects = [...document.querySelectorAll(`.source-rect[data-path="${CSS.escape(path)}"]`)];
  rects.forEach((rect) => {
    rect.classList.add("active");
    rect.getBoundingClientRect(); // restart the pulse animation
    rect.classList.add("pulse");
  });
  if (scroll && rects.length) scrollToRects(rects.filter((rect) => rect.closest(".pdf-page")?.dataset.page === String(pageIndex + 1)));
}
/** A line item value without its own location (e.g. a corrected OCR value): use the item's row instead. */
function rowContext(path) {
  const match = path.match(/^line_items\.(\d+)\.(\w+)$/);
  if (!match) return null;
  const sibling = ["supplier_article_number", "position", "description", "quantity", "line_total"]
    .map((key) => `line_items.${match[1]}.${key}`)
    .find((candidate) => review.sources[candidate]?.locations?.length);
  if (!sibling) return null;
  const position = review.order.line_items[Number(match[1])]?.position;
  const keyword = { unit_price: "price", line_total: "total", quantity: "quantit" }[match[2]];
  const notes = (review.order.validation?.warnings ?? []).filter((warning) => warning.startsWith(`Position ${position}:`));
  const fitting = notes.find((warning) => keyword && warning.includes(keyword));
  const occurrences = review.sources[path]?.occurrences ?? 0;
  const note = fitting?.replace(/^Position \d+:\s*/, "").replace(/\.$/, "")
    ?? (occurrences > 1 ? `value appears ${occurrences}× nearby, position not unique` : null);
  return { path: sibling, locations: review.sources[sibling].locations, note };
}
function scrollToRects(rects) {
  if (!rects.length) return;
  const container = $("#pdf-pages");
  const boxes = rects.map((rect) => rect.getBoundingClientRect());
  const top = Math.min(...boxes.map((box) => box.top));
  const bottom = Math.max(...boxes.map((box) => box.bottom));
  const left = Math.min(...boxes.map((box) => box.left));
  const right = Math.max(...boxes.map((box) => box.right));
  const view = container.getBoundingClientRect();
  container.scrollTo({
    top: container.scrollTop + (top + bottom) / 2 - view.top - view.height / 2,
    left: container.scrollLeft + (left + right) / 2 - view.left - view.width / 2,
    behavior: "smooth",
  });
}

// ---------------------------------------------------------------------------
// PDF rendering
// ---------------------------------------------------------------------------
function pageScale(index) {
  const availableWidth = Math.max(100, $("#pdf-pages").clientWidth - 48);
  const fitScale = Math.min(1.5, availableWidth / review.page_sizes[index][0]);
  return zoom * fitScale;
}
async function renderPdf() {
  const generation = ++renderGeneration;
  const container = $("#pdf-pages");
  const scrollRatio = container.scrollHeight ? container.scrollTop / container.scrollHeight : 0;
  const pages = [];
  for (let index = 0; index < pdfDocument.numPages; index++) {
    const pdfPage = await pdfDocument.getPage(index + 1);
    if (generation !== renderGeneration) return;
    const scale = pageScale(Math.min(index, review.page_sizes.length - 1));
    const viewport = pdfPage.getViewport({ scale });
    const page = document.createElement("div");
    page.className = "pdf-page";
    page.style.width = `${Math.floor(viewport.width)}px`;
    page.style.height = `${Math.floor(viewport.height)}px`;
    page.dataset.page = index + 1;
    page.dataset.scale = scale;
    page.append(Object.assign(document.createElement("span"), { className: "pdf-page-number", textContent: `${index + 1} / ${pdfDocument.numPages}` }));
    pages.push(page);
  }
  pageObserver?.disconnect();
  pageRenders.clear();
  container.replaceChildren(...pages);
  container.scrollTop = scrollRatio * container.scrollHeight;
  pageObserver = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      pageObserver.unobserve(entry.target);
      renderPdfPage(Number(entry.target.dataset.page) - 1).catch(reportError);
    });
  }, { root: container, rootMargin: "800px 0px" });
  pages.forEach((page) => pageObserver.observe(page));
}
function renderPdfPage(index) {
  if (!pageRenders.has(index)) pageRenders.set(index, drawPdfPage(index));
  return pageRenders.get(index);
}
async function drawPdfPage(index) {
  const page = document.querySelector(`.pdf-page[data-page="${index + 1}"]`);
  if (!page) return;
  const pdfPage = await pdfDocument.getPage(index + 1);
  const viewport = pdfPage.getViewport({ scale: Number(page.dataset.scale) });
  const canvas = document.createElement("canvas");
  const pixelRatio = window.devicePixelRatio || 1;
  canvas.width = Math.floor(viewport.width * pixelRatio);
  canvas.height = Math.floor(viewport.height * pixelRatio);
  page.style.setProperty("--total-scale-factor", viewport.scale);
  const textLayer = document.createElement("div");
  textLayer.className = "textLayer";
  const svg = document.createElementNS(NS, "svg");
  svg.classList.add("highlights");
  svg.setAttribute("viewBox", `0 0 ${viewport.width} ${viewport.height}`);
  const size = review.page_sizes[index] ?? [viewport.width / viewport.scale, viewport.height / viewport.scale];
  page.dataset.scaleX = viewport.width / size[0];
  page.dataset.scaleY = viewport.height / size[1];
  page.prepend(canvas, textLayer, svg);
  drawHighlights(page);
  const textContent = await pdfPage.getTextContent();
  if (!page.isConnected) return;
  const renderedText = new pdfjsLib.TextLayer({ textContentSource: textContent, container: textLayer, viewport });
  await Promise.all([
    pdfPage.render({
      canvasContext: canvas.getContext("2d", { alpha: false }), viewport,
      transform: pixelRatio === 1 ? null : [pixelRatio, 0, 0, pixelRatio, 0, 0],
    }).promise,
    renderedText.render(),
  ]);
  applySearchHighlights();
}
function drawHighlights(page) {
  const svg = page.querySelector("svg.highlights");
  if (!svg) return;
  const pageNumber = Number(page.dataset.page);
  const scaleX = Number(page.dataset.scaleX);
  const scaleY = Number(page.dataset.scaleY);
  const pad = 2;
  svg.replaceChildren();
  for (const [path, source] of Object.entries(review.sources)) {
    for (const location of source.locations ?? []) {
      if (location.page !== pageNumber) continue;
      const rect = document.createElementNS(NS, "rect");
      rect.setAttribute("x", location.x0 * scaleX - pad);
      rect.setAttribute("y", location.y0 * scaleY - pad);
      rect.setAttribute("width", Math.max(1, (location.x1 - location.x0) * scaleX) + pad * 2);
      rect.setAttribute("height", Math.max(1, (location.y1 - location.y0) * scaleY) + pad * 2);
      rect.setAttribute("rx", 3);
      rect.classList.add("source-rect");
      if (source.approximate) rect.classList.add("approx");
      if (path === selectedPath) rect.classList.add("active");
      rect.dataset.path = path;
      const title = document.createElementNS(NS, "title");
      title.textContent = describePath(path);
      rect.append(title);
      rect.addEventListener("click", () => focusField(path));
      svg.append(rect);
    }
  }
}
const redrawAllHighlights = () => document.querySelectorAll(".pdf-page").forEach(drawHighlights);

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------
function applySearchHighlights() {
  const query = $("#pdf-search").value.trim().toLocaleLowerCase();
  document.querySelectorAll(".textLayer .search-hit").forEach((span) => span.classList.remove("search-hit", "current"));
  if (!query) return;
  document.querySelectorAll(".pdf-page").forEach((page) => {
    page.querySelectorAll(".textLayer span").forEach((span) => {
      if (span.textContent.toLocaleLowerCase().includes(query)) {
        span.classList.add("search-hit");
        if (searchMatches[currentSearchMatch]?.index + 1 === Number(page.dataset.page)) span.classList.add("current");
      }
    });
  });
}
async function searchPdf(direction = 1) {
  const query = $("#pdf-search").value.trim().toLocaleLowerCase();
  if (!query) {
    searchMatches = [];
    currentSearchMatch = -1;
    lastSearchQuery = "";
    $("#search-count").textContent = "";
    applySearchHighlights();
    return;
  }
  if (query !== lastSearchQuery) {
    searchMatches = [];
    currentSearchMatch = -1;
    lastSearchQuery = query;
    for (let index = 0; index < pdfDocument.numPages; index++) {
      const page = await pdfDocument.getPage(index + 1);
      const content = await page.getTextContent();
      const matches = content.items.filter((item) => "str" in item && item.str.toLocaleLowerCase().includes(query));
      if (matches.length) searchMatches.push({ index, count: matches.length });
    }
  }
  if (!searchMatches.length) {
    $("#search-count").textContent = "0";
    applySearchHighlights();
    return;
  }
  currentSearchMatch = (currentSearchMatch + direction + searchMatches.length) % searchMatches.length;
  $("#search-count").textContent = `${currentSearchMatch + 1}/${searchMatches.length}`;
  const target = searchMatches[currentSearchMatch];
  await renderPdfPage(target.index);
  applySearchHighlights();
  const hit = document.querySelector(`.pdf-page[data-page="${target.index + 1}"] .search-hit`);
  if (hit) scrollToRects([hit]);
  else document.querySelector(`.pdf-page[data-page="${target.index + 1}"]`)?.scrollIntoView({ block: "start", behavior: "smooth" });
}

// ---------------------------------------------------------------------------
// Load, save, approve
// ---------------------------------------------------------------------------
function renderRunInfo() {
  const meta = review.order.meta ?? {};
  const box = $("#run-info");
  box.hidden = !meta.model;
  if (!meta.model) return;
  const seconds = (value) => (value == null ? "–" : `${Number(value).toFixed(1)} s`);
  const number = (value) => (value == null ? "–" : Number(value).toLocaleString());
  const showParams = meta.model_params && !meta.model.toLowerCase().includes(meta.model_params.toLowerCase());
  const model = showParams ? `${meta.model} · ${meta.model_params}` : meta.model;
  $("#run-summary").textContent = `${model} — ${seconds(meta.duration_s)}`
    + (meta.completion_tokens_per_s ? ` — ${meta.completion_tokens_per_s} tok/s` : "");
  const ocr = meta.ocr_pages?.length ? `${meta.ocr_pages.length} of ${meta.page_count} pages OCR` : `${meta.page_count} page(s), text layer`;
  const rows = [
    ["Model", model],
    ["Model file", meta.model_file ? `${meta.model_file}${meta.model_size_mb ? ` (${number(Math.round(meta.model_size_mb))} MB)` : ""}` : null],
    ["Context", meta.context_size ? `${number(meta.context_size)} tokens` : null],
    ["Total time", seconds(meta.duration_s)],
    ["Text extraction", `${seconds(meta.text_extraction_s)} · ${ocr}`],
    ["LLM", `${seconds(meta.llm_s)} · ${meta.llm_calls} call(s) · ${meta.llm_chunks} chunk(s)`],
    ["Prompt", `${number(meta.prompt_tokens)} tokens${meta.cached_prompt_tokens ? ` (+${number(meta.cached_prompt_tokens)} cached)` : ""}`
      + (meta.prompt_tokens_per_s ? ` · ${meta.prompt_tokens_per_s} tok/s` : "")],
    ["Generated", `${number(meta.completion_tokens)} tokens` + (meta.completion_tokens_per_s ? ` · ${meta.completion_tokens_per_s} tok/s` : "")],
    ["Line items from", { table_rows: "table row parser (LLM fills gaps)", llm: "LLM" }[meta.item_source] ?? null],
    ["Text size", `${number(meta.text_chars)} characters`],
  ];
  const list = $("#run-details");
  list.replaceChildren();
  for (const [label, value] of rows) {
    if (value == null) continue;
    list.append(Object.assign(document.createElement("dt"), { textContent: label }), Object.assign(document.createElement("dd"), { textContent: value }));
  }
}
function applyStatus() {
  const status = $("#status");
  status.textContent = review.status.toUpperCase();
  status.dataset.status = review.status;
  $("#approve-button").hidden = isApproved();
  $("#save-button").hidden = isApproved();
  $("#export-link").hidden = !isApproved();
  $("#export-link").href = `/api/reviews/${reviewId}/export`;
}
async function loadReview() {
  const response = await fetch(`/api/reviews/${reviewId}`);
  if (!response.ok) throw new Error(response.status === 404 ? "This review was not found." : "Could not load this review.");
  review = await response.json();
  savedOrder = structuredClone(review.order);
  syncConfirmed();
  pdfjsLib = await import("/static/vendor/pdfjs/pdf.mjs");
  pdfjsLib.GlobalWorkerOptions.workerSrc = "/static/vendor/pdfjs/pdf.worker.mjs";
  const pdfResponse = await fetch(`/api/reviews/${reviewId}/pdf`);
  if (!pdfResponse.ok) throw new Error("Could not load the source PDF.");
  pdfDocument = await pdfjsLib.getDocument({
    data: new Uint8Array(await pdfResponse.arrayBuffer()),
    cMapUrl: "/static/vendor/pdfjs/cmaps/",
    cMapPacked: true,
    standardFontDataUrl: "/static/vendor/pdfjs/standard_fonts/",
    wasmUrl: "/static/vendor/pdfjs/wasm/",
  }).promise;
  $("#upload-screen").hidden = true;
  $("#review-screen").hidden = false;
  document.body.classList.add("reviewing");
  renderedWidth = Math.round($("#pdf-pages").getBoundingClientRect().width);
  $("#document-title").textContent = review.order.document.order_number || "Review extracted data";
  $("#pdf-name").textContent = review.order.meta.source_filename || "Original PDF";
  applyStatus();
  renderRunInfo();
  renderFields();
  await renderPdf();
}
function collectOrder() {
  const order = structuredClone(review.order);
  const language = order.document.language;
  document.querySelectorAll("#field-list input[data-path]").forEach((input) => {
    const path = input.dataset.path;
    const raw = input.value.trim();
    const saved = getPath(savedOrder, path);
    const numeric = typeof saved === "number" || isNumericPath(path);
    let value = raw === "" ? null : raw;
    if (numeric && value !== null) {
      value = parseNumberInput(raw, language);
      if (Number.isNaN(value)) { focusField(path); throw new Error(`Enter a valid number for ${describePath(path)}.`); }
    }
    if (path.endsWith(".position") && (value === null || !Number.isInteger(value))) {
      focusField(path);
      throw new Error(`${describePath(path)} must be a whole number.`);
    }
    setPath(order, path, value);
  });
  return order;
}
function syncConfirmed() {
  confirmed = new Set(Object.entries(review.sources).filter(([, source]) => source.confirmed).map(([path]) => path));
}
async function save({ quiet = false } = {}) {
  const order = collectOrder();
  const sentRaw = new Map([...document.querySelectorAll("#field-list input[data-path]")].map((input) => [input.dataset.path, input.value]));
  const response = await fetch(`/api/reviews/${reviewId}`, {
    method: "PUT", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ order, confirmed: [...confirmed] }),
  });
  const result = await readJson(response);
  if (!response.ok) throw new Error(errorMessage(result, "Could not save corrections."));
  // keep what was typed or confirmed while the request was running
  const typedSince = [...document.querySelectorAll("#field-list input[data-path]")]
    .filter((input) => sentRaw.has(input.dataset.path) && sentRaw.get(input.dataset.path) !== input.value);
  const pendingConfirmations = [...confirmed];
  review = result;
  savedOrder = structuredClone(result.order);
  syncConfirmed();
  typedSince.forEach((input) => setPath(review.order, input.dataset.path, input.value === "" ? null : input.value));
  pendingConfirmations.forEach((path) => { if (review.sources[path]) confirmed.add(path); });
  applyStatus();
  renderFields();
  redrawAllHighlights();
  if (selectedPath) await selectField(selectedPath, { scroll: false });
  if (!quiet) toast("Corrections saved");
}
const queueSave = (options) => (saveChain = saveChain.then(() => save(options)).catch(reportError));
function confirmField(path, input) {
  if (input.value.trim() === "" && isRequired(path)) {
    toast(`${describePath(path)} is required — enter a value first.`, "error");
    return;
  }
  confirmed.add(path);
  updateFieldState(path);
  focusNextField(path);
  queueSave({ quiet: true });
}
function focusNextField(path) {
  const inputs = [...document.querySelectorAll("#field-list input[data-path]")].filter((input) => input.offsetParent !== null);
  const index = inputs.findIndex((input) => input.dataset.path === path);
  const after = inputs.slice(index + 1);
  const next = after.find((input) => ISSUE_STATES.includes(input.closest("[data-field]")?.dataset.state)) ?? after[0];
  if (next) {
    next.focus({ preventScroll: true });
    next.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }
}
async function approve() {
  await save();
  const warnings = review.order.validation?.warnings ?? [];
  const unmatched = review.unmatched_fields?.length ?? 0;
  if ((warnings.length || unmatched)
      && !window.confirm(`${warnings.length} extraction warning(s) and ${unmatched} value(s) without a source match remain.\n\nApprove anyway?`)) return;
  const response = await fetch(`/api/reviews/${reviewId}/approve`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ acknowledge_warnings: true }),
  });
  const result = await readJson(response);
  if (!response.ok) throw new Error(errorMessage(result, "Could not approve this order."));
  review = result;
  savedOrder = structuredClone(result.order);
  syncConfirmed();
  applyStatus();
  renderFields();
  toast("Approved — ERP export is ready");
}

// ---------------------------------------------------------------------------
// Wiring
// ---------------------------------------------------------------------------
$("#pdf-file").addEventListener("change", (event) => {
  $("#file-name").textContent = event.target.files[0]?.name || "PDF up to 25 MB";
});
$("#upload-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = event.currentTarget.querySelector("button");
  const error = $("#upload-error");
  button.disabled = true;
  button.textContent = "Extracting locally…";
  error.textContent = "Extraction can take a little while for scanned documents.";
  try {
    const form = new FormData(event.currentTarget);
    form.set("file", $("#pdf-file").files[0]);
    const response = await fetch("/api/reviews", { method: "POST", body: form });
    const result = await readJson(response);
    if (!response.ok) throw new Error(errorMessage(result, "Could not extract this PDF."));
    location.href = `/review/${result.id}`;
  } catch (problem) {
    error.textContent = problem.message;
    button.disabled = false;
    button.innerHTML = "Extract and start review <span>→</span>";
  }
});
$("#save-button").addEventListener("click", () => queueSave());
$("#approve-button").addEventListener("click", () => approve().catch(reportError));
const setZoom = (value) => {
  zoom = Math.min(2.5, Math.max(0.5, value));
  $("#zoom-value").textContent = `${Math.round(zoom * 100)}%`;
  renderPdf().catch(reportError);
};
$("#zoom-in").addEventListener("click", () => setZoom(zoom + 0.2));
$("#zoom-out").addEventListener("click", () => setZoom(zoom - 0.2));
// Re-render when the PDF column changes width (window resize, scrollbars, panel layout)
let resizeTimer;
let renderedWidth = 0;
new ResizeObserver(([entry]) => {
  const width = Math.round(entry.contentRect.width);
  if (!pdfDocument || Math.abs(width - renderedWidth) < 2) return;
  window.clearTimeout(resizeTimer);
  resizeTimer = window.setTimeout(() => { renderedWidth = width; renderPdf().catch(reportError); }, 150);
}).observe($("#pdf-pages"));
$("#pdf-search").addEventListener("keydown", (event) => {
  if (event.key === "Enter") searchPdf(event.shiftKey ? -1 : 1).catch(reportError);
  if (event.key === "Escape") {
    $("#pdf-search").value = "";
    searchPdf().catch(reportError);
  }
});
$("#search-next").addEventListener("click", () => searchPdf(1).catch(reportError));
$("#search-prev").addEventListener("click", () => searchPdf(-1).catch(reportError));
document.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "s" && reviewId) {
    event.preventDefault();
    if (!isApproved()) queueSave();
  }
});
window.addEventListener("beforeunload", (event) => {
  if (review && !isApproved() && (savedOrder.__dirty || document.querySelector('[data-state="edited"]'))) event.preventDefault();
});
// ---------------------------------------------------------------------------
// Start page: model in use
// ---------------------------------------------------------------------------
async function loadModelInfo() {
  let info;
  try {
    const response = await fetch("/api/model");
    info = await response.json();
  } catch {
    info = { status: "offline" };
  }
  const card = $("#model-card");
  card.dataset.status = info.status;
  const params = info.model_params && !String(info.name).toLowerCase().includes(info.model_params.toLowerCase()) ? ` · ${info.model_params}` : "";
  $("#model-name").textContent = info.name ? `${info.name}${params}` : "Unknown model";
  // configured alias (e.g. "qwen3.5-2b") vs. what llama-server actually loaded (e.g. 0.8B)
  const configured = Number(String(info.name).match(/(\d+(?:\.\d+)?)b\b/i)?.[1]);
  const loaded = Number(String(info.model_params ?? "").match(/[\d.]+/)?.[0]);
  const mismatch = configured && loaded && Math.abs(configured - loaded) / configured > 0.3;
  card.dataset.mismatch = mismatch ? "true" : "false";
  $("#model-status").textContent = mismatch
    ? `Configured ${info.name}, but llama-server has a ${info.model_params} model loaded – restart llama-server after changing the model.`
    : {
    ready: "Ready – runs locally on this machine",
    loading: "Loading the model … (first start downloads it)",
    offline: "llama-server not reachable – extraction is unavailable",
  }[info.status] ?? "";
  const quant = info.model_file?.match(/[-_.]((?:UD-)?(?:I?Q\d\w*|BF16|F16|F32))\.gguf$/i)?.[1];
  const rows = [
    ["File", info.model_file],
    ["Quantization", quant],
    ["Size", info.model_size_mb ? `${Math.round(info.model_size_mb).toLocaleString()} MB` : null],
    ["Context", info.context_size ? `${info.context_size.toLocaleString()} tokens` : null],
    ["OCR", info.ocr_languages],
  ];
  const list = $("#model-details");
  list.replaceChildren();
  for (const [label, value] of rows) {
    if (!value) continue;
    list.append(Object.assign(document.createElement("dt"), { textContent: label }), Object.assign(document.createElement("dd"), { textContent: value }));
  }
  if (info.status !== "ready" && !reviewId) window.setTimeout(loadModelInfo, 5000);
}

reviewId = location.pathname.match(/^\/review\/([0-9a-f-]+)\/?$/i)?.[1];
if (reviewId) loadReview().catch((error) => { $("#status").textContent = "REVIEW UNAVAILABLE"; reportError(error); });
else { $("#status").textContent = "READY"; $("#upload-screen").hidden = false; loadModelInfo(); }
