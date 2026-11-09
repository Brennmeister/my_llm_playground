const palette = ["#25816a", "#d18c26", "#527ec1", "#b15e78", "#805db0", "#4197a0", "#c26438", "#658744"];
const pathColor = new Map();
let reviewId;
let review;
let zoom = 1;
let savedOrder;
let pdfjsLib;
let pdfDocument;
let pageObserver;
let searchMatches = [];
let currentSearchMatch = -1;
let lastSearchQuery = "";
let selectedPath = null;

const $ = (selector) => document.querySelector(selector);
const setPath = (object, path, value) => {
  const parts = path.split(".");
  const key = parts.pop();
  const parent = parts.reduce((node, part) => node[Array.isArray(node) ? Number(part) : part], object);
  parent[Array.isArray(parent) ? Number(key) : key] = value;
};
const colorFor = (path) => {
  if (!pathColor.has(path)) pathColor.set(path, palette[pathColor.size % palette.length]);
  return pathColor.get(path);
};
const labelFor = (path) => path.split(".").at(-1).replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
const numericFields = new Set(["position", "quantity", "unit_price", "discount_percent", "line_total", "net_amount", "tax_rate_percent", "tax_amount", "gross_amount", "line_items_sum"]);
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
const visibleFields = (value, path = "") => {
  const fields = [];
  for (const [key, child] of Object.entries(value ?? {})) {
    if (["meta", "validation", "schema_version", "document_type"].includes(key)) continue;
    const childPath = path ? `${path}.${key}` : key;
    if (child && typeof child === "object" && !Array.isArray(child)) fields.push(...visibleFields(child, childPath));
    else if (Array.isArray(child)) child.forEach((item, index) => {
      if (item && typeof item === "object") fields.push(...visibleFields(item, `${childPath}.${index}`));
      else fields.push({ path: `${childPath}.${index}`, value: item });
    });
    else fields.push({ path: childPath, value: child });
  }
  return fields;
};
function toast(message) {
  const node = $("#toast");
  node.textContent = message;
  node.classList.add("show");
  window.setTimeout(() => node.classList.remove("show"), 2600);
}
function renderFields() {
  const root = $("#field-list");
  root.replaceChildren();
  const fields = visibleFields(review.order);
  $("#field-count").textContent = `${fields.length} FIELDS`;
  let group = "";
  fields.forEach(({ path, value }) => {
    const parts = path.split(".");
    const nextGroup = parts[0] === "line_items" ? `Line item ${Number(parts[1]) + 1}` : labelFor(parts[0]);
    if (group !== nextGroup) {
      group = nextGroup;
      const heading = document.createElement("h2");
      heading.className = "group-title";
      heading.textContent = group;
      root.append(heading);
    }
    const source = review.sources[path];
    const card = document.createElement("div");
    card.className = `field-card${source?.locations?.length ? "" : " unmatched"}`;
    card.dataset.path = path;
    card.style.setProperty("--field-color", colorFor(path));
    card.tabIndex = 0;
    const label = document.createElement("label");
    label.className = "field-label";
    label.htmlFor = `field-${path}`;
    const dot = document.createElement("span");
    dot.className = "match";
    label.append(dot, document.createTextNode(labelFor(path)));
    const input = document.createElement("input");
    input.id = `field-${path}`;
    input.dataset.path = path;
    input.value = value ?? "";
    input.placeholder = "Not detected";
    input.disabled = review.status === "approved";
    input.addEventListener("focus", () => selectField(path).catch((error) => toast(error.message)));
    input.addEventListener("mouseenter", () => selectField(path).catch((error) => toast(error.message)));
    const note = document.createElement("div");
    note.className = `field-note${source?.locations?.length ? " matched" : ""}`;
    note.textContent = source?.locations?.length
      ? `Source matched · ${Math.round(source.confidence * 100)}%`
      : value == null ? "Missing · please verify" : "Source not matched · please verify";
    card.append(label, input, note);
    card.addEventListener("mouseenter", () => selectField(path).catch((error) => toast(error.message)));
    root.append(card);
  });
  if (review.order.line_items.length < 60) {
    const add = document.createElement("button");
    add.className = "secondary";
    add.textContent = "+ Add line item";
    add.hidden = review.status === "approved";
    add.addEventListener("click", () => {
      review.order.line_items.push({
        position: review.order.line_items.length + 1, supplier_article_number: null, customer_article_number: null,
        description: null, quantity: null, unit_code: null, unit_raw: null, unit_price: null,
        discount_percent: null, line_total: null, requested_delivery_date: null,
      });
      renderFields();
      $("#field-list").lastElementChild?.scrollIntoView({ block: "nearest" });
    });
    root.append(add);
  }
}
async function selectField(path) {
  selectedPath = path;
  document.querySelectorAll(".source-rect.active").forEach((rect) => rect.classList.remove("active"));
  const source = review.sources[path];
  if (!source?.locations?.length) return;
  const firstPage = source.locations[0].page;
  const page = document.querySelector(`.pdf-page[data-page="${firstPage}"]`);
  page?.scrollIntoView({ block: "center", behavior: "smooth" });
  await renderPdfPage(firstPage - 1);
  source.locations.forEach((location) => {
    const rect = document.querySelector(`[data-word-id="${CSS.escape(location.word_id)}"]`);
    rect?.classList.add("active");
  });
}
function pageScale(index) {
  const availableWidth = Math.max(100, $("#pdf-pages").clientWidth - 48);
  const fitScale = Math.min(96 / 72, availableWidth / review.page_sizes[index][0]);
  return zoom * fitScale;
}
async function renderPdf() {
  const pages = $("#pdf-pages");
  pages.replaceChildren();
  pageObserver?.disconnect();
  pageObserver = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        pageObserver.unobserve(entry.target);
        renderPdfPage(Number(entry.target.dataset.page) - 1).catch((error) => toast(error.message));
      }
    });
  }, { root: pages, rootMargin: "800px 0px" });
  for (let index = 0; index < pdfDocument.numPages; index++) {
    const pdfPage = await pdfDocument.getPage(index + 1);
    const viewport = pdfPage.getViewport({ scale: pageScale(index) });
    const pageNumber = index + 1;
    const page = document.createElement("div");
    page.className = "pdf-page";
    page.style.width = `${Math.ceil(viewport.width)}px`;
    page.style.height = `${Math.ceil(viewport.height)}px`;
    page.dataset.page = pageNumber;
    const badge = document.createElement("span");
    badge.className = "pdf-page-number";
    badge.textContent = `Page ${pageNumber}`;
    page.append(badge);
    pages.append(page);
    pageObserver.observe(page);
  }
}
async function renderPdfPage(index) {
  const page = document.querySelector(`.pdf-page[data-page="${index + 1}"]`);
  if (!page || page.dataset.rendered === "true") return;
  page.dataset.rendered = "true";
  const pdfPage = await pdfDocument.getPage(index + 1);
  const scale = pageScale(index);
  const viewport = pdfPage.getViewport({ scale });
  const canvas = document.createElement("canvas");
  const context = canvas.getContext("2d", { alpha: false });
  const pixelRatio = window.devicePixelRatio || 1;
  canvas.width = Math.ceil(viewport.width * pixelRatio);
  canvas.height = Math.ceil(viewport.height * pixelRatio);
  canvas.style.width = `${viewport.width}px`;
  canvas.style.height = `${viewport.height}px`;
  page.style.width = `${viewport.width}px`;
  page.style.height = `${viewport.height}px`;
  page.style.setProperty("--total-scale-factor", scale);
  const textLayer = document.createElement("div");
  textLayer.className = "textLayer";
  textLayer.style.width = `${viewport.width}px`;
  textLayer.style.height = `${viewport.height}px`;
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", `0 0 ${viewport.width} ${viewport.height}`);
  const scaleX = viewport.width / review.page_sizes[index][0];
  const scaleY = viewport.height / review.page_sizes[index][1];
  Object.entries(review.sources).forEach(([path, source]) => {
    source.locations?.filter((location) => location.page === index + 1).forEach((location) => {
      const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      rect.setAttribute("x", location.x0 * scaleX);
      rect.setAttribute("y", location.y0 * scaleY);
      rect.setAttribute("width", Math.max(1, (location.x1 - location.x0) * scaleX));
      rect.setAttribute("height", Math.max(1, (location.y1 - location.y0) * scaleY));
      rect.classList.add("source-rect");
      rect.dataset.wordId = location.word_id;
      rect.style.setProperty("--field-color", colorFor(path));
      if (path === selectedPath) rect.classList.add("active");
      rect.setAttribute("aria-label", labelFor(path));
      rect.addEventListener("click", () => {
        document.querySelector(`[data-path="${CSS.escape(path)}"]`)?.focus();
        selectField(path).catch((error) => toast(error.message));
      });
      svg.append(rect);
    });
  });
  const textContent = await pdfPage.getTextContent();
  const renderedText = new pdfjsLib.TextLayer({ textContentSource: textContent, container: textLayer, viewport });
  page.prepend(canvas, textLayer, svg);
  await Promise.all([
    pdfPage.render({ canvasContext: context, viewport, transform: pixelRatio === 1 ? null : [pixelRatio, 0, 0, pixelRatio, 0, 0] }).promise,
    renderedText.render(),
  ]);
  applySearchHighlights();
}
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
  const pageElement = document.querySelector(`.pdf-page[data-page="${target.index + 1}"]`);
  pageElement?.scrollIntoView({ block: "start", behavior: "smooth" });
  await renderPdfPage(target.index);
  applySearchHighlights();
}
function renderAttention() {
  const attention = $("#attention");
  const missing = review.order.validation?.missing_required_fields ?? [];
  const warnings = review.order.validation?.warnings ?? [];
  const unmatched = Object.entries(review.sources).filter(([, source]) => !source.locations?.length).map(([path]) => path);
  const messages = [];
  if (missing.length) messages.push(`Missing required: ${missing.map(labelFor).join(", ")}.`);
  if (unmatched.length) messages.push(`${unmatched.length} value(s) have no confirmed source match.`);
  if (warnings.length) messages.push(...warnings.slice(0, 3));
  attention.hidden = !messages.length;
  attention.textContent = messages.join(" ");
}
async function loadReview() {
  const response = await fetch(`/api/reviews/${reviewId}`);
  if (!response.ok) throw new Error(response.status === 404 ? "This review was not found." : "Could not load this review.");
  review = await response.json();
  savedOrder = structuredClone(review.order);
  pdfjsLib = await import("/static/vendor/pdfjs/pdf.mjs");
  pdfjsLib.GlobalWorkerOptions.workerSrc = "/static/vendor/pdfjs/pdf.worker.mjs";
  const pdfResponse = await fetch(`/api/reviews/${reviewId}/pdf`);
  if (!pdfResponse.ok) throw new Error("Could not load the source PDF.");
  const pdfData = new Uint8Array(await pdfResponse.arrayBuffer());
  pdfDocument = await pdfjsLib.getDocument({
    data: pdfData,
    cMapUrl: "/static/vendor/pdfjs/cmaps/",
    cMapPacked: true,
    standardFontDataUrl: "/static/vendor/pdfjs/standard_fonts/",
    wasmUrl: "/static/vendor/pdfjs/wasm/",
  }).promise;
  $("#upload-screen").hidden = true;
  $("#review-screen").hidden = false;
  $("#status").textContent = review.status.toUpperCase();
  $("#document-title").textContent = review.order.document.order_number || "Review extracted data";
  $("#pdf-name").textContent = review.order.meta.source_filename || "Original PDF";
  $("#approve-button").hidden = review.status === "approved";
  $("#save-button").hidden = review.status === "approved";
  $("#export-link").hidden = review.status !== "approved";
  $("#export-link").href = `/api/reviews/${reviewId}/export`;
  renderFields();
  await renderPdf();
  renderAttention();
}
async function save() {
  const order = structuredClone(review.order);
  document.querySelectorAll("#field-list input[data-path]").forEach((input) => {
    const original = input.value;
    const current = input.dataset.path.split(".").reduce((node, part) => node == null ? undefined : node[Array.isArray(node) ? Number(part) : part], savedOrder);
    const numeric = typeof current === "number" || numericFields.has(input.dataset.path.split(".").at(-1));
    const parsed = numeric && original !== "" ? parseNumberInput(original, order.document.language) : original;
    if (Number.isNaN(parsed)) throw new Error(`Enter a valid number for ${labelFor(input.dataset.path)}.`);
    setPath(order, input.dataset.path, original === "" ? null : numeric ? parsed : original);
  });
  const response = await fetch(`/api/reviews/${reviewId}`, {
    method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ order }),
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.detail || "Could not save corrections.");
  review = result;
  savedOrder = structuredClone(result.order);
  $("#status").textContent = review.status.toUpperCase();
  renderAttention();
  toast("Corrections saved");
}
$("#pdf-file").addEventListener("change", (event) => {
  $("#file-name").textContent = event.target.files[0]?.name || "PDF up to 25 MB";
});
$("#upload-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = event.currentTarget.querySelector("button");
  const error = $("#upload-error");
  button.disabled = true;
  button.innerHTML = "Extracting locally…";
  error.textContent = "Extraction can take a little while for scanned documents.";
  try {
    const form = new FormData(event.currentTarget);
    form.set("file", $("#pdf-file").files[0]);
    const response = await fetch("/api/reviews", { method: "POST", body: form });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || "Could not extract this PDF.");
    location.href = `/review/${result.id}`;
  } catch (problem) {
    error.textContent = problem.message;
    button.disabled = false;
    button.innerHTML = "Extract and start review <span>→</span>";
  }
});
$("#save-button").addEventListener("click", async () => {
  try { await save(); } catch (error) { toast(error.message); }
});
$("#approve-button").addEventListener("click", async () => {
  try {
    await save();
    const warnings = review.order.validation?.warnings ?? [];
    const unmatched = Object.values(review.sources).filter((source) => !source.locations?.length).length;
    if ((warnings.length || unmatched) && !window.confirm(`${warnings.length} extraction warning(s) and ${unmatched} unmatched source value(s) remain. Approve anyway?`)) return;
    const response = await fetch(`/api/reviews/${reviewId}/approve`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ acknowledge_warnings: true }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || "Could not approve this order.");
    review = result;
    $("#status").textContent = "APPROVED";
    $("#approve-button").hidden = true;
    $("#export-link").hidden = false;
    toast("Approved — ERP export is ready");
  } catch (error) { toast(error.message); }
});
$("#zoom-in").addEventListener("click", async () => { zoom = Math.min(1.8, zoom + 0.15); $("#zoom-value").textContent = `${Math.round(zoom * 100)}%`; await renderPdf(); });
$("#zoom-out").addEventListener("click", async () => { zoom = Math.max(0.55, zoom - 0.15); $("#zoom-value").textContent = `${Math.round(zoom * 100)}%`; await renderPdf(); });
window.addEventListener("resize", () => {
  if (pdfDocument) renderPdf().catch((error) => toast(error.message));
});
$("#pdf-search").addEventListener("keydown", (event) => {
  if (event.key === "Enter") searchPdf(1).catch((error) => toast(error.message));
  if (event.key === "Escape") {
    $("#pdf-search").value = "";
    searchPdf().catch((error) => toast(error.message));
  }
});
$("#search-next").addEventListener("click", () => searchPdf(1).catch((error) => toast(error.message)));
$("#search-prev").addEventListener("click", () => searchPdf(-1).catch((error) => toast(error.message)));
document.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "s" && reviewId) {
    event.preventDefault();
    $("#save-button").click();
  }
});
reviewId = location.pathname.match(/^\/review\/([0-9a-f-]+)\/?$/i)?.[1];
if (reviewId) loadReview().catch((error) => { $("#status").textContent = "REVIEW UNAVAILABLE"; toast(error.message); });
else { $("#status").textContent = "READY"; $("#upload-screen").hidden = false; }
