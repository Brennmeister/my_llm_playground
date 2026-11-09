#!/bin/sh
# Installs the pinned PDF.js browser assets into app/static/vendor/pdfjs.
# POSIX sh on purpose: also runs in the node:alpine build stage (no bash there).
set -eu
cd "$(dirname "$0")/.."

npm ci --omit=optional --ignore-scripts --no-audit --no-fund
SRC="node_modules/pdfjs-dist"
DEST="app/static/vendor/pdfjs"
rm -rf "$DEST"
mkdir -p "$DEST"
cp "$SRC/build/pdf.mjs" "$SRC/build/pdf.worker.mjs" "$SRC/LICENSE" "$DEST/"
for dir in cmaps standard_fonts wasm; do
  cp -R "$SRC/$dir" "$DEST/"
done
