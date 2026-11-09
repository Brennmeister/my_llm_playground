#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

npm ci --omit=optional --ignore-scripts --no-audit --no-fund
DEST="app/static/vendor/pdfjs"
mkdir -p "$DEST"
cp node_modules/pdfjs-dist/build/pdf.mjs "$DEST/"
cp node_modules/pdfjs-dist/build/pdf.worker.mjs "$DEST/"
cp -R node_modules/pdfjs-dist/{cmaps,standard_fonts,wasm} "$DEST/"
cp node_modules/pdfjs-dist/LICENSE "$DEST/"
