#!/usr/bin/env bash
# Installs everything needed to run the order extractor:
#   1. System packages: Tesseract OCR (+ German, English, Polish, Italian language packs)
#   2. Python virtualenv (.venv) with requirements.txt
#   3. llama.cpp's llama-server (prebuilt release; CUDA build if an NVIDIA GPU is found)
#   4. Qwen3.5-0.8B GGUF model (from Hugging Face into ./models)
#
# Usage: ./install.sh [--no-system] [--cpu] [--force-llama]
#   --no-system    skip apt/dnf/brew (no sudo needed; OCR must be installed already)
#   --cpu          install the CPU build of llama-server even if an NVIDIA GPU exists
#   --force-llama  reinstall llama-server even if one is present
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"

SKIP_SYSTEM=0; FORCE_CPU=0; FORCE_LLAMA=0
for arg in "$@"; do
  case "$arg" in
    --no-system) SKIP_SYSTEM=1 ;;
    --cpu) FORCE_CPU=1 ;;
    --force-llama) FORCE_LLAMA=1 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "Unknown option: $arg"; exit 1 ;;
  esac
done

[ -f .env ] || cp .env.example .env
# Load .env without overriding variables that are already set in the environment
load_env() {
  local line key
  while IFS= read -r line || [ -n "$line" ]; do
    [[ "$line" =~ ^[[:space:]]*([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]] || continue
    key="${BASH_REMATCH[1]}"
    if [ -z "${!key+x}" ]; then export "$key=${BASH_REMATCH[2]}"; fi
  done < .env
}
load_env
MODEL_REPO="${MODEL_REPO:-unsloth/Qwen3.5-0.8B-GGUF}"
MODEL_FILE="${MODEL_FILE:-Qwen3.5-0.8B-Q8_0.gguf}"

log() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
OS="$(uname -s)"

# ---------------------------------------------------------------------------
# 1. System packages
# ---------------------------------------------------------------------------
if [ "$SKIP_SYSTEM" -eq 0 ]; then
  log "Installing system packages (Tesseract OCR, Python venv, tools)"
  SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
  if [ "$OS" = "Darwin" ]; then
    command -v brew >/dev/null || { echo "Homebrew required: https://brew.sh"; exit 1; }
    brew install tesseract tesseract-lang python@3.12 curl
  elif command -v apt-get >/dev/null; then
    $SUDO apt-get update
    $SUDO apt-get install -y python3 python3-venv python3-pip curl tar unzip ca-certificates libgomp1 \
      tesseract-ocr tesseract-ocr-deu tesseract-ocr-eng tesseract-ocr-pol tesseract-ocr-ita
  elif command -v dnf >/dev/null; then
    $SUDO dnf install -y python3 python3-pip curl tar unzip libgomp \
      tesseract tesseract-langpack-deu tesseract-langpack-eng tesseract-langpack-pol tesseract-langpack-ita
  else
    echo "Unsupported package manager - install tesseract (deu, eng, pol, ita) manually, then use --no-system."
  fi
fi

# ---------------------------------------------------------------------------
# 2. Python environment
# ---------------------------------------------------------------------------
log "Creating Python virtualenv (.venv) and installing requirements"
PY="$(command -v python3.12 || command -v python3)"
[ -d .venv ] || "$PY" -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

# ---------------------------------------------------------------------------
# 3. llama-server
# ---------------------------------------------------------------------------
LLAMA_DIR="$ROOT/bin/llama.cpp"

llama_works() { "$1" --version >/dev/null 2>&1; }

find_llama() {
  local c
  for c in "$LLAMA_DIR/llama-server" $(find "$LLAMA_DIR" -name llama-server -type f 2>/dev/null) "$(command -v llama-server || true)"; do
    [ -n "$c" ] && [ -x "$c" ] && llama_works "$c" && { echo "$c"; return 0; }
  done
  return 1
}

# Picks the release asset: CUDA build if an NVIDIA driver supports it, else CPU build.
pick_asset() {
  .venv/bin/python - "$FORCE_CPU" "$OS" "$(uname -m)" <<'PY'
import json, re, subprocess, sys, urllib.request
force_cpu, os_name, arch = sys.argv[1] == "1", sys.argv[2], sys.argv[3]
rel = json.load(urllib.request.urlopen("https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=5"))
rel = next(r for r in rel if any("bin-" in a["name"] for a in r["assets"]))
assets = {a["name"]: a["browser_download_url"] for a in rel["assets"]}
tag = rel["tag_name"]
arch = "arm64" if arch in ("aarch64", "arm64") else "x64"
if os_name == "Darwin":
    names = [f"llama-{tag}-bin-macos-{arch}.tar.gz", f"llama-{tag}-bin-macos-{arch}.zip"]
    print("\n".join(f"{assets[n]} -" for n in names if n in assets)); sys.exit()
cuda = None
if not force_cpu:
    try:
        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10).stdout
        m = re.search(r"CUDA (?:UMD )?Version:\s*(\d+)\.(\d+)", out)
        cuda = (int(m[1]), int(m[2])) if m else None
    except Exception:
        pass
out = []
if cuda:
    builds = []
    for n in assets:
        m = re.fullmatch(rf"llama-{tag}-bin-ubuntu-cuda-(\d+)\.(\d+)-{arch}\.(tar\.gz|zip)", n)
        if m and (int(m[1]), int(m[2])) <= cuda:
            builds.append(((int(m[1]), int(m[2])), n))
    if builds:
        _, n = max(builds)
        rt = "cudart-" + n
        out.append(f"{assets[n]} {assets.get(rt, '-')}")
for ext in ("tar.gz", "zip"):
    n = f"llama-{tag}-bin-ubuntu-{arch}.{ext}"
    if n in assets:
        out.append(f"{assets[n]} -")
print("\n".join(out))
PY
}

extract_archive() {  # $1=file $2=dir
  case "$1" in
    *.zip) unzip -oq "$1" -d "$2" ;;
    *) tar -xzf "$1" -C "$2" ;;
  esac
}

build_from_source() {
  log "Building llama.cpp from source (fallback)"
  command -v cmake >/dev/null && command -v git >/dev/null || {
    echo "cmake/git missing. Install: sudo apt-get install -y build-essential cmake git"; exit 1; }
  local src="$ROOT/bin/llama.cpp-src" flags=""
  [ -d "$src" ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp "$src"
  if [ "$FORCE_CPU" -eq 0 ] && command -v nvcc >/dev/null; then flags="-DGGML_CUDA=ON"; fi
  cmake -S "$src" -B "$src/build" -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF $flags
  cmake --build "$src/build" --config Release -j "$(nproc 2>/dev/null || echo 4)" --target llama-server
  mkdir -p "$LLAMA_DIR"; cp "$src"/build/bin/* "$LLAMA_DIR/"
}

install_llama() {
  if [ "$OS" = "Darwin" ] && command -v brew >/dev/null; then
    log "Installing llama.cpp via Homebrew (Metal acceleration)"
    brew install llama.cpp && return 0
  fi
  log "Downloading prebuilt llama-server from github.com/ggml-org/llama.cpp"
  local tmp; tmp="$(mktemp -d)"
  while read -r url rt; do
    [ -z "$url" ] && continue
    echo "Trying $(basename "$url")"
    rm -rf "$LLAMA_DIR"; mkdir -p "$LLAMA_DIR"
    curl -fL --progress-bar -o "$tmp/llama.${url##*.}" "$url" || continue
    extract_archive "$tmp/llama.${url##*.}" "$LLAMA_DIR"
    if [ "$rt" != "-" ]; then
      curl -fL --progress-bar -o "$tmp/cudart.${rt##*.}" "$rt" && extract_archive "$tmp/cudart.${rt##*.}" "$LLAMA_DIR"
    fi
    # flatten: put all binaries + libs in one folder so the rpath works
    find "$LLAMA_DIR" -mindepth 2 -type f \( -name 'llama-*' -o -name '*.so*' -o -name '*.dylib' \) -exec mv -f {} "$LLAMA_DIR/" \; 2>/dev/null || true
    find "$LLAMA_DIR" -mindepth 2 -type l \( -name '*.so*' -o -name '*.dylib' \) -exec mv -f {} "$LLAMA_DIR/" \; 2>/dev/null || true
    if LD_LIBRARY_PATH="$LLAMA_DIR:${LD_LIBRARY_PATH:-}" llama_works "$LLAMA_DIR/llama-server"; then
      rm -rf "$tmp"; return 0
    fi
    echo "  -> binary does not run on this system (glibc/driver mismatch?), trying next option"
  done < <(pick_asset)
  rm -rf "$tmp"
  build_from_source
}

export LD_LIBRARY_PATH="$LLAMA_DIR:${LD_LIBRARY_PATH:-}"
if [ "$FORCE_LLAMA" -eq 1 ] || ! LLAMA_BIN="$(find_llama)"; then
  install_llama
  LLAMA_BIN="$(find_llama)" || { echo "llama-server installation failed"; exit 1; }
fi
echo "llama-server: $LLAMA_BIN ($("$LLAMA_BIN" --version 2>&1 | grep -m1 -i version || true))"

# ---------------------------------------------------------------------------
# 4. Model
# ---------------------------------------------------------------------------
log "Downloading model $MODEL_REPO / $MODEL_FILE"
mkdir -p models
.venv/bin/python - "$MODEL_REPO" "$MODEL_FILE" <<'PY'
import sys
from huggingface_hub import hf_hub_download
path = hf_hub_download(repo_id=sys.argv[1], filename=sys.argv[2], local_dir="models")
print("Model:", path)
PY

# ---------------------------------------------------------------------------
# 5. Checks
# ---------------------------------------------------------------------------
log "Checks"
if command -v tesseract >/dev/null; then
  langs="$(tesseract --list-langs 2>/dev/null | tail -n +2 | tr '\n' ' ')"
  echo "Tesseract OCR languages: $langs"
  for l in deu eng pol ita; do echo " $langs " | grep -q " $l " || echo "  WARNING: OCR language '$l' missing"; done
else
  echo "WARNING: tesseract not found - OCR will not work (PDFs with text layer still work)."
fi
chmod +x start.sh
log "Done. Start everything with:  ./start.sh"
