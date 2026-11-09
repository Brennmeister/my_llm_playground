#!/usr/bin/env bash
# Starts llama-server (Qwen3.5-0.8B) and the FastAPI app with one command.
# Ctrl+C stops both.
#
# Usage: ./start.sh [--reload]
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"

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

MODEL_FILE="${MODEL_FILE:-Qwen3.5-0.8B-Q8_0.gguf}"
MODEL_PATH="$ROOT/models/$MODEL_FILE"
MODEL_REPO="${MODEL_REPO:-unsloth/Qwen3.5-0.8B-GGUF}"
MMPROJ_FILE="${MMPROJ_FILE-mmproj-BF16.gguf}"
MMPROJ_PATH="$ROOT/models/mmproj/${MODEL_REPO##*/}/$MMPROJ_FILE"
VISION_ARGS=()
if [ -n "$MMPROJ_FILE" ] && [ -f "$MMPROJ_PATH" ]; then
  VISION_ARGS=(--mmproj "$MMPROJ_PATH")   # the model also sees the page images
elif [ -n "$MMPROJ_FILE" ]; then
  echo "Vision projector $MMPROJ_PATH missing - run ./install.sh (text only until then)"
fi
LLM_PORT="${LLM_PORT:-8081}"
LLM_CTX="${LLM_CTX:-8192}"
LLM_NGL="${LLM_NGL:-99}"
API_HOST="${API_HOST:-127.0.0.1}"
API_PORT="${API_PORT:-8000}"
export LLM_BASE_URL="${LLM_BASE_URL:-http://127.0.0.1:$LLM_PORT}"
export LLM_CTX

LLAMA_DIR="$ROOT/bin/llama.cpp"
export LD_LIBRARY_PATH="$LLAMA_DIR:${LD_LIBRARY_PATH:-}"
LLAMA_BIN=""
for c in "$LLAMA_DIR/llama-server" "$(command -v llama-server || true)"; do
  [ -n "$c" ] && [ -x "$c" ] && { LLAMA_BIN="$c"; break; }
done
[ -n "$LLAMA_BIN" ] || { echo "llama-server not found - run ./install.sh first"; exit 1; }
[ -f "$MODEL_PATH" ] || { echo "Model $MODEL_PATH not found - run ./install.sh first"; exit 1; }
[ -x .venv/bin/uvicorn ] || { echo ".venv missing - run ./install.sh first"; exit 1; }

port_busy() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }
for p in "$LLM_PORT" "$API_PORT"; do
  if port_busy "$p"; then
    echo "Port $p is already in use (another instance running?). Stop it or set LLM_PORT/API_PORT."; exit 1
  fi
done

mkdir -p logs
THREAD_ARGS=()
[ -n "${LLM_THREADS:-}" ] && THREAD_ARGS=(--threads "$LLM_THREADS")

# Notebook-friendly settings:
#   --parallel 1     one slot -> all compute for one request, minimal KV-cache memory
#   --ctx-size 8192  enough for ~2-3 pages of order text + JSON output; keeps RAM low
#   --n-gpu-layers   offload everything if a GPU build is installed (0.8B fits in <2 GB VRAM)
#   --host 127.0.0.1 LLM is only reachable locally; the API is the only entry point
echo "Starting llama-server on port $LLM_PORT (log: logs/llama-server.log)"
"$LLAMA_BIN" \
  --model "$MODEL_PATH" \
  --alias "${LLM_MODEL_NAME:-qwen3.5-0.8b}" \
  --host 127.0.0.1 --port "$LLM_PORT" \
  --ctx-size "$LLM_CTX" \
  --parallel 1 \
  --n-gpu-layers "$LLM_NGL" \
  --jinja \
  --no-webui \
  "${THREAD_ARGS[@]}" \
  "${VISION_ARGS[@]}" \
  > logs/llama-server.log 2>&1 &
LLAMA_PID=$!

cleanup() {
  echo; echo "Stopping llama-server (pid $LLAMA_PID)"
  kill "$LLAMA_PID" 2>/dev/null || true
  wait "$LLAMA_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

printf "Waiting for model to load"
for _ in $(seq 1 180); do
  if ! kill -0 "$LLAMA_PID" 2>/dev/null; then
    echo; echo "llama-server exited. Last log lines:"; tail -n 25 logs/llama-server.log; exit 1
  fi
  curl -sf "http://127.0.0.1:$LLM_PORT/health" >/dev/null 2>&1 && { echo " ready."; break; }
  printf "."; sleep 1
done
curl -sf "http://127.0.0.1:$LLM_PORT/health" >/dev/null || { echo " timeout"; tail -n 25 logs/llama-server.log; exit 1; }
grep -iE "offloaded|using device|CUDA0|Vulkan0|Metal" logs/llama-server.log | head -n 3 || true

echo "Starting API on http://$API_HOST:$API_PORT  (Swagger UI: http://$API_HOST:$API_PORT/docs)"
.venv/bin/uvicorn app.main:app --host "$API_HOST" --port "$API_PORT" "$@"
