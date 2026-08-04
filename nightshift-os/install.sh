#!/usr/bin/env bash
# NightShift OS — one-command install of an on-prem, GPU-free agentic layer for Arm Linux.
# Turns any Arm64 Linux server into an AI-operated system: a local flagship-Llama server
# plus the `nsh` natural-language ops agent, managed by systemd.
#
#   sudo ./install.sh                      # flagship: Llama-4-Maverick Q4_0 (KleidiAI i8mm)
#   sudo ./install.sh --model qwen30b      # lighter/faster tier (30B MoE, 3B active)
#   sudo ./install.sh --model /path/to.gguf --threads 32
#
# Idempotent. No GPU. No data leaves the box.
set -euo pipefail

PREFIX=/opt/nightshift
MODELS=$PREFIX/models
THREADS=${THREADS:-$(( $(nproc) / 2 ))}   # one NUMA node is usually optimal for MoE
PORT=${PORT:-8080}
MODEL_ARG="maverick"
JOBS=$(nproc)

while [ $# -gt 0 ]; do case "$1" in
  --model) MODEL_ARG="$2"; shift 2;;
  --threads) THREADS="$2"; shift 2;;
  --port) PORT="$2"; shift 2;;
  *) echo "unknown arg: $1"; exit 1;;
esac; done

log(){ printf "\033[1;36m[nightshift]\033[0m %s\n" "$*"; }
[ "$(uname -m)" = "aarch64" ] || log "WARNING: not aarch64 — this appliance is tuned for Arm."
[ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }

log "1/5  dependencies"
if command -v apt-get >/dev/null; then
  apt-get update -qq && apt-get install -y -qq build-essential cmake curl git python3 libcurl4-openssl-dev >/dev/null
fi

log "2/5  build llama.cpp (KleidiAI i8mm for Arm Q4_0)"
mkdir -p "$PREFIX"
if [ ! -x "$PREFIX/llama.cpp/build/bin/llama-server" ]; then
  [ -d "$PREFIX/llama.cpp" ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp "$PREFIX/llama.cpp"
  cmake -S "$PREFIX/llama.cpp" -B "$PREFIX/llama.cpp/build" \
        -DGGML_NATIVE=ON -DGGML_CPU_KLEIDIAI=ON -DLLAMA_CURL=ON -DCMAKE_BUILD_TYPE=Release >/dev/null
  cmake --build "$PREFIX/llama.cpp/build" -j "$JOBS" --target llama-server >/dev/null
fi
SERVER="$PREFIX/llama.cpp/build/bin/llama-server"

log "3/5  model"
mkdir -p "$MODELS"
case "$MODEL_ARG" in
  maverick)
    MODEL="$MODELS/Maverick-Q4_0/Llama-4-Maverick-17B-128E-Instruct-Q4_0-00001-of-00005.gguf"
    if [ ! -f "$MODEL" ]; then
      log "    downloading Llama-4-Maverick Q4_0 (~225 GB, flagship) ..."
      mkdir -p "$MODELS/Maverick-Q4_0"
      B="https://huggingface.co/unsloth/Llama-4-Maverick-17B-128E-Instruct-GGUF/resolve/main/Q4_0"
      for i in 00001 00002 00003 00004 00005; do
        f="Llama-4-Maverick-17B-128E-Instruct-Q4_0-${i}-of-00005.gguf"
        curl -L -C - -o "$MODELS/Maverick-Q4_0/$f" "$B/$f"
      done
    fi ;;
  qwen30b)
    MODEL="$MODELS/Qwen3-30B-A3B-Instruct-2507-UD-Q4_K_XL.gguf"
    [ -f "$MODEL" ] || curl -L -C - -o "$MODEL" \
      "https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF/resolve/main/Qwen3-30B-A3B-Instruct-2507-UD-Q4_K_XL.gguf" ;;
  *) MODEL="$MODEL_ARG"; [ -f "$MODEL" ] || { echo "model not found: $MODEL"; exit 1; } ;;
esac

log "4/5  systemd services + nsh agent"
install -m 0755 "$(dirname "$0")/nsh.py" /usr/local/bin/nsh
sed -e "s#@SERVER@#$SERVER#g" -e "s#@MODEL@#$MODEL#g" -e "s#@THREADS@#$THREADS#g" -e "s#@PORT@#$PORT#g" \
    "$(dirname "$0")/systemd/nightshift-llm.service" > /etc/systemd/system/nightshift-llm.service
systemctl daemon-reload
systemctl enable --now nightshift-llm.service

log "5/5  waiting for the model server to come up (first load streams weights from disk) ..."
for _ in $(seq 1 120); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null)" = "200" ] && break
  sleep 5
done

cat <<EOF

\033[1;32mNightShift OS is up.\033[0m  GPU-free · on-prem · Arm64.
  model server : http://127.0.0.1:$PORT  (systemd: nightshift-llm)
  agent        : nsh "why is the root disk filling up?"
                 nsh                       # interactive
  audit log    : ~/.nightshift/audit.log
EOF
