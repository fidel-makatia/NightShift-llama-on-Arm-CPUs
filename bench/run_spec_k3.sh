#!/usr/bin/env bash
# Speculative-decoding prototype on Kimi K3 (2.8T) — Neoverse-N2, CPU only.
# K3-fork llama-lookup. Baseline (n-max 0) vs speculative (n-max 8). temp 0 => identical output.
set -uo pipefail
LK=/data/llama-k3/build/bin/llama-lookup
M=$(ls /data/models/Kimi-K3-GGUF/UD-IQ1_S/*00001-of-*.gguf)
T=64; N=160
rm -f /data/SPECK3_DONE

read -r -d '' PROMPT <<'EOF'
Repeat the following Python module verbatim, but add a type hint to every function
signature and a one-line docstring to each function. Output only the code.

import json
def load_config(path):
    with open(path) as f:
        return json.load(f)
def merge(a, b):
    out = dict(a)
    for k, v in b.items():
        out[k] = v
    return out
def get(cfg, key, default):
    return cfg.get(key, default)
def save_config(path, cfg):
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
def validate(cfg):
    return "name" in cfg and "version" in cfg
EOF

run () {  # $1 label, $2 n-max
  echo "===== $1 (spec-draft-n-max=$2) ====="
  "$LK" -m "$M" -t $T -c 4096 -n $N --temp 0 --spec-draft-n-max "$2" \
    -p "$PROMPT" >/dev/null 2> "/data/speck3_$1.err"
  grep -aiE "eval time|tokens per second|n_draft|n_accept|accept|drafted|n_predict" "/data/speck3_$1.err" | tail -10
  echo
}
run base 0
run spec 8
touch /data/SPECK3_DONE
echo SPECK3_DONE
