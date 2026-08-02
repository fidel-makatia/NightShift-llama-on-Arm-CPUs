#!/usr/bin/env bash
# Benchmark Llama 4 Maverick (17B active MoE) on Neoverse-N2 — same quant/tool as K2.
set -uo pipefail
B=/opt/llama.cpp/build/bin/llama-bench
M=$(ls /data/models/Llama4-Maverick-GGUF/UD-Q2_K_XL/*00001-of-*.gguf)
rm -f /data/MAVERICK_BENCH_DONE
"$B" -m "$M" -p 512 -n 128 -t 48,64,96 --load-mode mmap -o md > /data/maverick_bench.md 2>/data/maverick_bench.log
echo "BENCH_DONE rc=$?" >> /data/maverick_bench.log
touch /data/MAVERICK_BENCH_DONE
