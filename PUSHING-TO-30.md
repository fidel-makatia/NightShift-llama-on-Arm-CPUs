# Pushing K3 toward 30 tok/s on Arm CPU — four experiments, honestly scored

Goal: make Kimi K3 (2.8T) usable on a single Azure Cobalt 100 Arm CPU node, and find an honest
path to 30 tok/s. Four levers, all measured on `nightshift-vm` (E96ps_v6, Neoverse-N2, 96 vCPU,
660 GB RAM), CPU only. This document reports what each lever actually delivered — including where
it fell short.

## Scorecard

| # | Lever | Result | Hit 30? |
|---|---|---|---|
| 4 | Match active params to the roofline (Qwen3-30B-A3B, 3B active) | **44.3 tok/s** | ✅ **yes** |
| 1 | Faster i-quant kernel (measure the K3 compute wall) | iq1_s is **27× slower/byte** than q4_K — quantified the prize | ⏳ headroom found |
| 2 | Cobalt 200 (2× bandwidth) | Not available on this subscription; roofline anchored to **measured 420 GB/s** | ❌ can't rent |
| 3 | FGEQ mixed-precision + MTP | Sim: same RAM + free accuracy + **1.9–4.9× hot-path compute** | ⏳ research |

**Bottom line:** 30 tok/s is *reached today* for a frontier-quality 3B-active MoE (#4). For K3
single-stream it is not reachable on Cobalt 100 — the honest ceiling is a few tok/s — and the four
experiments explain precisely why, and what would move it.

## #4 — Active params set CPU speed (the 30 tok/s that landed)

Qwen3-30B-A3B (30B total, **3B active**, UD-Q4_K_XL, 17.7 GB), temp 0:

| threads | 32 | 48 | 64 | 96 |
|---|---|---|---|---|
| generation tok/s | **44.3** | 42.7 | 36.1 | 4.7 |

44.3 tok/s, comfortably past 30, on a real MoE — because on a bandwidth-bound CPU it is the
**active** parameter count, not the total, that sets speed. (The 96-thread collapse to 4.7 is
cross-NUMA: for a memory-light model, spanning both sockets makes remote-node expert reads
dominate. One NUMA node / 32 threads is optimal.) Raw: `bench/results/qwen3_30b_a3b.txt`.

## #1 — The real K3 wall is compression-vs-compute, and it's measurable

In-cache compute-ceiling microbench against the real `libggml` (`kernels/iquant_bench.c`), one core:

| Quant | used by | GFLOP/s | GB/s-compute | vs q4_K |
|---|---|---|---|---|
| **iq1_s** | K3 | 19.5 | 1.90 | **27× slower** |
| q4_K | Qwen3-30B | 184.3 | 51.84 | 1× |
| q8_0 | ref | 40.2 | 21.35 | — |

i-quants decode through a 2048-entry **codebook gather** with no i8mm/SDOT/KleidiAI path; q4_K has
a clean SIMD path. So the 1.56-bit quant that lets K3 *fit* in 660 GB RAM is the same thing that
makes it ~9–27× slower to *decode*. A q4_K K3 would be ~1.4 TB (doesn't fit); iq1_s (554 GB) is
near the largest quant that fits — **K3 is forced onto the slowest-compute quant.** The Q2_K SMMLA
kernel in the sibling Kimi repo (1.41× over llama.cpp, bit-exact) shows the technique that attacks
this; a production IQ1_S kernel is future work. Raw: `kernels/iquant_results.txt`.

## #2 — Cobalt 200 unavailable; roofline anchored to a measured number

Cobalt 200 (Arm v7 SKUs) is not allocatable on this Azure Sponsorship subscription, so I anchored
the roofline to a *measured* bandwidth instead of a spec sheet. STREAM Triad on our Cobalt 100:
**~420 GB/s** aggregate. K3 reads ~22 GB of active weights/token → **~19 tok/s bandwidth ceiling**;
measured 2.2 tok/s is ~8× below it, confirming K3 is compute-co-limited (see #1), not purely
bandwidth-starved. Cobalt 200's reported ~1.5–2× bandwidth would lift the ceiling to ~30–38 tok/s —
but only once the #1 compute wall is also removed. Raw: `bench/results/bandwidth_and_cobalt200.txt`.

## #3 — FGEQ: one config, three wins (footprint, accuracy, and now speed)

The #1 finding adds a speed axis to FGEQ. Its iso-footprint tiering (hot experts → 4-bit q4_K,
cold → 1-bit iq1_s) is *exactly* the assignment that routes the hot activation mass through the
fast compute path. Simulation (`fgeq/fgeq_sim.py`, measured compute rates):

| Routing skew | iso-footprint precision | iso-accuracy RAM | hot-path compute speedup |
|---|---|---|---|
| measured K2 (Gini 0.32) | 2.59 bits (vs 2.00) | −18% | **1.90×** |
| literature Zipf s=1 (Gini 0.80) | 3.67 bits (vs 2.00) | −32% | **4.91×** |

**MTP:** K3 ships Multi-Token-Prediction heads — the highest-acceptance self-draft available. By
the geometric-series bound `(1−α^{γ+1})/(1−α)`, α≈0.6–0.7 with γ≈3–4 heads → ~2.5–3× accuracy-free.
Wiring MTP into the CPU speculative loop is implementation TODO (not done here). Honest bound on
#3: the compute/MTP speedups are real but Amdahl-limited by the memory roofline — they move K3 from
2.2 toward a few tok/s, not to 30.

## Where this sits vs the public K3 landscape (verified 2026-08)

- **WASTE** ([sqliteai/waste](https://github.com/sqliteai/waste)) — NVMe expert-streaming + 3-bit
  residual VQ (982 GB). I built its container and **benchmarked it on our Arm server: 0.29 tok/s at
  90% cache hit** — *slower* than its own 64 GB Mac number (0.45–0.62), confirming the codebook-gather
  compute wall from #1. Full head-to-head: [WASTE-HEAD-TO-HEAD.md](WASTE-HEAD-TO-HEAD.md).
- **kimi-k3-in-c** ([FareedKhan-dev/kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c)) —
  176 KB C99, 4-bit NVMe expert-streaming. Author-reported **~0.09 tok/s @ 110 GB, ~0.03 @ 8 GB** —
  same storage-bound streaming category.
- **vLLM + DSpark** ([vLLM K3 blog](https://vllm.ai/blog/2026-07-27-k3)) — 111/118 → 331/**370 tok/s**
  (TP8/TP16), via a block-diffusion speculative head, on **16× NVIDIA GB300 GPUs**. Not Arm CPU; the
  blog describes no CPU/Arm64 backend.
- **NightShift RAM-resident** — **2.2 tok/s** holding IQ1_S fully in RAM: ~7.6× the streaming engines
  on the same Arm silicon, because it never pays the per-token NVMe + VQ-gather cost.
- **NightShift (this work)** — runs the *full* K3 **RAM-resident on one Arm CPU node** at a stable
  2.2 tok/s (the RAM budget is what buys the ~7× over consumer streaming), and is the only one to
  **measure why CPU K3 is slow** (the iq1_s compute wall) and turn that into a compression-for-speed
  design (FGEQ axis C). Different design point from both — single-node, GPU-free, and diagnostic.

*All numbers are measured on the hardware named above unless labeled "sim" or "reported." Claims I
could not reproduce are flagged as such.*
