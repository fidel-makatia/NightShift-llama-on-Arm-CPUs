# FGEQ — Frequency-Graded Expert Quantization

A workload-aware compression scheme for MoE models, built on ExpertAtlas's measured routing data.
**Status: a design + feasibility simulation ([`fgeq_sim.py`](fgeq_sim.py)), not a finished codec.**

## The honest premise

You cannot beat the resources / speed / accuracy tradeoff triangle by wishing. Uniform
quantization (GPTQ, AWQ, the K/IQ-quants) already sits near the Pareto frontier. FGEQ doesn't
beat physics — it exploits **one thing those schemes ignore: not all experts are equal.**

MoE routing is skewed — a minority of experts serve the majority of tokens. Uniform quantization
spends the same bits on an expert hit 100,000× and one hit 5×. FGEQ instead **allocates
bit-width by measured activation frequency** (the exact data [ExpertAtlas](expert-capture.cpp)
captures): hot experts get more bits, the cold tail gets fewer or is evicted to NVMe.

## Two honest operating points

Let $f_e$ be expert $e$'s measured activation frequency and $b_e$ its assigned bit-width. Total
footprint $\propto \sum_e b_e$; the precision *where compute actually happens* is the
**activation-weighted** bit-width $\bar b = \frac{\sum_e f_e\, b_e}{\sum_e f_e}$.

- **(A) Iso-footprint** — hold $\sum_e b_e$ equal to uniform 2-bit, but assign hot→4-bit,
  cold→1-bit. Because hot experts carry most of the mass, $\bar b > 2$: **the running compute is
  more accurate at the same memory.** Free accuracy.
- **(B) Iso-accuracy** — hold $\bar b \approx 2$, compress the cold tail: **RAM drops.**

## The third axis: speed (added from a measured kernel finding)

The original FGEQ pitch was footprint + accuracy. A microbench of the real `libggml` dot-products
([`../kernels/iquant_results.txt`](../kernels/iquant_results.txt)) surfaced a **third payoff that
comes free with the same config.** Per-weight decode-and-MAC cost, in-cache, one Neoverse-N2 core:

| Quant | bits/weight | GFLOP/s (measured) | relative compute cost |
|---|---|---|---|
| iq1_s (what a 1-bit K3 must use) | 1.56 | 19.5 | **9.4× slower** |
| q4_K (what a 4-bit expert uses) | 4.50 | 184.3 | 1× |

i-quants decode through a 2048-entry **codebook gather** (no i8mm/SDOT/KleidiAI path); q4_K has a
clean SIMD path. So the 1-bit quant that lets a trillion-param model *fit* is also ~9× slower to
*decode*. Crucially, FGEQ's iso-footprint tiering — **hot experts at 4-bit (q4_K), cold at 1-bit
(iq1_s)** — is exactly the assignment that routes the hot activation mass through the *fast* path.
The same config that buys free accuracy also speeds up the compute that actually runs.

## Simulated results ([`fgeq_sim.py`](fgeq_sim.py))

| Routing distribution | (A) iso-footprint precision | (B) iso-accuracy RAM | (C) hot-path compute speedup |
|---|---|---|---|
| measured K2 (ExpertAtlas, 509-token sample) — Gini 0.32 | **2.59 bits** (vs 2.00) | **−18%** | **1.90×** |
| literature-typical skew (Zipf s=1) — Gini 0.80 | **3.67 bits** (vs 2.00) | **−32%** | **4.91×** |

One tiering, three wins: same RAM, higher effective precision, *and* the hot path decodes faster.
The win scales with skew. Our 509-token sample under-measures it; larger traces and K3's 896
experts should push toward the Zipf row. **Honest bound:** (C) is the speedup on the *compute*
fraction; K3 end-to-end is co-limited by compute and scattered-memory access (both ~same order,
see kernel results), so the end-to-end gain is Amdahl-bounded by the memory roofline — this moves
K3 from 2.2 toward a few tok/s, not to 30. It is a real lever, not a magic one.

## Where the "very high TPS" actually comes from (being precise)

Compression alone **cannot** give high TPS *and* high accuracy — per token you still read the
active experts, and making the hot ones smaller (for speed) is exactly what hurts accuracy. So
FGEQ's honest contribution is **resources + accuracy** (less RAM at equal precision, or more
precision at equal RAM). The **TPS** lever that preserves accuracy is orthogonal: **speculative
decoding** (a draft model proposes tokens, full K3 verifies — multiple tokens per weight-read,
accuracy unchanged). FGEQ + a hot-expert speculative draft is the combination that moves all
three corners; each does the part it honestly can.

## What it would take to make it real

1. A GGUF variant that stores per-expert bit-widths (extend the K-quant block header).
2. A packer that reads an ExpertAtlas activation CSV and assigns tiers (the sim's logic).
3. A ggml kernel path that dispatches per-expert precision (this is where our
   [SMMLA K-quant kernel](../kernels/q2k_smmla.c) work plugs in — mixed-precision GEMM).
4. Real accuracy eval (perplexity / task suite) at each operating point — the missing proof.

FGEQ is a research direction with a data-backed feasibility case, not a shipped result. Stated
honestly so it reads as engineering, not hype.
