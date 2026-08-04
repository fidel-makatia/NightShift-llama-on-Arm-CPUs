# RAM-resident vs streaming — a measured K3 head-to-head on one Arm CPU node

The public "run Kimi K3 (2.8T) locally" projects (WASTE, kimi-k3-in-c) all **stream experts from
NVMe** to fit the model on small machines. NightShift instead holds a smaller quant **fully in RAM**
on a large Arm server. This is the first measurement of both on the *same* Arm silicon — an Azure
Cobalt 100 (E96ps_v6, 96-core Neoverse-N2, 660 GB), CPU only.

To do it I built the WASTE 3-bit-VQ container myself from the 1.56 TB native MXFP4 weights. The
converter crashed at the final assembly with a codebook-base collision (`base 792 overlaps 801`);
I traced it to `source_ok` being a false-negative on the tail layers, so `next_base` didn't advance
and two layers collided. One-line fix (advance `next_base` for every queued layer) → a clean 982 GB
container that produces correct, coherent K3 output on Arm. (Fix may be worth upstreaming.)

## The numbers

| Approach | Config on the Cobalt 100 | Throughput | Cache hit |
|---|---|---|---|
| WASTE (3-bit VQ, 982 GB, streamed) | default 80 GB budget, cold | 0.08 tok/s | 57% |
| WASTE (3-bit VQ, 982 GB, streamed) | 600 GB budget, warm | 0.26 tok/s | 88% |
| WASTE (3-bit VQ, 982 GB, streamed) | 450 GB budget, warm | **0.29 tok/s** | 90% |
| **NightShift RAM-resident (IQ1_S, 554 GB)** | fully in RAM, no streaming | **2.2 tok/s** | — |

Reported by their authors on other hardware: WASTE on a 64 GB M5 Pro Mac **0.45–0.62 tok/s**;
`kimi-k3-in-c` (4-bit stream) **~0.09 tok/s @ 110 GB / ~0.03 @ 8 GB**.

## Three findings

1. **Fit-in-RAM beats stream-from-NVMe ~7.6× on identical silicon** (2.2 vs 0.29 tok/s). When the
   machine has the RAM, holding the model resident is the decisive win.
2. **WASTE on our 96-core / 660 GB Arm server (0.29) is *slower* than on a 64 GB M5 Pro Mac
   (0.45–0.62)** — despite 10× the RAM and far more cores. At 90% cache hit the model is no longer
   I/O-bound; the limiter is **per-core VQ-decode compute** (the 3-bit codebook gather) plus memory
   latency, where Apple Silicon's fast cores beat Neoverse-N2 efficiency cores. RAM lifts hit-rate;
   it can't beat the per-token compute floor.
3. **This confirms the [i-quant compute-wall finding](kernels/iquant_results.txt).** WASTE's 3-bit VQ
   is codebook-gather — the same class as `iq1_s`, which we measured at 27× slower/byte than `q4_K`.
   At 90% hit the experts are *in RAM* and it's still ~0.3 tok/s, because **decoding them is the
   wall.** Compression schemes that gather from a codebook trade decode-speed for footprint.

## The trade-off, made concrete

WASTE's 982 GB (3-bit VQ, higher quality) doesn't fit in 660 GB → must stream → 0.29 tok/s.
Our IQ1_S 554 GB (1.56-bit, lower quality) fits → 2.2 tok/s. **Quality vs fit vs speed is a
triangle**; on a big-RAM Arm server, the fit-in-RAM corner is the fast one — and the fastest corner
of all is a model whose *active* params already fit the roofline (a 3B-active MoE hits 44 tok/s here,
see [PUSHING-TO-30.md](PUSHING-TO-30.md)).

*Fairness: WASTE is built for machines that can't hold the model (8–64 GB), where streaming is its
strength; on 660 GB it's simply the wrong tool, not a bad one. Raw logs:
[`bench/results/waste_head_to_head.txt`](bench/results/waste_head_to_head.txt).*
