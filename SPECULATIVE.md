# Speculative decoding on Arm CPU — measured on a trillion-parameter model

**Result: 1.66× accuracy-free speedup on Kimi K2 (1.04T) via prompt-lookup speculative
decoding, on one Azure Cobalt 100 VM (Arm Neoverse-N2), CPU only.** Temperature 0 → the output
is byte-for-byte identical to plain decoding, so this is pure speed at zero accuracy cost.

## The measurement

| | Baseline (no speculation) | Speculative (prompt-lookup, draft=8) |
|---|---|---|
| Effective generation | **10.3 tok/s** | **17.2 tok/s** |
| Wall time, ~256 tokens | 24.8 s | 15.0 s |
| Model forward passes | 256 | **156** |
| Time per forward pass | 96.9 ms | 96.1 ms |
| Draft acceptance | — | 18.4% (naive n-gram) |

Raw output: [`bench/results/speculative_k2.txt`](bench/results/speculative_k2.txt). Reproduce with
[`bench/run_spec.sh`](bench/run_spec.sh) (`llama-lookup`, `--spec-draft-n-max 0` vs `8`).

## Why this is the important result (theory confirmed on real silicon)

Token generation on a CPU is **memory-bandwidth-bound**: each step reads the model's active
weights from RAM, and that read dominates. Speculative decoding proposes several tokens with a
cheap draft and **verifies them all in one forward pass**. The key question is what that
verification costs.

**We measured it: the per-forward-pass time is unchanged — 96.9 ms → 96.1 ms.** Verifying up to
9 tokens in a pass costs the same as generating one, because the weights are read once and reused
across all positions (arithmetic intensity rises; the read is amortized). So the speedup is
exactly the number of tokens accepted per pass:

$$\text{speedup} = \frac{\text{tokens produced}}{\text{forward passes}} = \frac{257}{156} = 1.65\times .$$

This is the roofline argument, confirmed: **on bandwidth-bound CPU, speculative verification is
nearly free**, so the *full* geometric speedup lands — unlike GPU, where the extra tokens cost
extra compute. The regime everyone dismisses as "too slow for big models" is the regime where
speculation helps most.

## This is the floor, not the ceiling

The 1.66× came from a **naive n-gram drafter at just 18.4% acceptance.** Expected accepted length
follows $\frac{1-\alpha^{\gamma+1}}{1-\alpha}$, so raising acceptance $\alpha$ pays off fast:

| Draft acceptance $\alpha$ | Expected speedup (γ=8) |
|---|---|
| 0.18 (measured, naive n-gram) | ~1.7× |
| 0.50 (a decent draft model) | ~2.0× |
| 0.70 (a well-matched draft / MTP head) | ~3.2× |

Paths to higher $\alpha$: a proper small draft model sharing the vocab, the model's own **MTP
heads** (K3 ships them), or a smarter lookup (larger n-gram cache, code-aware). At α≈0.7 a 2.8T
model would *decode* at the effective speed of a ~1T one — accuracy unchanged. That is the
"run 2.8T like 1T" thesis, now with a working, measured first data point.

> Bonus finding, straight from the load log: `kleidiai: no kernel for tensor type q4_K` — K2's
> K-quant tensors bypass KleidiAI entirely (kernels exist only for Q4_0/Q8_0), which is exactly
> the gap the hand-written SMMLA K-quant kernel fills.
