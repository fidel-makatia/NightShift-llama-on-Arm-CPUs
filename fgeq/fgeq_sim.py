#!/usr/bin/env python3
"""FGEQ — Frequency-Graded Expert Quantization: a workload-aware MoE compression scheme.

Uniform quantization spends equal bits on every expert. MoE routing is skewed, so FGEQ assigns
bit-width by MEASURED activation frequency (from ExpertAtlas). Two honest operating points:

  (A) ISO-FOOTPRINT  — keep total RAM = uniform 2-bit, but put the bits where the tokens go.
                       Payoff: the compute that actually runs is at HIGHER effective precision
                       (activation-weighted bits > 2) for the SAME memory. Free accuracy.
  (B) ISO-ACCURACY   — keep effective precision ~2 bits, compress the cold tail to cut RAM.

The size of the win scales with routing skew. This is a feasibility simulation on real and
literature-typical distributions — not a finished codec.
"""
import csv, sys

def load_freqs(path):
    with open(path) as f:
        return [int(r["count"]) for r in csv.DictReader(f)]

def synth_zipf(n=13000, s=1.0):
    return [1.0/((i+1)**s) for i in range(n)]

def gini(xs):
    xs = sorted(xs); n = len(xs); tot = sum(xs)
    if not tot: return 0.0
    c = sum(i*x for i, x in enumerate(xs, 1))
    return (2*c)/(n*tot) - (n+1)/n

# Measured per-weight COMPUTE cost (in-cache, single core, real libggml on Neoverse-N2;
# see ../kernels/iquant_results.txt). GFLOP/s -> weights/s = GFLOP/s * 1e9 / 2.
# This is the dimension uniform quantization ignores: the 1-bit i-quant that makes a
# trillion-param model FIT is also the slowest to DECODE (codebook gather, no i8mm path).
GFLOPS = {"iq1_s": 19.5, "q4_K": 184.3}          # measured
T_W = {q: 2.0/(g*1e9) for q, g in GFLOPS.items()} # seconds per weight decoded+MACed
T_HOT, T_COLD = T_W["q4_K"], T_W["iq1_s"]          # FGEQ iso-footprint: hot=4-bit(q4_K), cold=1-bit(iq1_s)

def evaluate(freqs, label):
    n = len(freqs); total = sum(freqs)
    ranked = sorted(freqs, reverse=True)
    # (A) ISO-FOOTPRINT: top 1/3 experts -> 4-bit, bottom 2/3 -> 1-bit  => mean = 2.0 bits (== uniform 2-bit)
    k = n // 3
    hot_mass = sum(ranked[:k]) / total
    mean_bits = (k*4 + (n-k)*1)/n
    aw_bits_iso = hot_mass*4 + (1-hot_mass)*1
    # (C) SPEED, same iso-footprint config: activation-weighted per-weight compute time.
    # Uniform 1-bit i-quant decodes every active weight at the slow rate; FGEQ routes the hot
    # mass through the fast q4_K path. Speedup on the COMPUTE fraction of the hot path:
    fgeq_tw = hot_mass*T_HOT + (1-hot_mass)*T_COLD
    compute_speedup = T_COLD / fgeq_tw
    # (B) ISO-ACCURACY: keep activation-weighted bits ~2 by giving hot 2-bit, cold 1-bit; report RAM saved.
    #     choose hot fraction p so that hot experts cover ~all the mass at 2-bit, tail at 1-bit.
    #     simple: hot = experts covering 90% of mass at 2-bit, rest at 1-bit.
    cum = 0.0; hot_cells = 0
    for f in ranked:
        cum += f/total; hot_cells += 1
        if cum >= 0.90: break
    mean_bits_isoacc = (hot_cells*2 + (n-hot_cells)*1)/n
    ram_saved = 1 - mean_bits_isoacc/2.0

    print(f"== {label}")
    print(f"   n={n} experts·layers, routing Gini = {gini(freqs):.3f}, top-1/3 hold {hot_mass*100:.0f}% of activations")
    print(f"   (A) iso-footprint (same RAM as uniform 2-bit): effective precision "
          f"{aw_bits_iso:.2f} bits  vs  2.00  ->  {'+' if aw_bits_iso>2 else ''}{(aw_bits_iso-2):.2f} bits of free accuracy")
    print(f"   (B) iso-accuracy  (keep ~2-bit precision):     RAM {ram_saved*100:+.0f}% "
          f"({'saved' if ram_saved>0 else 'cost'})")
    print(f"   (C) SPEED (same iso-footprint config): hot mass {hot_mass*100:.0f}% via fast q4_K path "
          f"-> {compute_speedup:.2f}x on the hot-path COMPUTE fraction (measured iq1_s vs q4_K rates)\n")

if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "sample_activations.csv"
    print("FGEQ — Frequency-Graded Expert Quantization  ·  feasibility simulation")
    print("(bits allocated by measured routing frequency; win scales with skew)\n")
    evaluate(load_freqs(csv_path), "measured K2 routing (ExpertAtlas, 509-token sample — under-samples skew)")
    evaluate(synth_zipf(), "literature-typical skew (Zipf s=1.0) — what fuller samples show")
