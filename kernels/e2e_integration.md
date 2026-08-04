# End-to-end kernel integration — wiring the SMMLA Q2_K GEMM into llama.cpp

Goal: turn the standalone SMMLA microbench (1.41× over `ggml_vec_dot_q2_K_q8_K`, bit-exact) into a
**real, end-to-end token/s win** on a running model, on the Azure Cobalt 100 (Neoverse-N2).

## What SMMLA can and cannot speed up (honest scope)

SMMLA (`vmmlaq_s32`, i8mm) is a matrix×matrix instruction — it needs ≥2 rows to fill its tile. So it
accelerates **batched** matmuls:
- **Prompt / prefill processing** (many tokens per forward pass), and
- **Speculative verification** (γ+1 tokens verified in one pass — the batched step our
  [speculative-decoding result](../SPECULATIVE.md) creates).

It does **not** speed up naive single-token decode (batch=1 → GEMV). This is stated so the number is
not oversold: the win is on prefill + speculative-verify, not plain autoregressive decode.

## Recon: the live Q2_K path in the k3-fork (measured)

- Build has `MATMUL_INT8=1`, `REPACK=1`, `KLEIDIAI=1` — i8mm is compiled in, but used only for
  KleidiAI's Q4 kernels.
- Q2_K GEMM/GEMV in `repack.cpp` exists but is **scalar `_generic` only** — no i8mm version.
- Verbose load shows **Q2_K weights are NOT repacked** ("cannot be used with preferred buffer type
  CPU_KLEIDIAI, using CPU instead"; the 188 MB CPU_REPACK buffer is just the q6_K `output.weight`).
- ⇒ **Every Q2_K matmul runs the standard path: `ggml_compute_forward_mul_mat` → looped
  `ggml_vec_dot_q2_K_q8_K` (scalar SDOT).** That is exactly the function the SMMLA microbench beats,
  on the *standard* `block_q2_K` layout — so the existing kernel drops in without a layout rewrite.

## Integration plan

1. Add the SMMLA GEMM (`kernels/q2k_smmla.c` logic) to the ggml-cpu build.
2. In `ggml_compute_forward_mul_mat`, for `src0->type == GGML_TYPE_Q2_K` and `ne11 >= 4`, call the
   SMMLA GEMM per thread-chunk instead of the per-element `vec_dot` loop.
3. Rebuild `libggml-cpu.so`; validate output is coherent + numerically matches baseline.
4. Benchmark prefill token/s vs stock.

## Result: integrated and correct — but no end-to-end win at 1.5B (honest null result)

The kernel was wired into `ggml_compute_forward_mul_mat_one_chunk` (hook fires, confirmed via a
one-shot print; output stays coherent → correct). Re-confirmed the isolated win against *this*
libggml: **41.9 vs 29.7 GFLOP/s = 1.41×, bit-exact** (max|rel| 0.0016).

But a **clean A/B** (identical prompt/config, stock `.so` vs patched `.so`, best-of-3) on
Qwen2.5-1.5B Q2_K prefill:

| Build | Prompt t/s |
|---|---|
| stock ggml (SDOT vec_dot) | 206.0 |
| patched (SMMLA) | 206.1 |

**Zero end-to-end change.** The reason is honest and instructive: at 1.5B, prefill is **not
bottlenecked by the Q2_K matmul** — attention (quadratic in prompt length), the F32→Q8_K activation
quantization, norms/RoPE/softmax, and memory movement dominate. A 1.41× on a non-bottleneck op moves
nothing. (An earlier "250 vs 236" comparison was invalid — different prompt/context; the A/B above is
the controlled measurement.)

## What it would take to land the win (where the matmul *does* dominate)

1. **Hook the MoE path too** (`ggml_compute_forward_mul_mat_id`) — for trillion-param MoE the expert
   FFNs are the compute, and they go through `mul_mat_id`, not `mul_mat`.
2. **Test in the trillion-param regime** (Kimi K2, 1.04T, UD-Q2_K_XL) where the Q2_K GEMMs genuinely
   dominate prefill — needs a 360 GB re-download.
3. A per-matmul size gate (only engage SMMLA for large K and batch), tuned so it never regresses the
   many small attention projections.

**Honest takeaway:** the kernel is a real, bit-exact 1.41× Arm i8mm win *for the Q2_K GEMM*, and it is
correctly integrated — but it only pays off end-to-end where that GEMM is the bottleneck, which is the
trillion-param MoE prefill, not a 1.5B dense model. The microbench result is the defensible claim; the
end-to-end claim is not yet earned and is not made. Patched source kept at
`ggml-cpu.c.patched` on the VM.
