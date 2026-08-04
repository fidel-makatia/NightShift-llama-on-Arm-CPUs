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

## Baselines to beat (stock scalar path, Neoverse-N2)

| Model | Prompt (prefill) | Generation |
|---|---|---|
| Qwen2.5-1.5B Q2_K (iteration model) | **250 t/s** | 96 t/s |
| Kimi K2 (1.04T, UD-Q2_K_XL) | *to measure (needs 360 GB re-download)* | ~11 t/s |

Generation is expected to be unchanged (GEMV); the target is prefill (and, composed with speculation,
the verify step).
