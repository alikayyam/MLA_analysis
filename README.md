# The Geometry of Latent Compression

Experimental codebase for the paper:

> **The Geometry of Latent Compression: Subspace Collapse and Gauge Freedom in Multi-Head Latent Attention**
> Ali Kayyam, June 2026

## Overview

This repo reproduces three experiments from the paper:

1. **Exp 1 — Gauge-Invariant Weight Audit**: compute per-layer Gram distance (D_Gram) and mean principal angle (θ̄) between the effective key and value operators W̃_K = W_c W_K↑ and W̃_V = W_c W_V↑. Produces the U-shaped alignment profile across network depth.

2. **Exp 2 — Activation-Based Alignment**: hook into `kv_b_proj` at runtime and measure token-level cosine similarity ρ between K and V activations on WikiText. Correlates with static weight metrics.

3. **Exp 3 — Surgical Parameter Clamping**: set W_K↑ ← W_V↑ in selected layers under three schedules (middle-block, progressive outward, global) and measure ΔPPL and downstream accuracy.

## Setup

```bash
# Install dependencies
uv sync

# Run the unit test suite (no model download needed)
uv run --with pytest pytest tests/ -v
```

All tests run against a synthetic `ToyMLAModel` with the same weight structure as DeepSeek. Expected: **27 passed**.

## Hardware Notes

- **GPU**: GTX 1080 Ti (compute capability 6.1) is incompatible with the installed PyTorch 2.12+cu130 — all runs are **CPU-only**.
- **RAM**: 62 GB. DeepSeek-V2-Lite (~31 GB bfloat16) fits. Full V2 (236B) and V3 (671B) do not — Exp 1 uses shard-by-shard loading (< 1 GB peak RAM).

## Running the Experiments

### Download models

```bash
# V2-Lite (16B, ~31 GB) — proxy model for Exp 2 and 3
hf download deepseek-ai/DeepSeek-V2-Lite --local-dir models/deepseek-v2-lite

# V2 (236B, ~472 GB) — required for U-curve paper results
hf download deepseek-ai/DeepSeek-V2 --local-dir models/deepseek-v2

# V3 (671B, ~1.3 TB) — required for U-curve paper results (FP8, auto-dequantized)
hf download deepseek-ai/DeepSeek-V3 --local-dir models/deepseek-v3
```

### Experiment 1: Weight Audit

```bash
uv run scripts/run_exp1_weight_audit.py \
    --model deepseek-v2-lite \
    --model-dir models/deepseek-v2-lite \
    --output-dir results
```

Options:
- `--dtype bfloat16` (default) — halves memory during shard loading
- `--layers 0 5 10 ...` — audit a subset of layers
- `--gram-method latent` (default, O(r²)) or `full_gram` (small models only)
- `--angle-method svd` (default, exact) or `latent` (approximation, skips W_c)

Outputs `results/deepseek-v2-lite/exp1_metrics.npz` and a U-curve plot.

### Experiment 2: Activation Alignment

Requires the full model in RAM (V2-Lite only on this machine).

```bash
uv run scripts/run_exp2_activation.py \
    --model deepseek-v2-lite \
    --model-dir models/deepseek-v2-lite \
    --exp1-results results/deepseek-v2-lite/exp1_metrics \
    --dataset wikitext-2-raw-v1 \
    --max-tokens 50000
```

### Experiment 3: Surgical Clamping

```bash
uv run scripts/run_exp3_clamping.py \
    --model deepseek-v2-lite \
    --model-dir models/deepseek-v2-lite \
    --exp1-results results/deepseek-v2-lite/exp1_metrics \
    --schedules middle_block progressive_outward global \
    --skip-downstream   # add lm-eval benchmarks when needed
```

## Project Structure

```
configs/                  # per-model YAML (d, r, num_heads, HF weight key patterns)
src/
  proxy/toy_mla.py        # synthetic MLA for unit tests
  model_io/
    weight_extractor.py   # W_c / W_K↑ / W_V↑ slicing — single source of truth
    shard_loader.py       # shard-by-shard safetensors iteration
    model_loader.py       # full model load (transformers AutoModel)
    weight_clamper.py     # W_K↑ ← W_V↑ in-place or on state dict
  metrics/
    gram_distance.py      # D_Gram via [r×r] trace identities (O(r²), no [d×d] matrix)
    principal_angles.py   # θ̄ via SVD → W_c-projection → QR (O(d·r) memory)
    activation_alignment.py  # online ρ with Welford accumulator
  eval/
    clamp_schedules.py    # middle_block / progressive_outward / global
    perplexity.py         # sliding-window PPL on WikiText / C4
    downstream.py         # lm-eval subprocess wrapper
  experiments/
    exp1_weight_audit.py
    exp2_activation.py
    exp3_clamping.py
  utils/plotting.py       # U-curve, correlation scatter, ΔPPL plots
scripts/
  run_exp1_weight_audit.py
  run_exp2_activation.py
  run_exp3_clamping.py
tests/                    # 27 unit tests, all run without downloading any model
results/                  # .npz outputs and plots (created at runtime)
```

## Key Implementation Notes

**Gauge freedom**: the individual factor matrices W_c, W_K↑, W_V↑ are non-identifiable under invertible reparametrisations. All metrics are computed on the effective operators W̃_K = W_c W_K↑ and W̃_V = W_c W_V↑, which are gauge-invariant.

**Memory-efficient Gram distance**: avoids forming any [d × output_dim] matrix. Uses the identity tr(W̃_K W̃_K^T) = tr(C M_KK) where C = W_c^T W_c and M_KK = W_K↑^T W_K↑, all [r × r].

**Principal angles**: for full-rank up-projections, Col(W̃_K) = Col(W̃_V) = Col(W_c) exactly — angles are non-trivial only when effective rank < r (as in trained models). The SVD method computes Q_K via: SVD of W_K↑ → right singular vectors V_K → QR of W_c @ V_K, all in O(d·r) memory.

**HuggingFace weight names** (differ from paper notation):
| Paper | HuggingFace key | Shape |
|---|---|---|
| W_c | `kv_a_proj_with_mqa.weight[:r, :].T` | [d, r] |
| W_K↑ | `kv_b_proj.weight[:nh*qk_nope, :]` | [nh·qk_nope, r] |
| W_V↑ | `kv_b_proj.weight[nh*qk_nope:, :]` | [nh·v, r] |
