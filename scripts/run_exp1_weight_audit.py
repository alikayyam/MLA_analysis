#!/usr/bin/env python
"""CLI for Experiment 1: gauge-invariant layer-wise weight audit.

Example:
    uv run scripts/run_exp1_weight_audit.py \
        --model deepseek-v2-lite \
        --model-dir models/deepseek-v2-lite \
        --output-dir results/deepseek-v2-lite
"""

import argparse
import sys
from pathlib import Path

# Add src to path when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import yaml
from experiments.exp1_weight_audit import run_weight_audit
from utils.plotting import plot_u_curve

MODEL_CONFIG_MAP = {
    "deepseek-v2": "configs/deepseek_v2.yaml",
    "deepseek-v2-lite": "configs/deepseek_v2_lite.yaml",
    "deepseek-v3": "configs/deepseek_v3.yaml",
}


def parse_args():
    p = argparse.ArgumentParser(description="MLA weight audit (Experiment 1)")
    p.add_argument("--model", choices=list(MODEL_CONFIG_MAP.keys()), required=True)
    p.add_argument("--model-dir", required=True, help="Path to local safetensors directory")
    p.add_argument("--output-dir", default="results", help="Directory to save .npz and plots")
    p.add_argument(
        "--dtype", choices=["float32", "bfloat16"], default="bfloat16",
        help="Dtype for weight loading (bfloat16 halves memory usage)",
    )
    p.add_argument(
        "--gram-method", choices=["latent", "full_gram"], default="latent",
        help="latent: memory-safe O(r^2); full_gram: explicit O(d^2), small models only",
    )
    p.add_argument(
        "--angle-method", choices=["svd", "latent"], default="svd",
        help="svd: correct O(d*r); latent: O(r^2) approximation (ignores W_c geometry)",
    )
    p.add_argument(
        "--layers", type=int, nargs="+", default=None,
        help="Subset of layer indices to audit (default: all)",
    )
    p.add_argument("--no-plot", action="store_true", help="Skip plot generation")
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(__file__).parent.parent

    config_path = root / MODEL_CONFIG_MAP[args.model]
    with open(config_path) as f:
        config = yaml.safe_load(f)

    output_dir = Path(args.output_dir) / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = str(output_dir / "exp1_metrics")

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    print(f"Model:       {args.model}")
    print(f"Model dir:   {args.model_dir}")
    print(f"Dtype:       {args.dtype}")
    print(f"Gram method: {args.gram_method}")
    print(f"Angle method:{args.angle_method}")

    results = run_weight_audit(
        model_dir=args.model_dir,
        config=config,
        output_path=npz_path,
        layer_indices=args.layers,
        dtype=dtype,
        gram_method=args.gram_method,
        angle_method=args.angle_method,
    )

    if not args.no_plot:
        plot_path = str(output_dir / "u_curve")
        plot_u_curve(results, model_name=args.model, output_path=plot_path)

    print("\nSummary:")
    print(f"  Layers audited: {len(results['layer_idx'])}")
    print(f"  D_Gram  min/mean/max: {results['gram_dist'].min():.3f} / "
          f"{results['gram_dist'].mean():.3f} / {results['gram_dist'].max():.3f}")
    print(f"  θ̄ (°)  min/mean/max: {results['mean_angle'].min():.4f} / "
          f"{results['mean_angle'].mean():.4f} / {results['mean_angle'].max():.4f}")


if __name__ == "__main__":
    main()
