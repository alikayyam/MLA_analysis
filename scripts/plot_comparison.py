#!/usr/bin/env python
"""Generate a multi-model U-curve comparison plot from Exp 1 results.

Usage:
    uv run scripts/plot_comparison.py \
        --results results/deepseek-v2-lite/exp1_metrics.npz \
                  results/deepseek-v2/exp1_metrics.npz \
                  results/deepseek-v3/exp1_metrics.npz \
        --labels "V2-Lite (16B)" "V2 (236B)" "V3 (671B)" \
        --output results/comparison_u_curve
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from utils.plotting import plot_u_curve_multi


def parse_args():
    p = argparse.ArgumentParser(description="Multi-model U-curve comparison plot")
    p.add_argument(
        "--results", nargs="+", required=True,
        help=".npz files from run_exp1_weight_audit.py (one per model)",
    )
    p.add_argument(
        "--labels", nargs="+", required=True,
        help="Model labels (same order as --results)",
    )
    p.add_argument(
        "--output", default="results/comparison_u_curve",
        help="Output path prefix (no extension)",
    )
    p.add_argument(
        "--shade-regimes", action="store_true",
        help="Shade Early/Middle/Late regimes",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if len(args.results) != len(args.labels):
        print(f"ERROR: {len(args.results)} result files but {len(args.labels)} labels")
        sys.exit(1)

    model_results = []
    for path, label in zip(args.results, args.labels):
        data = np.load(path)
        model_results.append((dict(data), label))
        print(f"Loaded {label}: {len(data['layer_idx'])} layers")

    plot_u_curve_multi(
        model_results,
        output_path=args.output,
        shade_regimes=args.shade_regimes,
    )


if __name__ == "__main__":
    main()
