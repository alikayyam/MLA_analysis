#!/usr/bin/env python
"""CLI for Experiment 2: activation-based K-V alignment.

Requires a fully loaded model (fits in RAM). Use DeepSeek-V2-Lite on this machine.

Example:
    uv run scripts/run_exp2_activation.py \
        --model deepseek-v2-lite \
        --model-dir models/deepseek-v2-lite \
        --exp1-results results/deepseek-v2-lite/exp1_metrics \
        --output-dir results/deepseek-v2-lite \
        --dataset wikitext-2-raw-v1 \
        --max-tokens 50000
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yaml
from experiments.exp2_activation import run_activation_experiment
from utils.plotting import plot_correlation_scatter
import numpy as np

MODEL_CONFIG_MAP = {
    "deepseek-v2": "configs/deepseek_v2.yaml",
    "deepseek-v2-lite": "configs/deepseek_v2_lite.yaml",
    "deepseek-v3": "configs/deepseek_v3.yaml",
}


def parse_args():
    p = argparse.ArgumentParser(description="MLA activation alignment (Experiment 2)")
    p.add_argument("--model", choices=list(MODEL_CONFIG_MAP.keys()), required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--exp1-results", required=True,
                   help="Path to Exp 1 .npz file (without .npz extension)")
    p.add_argument("--output-dir", default="results")
    p.add_argument("--dataset", default="wikitext-2-raw-v1",
                   choices=["wikitext-2-raw-v1", "wikitext-103-raw-v1"])
    p.add_argument("--max-tokens", type=int, default=50_000)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(__file__).parent.parent

    config_path = root / MODEL_CONFIG_MAP[args.model]
    with open(config_path) as f:
        config = yaml.safe_load(f)

    output_dir = Path(args.output_dir) / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    results = run_activation_experiment(
        model_dir=args.model_dir,
        config=config,
        exp1_results_path=args.exp1_results,
        output_path=str(output_dir / "exp2_activation"),
        dataset_name=args.dataset,
        max_tokens=args.max_tokens,
        seq_len=args.seq_len,
    )

    if not args.no_plot and len(results.get("rho_common", [])) > 2:
        plot_correlation_scatter(
            gram_dist=results["gram_dist_common"],
            rho=results["rho_common"],
            model_name=args.model,
            output_path=str(output_dir / "correlation_scatter"),
        )


if __name__ == "__main__":
    main()
