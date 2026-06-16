#!/usr/bin/env python
"""CLI for Experiment 3: surgical K=V parameter clamping.

Example:
    uv run scripts/run_exp3_clamping.py \
        --model deepseek-v2-lite \
        --model-dir models/deepseek-v2-lite \
        --exp1-results results/deepseek-v2-lite/exp1_metrics \
        --output-dir results/deepseek-v2-lite \
        --schedules middle_block global \
        --skip-downstream
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yaml
from experiments.exp3_clamping import run_clamping_experiment
from utils.plotting import plot_delta_ppl

MODEL_CONFIG_MAP = {
    "deepseek-v2": "configs/deepseek_v2.yaml",
    "deepseek-v2-lite": "configs/deepseek_v2_lite.yaml",
    "deepseek-v3": "configs/deepseek_v3.yaml",
}

VALID_SCHEDULES = ["middle_block", "progressive_outward", "global"]


def parse_args():
    p = argparse.ArgumentParser(description="MLA surgical clamping (Experiment 3)")
    p.add_argument("--model", choices=list(MODEL_CONFIG_MAP.keys()), required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--exp1-results", required=True,
                   help="Path to Exp 1 .npz (without .npz extension)")
    p.add_argument("--output-dir", default="results")
    p.add_argument("--schedules", nargs="+", choices=VALID_SCHEDULES,
                   default=["middle_block", "global"])
    p.add_argument("--dataset", default="wikitext-2-raw-v1",
                   choices=["wikitext-2-raw-v1", "wikitext-103-raw-v1", "c4"])
    p.add_argument("--skip-downstream", action="store_true",
                   help="Skip lm-eval downstream benchmarks (PPL only)")
    p.add_argument("--max-ppl-samples", type=int, default=None,
                   help="Limit PPL computation to N chunks (for fast iteration)")
    p.add_argument("--ppl-max-length", type=int, default=2048,
                   help="Context window per PPL chunk (use 512 on CPU-only machines)")
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(__file__).parent.parent

    config_path = root / MODEL_CONFIG_MAP[args.model]
    with open(config_path) as f:
        config = yaml.safe_load(f)

    output_dir = str(Path(args.output_dir) / args.model)

    results = run_clamping_experiment(
        model_dir=args.model_dir,
        config=config,
        exp1_results_path=args.exp1_results,
        output_dir=output_dir,
        schedules=args.schedules,
        eval_dataset=args.dataset,
        run_downstream=not args.skip_downstream,
        max_ppl_samples=args.max_ppl_samples,
        ppl_max_length=args.ppl_max_length,
    )

    if not args.no_plot:
        # Build {schedule: {n_layers: delta_ppl}} for plotting
        plot_data = {}
        for sched in args.schedules:
            if sched in results:
                plot_data[sched] = {
                    k: v["delta_ppl"] for k, v in results[sched].items()
                }
        if plot_data:
            plot_delta_ppl(
                results=plot_data,
                output_path=str(Path(output_dir) / "delta_ppl"),
                title=f"ΔPPL vs clamped layers — {args.model}",
            )


if __name__ == "__main__":
    main()
