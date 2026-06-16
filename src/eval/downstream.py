"""Downstream task evaluation using lm-evaluation-harness.

Shells out to `lm_eval` CLI to evaluate a model on zero-shot benchmarks.
The model must be saved to disk before calling (use save_model_checkpoint).
"""

from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path
import torch
import torch.nn as nn


DEFAULT_TASKS = ["hellaswag", "winogrande", "arc_challenge", "piqa"]


def run_lm_eval(
    model_path: str,
    tasks: list[str] = DEFAULT_TASKS,
    num_fewshot: int = 0,
    output_path: str | None = None,
    device: str = "cpu",
    trust_remote_code: bool = True,
) -> dict[str, float]:
    """Run lm-evaluation-harness on a local model checkpoint.

    Args:
        model_path:   Path to a saved HuggingFace model directory.
        tasks:        Task names as accepted by lm_eval (e.g. "hellaswag").
        num_fewshot:  Number of few-shot examples (0 = zero-shot).
        output_path:  Directory to write lm_eval JSON results (optional).
        device:       "cpu" or "cuda".
        trust_remote_code: Pass through to the model loader.

    Returns:
        dict mapping task_name -> accuracy (normalized).
    """
    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_path},trust_remote_code={trust_remote_code}",
        "--tasks", ",".join(tasks),
        "--num_fewshot", str(num_fewshot),
        "--device", device,
        "--output_path", output_path or "/tmp/lm_eval_results",
        "--log_samples",
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("lm_eval stderr:", result.stderr[-2000:])
        raise RuntimeError(f"lm_eval failed with code {result.returncode}")

    # Parse results from the JSON output file
    results_dir = Path(output_path or "/tmp/lm_eval_results")
    json_files = list(results_dir.glob("results_*.json"))
    if not json_files:
        raise FileNotFoundError(f"No lm_eval results JSON found in {results_dir}")

    with open(sorted(json_files)[-1]) as f:
        raw = json.load(f)

    scores = {}
    for task, metrics in raw.get("results", {}).items():
        # Primary metric is usually "acc_norm" or "acc"
        for key in ("acc_norm,none", "acc,none", "acc_norm", "acc"):
            if key in metrics:
                scores[task] = metrics[key]
                break

    return scores


def save_model_checkpoint(
    model: nn.Module,
    tokenizer,
    output_dir: str,
) -> str:
    """Save a (possibly modified) model and tokenizer to disk for lm_eval."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved model to {output_dir}")
    return output_dir
