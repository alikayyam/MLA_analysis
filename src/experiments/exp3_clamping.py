"""Experiment 3: Surgical K=V parameter clamping.

Evaluates the perplexity and downstream task impact of forcing W_K_up ← W_V_up
in selected layers, under three scheduling strategies:
  - middle_block:       layers in [0.25L, 0.75L]
  - progressive_outward: iteratively add layers, sorted by ascending D_Gram
  - global:             all layers

Results are saved as a nested JSON/npz: {schedule: {n_layers: {delta_ppl, ...}}}.
"""

from __future__ import annotations
import json
import numpy as np
from pathlib import Path
import torch

from model_io.model_loader import load_model_and_tokenizer
from model_io.weight_clamper import clamp_layers_inplace
from eval.clamp_schedules import (
    middle_block_schedule,
    progressive_outward_schedule,
    global_schedule,
)
from eval.perplexity import compute_perplexity, compute_delta_ppl
from experiments.exp1_weight_audit import load_audit_results


def _eval_model(
    model,
    tokenizer,
    dataset: str,
    device: str,
    max_samples: int | None = None,
    run_downstream: bool = False,
    downstream_save_dir: str | None = None,
    tasks: list[str] | None = None,
) -> dict:
    ppl = compute_perplexity(
        model, tokenizer, dataset_name=dataset, device=device, max_samples=max_samples
    )
    result = {"ppl": ppl}

    if run_downstream and downstream_save_dir is not None:
        from eval.downstream import run_lm_eval, save_model_checkpoint
        save_model_checkpoint(model, tokenizer, downstream_save_dir)
        scores = run_lm_eval(
            model_path=downstream_save_dir,
            tasks=tasks or ["hellaswag", "winogrande", "arc_challenge", "piqa"],
            device=device,
        )
        result["downstream"] = scores

    return result


def run_clamping_experiment(
    model_dir: str,
    config: dict,
    exp1_results_path: str,
    output_dir: str,
    schedules: list[str] | None = None,
    eval_dataset: str = "wikitext-2-raw-v1",
    run_downstream: bool = False,
    device: str = "cpu",
    max_ppl_samples: int | None = None,
    ppl_max_length: int = 2048,
) -> dict:
    """Run all clamping schedules and return results.

    Args:
        model_dir:          Local model directory.
        config:             YAML config dict.
        exp1_results_path:  Path to Exp 1 .npz (needed for progressive schedule).
        output_dir:         Directory to save results.
        schedules:          Which schedules to run (default: all three).
        eval_dataset:       Dataset for perplexity evaluation.
        run_downstream:     Whether to run lm-eval downstream tasks.
        device:             "cpu" or "cuda".
        max_ppl_samples:    Limit number of PPL chunks (for fast iteration).
        ppl_max_length:     Context window for PPL (default 2048; use 512 on CPU-only machines).

    Returns:
        Nested dict: {schedule_name: {n_layers_clamped: {ppl, delta_ppl, ...}}}
    """
    if schedules is None:
        schedules = ["middle_block", "progressive_outward", "global"]

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    num_layers = config["num_hidden_layers"]

    # Load Exp 1 results for progressive schedule
    exp1 = load_audit_results(exp1_results_path + ".npz")
    gram_dist = exp1["gram_dist"]
    layer_indices = exp1["layer_idx"]

    model, tokenizer = load_model_and_tokenizer(model_dir, device_map=device)

    # Baseline perplexity (unmodified model)
    print("Computing baseline perplexity ...")
    baseline_ppl = compute_perplexity(
        model, tokenizer, dataset_name=eval_dataset, device=device,
        max_samples=max_ppl_samples, max_length=ppl_max_length,
        stride=min(ppl_max_length, 512),
    )
    print(f"Baseline PPL: {baseline_ppl:.3f}")

    all_results = {"baseline_ppl": baseline_ppl}

    for schedule_name in schedules:
        print(f"\n--- Schedule: {schedule_name} ---")
        sched_results = {}

        if schedule_name == "middle_block":
            target_layers = middle_block_schedule(num_layers)
            _run_single_clamp(
                model, tokenizer, config, target_layers, baseline_ppl,
                eval_dataset, device, max_ppl_samples, run_downstream,
                output_dir, schedule_name, sched_results,
                ppl_max_length=ppl_max_length,
            )

        elif schedule_name == "global":
            target_layers = global_schedule(num_layers)
            _run_single_clamp(
                model, tokenizer, config, target_layers, baseline_ppl,
                eval_dataset, device, max_ppl_samples, run_downstream,
                output_dir, schedule_name, sched_results,
                ppl_max_length=ppl_max_length,
            )

        elif schedule_name == "progressive_outward":
            layer_sets = progressive_outward_schedule(gram_dist, layer_indices)
            all_layers = layer_sets[-1] if layer_sets else []
            full_saved = _save_kv_weights(model, all_layers)
            for step_idx, cumulative_layers in enumerate(layer_sets):
                n = len(cumulative_layers)
                print(f"  Step {step_idx+1}/{len(layer_sets)}: clamping {n} layers ...")
                _restore_kv_weights(model, full_saved)
                clamp_layers_inplace(model, cumulative_layers, config)
                ppl = compute_perplexity(
                    model, tokenizer, dataset_name=eval_dataset, device=device,
                    max_samples=max_ppl_samples, max_length=ppl_max_length,
                    stride=min(ppl_max_length, 512),
                )
                delta = compute_delta_ppl(baseline_ppl, ppl)
                sched_results[n] = {"ppl": ppl, "delta_ppl": delta, "layers": cumulative_layers}
                print(f"    n_clamped={n}, PPL={ppl:.3f}, ΔPPL={delta:+.3f}")
            _restore_kv_weights(model, full_saved)

        all_results[schedule_name] = sched_results

    # Save results
    results_path = Path(output_dir) / "exp3_clamping_results.json"
    # Convert numpy types to native Python for JSON serialisation
    _save_json(all_results, results_path)
    print(f"\nSaved Exp 3 results to {results_path}")

    return all_results


def _save_kv_weights(model, layer_indices: list[int]) -> dict:
    """Save kv_b_proj weights for the given layers (shallow clone, not full model copy)."""
    saved = {}
    layers = getattr(model, "model", model).layers
    for idx in layer_indices:
        attn = _get_attn(layers[idx])
        if attn is not None and hasattr(attn, "kv_b_proj"):
            saved[idx] = attn.kv_b_proj.weight.data.clone()
    return saved


def _restore_kv_weights(model, saved: dict) -> None:
    layers = getattr(model, "model", model).layers
    for idx, weight_data in saved.items():
        attn = _get_attn(layers[idx])
        if attn is not None and hasattr(attn, "kv_b_proj"):
            attn.kv_b_proj.weight.data.copy_(weight_data)


def _get_attn(layer):
    for attr in ("self_attn", "attention", "attn"):
        if hasattr(layer, attr):
            return getattr(layer, attr)
    return None


def _run_single_clamp(
    model, tokenizer, config, target_layers, baseline_ppl,
    eval_dataset, device, max_ppl_samples, run_downstream,
    output_dir, schedule_name, sched_results, ppl_max_length=2048,
):
    n = len(target_layers)
    print(f"  Clamping {n} layers: {target_layers[:5]}{'...' if n > 5 else ''}")

    saved = _save_kv_weights(model, target_layers)
    clamp_layers_inplace(model, target_layers, config)
    ppl = compute_perplexity(
        model, tokenizer, dataset_name=eval_dataset, device=device,
        max_samples=max_ppl_samples, max_length=ppl_max_length,
        stride=min(ppl_max_length, 512),
    )
    _restore_kv_weights(model, saved)

    delta = compute_delta_ppl(baseline_ppl, ppl)
    result = {"ppl": ppl, "delta_ppl": delta, "layers": target_layers}
    sched_results[n] = result
    print(f"  PPL={ppl:.3f}, ΔPPL={delta:+.3f}")


def _save_json(obj, path):
    def _convert(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o

    def _recurse(o):
        if isinstance(o, dict):
            return {str(k): _recurse(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_recurse(v) for v in o]
        return _convert(o)

    with open(path, "w") as f:
        json.dump(_recurse(obj), f, indent=2)
