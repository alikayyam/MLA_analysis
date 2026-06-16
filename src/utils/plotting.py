"""Plotting utilities for MLA geometry experiments."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


def plot_u_curve(
    results: dict,
    model_name: str,
    output_path: str,
    second_results: dict | None = None,
    second_model_name: str | None = None,
) -> None:
    """Plot layer-wise D_Gram and θ̄ as a U-shaped curve.

    Args:
        results:           dict with keys layer_idx, gram_dist, mean_angle.
        model_name:        Label for the primary model (solid line).
        output_path:       Save path (without extension; saves .pdf and .png).
        second_results:    Optional second model results (dashed line).
        second_model_name: Label for second model.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    _plot_panel(axes[0], results, model_name, "gram_dist", "D_Gram",
                second_results, second_model_name)
    _plot_panel(axes[1], results, model_name, "mean_angle", "θ̄ (degrees)",
                second_results, second_model_name)

    fig.suptitle(
        f"Layer-wise K–V subspace alignment: {model_name}",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out) + ".pdf", bbox_inches="tight")
    fig.savefig(str(out) + ".png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved U-curve plot to {output_path}.{{pdf,png}}")


def _plot_panel(
    ax,
    results: dict,
    model_name: str,
    key: str,
    ylabel: str,
    second_results: dict | None,
    second_name: str | None,
) -> None:
    layers = results["layer_idx"]
    values = results[key]
    L = len(layers)

    ax.plot(layers, values, color="steelblue", linewidth=2, label=model_name)

    if second_results is not None:
        ax.plot(
            second_results["layer_idx"], second_results[key],
            color="tomato", linewidth=2, linestyle="--",
            label=second_name or "model 2",
        )

    # Shade the three functional regimes
    lo, hi = layers[0], layers[-1]
    early_end = lo + int(0.2 * (hi - lo))
    mid_end = lo + int(0.8 * (hi - lo))
    ax.axvspan(lo, early_end, alpha=0.08, color="green", label="Early (decoupled)")
    ax.axvspan(early_end, mid_end, alpha=0.08, color="orange", label="Middle (collapsed)")
    ax.axvspan(mid_end, hi, alpha=0.08, color="purple", label="Late (diverging)")

    ax.set_xlabel("Layer index", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_u_curve_multi(
    model_results: list[tuple[dict, str]],
    output_path: str,
    title: str = "Layer-wise K–V subspace alignment",
    shade_regimes: bool = False,
) -> None:
    """Plot layer-wise D_Gram and θ̄ for multiple models on the same axes.

    Args:
        model_results: list of (results_dict, model_name) pairs.
        output_path:   Save path prefix (no extension); saves .pdf and .png.
        title:         Figure suptitle.
        shade_regimes: If True, shade Early/Middle/Late regimes from the first model.
    """
    colors = ["steelblue", "tomato", "seagreen", "darkorange", "purple"]
    linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for metric, ylabel, ax in [
        ("gram_dist", "D_Gram", axes[0]),
        ("mean_angle", "θ̄ (degrees)", axes[1]),
    ]:
        for k, (res, name) in enumerate(model_results):
            layers = res["layer_idx"]
            # Normalise layer index to [0, 1] for cross-model comparison
            norm_layers = (layers - layers[0]) / max(layers[-1] - layers[0], 1)
            ax.plot(
                norm_layers, res[metric],
                color=colors[k % len(colors)],
                linestyle=linestyles[k % len(linestyles)],
                linewidth=2, label=name,
            )

        if shade_regimes and model_results:
            ax.axvspan(0.0, 0.2, alpha=0.06, color="green")
            ax.axvspan(0.2, 0.8, alpha=0.06, color="orange")
            ax.axvspan(0.8, 1.0, alpha=0.06, color="purple")

        ax.set_xlabel("Normalised layer depth", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out) + ".pdf", bbox_inches="tight")
    fig.savefig(str(out) + ".png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved multi-model U-curve plot to {output_path}.{{pdf,png}}")


def plot_correlation_scatter(
    gram_dist: np.ndarray,
    rho: np.ndarray,
    model_name: str,
    output_path: str,
) -> None:
    """Scatter plot of static D_Gram vs runtime activation alignment ρ."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(gram_dist, rho, color="steelblue", alpha=0.7, s=60)
    ax.set_xlabel("D_Gram (static weight metric)", fontsize=11)
    ax.set_ylabel("ρ (activation cosine similarity)", fontsize=11)
    ax.set_title(f"Static vs runtime alignment: {model_name}", fontsize=12)
    ax.grid(True, alpha=0.3)

    # Fit and display linear trendline
    if len(gram_dist) > 2:
        m, b = np.polyfit(gram_dist, rho, 1)
        x_line = np.linspace(gram_dist.min(), gram_dist.max(), 100)
        ax.plot(x_line, m * x_line + b, color="tomato", linestyle="--", linewidth=1.5)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out) + ".pdf", bbox_inches="tight")
    fig.savefig(str(out) + ".png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved correlation plot to {output_path}.{{pdf,png}}")


def plot_delta_ppl(
    results: dict,
    output_path: str,
    title: str = "ΔPPL vs clamped layers",
) -> None:
    """Plot ΔPPL as a function of number of clamped layers per schedule.

    Args:
        results: {schedule_name: {n_layers: delta_ppl, ...}, ...}
        output_path: Save path prefix.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"middle_block": "steelblue", "progressive_outward": "darkorange", "global": "tomato"}

    for schedule, data in results.items():
        n_layers = sorted(data.keys())
        delta_ppls = [data[n] for n in n_layers]
        color = colors.get(schedule, "gray")
        ax.plot(n_layers, delta_ppls, marker="o", label=schedule, color=color, linewidth=2)

    ax.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("Number of clamped layers", fontsize=11)
    ax.set_ylabel("ΔPPL (positive = degradation)", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out) + ".pdf", bbox_inches="tight")
    fig.savefig(str(out) + ".png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved ΔPPL plot to {output_path}.{{pdf,png}}")
