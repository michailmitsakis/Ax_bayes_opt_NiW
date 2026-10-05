"""Regenerate the figures in docs/img from data/experiments.csv.

The results figures use the lab rows only; the hypervolume figure shows the whole table
to demonstrate the loop, with the illustrative trials marked. Run after each round:

    python scripts/make_figures.py
"""

import logging
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from ax.utils.common.logger import set_stderr_log_level  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from niw_bo import (  # noqa: E402
    attach_experiments,
    build_client,
    lab_only,
    load_experiments,
    plot_contour,
    plot_cross_validation,
    plot_hypervolume,
    plot_observed_front,
    plot_sensitivity,
    plot_slices,
)
from niw_bo.plotting import INK, INK_SECONDARY, LEGEND, MODEL, SERIES, TITLE  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "experiments.csv"
OUT = ROOT / "docs" / "img"


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / f"{name}.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {name}.png")


def plot_workflow() -> plt.Figure:
    """The optimization loop as a diagram: table -> model -> batch -> lab -> table."""
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 3.6)
    ax.axis("off")
    steps = [
        ("Experiments table", "data/experiments.csv\nall recipes and results", SERIES[0]),
        ("Surrogate model", "Ax fits one Gaussian\nprocess per objective", MODEL),
        ("Next batch", "qLogNEHVI picks\nthe next 3 recipes", MODEL),
        ("Lab", "deposit Ni-W films,\nmeasure HER activity", SERIES[1]),
    ]
    width, height, gap, y = 2.35, 1.5, 0.43, 1.75
    centers = []
    for i, (title, body, color) in enumerate(steps):
        x = 0.15 + i * (width + gap)
        ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0,rounding_size=0.18",
                                    facecolor="white", edgecolor=color, linewidth=2))
        ax.add_patch(FancyBboxPatch((x, y + height - 0.5), width, 0.5,
                                    boxstyle="round,pad=0,rounding_size=0.18",
                                    facecolor=color, edgecolor=color, linewidth=2))
        ax.text(x + width / 2, y + height - 0.25, title, ha="center", va="center",
                color="white", fontsize=13, fontweight="bold")
        ax.text(x + width / 2, y + (height - 0.5) / 2, body, ha="center", va="center",
                color=INK, fontsize=LEGEND + 0.5, linespacing=1.35)
        centers.append(x + width / 2)
        if i:
            ax.add_patch(FancyArrowPatch((x - gap + 0.04, y + height / 2), (x - 0.04, y + height / 2),
                                         arrowstyle="-|>", mutation_scale=20, color=INK_SECONDARY,
                                         linewidth=1.8))
    ax.add_patch(FancyArrowPatch((centers[-1], y - 0.04), (centers[0], y - 0.04),
                                 connectionstyle="arc3,rad=-0.22", arrowstyle="-|>",
                                 mutation_scale=20, color=INK_SECONDARY, linewidth=1.8))
    ax.text((centers[0] + centers[-1]) / 2, 0.42, "measured values go into the table; repeat",
            ha="center", va="center", color=INK_SECONDARY, fontsize=LEGEND + 0.5, style="italic")
    return fig


def main() -> None:
    set_stderr_log_level(logging.WARNING)
    warnings.filterwarnings("ignore")
    OUT.mkdir(parents=True, exist_ok=True)

    table = load_experiments(DATA)
    lab = lab_only(table)
    client = build_client(seed=0)
    attach_experiments(client, lab)
    demo_client = build_client(seed=0)
    attach_experiments(demo_client, table)

    save(plot_workflow(), "workflow")
    save(plot_observed_front(lab, client=client).figure, "pareto_front")
    save(plot_hypervolume(demo_client, table).figure, "hypervolume")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    plot_cross_validation(client, lab, "overpotential", ax=axes[0])
    plot_cross_validation(client, lab, "overpotential_slope", ax=axes[1])
    fig.tight_layout(w_pad=3)
    save(fig, "cross_validation")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), gridspec_kw={"width_ratios": [1, 1.3]})
    plot_sensitivity(client, "overpotential", ax=axes[0])
    plot_contour(client, lab, "overpotential", "current_density", "pH", ax=axes[1])
    axes[0].set_title(axes[0].get_title(), color=INK, fontsize=TITLE, pad=42)  # align titles
    fig.tight_layout(w_pad=3)
    save(fig, "model_insights")

    save(plot_slices(client, lab, "overpotential"), "slices")


if __name__ == "__main__":
    main()
