"""Regenerate the figures in docs/img from the lab rows of data/experiments.csv.

Rows marked ``illustrative`` are left out, so the README figures show measurements only.
Run after each optimization round:

    python scripts/make_figures.py
"""

import logging
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from ax.utils.common.logger import set_stderr_log_level  # noqa: E402

from niw_bo import (  # noqa: E402
    attach_experiments,
    build_client,
    load_experiments,
    plot_contour,
    plot_cross_validation,
    plot_observed_front,
    plot_sensitivity,
    plot_slices,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "experiments.csv"
OUT = ROOT / "docs" / "img"


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / f"{name}.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {name}.png")


def main() -> None:
    set_stderr_log_level(logging.WARNING)
    warnings.filterwarnings("ignore")
    OUT.mkdir(parents=True, exist_ok=True)

    df = load_experiments(DATA, include_illustrative=False)
    client = build_client(seed=0)
    attach_experiments(client, df)

    save(plot_observed_front(df).figure, "observed_front")
    save(plot_sensitivity(client, "overpotential").figure, "sensitivity")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    plot_cross_validation(client, df, "overpotential", ax=axes[0])
    plot_cross_validation(client, df, "overpotential_slope", ax=axes[1])
    axes[0].legend(frameon=False, fontsize=9, labelcolor="#52514e", loc="upper left")
    fig.tight_layout()
    save(fig, "cross_validation")

    save(plot_contour(client, df, "overpotential", "current_density", "pH").figure, "contour")
    save(plot_slices(client, df, "overpotential"), "slices")


if __name__ == "__main__":
    main()
