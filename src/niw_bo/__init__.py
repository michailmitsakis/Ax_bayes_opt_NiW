"""Multi-objective Bayesian optimization of Ni-W electrodeposition with Ax."""

from niw_bo.campaign import (
    attach_experiments,
    build_client,
    lab_only,
    load_experiments,
    suggest_next_batch,
)
from niw_bo.plotting import (
    plot_contour,
    plot_cross_validation,
    plot_hypervolume,
    plot_observed_front,
    plot_sensitivity,
    plot_slices,
    predicted_front,
)

__all__ = [
    "attach_experiments",
    "build_client",
    "lab_only",
    "load_experiments",
    "plot_contour",
    "plot_cross_validation",
    "plot_hypervolume",
    "plot_observed_front",
    "plot_sensitivity",
    "plot_slices",
    "predicted_front",
    "suggest_next_batch",
]
