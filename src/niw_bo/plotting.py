"""Static matplotlib figures for the campaign.

Ax computes the numbers (hypervolume, cross-validation, Sobol indices, model
predictions); these functions only draw them, so the figures look the same in Jupyter,
on GitHub and in the README, without Ax's interactive cards.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ax import Client
from ax.analysis import CrossValidationPlot, SensitivityAnalysisPlot, UtilityProgressionAnalysis
from matplotlib.colors import LinearSegmentedColormap

from niw_bo.config import LABELS, OBJECTIVE_THRESHOLDS, OBJECTIVES, PARAMETERS

# Colours: three categorical slots (one per batch group), a one-hue sequential ramp for
# model predictions, and neutral inks for text, axes and grid.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blues", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"


def label(name: str) -> str:
    """Axis label with unit, e.g. "Current density (mA/cm²)"."""
    return LABELS.get(name, name)


def short_label(name: str) -> str:
    """Label without unit, for use inside a sentence: "current density", "pH"."""
    base = label(name).split(" (")[0]
    return base[:1].lower() + base[1:]


def _unit(name: str) -> str:
    text = label(name)
    return text[text.find("(") + 1 : text.rfind(")")] if "(" in text else ""


# --- Observed data ---------------------------------------------------------------


def pareto_mask(values: np.ndarray) -> np.ndarray:
    """Boolean mask of the non-dominated rows of ``values`` (all columns maximized)."""
    values = np.asarray(values, dtype=float)
    mask = np.ones(len(values), dtype=bool)
    for i, point in enumerate(values):
        dominated_by = np.all(values >= point, axis=1) & np.any(values > point, axis=1)
        mask[i] = not dominated_by.any()
    return mask


def batch_groups(batches: pd.Series) -> pd.Series:
    """Map batch numbers to at most three groups: initial design, earlier Ax batches,
    and the latest Ax batch. Keeps the colours readable however many rounds are run."""
    last = int(batches.max())

    def group(batch: int) -> str:
        if batch == 0:
            return "initial design"
        if batch == last:
            return f"Ax batch {batch} (latest)"
        return "Ax batch 1" if last == 2 else f"Ax batches 1–{last - 1}"

    return batches.astype(int).map(group)


def plot_observed_front(
    df: pd.DataFrame, ax: plt.Axes | None = None, symlog_y: float | None = None
) -> plt.Axes:
    """Scatter the measured objectives by batch group and outline the observed Pareto front.

    Unmeasured rows are skipped and rows marked ``illustrative`` are drawn hollow.
    Dashed lines mark the objective thresholds: only points up and to the right of both
    lines count towards the hypervolume. ``symlog_y`` sets a
    symmetric-log y axis that is linear within ``±symlog_y``, which keeps a single large
    outlier from flattening all other points.
    """
    x_name, y_name = OBJECTIVES
    measured = df.dropna(subset=OBJECTIVES)
    ax = ax or plt.subplots(figsize=(7, 5))[1]
    if symlog_y is not None:
        ax.set_yscale("symlog", linthresh=symlog_y)

    _scatter_by_group(ax, measured[x_name], measured[y_name], measured["batch"],
                      _illustrative(measured))

    front = measured[pareto_mask(measured[OBJECTIVES].to_numpy())].sort_values(x_name)
    ax.step(front[x_name], front[y_name], where="pre", color=INK, lw=1, zorder=2)
    ax.scatter(front[x_name], front[y_name], s=170, facecolors="none", edgecolors=INK,
               linewidths=1.2, label="observed Pareto front", zorder=4)

    thresholds = _threshold_values()
    ax.axvline(thresholds[x_name], color=MUTED, ls="--", lw=1, zorder=1)
    ax.axhline(thresholds[y_name], color=MUTED, ls="--", lw=1, zorder=1)

    _finish(ax, "Measured objectives (both maximized)", label(x_name), label(y_name))
    _legend(ax)
    return ax


# --- Ax analyses, drawn with matplotlib ------------------------------------------


def plot_hypervolume(client: Client, df: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
    """Hypervolume dominated by the measured Pareto front after each completed trial."""
    data = _analysis_df(client, UtilityProgressionAnalysis())
    ax = ax or plt.subplots(figsize=(7, 4))[1]
    ax.plot(data["trial_index"], data["utility"], drawstyle="steps-post",
            color=SERIES[0], lw=2, zorder=3)
    ax.scatter(data["trial_index"], data["utility"], s=30, color=SERIES[0], zorder=4)

    # Mark where each batch starts; trial index = row number in the table.
    starts = df.reset_index().groupby("batch")["index"].min()
    top = data["utility"].max() * 1.12
    for batch, start in starts.items():
        if batch > 0:
            ax.axvline(start - 0.5, color=AXIS, lw=1, zorder=1)
        ax.text(start - 0.3, top, "initial design" if batch == 0 else f"batch {batch}",
                color=INK_SECONDARY, fontsize=9, va="top")
    ax.set_ylim(0, top * 1.05)
    ax.set_xticks(data["trial_index"])
    _finish(ax, "Hypervolume of the measured Pareto front", "Trial", "Hypervolume")
    return ax


def plot_cross_validation(
    client: Client,
    df: pd.DataFrame,
    metric: str,
    ax: plt.Axes | None = None,
    symlog: float | None = None,
) -> plt.Axes:
    """Leave-one-out cross-validation: each trial predicted from all the others.

    Error bars are the 95% predictive interval. Points on the diagonal are predicted
    exactly; a flat band means the model predicts little more than the mean. R² is
    computed in measured units. ``symlog`` works as in ``plot_observed_front``.
    """
    cards = _analysis_cards(client, CrossValidationPlot(metric_names=[metric], untransform=True))
    data = next(c.df for c in cards if "observed" in c.df.columns)
    r2 = next(c.df for c in cards if "R²" in c.df.columns)["R²"].iloc[0]
    trial = data["arm_name"].str.split("_").str[0].astype(int).to_numpy()
    batches = df["batch"].iloc[trial].reset_index(drop=True)
    illustrative = _illustrative(df)[trial]

    ax = ax or plt.subplots(figsize=(5, 5))[1]
    if symlog is not None:
        ax.set_xscale("symlog", linthresh=symlog)
        ax.set_yscale("symlog", linthresh=symlog)
    low = min(data["observed"].min(), (data["predicted"] - data["predicted_95_ci"]).min())
    high = max(data["observed"].max(), (data["predicted"] + data["predicted_95_ci"]).max())
    pad = 0.05 * (high - low)
    ax.plot([low - pad, high + pad], [low - pad, high + pad], color=MUTED, ls="--", lw=1)
    groups = batch_groups(batches)
    for i, name in enumerate(dict.fromkeys(groups)):
        for hollow in (False, True):
            rows = (groups == name).to_numpy() & (illustrative == hollow)
            if not rows.any():
                continue
            ax.errorbar(data["observed"][rows], data["predicted"][rows],
                        yerr=data["predicted_95_ci"][rows], fmt="o", ms=6, color=SERIES[i],
                        mfc="white" if hollow else SERIES[i], ecolor=SERIES[i],
                        elinewidth=1, alpha=0.9, zorder=3,
                        label=f"{name}, illustrative values" if hollow else name)
    ax.set_xlim(low - pad, high + pad)
    ax.set_ylim(low - pad, high + pad)
    if symlog is None:
        ax.set_aspect("equal")
    _finish(ax, f"{label(metric)}: R² = {r2:.2f}", "Measured", "Predicted (leave-one-out)")
    return ax


def plot_sensitivity(client: Client, metric: str, ax: plt.Axes | None = None) -> plt.Axes:
    """Total-order Sobol indices: the share of the model's predicted variation in
    ``metric`` that involves each parameter. Ax estimates them by Monte Carlo, so an
    unimportant parameter can come out slightly negative; those are shown as 0."""
    data = _analysis_df(client, SensitivityAnalysisPlot(metric_name=metric))
    data = data.assign(sensitivity=data["sensitivity"].clip(lower=0)).sort_values("sensitivity")
    ax = ax or plt.subplots(figsize=(7, 3))[1]
    names = [label(n) for n in data["parameter_name"]]
    ax.barh(names, data["sensitivity"], color=SERIES[0], height=0.6, zorder=3)
    for y, value in enumerate(data["sensitivity"]):
        ax.text(value + 0.01, y, f"{value:.2f}", va="center", color=INK_SECONDARY, fontsize=9)
    ax.set_xlim(0, max(1.0, data["sensitivity"].max() * 1.15))
    ax.grid(axis="y", visible=False)
    _finish(ax, f"Parameter importance for {short_label(metric)}",
            "Sobol index (total order)", None)
    return ax


def plot_contour(
    client: Client,
    df: pd.DataFrame,
    metric: str,
    x: str,
    y: str,
    fixed: dict | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Model-predicted ``metric`` over two parameters, with the others held fixed.

    By default the other parameters are fixed at the measured trial with the best value
    of ``metric``. Crosses show where experiments were run, projected onto these two
    parameters (their other parameters differ).
    """
    fixed, trial = _reference(df, metric, fixed)
    xs, ys = _grid(x, 60), _grid(y, 60)
    X, Y = np.meshgrid(xs, ys)
    Z = _predict_grid(client, fixed, {x: X.ravel(), y: Y.ravel()}, metric)[0].reshape(X.shape)

    ax = ax or plt.subplots(figsize=(7, 5))[1]
    filled = ax.contourf(X, Y, Z, levels=12, cmap=SEQUENTIAL)
    ax.contour(X, Y, Z, levels=filled.levels, colors="white", linewidths=0.4, alpha=0.6)
    measured = df.dropna(subset=[metric])
    ax.scatter(measured[x], measured[y], marker="x", s=30, color=INK, linewidths=1.2, zorder=3)
    colorbar = ax.figure.colorbar(filled, ax=ax, pad=0.02)
    colorbar.set_label(f"Predicted {short_label(metric)}"
                       + (f" ({_unit(metric)})" if _unit(metric) else ""), color=INK_SECONDARY)
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(colors=MUTED, labelcolor=INK_SECONDARY)

    _finish(ax, f"Predicted {short_label(metric)}", label(x), label(y),
            grid=False)
    ax.text(0, 1.015, _fixed_note(fixed, [x, y], trial), transform=ax.transAxes,
            fontsize=8.5, color=INK_SECONDARY)
    return ax


def plot_slices(
    client: Client, df: pd.DataFrame, metric: str, fixed: dict | None = None
) -> plt.Figure:
    """Model-predicted ``metric`` along each parameter in turn, with a 95% band.

    Other parameters are fixed as in ``plot_contour``. Ticks along the bottom show the
    values at which experiments were run.
    """
    fixed, trial = _reference(df, metric, fixed)
    names = [p.name for p in PARAMETERS]
    fig, axes = plt.subplots(1, len(names), figsize=(3.2 * len(names), 3.4), sharey=True)
    for ax, name in zip(axes, names):
        xs = _grid(name, 80)
        mean, sem = _predict_grid(client, fixed, {name: xs}, metric)
        ax.fill_between(xs, mean - 1.96 * sem, mean + 1.96 * sem, color=SERIES[0],
                        alpha=0.18, lw=0)
        ax.plot(xs, mean, color=SERIES[0], lw=2)
        ax.axvline(fixed[name], color=MUTED, ls=":", lw=1)
        ax.plot(df[name], np.zeros(len(df)), "|", color=INK_SECONDARY, ms=8,
                transform=ax.get_xaxis_transform())
        _finish(ax, None, label(name), label(metric) if ax is axes[0] else None)
    fig.suptitle(f"Predicted {short_label(metric)} along each parameter (95% band)",
                 x=0.01, y=0.99, ha="left", color=INK, fontsize=12)
    fig.text(0.01, 0.87, _fixed_note(fixed, [], trial) + " (dotted lines)",
             fontsize=8.5, color=INK_SECONDARY)
    fig.tight_layout()
    fig.subplots_adjust(top=0.78)
    return fig


# --- Helpers -----------------------------------------------------------------------


def ensure_fitted(client: Client) -> None:
    """Fit the surrogate model if it has not been fitted yet, so ``predict`` works
    before any trials are generated. ``Client`` has no public method for this;
    ``GenerationStrategy.fit`` is Ax's public entry point for fitting outside ``gen``."""
    strategy = client._generation_strategy_or_choose()
    if strategy.adapter is None:
        strategy.fit(experiment=client._experiment)


def _analysis_cards(client: Client, analysis) -> list:
    (card,) = client.compute_analyses([analysis], display=False)
    leaves = [c for c in card.flatten() if hasattr(c, "df") and not c.df.empty]
    if not leaves:
        raise RuntimeError(f"Ax could not compute {type(analysis).__name__}: {card.subtitle}")
    return leaves


def _analysis_df(client: Client, analysis) -> pd.DataFrame:
    return _analysis_cards(client, analysis)[0].df


def _reference(df: pd.DataFrame, metric: str, fixed: dict | None) -> tuple[dict, int | None]:
    if fixed is not None:
        return dict(fixed), None
    names = [p.name for p in PARAMETERS]
    best = df[metric].idxmax()
    return {n: _cast(n, df.loc[best, n]) for n in names}, int(best)


def _grid(name: str, n: int) -> np.ndarray:
    config = next(p for p in PARAMETERS if p.name == name)
    low, high = config.bounds
    if config.step_size:
        return np.arange(low, high + config.step_size / 2, config.step_size)
    values = np.linspace(low, high, n)
    return np.unique(values.round()) if config.parameter_type == "int" else values


def _cast(name: str, value) -> int | float:
    config = next(p for p in PARAMETERS if p.name == name)
    return int(round(value)) if config.parameter_type == "int" else float(value)


def _predict_grid(client, fixed, varying: dict, metric: str) -> tuple[np.ndarray, np.ndarray]:
    ensure_fitted(client)
    n = len(next(iter(varying.values())))
    points = [{**fixed, **{k: _cast(k, v[i]) for k, v in varying.items()}} for i in range(n)]
    predictions = client.predict(points)
    mean = np.array([p[metric][0] for p in predictions], dtype=float)
    sem = np.array([p[metric][1] for p in predictions], dtype=float)
    return mean, sem


def _fixed_note(fixed: dict, skip: list[str], trial: int | None) -> str:
    parts = [f"{short_label(k)} {v:.3g} {_unit(k)}".strip()
             for k, v in fixed.items() if k not in skip]
    source = f"trial {trial}" if trial is not None else "chosen values"
    return f"Other parameters fixed at {source}: " + ", ".join(parts)


def _illustrative(df: pd.DataFrame) -> np.ndarray:
    """Boolean array: which rows hold illustrative (not measured) objective values."""
    if "source" not in df.columns:
        return np.zeros(len(df), dtype=bool)
    return (df["source"] == "illustrative").to_numpy()


def _scatter_by_group(ax, x, y, batches, illustrative) -> None:
    groups = batch_groups(batches)
    for i, name in enumerate(dict.fromkeys(groups)):
        for hollow in (False, True):
            rows = (groups == name).to_numpy() & (illustrative == hollow)
            if not rows.any():
                continue
            ax.scatter(x[rows], y[rows], s=55, zorder=3,
                       color="white" if hollow else SERIES[i],
                       edgecolors=SERIES[i] if hollow else "white",
                       linewidths=1.5 if hollow else 0.8,
                       label=f"{name}, illustrative values" if hollow else name)


def _threshold_values() -> dict[str, float]:
    """Parse ``"metric >= value"`` strings from the config into ``{metric: value}``."""
    thresholds = {}
    for constraint in OBJECTIVE_THRESHOLDS:
        name, value = (part.strip() for part in constraint.split(">="))
        thresholds[name] = float(value)
    return thresholds


def _finish(ax, title, xlabel, ylabel, grid: bool = True) -> None:
    if title:
        ax.set_title(title, loc="left", color=INK, fontsize=12, pad=18)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_SECONDARY)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_SECONDARY)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=AXIS, labelcolor=INK_SECONDARY, labelsize=9)
    if grid:
        ax.grid(True, color=GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)


def _legend(ax) -> None:
    legend = ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    legend.set_zorder(5)
