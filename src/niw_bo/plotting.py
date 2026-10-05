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
from scipy.stats import qmc

from niw_bo.config import LABELS, OBJECTIVE_THRESHOLDS, OBJECTIVES, PARAMETERS

# Colours: three categorical slots (one per batch group), a separate hue for model
# predictions, a one-hue sequential ramp for contour plots, and neutral inks.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
MODEL = "#4a3aa7"
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blues", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SHADE = "#f3f2ee"

# Font sizes in points, chosen so figures stay legible when GitHub scales them down.
TITLE, LABEL, TICK, LEGEND, NOTE = 14, 12, 11, 10.5, 10


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


# --- Pareto front --------------------------------------------------------------------


def pareto_mask(values: np.ndarray) -> np.ndarray:
    """Boolean mask of the non-dominated rows of ``values`` (all columns maximized)."""
    values = np.asarray(values, dtype=float)
    mask = np.ones(len(values), dtype=bool)
    for i, point in enumerate(values):
        dominated_by = np.all(values >= point, axis=1) & np.any(values > point, axis=1)
        mask[i] = not dominated_by.any()
    return mask


def predicted_front(client: Client, n: int = 4096, seed: int = 0) -> pd.DataFrame:
    """The model's predicted Pareto front.

    Predicts both objectives at ``n`` quasi-random recipes spread over the search space
    and keeps the non-dominated predicted means. Returns one row per recipe with its
    parameters and, for each objective, the predicted mean and standard error
    (``<objective>_sem``), sorted by the first objective.
    """
    ensure_fitted(client)
    unit = qmc.Sobol(d=len(PARAMETERS), seed=seed).random(n)
    points = []
    for row in unit:
        point = {}
        for u, config in zip(row, PARAMETERS):
            low, high = config.bounds
            value = low + u * (high - low)
            if config.step_size:
                value = low + round((value - low) / config.step_size) * config.step_size
            point[config.name] = _cast(config.name, value)
        points.append(point)
    predictions = client.predict(points)
    means = np.array([[p[m][0] for m in OBJECTIVES] for p in predictions], dtype=float)
    sems = np.array([[p[m][1] for m in OBJECTIVES] for p in predictions], dtype=float)
    keep = pareto_mask(means)
    front = pd.DataFrame(points)[keep].reset_index(drop=True)
    for j, name in enumerate(OBJECTIVES):
        front[name] = means[keep, j]
        front[f"{name}_sem"] = sems[keep, j]
    return front.drop_duplicates(subset=OBJECTIVES).sort_values(OBJECTIVES[0]).reset_index(drop=True)


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
    df: pd.DataFrame,
    client: Client | None = None,
    ax: plt.Axes | None = None,
    symlog_y: float | None = None,
) -> plt.Axes:
    """Measured objectives by batch group, the observed Pareto front, and, if ``client``
    is given, the model's predicted Pareto front with 95% intervals on both objectives.

    Unmeasured rows are skipped and rows marked ``illustrative`` are drawn hollow.
    Dashed lines mark the objective thresholds: only points up and to the right of both
    lines count towards the hypervolume. ``symlog_y`` sets a symmetric-log y axis that
    is linear within ``±symlog_y``, which keeps a single large outlier from flattening
    all other points.
    """
    x_name, y_name = OBJECTIVES
    measured = df.dropna(subset=OBJECTIVES)
    ax = ax or plt.subplots(figsize=(9, 5.5))[1]
    if symlog_y is not None:
        ax.set_yscale("symlog", linthresh=symlog_y)

    if client is not None:
        front = predicted_front(client)
        ax.errorbar(front[x_name], front[y_name],
                    xerr=1.96 * front[f"{x_name}_sem"], yerr=1.96 * front[f"{y_name}_sem"],
                    fmt="none", ecolor=MODEL, elinewidth=1, capsize=2.5, alpha=0.3, zorder=2)
        ax.plot(front[x_name], front[y_name], "-D", color=MODEL, lw=2, ms=6, zorder=3,
                label="model-predicted Pareto front (95% intervals)")

    _scatter_by_group(ax, measured[x_name], measured[y_name], measured["batch"],
                      _illustrative(measured))
    observed = measured[pareto_mask(measured[OBJECTIVES].to_numpy())]
    ax.scatter(observed[x_name], observed[y_name], s=220, facecolors="none", edgecolors=INK,
               linewidths=1.4, label="measured Pareto front", zorder=5)

    thresholds = _threshold_values()
    ax.axvline(thresholds[x_name], color=MUTED, ls="--", lw=1.2, zorder=1,
               label="objective thresholds")
    ax.axhline(thresholds[y_name], color=MUTED, ls="--", lw=1.2, zorder=1)

    title = "Measured objectives" + (" and predicted Pareto front" if client else "")
    _finish(ax, f"{title} (both maximized)", label(x_name), label(y_name))
    _legend(ax, loc="upper left")
    return ax


# --- Ax analyses, drawn with matplotlib ------------------------------------------


def plot_hypervolume(client: Client, df: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
    """Hypervolume dominated by the measured Pareto front after each completed trial.

    Batches are labelled along the top. Trials with illustrative values are shaded and
    drawn with a dashed line and hollow markers.
    """
    data = _analysis_df(client, UtilityProgressionAnalysis())
    trials = data["trial_index"].to_numpy()
    utility = data["utility"].to_numpy()
    illustrative = _illustrative(df)[trials]
    ax = ax or plt.subplots(figsize=(9, 4.5))[1]

    top = utility.max() * 1.18
    cut = trials[illustrative].min() if illustrative.any() else trials.max() + 1
    if illustrative.any():
        ax.axvspan(cut - 0.5, trials.max() + 0.5, color=SHADE, zorder=0)
        ax.text(trials.max() + 0.4, top * 0.06, "illustrative values", ha="right",
                color=INK_SECONDARY, fontsize=NOTE, style="italic")
        dashed = trials >= cut - 1
        ax.plot(trials[dashed], utility[dashed], drawstyle="steps-post", color=SERIES[0],
                lw=2.2, ls="--", zorder=3)
    solid = trials < cut
    ax.plot(trials[solid], utility[solid], drawstyle="steps-post", color=SERIES[0], lw=2.2, zorder=3)
    ax.scatter(trials[~illustrative], utility[~illustrative], s=45, color=SERIES[0], zorder=4)
    ax.scatter(trials[illustrative], utility[illustrative], s=45, color="white",
               edgecolors=SERIES[0], linewidths=1.6, zorder=4)

    # Batch labels; trial index = row number in the table.
    starts = df.reset_index().groupby("batch")["index"].min()
    for batch, start in starts.items():
        if batch > 0:
            ax.axvline(start - 0.5, color=AXIS, lw=1, zorder=1)
        ax.text(start - 0.35, top * 0.98, "initial design" if batch == 0 else f"Ax batch {batch}",
                color=INK_SECONDARY, fontsize=NOTE, va="top")
    ax.set_ylim(0, top)
    ax.set_xlim(trials.min() - 0.5, trials.max() + 0.5)
    ax.set_xticks(trials)
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

    ax = ax or plt.subplots(figsize=(5.5, 5.5))[1]
    if symlog is not None:
        ax.set_xscale("symlog", linthresh=symlog)
        ax.set_yscale("symlog", linthresh=symlog)
    low = min(data["observed"].min(), (data["predicted"] - data["predicted_95_ci"]).min())
    high = max(data["observed"].max(), (data["predicted"] + data["predicted_95_ci"]).max())
    pad = 0.05 * (high - low)
    ax.plot([low - pad, high + pad], [low - pad, high + pad], color=MUTED, ls="--", lw=1.2)
    groups = batch_groups(batches)
    for i, name in enumerate(dict.fromkeys(groups)):
        for hollow in (False, True):
            rows = (groups == name).to_numpy() & (illustrative == hollow)
            if not rows.any():
                continue
            ax.errorbar(data["observed"][rows], data["predicted"][rows],
                        yerr=data["predicted_95_ci"][rows], fmt="o", ms=7, color=SERIES[i],
                        mfc="white" if hollow else SERIES[i], ecolor=SERIES[i],
                        elinewidth=1.2, capsize=2.5, alpha=0.9, zorder=3,
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
    ax = ax or plt.subplots(figsize=(8, 3.6))[1]
    names = [label(n).split(" (")[0] for n in data["parameter_name"]]
    ax.barh(names, data["sensitivity"], color=SERIES[0], height=0.6, zorder=3)
    for y, value in enumerate(data["sensitivity"]):
        ax.text(value + 0.015, y, f"{value:.2f}", va="center", color=INK_SECONDARY,
                fontsize=LEGEND)
    ax.set_xlim(0, min(1.0, max(0.3, data["sensitivity"].max() * 1.25)))
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

    ax = ax or plt.subplots(figsize=(8, 5.5))[1]
    filled = ax.contourf(X, Y, Z, levels=12, cmap=SEQUENTIAL)
    ax.contour(X, Y, Z, levels=filled.levels, colors="white", linewidths=0.4, alpha=0.6)
    measured = df.dropna(subset=[metric])
    ax.scatter(measured[x], measured[y], marker="x", s=45, color=INK, linewidths=1.5, zorder=3,
               label="experiments (projected)")
    colorbar = ax.figure.colorbar(filled, ax=ax, pad=0.02)
    colorbar.set_label(f"Predicted {short_label(metric)}"
                       + (f" ({_unit(metric)})" if _unit(metric) else ""),
                       color=INK_SECONDARY, fontsize=LABEL)
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(colors=MUTED, labelcolor=INK_SECONDARY, labelsize=TICK)

    _finish(ax, f"Predicted {short_label(metric)}", label(x), label(y), grid=False)
    ax.title.set_position((0.5, 1.0))
    ax.set_title(ax.get_title(), color=INK, fontsize=TITLE, pad=42)
    ax.text(0.5, 1.025, _fixed_note(fixed, [x, y], trial).replace(": ", ":\n", 1),
            transform=ax.transAxes, ha="center", va="bottom", fontsize=NOTE,
            color=INK_SECONDARY, linespacing=1.3)
    return ax


def plot_slices(
    client: Client, df: pd.DataFrame, metric: str, fixed: dict | None = None, ncols: int = 2
) -> plt.Figure:
    """Model-predicted ``metric`` along each parameter in turn, with a 95% band.

    Other parameters are fixed as in ``plot_contour`` (dotted lines). Ticks along the
    bottom show the values at which experiments were run.
    """
    fixed, trial = _reference(df, metric, fixed)
    names = [p.name for p in PARAMETERS]
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.6 * nrows + 0.8), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for i, (ax, name) in enumerate(zip(axes, names)):
        xs = _grid(name, 80)
        mean, sem = _predict_grid(client, fixed, {name: xs}, metric)
        ax.fill_between(xs, mean - 1.96 * sem, mean + 1.96 * sem, color=SERIES[0],
                        alpha=0.18, lw=0)
        ax.plot(xs, mean, color=SERIES[0], lw=2.2)
        ax.axvline(fixed[name], color=MUTED, ls=":", lw=1.2)
        ax.plot(df[name], np.zeros(len(df)), "|", color=INK_SECONDARY, ms=10, mew=1.5,
                transform=ax.get_xaxis_transform())
        _finish(ax, None, label(name), label(metric) if i % ncols == 0 else None)
    for ax in axes[len(names):]:
        ax.set_visible(False)
    fig.suptitle(f"Predicted {short_label(metric)} along each parameter, with 95% band",
                 y=0.99, color=INK, fontsize=TITLE)
    fig.text(0.5, 0.945, _fixed_note(fixed, [], trial) + " (dotted lines)",
             ha="center", fontsize=NOTE, color=INK_SECONDARY)
    fig.tight_layout()
    fig.subplots_adjust(top=1 - 0.7 / fig.get_figheight())
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
            label_ = f"{name}, illustrative values" if hollow else f"{name} (measured)"
            ax.scatter(x[rows], y[rows], s=75, zorder=4,
                       color="white" if hollow else SERIES[i],
                       edgecolors=SERIES[i] if hollow else "white",
                       linewidths=1.8 if hollow else 1.0, label=label_)


def _threshold_values() -> dict[str, float]:
    """Parse ``"metric >= value"`` strings from the config into ``{metric: value}``."""
    thresholds = {}
    for constraint in OBJECTIVE_THRESHOLDS:
        name, value = (part.strip() for part in constraint.split(">="))
        thresholds[name] = float(value)
    return thresholds


def _finish(ax, title, xlabel, ylabel, grid: bool = True) -> None:
    if title:
        ax.set_title(title, color=INK, fontsize=TITLE, pad=20)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=LABEL)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=LABEL)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=AXIS, labelcolor=INK_SECONDARY, labelsize=TICK)
    if grid:
        ax.grid(True, color=GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)


def _legend(ax, **kwargs) -> None:
    legend = ax.legend(frameon=True, framealpha=0.92, edgecolor=GRID, fontsize=LEGEND,
                       labelcolor=INK_SECONDARY, **kwargs)
    legend.set_zorder(6)
