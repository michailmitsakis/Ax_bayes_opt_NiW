"""Website versions of the Ax figures: card-style titles, colour by role, legends.

Numbers come from data.json (computed by extract.py through Ax 1.3.1 from
data/experiments.csv); this script only draws them.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Patch  # noqa: E402

HERE = Path(__file__).parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
D = json.loads((HERE / "data.json").read_text())

# --- style ---------------------------------------------------------------------------
FONT_DIR = Path("/usr/local/lib/python3.11/dist-packages/font_roboto/files")
for f in FONT_DIR.glob("Roboto-*.ttf"):
    font_manager.fontManager.addfont(str(f))
plt.rcParams.update({"font.family": "Roboto", "axes.unicode_minus": True})

BLUE, ORANGE, GREEN, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SHADE = "#e6e4dc", "#c3c2b7", "#ecebe6"
TINT = {"blue": "#eaf2fc", "orange": "#fdeee7", "green": "#e5f6ef"}
BLUES = LinearSegmentedColormap.from_list("blues", ["#dbe9fb", "#9cc3f2", "#4f93e6", "#2a6fc4", "#1b5299", "#0d366b"])

THR_ETA, THR_SLOPE = -350.0, -0.001


def header(fig, title, subtitle, top_in=0.95):
    """Bold left-aligned title and grey subtitle, like a card header."""
    h = fig.get_figheight()
    fig.text(0.012, 1 - 0.22 / h, title, fontsize=17, fontweight="bold", color=INK, va="top")
    fig.text(0.012, 1 - 0.58 / h, subtitle, fontsize=11.5, color=INK2, va="top")
    return 1 - top_in / h


def style(ax, xlabel=None, ylabel=None, grid="both"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=AXIS, labelcolor=INK2, labelsize=11)
    if grid:
        ax.grid(True, axis=grid, color=GRID, lw=0.9, zorder=0)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=12, labelpad=8)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=12, labelpad=8)


def legend(ax, handles, frameon=True, **kw):
    leg = ax.legend(handles=handles, frameon=frameon, fontsize=10.5, borderpad=0.7, labelspacing=0.55, handletextpad=0.6, **kw)
    leg.get_frame().set(facecolor="white", edgecolor=GRID, alpha=0.96, linewidth=1)
    for t in leg.get_texts():
        t.set_color(INK2)
    return leg


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=170, facecolor="white")
    plt.close(fig)
    print("wrote", name)


def pareto_mask(v):
    v = np.asarray(v, float)
    return np.array([not (np.all(v >= p, 1) & np.any(v > p, 1)).any() for p in v])


table = {k: np.array(v) for k, v in D["table"].items()}
lab = table["source"] == "lab"
eta, slope = table["overpotential"].astype(float), table["overpotential_slope"].astype(float)


# --- 1. Measured results and predicted Pareto front -----------------------------------
def fig_pareto():
    fig, ax = plt.subplots(figsize=(10, 7.3))
    top = header(fig, "Measured results and predicted Pareto front",
                 "10 lab measurements (initial design)  ·  front predicted by the Ax model  ·  both objectives maximized")
    fig.subplots_adjust(left=0.1, right=0.975, bottom=0.25, top=top)

    x, y = eta[lab], slope[lab]
    front_mask = pareto_mask(np.c_[x, y])
    inside = (x >= THR_ETA) & (y >= THR_SLOPE)

    # Hypervolume region: what the measured front dominates, relative to the thresholds.
    pts = sorted(zip(x[front_mask & inside], y[front_mask & inside]), key=lambda p: -p[1])
    sx, sy = [THR_ETA], [pts[0][1]]
    for i, (px, py) in enumerate(pts):
        sx += [px, px]
        sy += [py, pts[i + 1][1] if i + 1 < len(pts) else THR_SLOPE]
    poly_x, poly_y = sx + [THR_ETA], sy + [THR_SLOPE]
    ax.fill(poly_x, poly_y, color=TINT["blue"], zorder=1, lw=0)
    ax.plot(sx, sy, color=INK2, lw=1.4, zorder=2)
    ax.text(-346, -0.00078, "Shaded area = hypervolume", fontsize=10.5, color=INK2, va="center")

    # Thresholds, labelled inline.
    ax.axvline(THR_ETA, color=MUTED, lw=1.3, zorder=1)
    ax.axhline(THR_SLOPE, color=MUTED, lw=1.3, zorder=1)
    ax.text(THR_ETA + 1.5, 0.00228, "η threshold −350 mV", fontsize=10, color=INK2, va="top")
    ax.text(-383, THR_SLOPE + 0.00007, "slope threshold −0.001", fontsize=10, color=INK2)

    # Model-predicted front with 95% intervals.
    f = {k: np.array(v) for k, v in D["front"].items()}
    ax.errorbar(f["overpotential"], f["overpotential_slope"], xerr=1.96 * f["overpotential_sem"],
                yerr=1.96 * f["overpotential_slope_sem"], fmt="none", ecolor=PURPLE, elinewidth=1.1,
                capsize=3, alpha=0.35, zorder=3)
    ax.plot(f["overpotential"], f["overpotential_slope"], "-D", color=PURPLE, lw=2.2, ms=7,
            mec="white", mew=1.2, zorder=4)

    # Measurements: inside both thresholds vs outside.
    ax.scatter(x[inside], y[inside], s=80, color=BLUE, edgecolor="white", lw=1.4, zorder=5)
    ax.scatter(x[~inside], y[~inside], s=80, color="white", edgecolor=BLUE, lw=1.8, zorder=5)
    ax.scatter(x[front_mask], y[front_mask], s=330, facecolor="none", edgecolor=INK, lw=1.6, zorder=6)

    # Name the two non-dominated recipes.
    trials = np.arange(len(eta))[lab]
    for t, px, py in zip(trials[front_mask], x[front_mask], y[front_mask]):
        text, dx, dy = ("trial 4 · best η\n−286 mV", -31, -0.00046) if t == 4 else ("trial 7 · best slope\n+0.0017", -36, 0.00042)
        ax.annotate(text, (px, py), xytext=(px + dx, py + dy), fontsize=10, color=INK, ha="left", va="center",
                    arrowprops=dict(arrowstyle="-", color=INK2, lw=1, shrinkA=2, shrinkB=11), zorder=7)

    ax.add_patch(FancyArrowPatch((-268, 0.00075), (-256, 0.0012), arrowstyle="-|>", mutation_scale=16, color=INK2, lw=1.4))
    ax.text(-270, 0.00072, "better", fontsize=10.5, color=INK2, ha="right", va="top")

    ax.set_xlim(-385, -250)
    ax.set_ylim(-0.00125, 0.0023)
    style(ax, "Overpotential η (mV)  ·  higher (less negative) is better", "Overpotential slope (drift over cycling)")
    legend(ax, [
        Line2D([], [], color=PURPLE, marker="D", lw=2.2, ms=7, mec="white", label="model-predicted Pareto front (95% intervals)"),
        Line2D([], [], ls="", marker="o", ms=9, color=BLUE, mec="white", label="lab measurement inside both thresholds"),
        Line2D([], [], ls="", marker="o", ms=9, mfc="white", mec=BLUE, mew=1.8, label="lab measurement outside a threshold"),
        Line2D([], [], ls="", marker="o", ms=16, mfc="none", mec=INK, mew=1.6, label="measured Pareto front (non-dominated)"),
        Patch(facecolor=TINT["blue"], edgecolor=INK2, lw=1, label="hypervolume"),
    ], loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)
    save(fig, "ax-pareto")


# --- 2. Hypervolume across the campaign -------------------------------------------------
def fig_hypervolume():
    fig, ax = plt.subplots(figsize=(10, 5.6))
    top = header(fig, "Hypervolume across the campaign",
                 "Reference point −350 mV, −0.001  ·  trials 10–15 are real Ax suggestions with illustrative values")
    fig.subplots_adjust(left=0.08, right=0.975, bottom=0.13, top=top)

    t = np.array(D["hv"]["trial"])
    u = np.array(D["hv"]["utility"])
    batch = table["batch"].astype(int)[t]
    colors = {0: BLUE, 1: ORANGE, 2: GREEN}
    markers = {0: "o", 1: "s", 2: "^"}
    tints = {0: TINT["blue"], 1: TINT["orange"], 2: TINT["green"]}
    names = {0: "Initial design\nlab measurements", 1: "Ax batch 1\nillustrative", 2: "Ax batch 2\nillustrative"}
    ymax = 0.245
    for b in (0, 1, 2):
        lo, hi = t[batch == b].min() - 0.5, t[batch == b].max() + 0.5
        ax.axvspan(lo, hi, color=tints[b], zorder=0, lw=0)
        ax.text((lo + hi) / 2, ymax * 0.975, names[b], ha="center", va="top", fontsize=10.5, color=INK2, linespacing=1.3)

    cut = t[batch > 0].min()
    solid = t <= cut - 1
    ax.plot(t[solid], u[solid], drawstyle="steps-post", color=INK2, lw=2.2, zorder=3)
    ax.plot(t[t >= cut - 1], u[t >= cut - 1], drawstyle="steps-post", color=INK2, lw=2.2, ls=(0, (4, 2.5)), zorder=3)
    for b in (0, 1, 2):
        m = batch == b
        ax.scatter(t[m], u[m], s=95 if b == 2 else 75, marker=markers[b], color=colors[b], edgecolor="white", lw=1.4, zorder=4)

    ax.annotate(f"{u[t == cut - 1][0]:.3f}", (cut - 1, u[t == cut - 1][0]), xytext=(0, 10), textcoords="offset points",
                ha="center", fontsize=10.5, color=INK, fontweight="medium")
    ax.annotate(f"{u[-1]:.3f}", (t[-1], u[-1]), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=10.5,
                color=INK, fontweight="medium")

    ax.set_xlim(t.min() - 0.5, t.max() + 0.5)
    ax.set_ylim(0, ymax)
    ax.set_xticks(t)
    style(ax, "Trial", "Hypervolume", grid="y")
    legend(ax, [
        Line2D([], [], ls="", marker="o", ms=9, color=BLUE, mec="white", label="lab measurement"),
        Line2D([], [], ls="", marker="s", ms=8.5, color=ORANGE, mec="white", label="Ax batch 1 (illustrative values)"),
        Line2D([], [], ls="", marker="^", ms=10, color=GREEN, mec="white", label="Ax batch 2 (illustrative values)"),
        Line2D([], [], color=INK2, lw=2.2, ls=(0, (4, 2.5)), label="extrapolated with illustrative values"),
    ], loc="lower right", bbox_to_anchor=(0.995, 0.03))
    save(fig, "ax-hypervolume")


# --- 3. Leave-one-out cross-validation --------------------------------------------------
def fig_cv():
    fig, axes = plt.subplots(1, 2, figsize=(10, 6.0))
    top = header(fig, "How well does the model predict?",
                 "Leave-one-out cross-validation: each lab recipe predicted by a model fitted to the other nine")
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.2, top=top - 0.05, wspace=0.32)
    panels = [("overpotential", "Overpotential (mV)", BLUE, "usable", 1.0),
              ("overpotential_slope", "Overpotential slope (×10⁻³)", ORANGE, "no pattern yet", 1e3)]
    for ax, (m, title, color, verdict, scale) in zip(axes, panels):
        c = D["cv"][m]
        obs, pred, ci = (np.array(c[k]) * scale for k in ("observed", "predicted", "ci95"))
        lo = min(obs.min(), (pred - ci).min())
        hi = max(obs.max(), (pred + ci).max())
        pad = 0.06 * (hi - lo)
        lo, hi = lo - pad, hi + pad
        ax.fill_between([lo, hi], [lo, hi], [hi, hi], color="white", zorder=0)
        ax.plot([lo, hi], [lo, hi], color=MUTED, ls="--", lw=1.3, zorder=1)
        ax.errorbar(obs, pred, yerr=ci, fmt="none", ecolor=color, elinewidth=1.4, capsize=3.5, alpha=0.55, zorder=2)
        ax.scatter(obs, pred, s=80, color=color, edgecolor="white", lw=1.4, zorder=3)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        style(ax, "Measured", "Predicted (leave-one-out)")
        ax.set_title(title, loc="left", fontsize=13, color=INK, fontweight="medium", pad=10)
        ax.text(0.04, 0.95, f"R² = {c['r2']:.2f}", transform=ax.transAxes, fontsize=13, fontweight="bold", color=INK,
                va="top", bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor=color, lw=1.6))
        ax.text(0.04, 0.84, verdict, transform=ax.transAxes, fontsize=10.5, color=INK2, va="top", style="italic")
    fig.legend(handles=[
        Line2D([], [], ls="", marker="o", ms=9, color=BLUE, mec="white", label="prediction (overpotential)"),
        Line2D([], [], ls="", marker="o", ms=9, color=ORANGE, mec="white", label="prediction (slope)"),
        Line2D([], [], color=INK2, lw=1.4, marker="_", ms=10, label="95% predictive interval"),
        Line2D([], [], color=MUTED, ls="--", lw=1.3, label="perfect prediction"),
    ], loc="lower center", ncol=4, frameon=False, fontsize=10.5, labelcolor=INK2, bbox_to_anchor=(0.5, 0.015))
    save(fig, "ax-cross-validation")


# --- 4. What the overpotential model has learned ----------------------------------------
def fig_insights():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5.8), gridspec_kw={"width_ratios": [1, 1.35]})
    top = header(fig, "What the overpotential model has learned",
                 "Model output, so only as reliable as its cross-validation (R² = 0.64)  ·  based on 10 lab measurements")
    fig.subplots_adjust(left=0.135, right=0.97, bottom=0.2, top=top - 0.06, wspace=0.42)

    labels = {"tungstate_concentration": "Tungstate\nconcentration", "current_density": "Current density",
              "deposition_time": "Deposition time", "pH": "pH"}
    s = sorted(D["sobol"].items(), key=lambda kv: kv[1])
    vals = np.array([v for _, v in s])
    bar_colors = [BLUES(0.25 + 0.75 * v / vals.max()) for v in vals]
    a1.barh([labels[k] for k, _ in s], vals, color=bar_colors, height=0.62, zorder=3)
    for i, v in enumerate(vals):
        a1.text(v + 0.015, i, f"{v:.2f}", va="center", fontsize=11, color=INK, fontweight="medium")
    a1.set_xlim(0, 0.82)
    style(a1, "Sobol index (total order)", None, grid="x")
    a1.tick_params(axis="y", length=0, labelsize=11.5, labelcolor=INK)
    a1.set_title("Parameter importance", loc="left", fontsize=13, color=INK, fontweight="medium", pad=10)

    c = D["contour"]
    X, Y = np.meshgrid(c["x"], c["y"])
    Z = np.array(c["z"])
    levels = np.arange(np.floor(Z.min() / 5) * 5, np.ceil(Z.max() / 5) * 5 + 0.1, 5)
    filled = a2.contourf(X, Y, Z, levels=levels, cmap=BLUES)
    a2.contour(X, Y, Z, levels=levels, colors="white", linewidths=0.5, alpha=0.7, linestyles="solid")
    best_level = levels[-3]
    a2.contour(X, Y, Z, levels=[best_level], colors=ORANGE, linewidths=2.2, linestyles="solid")
    iy, ix = np.unravel_index(np.argmax(Z), Z.shape)
    a2.scatter(X[iy, ix], Y[iy, ix], marker="*", s=320, color=ORANGE, edgecolor="white", lw=1.3, zorder=5)

    x, y = table["current_density"][lab].astype(float), table["pH"][lab].astype(float)
    ref = np.arange(len(eta))[lab] == c["trial"]
    a2.scatter(x[~ref], y[~ref], s=60, color="white", edgecolor=INK, lw=1.4, zorder=4, clip_on=False)
    a2.scatter(x[ref], y[ref], s=95, marker="D", color="white", edgecolor=INK, lw=1.8, zorder=4)
    cb = fig.colorbar(filled, ax=a2, pad=0.025)
    cb.set_label("Predicted overpotential (mV)", color=INK2, fontsize=11.5)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=AXIS, labelcolor=INK2, labelsize=10.5)
    style(a2, "Current density (mA/cm²)", "pH", grid=None)
    fx = c["fixed"]
    a2.set_title(f"Predicted overpotential  ·  tungstate {fx['tungstate_concentration']:g} M, {fx['deposition_time']} s",
                 loc="left", fontsize=13, color=INK, fontweight="medium", pad=10)

    fig.legend(handles=[
        Line2D([], [], ls="", marker="*", ms=17, color=ORANGE, mec="white", label="predicted best"),
        Line2D([], [], color=ORANGE, lw=2.2, label=f"predicted η ≥ {best_level:.0f} mV".replace("-", "−")),
        Line2D([], [], ls="", marker="o", ms=8, mfc="white", mec=INK, mew=1.4, label="tested recipes (projected)"),
        Line2D([], [], ls="", marker="D", ms=8, mfc="white", mec=INK, mew=1.8, label="trial 4 (other parameters fixed here)"),
    ], loc="lower right", ncol=2, frameon=False, fontsize=10.5, labelcolor=INK2, bbox_to_anchor=(0.97, 0.0))
    save(fig, "ax-model-insights")


fig_pareto()
fig_hypervolume()
fig_cv()
fig_insights()
