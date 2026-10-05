"""Compute every number the figures need, from data/experiments.csv via Ax."""
import json, logging, warnings
from pathlib import Path
import numpy as np
from ax.analysis import CrossValidationPlot, SensitivityAnalysisPlot, UtilityProgressionAnalysis
from ax.utils.common.logger import set_stderr_log_level
from niw_bo import attach_experiments, build_client, lab_only, load_experiments
from niw_bo.plotting import predicted_front, _analysis_cards, _analysis_df, _reference, _grid, _predict_grid

set_stderr_log_level(logging.WARNING); warnings.filterwarnings("ignore")
REPO = Path("/home/claude/michailmitsakis/ax_bayes_opt_niw")
table = load_experiments(REPO / "data/experiments.csv")
lab = lab_only(table)
client = build_client(seed=0); attach_experiments(client, lab)
demo = build_client(seed=0); attach_experiments(demo, table)
out = {"table": table.reset_index(drop=True).to_dict(orient="list")}

front = predicted_front(client)
out["front"] = front.to_dict(orient="list")

out["cv"] = {}
for m in ["overpotential", "overpotential_slope"]:
    cards = _analysis_cards(client, CrossValidationPlot(metric_names=[m], untransform=True))
    d = next(c.df for c in cards if "observed" in c.df.columns)
    r2 = next(c.df for c in cards if "R²" in c.df.columns)["R²"].iloc[0]
    out["cv"][m] = {"trial": d["arm_name"].str.split("_").str[0].astype(int).tolist(),
                    "observed": d["observed"].tolist(), "predicted": d["predicted"].tolist(),
                    "ci95": d["predicted_95_ci"].tolist(), "r2": float(r2)}

s = _analysis_df(client, SensitivityAnalysisPlot(metric_name="overpotential"))
out["sobol"] = dict(zip(s["parameter_name"], s["sensitivity"].clip(lower=0).astype(float)))

fixed, trial = _reference(lab, "overpotential", None)
xs, ys = _grid("current_density", 60), _grid("pH", 60)
X, Y = np.meshgrid(xs, ys)
Z = _predict_grid(client, fixed, {"current_density": X.ravel(), "pH": Y.ravel()}, "overpotential")[0].reshape(X.shape)
out["contour"] = {"x": xs.tolist(), "y": ys.tolist(), "z": Z.tolist(), "fixed": fixed, "trial": trial}

hv = _analysis_df(demo, UtilityProgressionAnalysis())
out["hv"] = {"trial": hv["trial_index"].astype(int).tolist(), "utility": hv["utility"].astype(float).tolist()}

Path("/home/claude/sitefigs/data.json").write_text(json.dumps(out, default=float, indent=1))
print("front rows", len(front), "| R2", {k: round(v["r2"], 2) for k, v in out["cv"].items()},
      "| sobol", {k: round(v, 2) for k, v in out["sobol"].items()}, "| hv", [round(u, 3) for u in out["hv"]["utility"]])
