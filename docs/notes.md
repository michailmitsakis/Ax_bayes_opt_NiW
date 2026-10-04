# Notes on the method

Background for the choices in `src/niw_bo/config.py` and for reading the output of `niw_optimization.ipynb`. Written for Ax 1.3.

## Objective thresholds

In a multi-objective problem each objective gets a threshold: the worst value still worth having. Together they form the reference point of the hypervolume. Points that miss any threshold do not count towards the hypervolume, and Ax does not try to improve the front in that region.

A rule of thumb from the Ax/BoTorch developers: for each objective, ask what value would rule a material out even if every other objective were excellent, then set the threshold about 10% worse than that. If an overpotential worse than −320 mV would make a catalyst useless, a threshold around −350 mV follows.

- Too strict: if no measured point meets every threshold, Ax switches the acquisition function to maximizing the probability of feasibility until one does.
- Too loose: the hypervolume rewards improvements in regions nobody cares about.
- Not set: Ax infers thresholds from the data. This works, but hypervolume values are then not comparable between runs.

In the `Client` API, thresholds are written as outcome constraints on the objective metrics (`"overpotential >= -350"`); for multi-objective problems Ax converts them into objective thresholds.

## Measurement noise

`attach_experiments` passes each measurement as a bare float, which tells Ax the noise level is unknown; the Gaussian process then infers a single noise level per objective. With replicate measurements, pass `(mean, sem)` tuples instead so the model uses the measured uncertainty of each point.

## Batch size

Running experiments one at a time gives the best result per experiment, because every suggestion uses all data available. Running them in batches finishes the campaign in fewer lab sessions, at the cost of some efficiency per experiment (the "adaptivity gap"). A batch of three is a compromise between the two. `get_next_trials(max_trials=3)` builds the batch one point at a time and treats the points already proposed, and any experiments still unmeasured, as pending, so the batch spreads out instead of clustering.

## Model and acquisition function

The default strategy (`method="fast"`) fits one Gaussian process per objective and selects candidates with qLogNEHVI, the log-space version of the noisy expected hypervolume improvement (qNEHVI). It targets the same quantity as qNEHVI and is easier to optimize numerically.

`method="quality"` uses a fully Bayesian GP with a sparsity prior (SAAS) and input warping. It tends to do better when data are scarce, and fitting takes considerably longer.

`initialization_budget=0` skips the quasi-random (Sobol) warm-up, because the hand-picked initial experiments already cover the search space.

## Search space

- pH is varied in steps of 0.5, as in the lab. `step_size=0.5` makes Ax represent it as an ordered parameter with 11 values, and Ax rejects any value off that grid.
- `current_density` and `deposition_time` are integers. `attach_experiments` refuses a fractional value for them instead of rounding it.
- Temperature is held at 25 °C.

## Reading the results

**Model-based vs. measured Pareto front.** `get_pareto_frontier(use_model_predictions=True)` ranks trials by the model's predicted means, which smooths noise; `False` uses the raw measurements. With model predictions, the second element of each metric pair is the predicted *variance*, not the standard error, despite what the type hints suggest. Take the square root for a standard deviation.

**Zero covariance between objectives.** Each objective has its own independent Gaussian process, so the predicted covariance between the two objectives is zero by construction. It does not mean the objectives are unrelated; look at the measured scatter plot to judge the trade-off.

**Cross-validation.** Leave-one-out: each trial is predicted from all others. Points near the diagonal and a high R² mean the model is useful for that objective. A low R² means suggestions for that objective are close to exploration; check for outliers, noisy measurements, or whether the quantity needs a transform.

**Utility progression.** For multi-objective problems this is the hypervolume dominated by the measured Pareto front. Flat stretches mean those trials added no new non-dominated point.

## Moving from `AxClient` (Ax 0.3) to `Client` (Ax 1.x)

`AxClient` is deprecated and is scheduled for removal in Ax 1.4.

| `AxClient` | `Client` |
|---|---|
| `create_experiment(parameters=[{...}], objectives={name: ObjectiveProperties(...)})` | `configure_experiment(parameters=[RangeParameterConfig(...)])` and `configure_optimization(objective="a, b", outcome_constraints=["a >= x"])` |
| `GenerationStrategy([GenerationStep(Models.MOO, num_trials=-1, max_parallelism=3)])` | `configure_generation_strategy(initialization_budget=0)` |
| `get_next_trial()` in a loop | `get_next_trials(max_trials=3)` |
| `complete_trial(i, raw_data={m: (mean, None)})` | `complete_trial(i, raw_data={m: mean})` |
| `get_pareto_optimal_parameters()` | `get_pareto_frontier()` |
| `get_hypervolume()`, `get_trace()` | `UtilityProgressionAnalysis`, part of `compute_analyses()` |
| `ax.plot.*` with `render()`, `cross_validate()` + `compute_diagnostics()` | `compute_analyses()`, or individual classes from `ax.analysis` |
| `get_model_predictions()` | `predict(points)` |

## Figures

`src/niw_bo/plotting.py` draws every figure with matplotlib. The numbers come from Ax: each analysis in `ax.analysis` returns its data as a table (`card.df`), and contour and slice plots call `client.predict()` on a grid. The analysis tables are not part of Ax's stable API and may change between minor versions, which is why `ax-platform` is pinned to 1.3.x; the slow test in `tests/` draws every figure, so a breaking change shows up there.

Cross-validation is shown in measured units. Ax's own card computes R² on Ax's internal transformed scale, so the two values can differ slightly.
| `generation_strategy.trials_as_df`, `experiment.fetch_data().df` | `summarize()` |

## References

- Ax documentation: <https://ax.dev/docs/>
- Daulton, Balandat, Bakshy. *Parallel Bayesian Optimization of Multiple Noisy Objectives with Expected Hypervolume Improvement.* NeurIPS 2021.
- Ament et al. *Unexpected Improvements to Expected Improvement for Bayesian Optimization.* NeurIPS 2023 (the log-space acquisition functions).
