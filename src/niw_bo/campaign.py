"""Human-in-the-loop campaign: an experiments table on disk, replayed into an Ax ``Client``.

The CSV is the source of truth. Each row is one experiment; rows whose objectives are
still empty are experiments suggested by Ax but not yet measured. Rebuilding the client
from the CSV each session keeps the record human-readable and independent of Ax's
serialization format, which changes between Ax versions.

The optional ``source`` column marks where the objective values come from: ``lab`` for
measurements, ``illustrative`` for made-up values entered to demonstrate the loop.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
from ax import Client

from niw_bo.config import BATCH_SIZE, OBJECTIVE_THRESHOLDS, OBJECTIVES, PARAMETERS

PARAMETER_NAMES = [p.name for p in PARAMETERS]
COLUMNS = ["batch", *PARAMETER_NAMES, *OBJECTIVES]
ILLUSTRATIVE = "illustrative"


def load_experiments(path: str | Path, include_illustrative: bool = True) -> pd.DataFrame:
    """Read the experiments table and check that it has the expected columns.

    With ``include_illustrative=False``, rows whose ``source`` is ``illustrative`` are
    dropped, so that models and figures use lab measurements only.
    """
    df = pd.read_csv(path)
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if not include_illustrative:
        df = lab_only(df)
    return df


def lab_only(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows marked ``illustrative`` and renumber the rest from 0 (trial order)."""
    if "source" not in df.columns:
        return df.reset_index(drop=True)
    return df[df["source"] != ILLUSTRATIVE].reset_index(drop=True)


def build_client(
    seed: int | None = None,
    method: str = "fast",
    initialization_budget: int | None = 0,
) -> Client:
    """Create a ``Client`` with the search space and objectives from ``config.py``.

    ``initialization_budget=0`` goes straight to Bayesian optimization, using the
    attached lab data as the initial design. Pass ``None`` when starting with no data,
    so that Ax first proposes a center point and quasi-random (Sobol) points.
    ``method="quality"`` swaps the default GP for a fully Bayesian (SAAS) GP with
    input warping: often better with very little data, but much slower to fit.
    """
    client = Client(random_seed=seed)
    client.configure_experiment(name="NiW", parameters=PARAMETERS)
    client.configure_optimization(
        objective=", ".join(OBJECTIVES),
        outcome_constraints=OBJECTIVE_THRESHOLDS,
    )
    client.configure_generation_strategy(
        method=method, initialization_budget=initialization_budget
    )
    return client


def attach_experiments(client: Client, df: pd.DataFrame) -> None:
    """Attach every row of ``df`` to ``client`` as a trial, in row order.

    Rows with all objectives measured are completed. Rows with a missing objective stay
    RUNNING, so Ax treats them as pending and does not suggest the same point again.
    """
    for _, row in df.iterrows():
        parameters = {name: _coerce(name, row[name]) for name in PARAMETER_NAMES}
        trial_index = client.attach_trial(parameters=parameters)
        if row[OBJECTIVES].notna().all():
            # A bare float means "noise unknown": Ax infers the noise level.
            # Pass (mean, sem) tuples instead if you have replicate measurements.
            raw_data = {name: float(row[name]) for name in OBJECTIVES}
            client.complete_trial(trial_index=trial_index, raw_data=raw_data)


def suggest_next_batch(
    client: Client, df: pd.DataFrame, batch_size: int = BATCH_SIZE
) -> pd.DataFrame:
    """Ask Ax for the next batch and return it as new, unmeasured rows for the table.

    ``df`` is only used to number the batch, so pass the full table (including any
    illustrative rows) to avoid reusing a batch number. New rows are marked ``lab``.
    """
    trials = client.get_next_trials(max_trials=batch_size)
    batch = int(df["batch"].max()) + 1 if len(df) else 0
    rows = [{"batch": batch, **parameters, "source": "lab"} for parameters in trials.values()]
    return pd.DataFrame(rows, columns=[*COLUMNS, "source"])


def _coerce(name: str, value) -> int | float:
    """Convert a table value to the parameter's declared type, refusing silent rounding."""
    config = next(p for p in PARAMETERS if p.name == name)
    value = float(value)
    if config.parameter_type == "float":
        return value
    if not math.isclose(value, round(value)):
        raise ValueError(f"{name} is an integer parameter, got {value}")
    return int(round(value))
