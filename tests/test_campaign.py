from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from niw_bo import (
    attach_experiments,
    build_client,
    lab_only,
    load_experiments,
    suggest_next_batch,
)
from niw_bo.campaign import COLUMNS
from niw_bo.config import PARAMETERS
from niw_bo.plotting import batch_groups, pareto_mask

DATA = Path(__file__).parents[1] / "data" / "experiments.csv"


def one_row(**overrides) -> pd.DataFrame:
    row = {
        "batch": 0,
        "tungstate_concentration": 0.1,
        "current_density": 50,
        "deposition_time": 600,
        "pH": 7.5,
        "overpotential": -286.0,
        "overpotential_slope": 8e-5,
    }
    row.update(overrides)
    return pd.DataFrame([row], columns=COLUMNS)


def assert_in_search_space(df: pd.DataFrame) -> None:
    for p in PARAMETERS:
        values = df[p.name].astype(float)
        low, high = p.bounds
        assert values.between(low, high).all(), p.name
        if p.parameter_type == "int":
            assert (values == values.round()).all(), p.name
        if p.step_size:
            steps = (values - low) / p.step_size
            assert np.allclose(steps, steps.round()), p.name


def test_data_file_is_in_search_space():
    assert_in_search_space(load_experiments(DATA))


def test_illustrative_rows_can_be_excluded():
    full = load_experiments(DATA)
    lab = load_experiments(DATA, include_illustrative=False)
    assert (lab["source"] == "lab").all()
    assert len(lab) == (full["source"] == "lab").sum()
    assert lab.index.tolist() == list(range(len(lab)))  # trial index = row number
    assert lab_only(full.drop(columns="source")).shape[0] == len(full)


def test_half_unit_ph_is_kept():
    # The old helper cast pH to int, silently turning 7.5 into 7.
    client = build_client(seed=0)
    attach_experiments(client, one_row(pH=7.5))
    assert client.summarize()["pH"].tolist() == [7.5]


def test_fractional_value_for_int_parameter_is_rejected():
    client = build_client(seed=0)
    with pytest.raises(ValueError, match="current_density"):
        attach_experiments(client, one_row(current_density=10.5))


def test_unmeasured_rows_stay_pending():
    client = build_client(seed=0)
    attach_experiments(client, one_row(overpotential=np.nan, overpotential_slope=np.nan))
    assert client.summarize()["trial_status"].tolist() == ["RUNNING"]


def test_batch_groups_use_at_most_three_labels():
    groups = batch_groups(pd.Series([0, 0, 1, 2, 3, 4]))
    assert groups.tolist() == [
        "initial design", "initial design",
        "Ax batches 1–3", "Ax batches 1–3", "Ax batches 1–3",
        "Ax batch 4 (latest)",
    ]


def test_pareto_mask_maximizes_all_columns():
    points = np.array([[1, 1], [2, 0], [0, 2], [0, 0], [1, 1]])
    assert pareto_mask(points).tolist() == [True, True, True, False, True]


@pytest.mark.slow
def test_suggest_next_batch():
    df = load_experiments(DATA)
    client = build_client(seed=0)
    attach_experiments(client, df)

    batch = suggest_next_batch(client, df, batch_size=2)

    assert list(batch.columns) == [*COLUMNS, "source"]
    assert (batch["source"] == "lab").all()
    assert len(batch) == 2
    assert (batch["batch"] == df["batch"].max() + 1).all()
    assert batch[["overpotential", "overpotential_slope"]].isna().all().all()
    assert_in_search_space(batch)


@pytest.mark.slow
def test_figures_draw():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from niw_bo import (
        plot_contour,
        plot_cross_validation,
        plot_hypervolume,
        plot_observed_front,
        plot_sensitivity,
        plot_slices,
    )

    df = load_experiments(DATA)  # all rows, so the illustrative styling is drawn too
    client = build_client(seed=0)
    attach_experiments(client, df)

    plot_observed_front(df, client=client, symlog_y=1e-3)
    plot_hypervolume(client, df)
    plot_sensitivity(client, "overpotential")
    for metric in ("overpotential", "overpotential_slope"):
        plot_cross_validation(client, df, metric)
    plot_contour(client, df, "overpotential", "current_density", "pH")
    plot_slices(client, df, "overpotential")
    plt.close("all")


@pytest.mark.slow
def test_predicted_front_is_non_dominated_and_in_bounds():
    from niw_bo import predicted_front

    df = load_experiments(DATA, include_illustrative=False)
    client = build_client(seed=0)
    attach_experiments(client, df)

    front = predicted_front(client, n=512)

    assert len(front) >= 1
    assert pareto_mask(front[["overpotential", "overpotential_slope"]].to_numpy()).all()
    assert_in_search_space(front)
    assert (front[["overpotential_sem", "overpotential_slope_sem"]] > 0).all().all()
