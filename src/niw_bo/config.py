"""Search space and goals of the Ni-W electrodeposition campaign.

This is the only file to edit when adapting the workflow to another process.
"""

from ax import RangeParameterConfig

PARAMETERS = [
    # Tungstate concentration in the bath [M]
    RangeParameterConfig(name="tungstate_concentration", bounds=(0.05, 0.20), parameter_type="float"),
    # Applied current density [mA/cm²]
    RangeParameterConfig(name="current_density", bounds=(5, 125), parameter_type="int"),
    # Deposition time [s]
    RangeParameterConfig(name="deposition_time", bounds=(60, 600), parameter_type="int"),
    # Bath pH, on the 0.5-unit grid used in the lab
    RangeParameterConfig(name="pH", bounds=(5.0, 10.0), parameter_type="float", step_size=0.5),
]
# Temperature is held at 25 °C and is not optimized.

# Both objectives are maximized. The HER overpotential is negative, so "higher" means
# closer to zero; a slope >= 0 means the overpotential does not degrade over cycling.
OBJECTIVES = ["overpotential", "overpotential_slope"]

# In a multi-objective problem, Ax reads constraints on the objective metrics as
# objective thresholds, i.e. the reference point for the hypervolume and the worst
# values still worth having on the Pareto front. See docs/notes.md.
OBJECTIVE_THRESHOLDS = [
    "overpotential >= -350",
    "overpotential_slope >= -0.001",
]

# Experiments run in parallel per lab session.
BATCH_SIZE = 3

# Axis labels for the figures; names missing here are shown as they are.
LABELS = {
    "tungstate_concentration": "Tungstate concentration (M)",
    "current_density": "Current density (mA/cm²)",
    "deposition_time": "Deposition time (s)",
    "pH": "pH",
    "overpotential": "Overpotential (mV)",
    "overpotential_slope": "Overpotential slope",
}
