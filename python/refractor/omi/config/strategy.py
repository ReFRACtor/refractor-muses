import numpy as np

from refractor import framework as rf


@rf.strategy_list
def retrieval_steps():
    steps = [
        {
            "retrieval_elements": ["O3", "ground", "dispersion"],
            "micro_windows": [
                [268.00, 311.00],
                [307.00, 378.00],
            ],
            "solver_parameters": {
                "max_iteration": 20,
                # Setting a tolerance to 0 effectively disables it
                "dx_tol_abs": 0.0,
                "dx_tol_rel": 0.0,
                "g_tol_abs": 0.0,
                "g_tol_rel": 0.0,
            },
        },
    ]

    # Change micro_windows into the proper ArrayWithUnit objects
    # Do this here afterwards to make it easier to modify steps dictionary
    for step_index in range(len(steps)):
        steps[step_index]["micro_windows"] = rf.ArrayWithUnit(
            np.array(steps[step_index]["micro_windows"]), "nm"
        )

    return steps
