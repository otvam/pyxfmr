"""
Compute single transformer designs with the numerical optima.
"""

__author__ = "Thomas Guillod"
__copyright__ = "Thomas Guillod - Dartmouth College"
__license__ = "Mozilla Public License Version 2.0"

import numpy as np
from transformer_scaling import optimizer
from transformer_scaling import display
import param


def fct_solve(geom, conv, split):
    """
    Optimize, solve, and plot a single transformer design (numerical optima).
    """

    # geometry target
    trg = "volume"

    # optimization type
    opt = "none"

    # use the simplified design
    simplified = False

    # define the objective function
    #   - the relative losses are the objective
    #   - a penalty is added for invalid designs
    def fct_obj(design_tmp):
        # extract the data
        loss = design_tmp["loss"]
        penalty = design_tmp["penalty"]

        # loss parameters
        fact_loss = 1.0
        fact_penalty = 5.0
        obj_invalid = 1.0

        # compute the objective
        obj = loss * (fact_loss + fact_penalty * penalty)
        obj = np.minimum(obj, obj_invalid)

        return obj

    # optimization parameters
    optim = {
        "fct_obj": fct_obj,  # objective function for the optimization
        "var_list": {  # options for the variables to be optimized
            "ratio_cw": {"bnd": (0.2, 10.0), "log": True, "use": True},
            "ratio_c": {"bnd": (0.5, 5.0), "log": True, "use": True},
            "ratio_w": {"bnd": (0.5, 6.0), "log": True, "use": True},
            "n_def": {"bnd": (2, 50), "log": True, "use": True},
            "f_def": {"bnd": (10e3, 500e3), "log": True, "use": True},
        },
        "options": {  # options for the global optimizer (differential evolution)
            "maxiter": 250,
            "popsize": 200,
            "tol": 1e-5,
        },
    }

    # get the parameters
    constant = param.get_constant()
    design = param.get_design(geom, conv, split, simplified)

    # solve the transformer design problem
    design = optimizer.get_optimal(geom, trg, opt, constant, design, optim)

    # display the solution (with or without details)
    tag = f"{geom}_{split}_{conv}"
    display.get_summary(tag, design)


if __name__ == "__main__":
    # transformer excitation
    conv_list = ["sin", "dab", "src"]

    # different transformer configurations
    shape_list = [
        ("shell_inter", "1p"),
        ("shell_simple", "1p"),
        ("core_type", "1p"),
        ("shell_inter", "sp_wye"),
        ("shell_inter", "sp_delta"),
        ("shell_simple", "sp_wye"),
        ("shell_simple", "sp_delta"),
        ("core_type", "sp_wye"),
        ("core_type", "sp_delta"),
        ("three_phase", "3p_wye"),
        ("three_phase", "3p_delta"),
    ]

    # run the sweeps for the different configurations
    for geom, split in shape_list:
        print(f"========================================== {geom} / {split}")
        for conv in conv_list:
            fct_solve(geom, conv, split)
