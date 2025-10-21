"""
Compute single transformer designs with the numerical optima.
"""

__author__ = "Thomas Guillod"
__copyright__ = "Thomas Guillod - Dartmouth College"
__license__ = "Mozilla Public License Version 2.0"

import numpy as np
from transformer_scaling import optimizer
from transformer_scaling import display
from transformer_scaling import model
from transformer_scaling import vector
import param


def _get_objective(design_tmp, mode):
    """
    Compute the objective function for the optimization.
    Optimization with a constant volume and mass.
    Optimization with a loss constraint.
    """

    # extract the data
    rho = design_tmp["rho"]
    gamma = design_tmp["gamma"]
    loss = design_tmp["loss"]
    penalty = design_tmp["penalty"]

    # objective parameters
    fact_obj = 1.0
    fact_bound = 8.0
    fact_penalty = 2.0

    # scaling parameters
    loss_max = 0.2e-2
    loss_scl = 0.2e-2
    rho_scl = 20e6
    gamma_scl = 7e3

    # scale the loss violation
    bound = np.maximum((loss - loss_max) / loss_max, 0.0)

    # compute the objective
    if mode == "volume":
        obj = (loss / loss_scl) * (fact_obj + fact_penalty * penalty)
    elif mode == "mass":
        obj = (loss / loss_scl) * (fact_obj + fact_penalty * penalty)
    elif mode == "loss_volume":
        obj = (rho_scl / rho) * (fact_obj + fact_penalty * penalty + fact_bound * bound)
    elif mode == "loss_mass":
        obj = (gamma_scl / gamma) * (fact_obj + fact_penalty * penalty + fact_bound * bound)
    else:
        raise ValueError("invalid objective")

    return obj


def _get_design(geom, constant, design, trg, var_tmp, n_sweep):
    """
    Compute transformer designs from optimized parameters.
    """

    # optimization type (frequency and number of turns are optimized numerically)
    opt = "none"

    # merge parameters
    design_tmp = design | var_tmp

    # vectorize the design
    design_tmp = vector.get_vectorize(design_tmp, n_sweep)

    # solve the design
    design_tmp = model.get_solve(geom, trg, opt, constant, design_tmp)

    return design_tmp


def fct_solve(geom, conv, split, mode):
    """
    Optimize, solve, and plot a single transformer design (numerical optima).
    """

    # use the simplified design
    simplified = False

    # get the geometry metric and optim variables
    if mode == "volume":
        trg = "volume"
        use_rho = False
        use_gamma = False
    elif mode == "mass":
        trg = "mass"
        use_rho = False
        use_gamma = False
    elif mode == "loss_volume":
        trg = "volume"
        use_rho = True
        use_gamma = False
    elif mode == "loss_mass":
        trg = "mass"
        use_rho = False
        use_gamma = True
    else:
        raise ValueError("invalid objective")

    # get the parameters
    constant = param.get_constant()
    design = param.get_design(geom, conv, split, simplified)

    # define the objective function
    #   - the relative losses are the objective
    #   - a penalty is added for invalid designs
    def fct_obj(var_tmp, n_sweep):
        # solve the design
        design_tmp = _get_design(geom, constant, design, trg, var_tmp, n_sweep)

        # compute the objective
        obj = _get_objective(design_tmp, mode)

        return obj

    #  options for the variables to be optimized
    var_list = {
        "ratio_cw": {"bnd": (0.2, 10.0), "log": True, "use": True},
        "ratio_c": {"bnd": (0.5, 5.0), "log": True, "use": True},
        "ratio_w": {"bnd": (0.5, 6.0), "log": True, "use": True},
        "n_def": {"bnd": (2, 50), "log": True, "use": True},
        "f_def": {"bnd": (10e3, 500e3), "log": True, "use": True},
        "rho_def": {"bnd": (4e6, 80e6), "log": True, "use": use_rho},
        "gamma_def": {"bnd": (1e3, 30e3), "log": True, "use": use_gamma},
    }

    # options for the global optimizer (differential evolution)
    options = {
        "maxiter": 250,
        "popsize": 250,
        "tol": 1e-5,
        "rng": 1234,
    }

    # solve the transformer design problem
    (var, res) = optimizer.get_optimal(fct_obj, var_list, options)

    # extract the optimal design
    design = _get_design(geom, constant, design, trg, var, None)

    return design


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
            print(f"========= {conv}")

            design = fct_solve(geom, conv, split, "volume")
            summary = display.get_summary(design)
            print(f"volume / {summary}")

            design = fct_solve(geom, conv, split, "mass")
            summary = display.get_summary(design)
            print(f"mass / {summary}")

            design = fct_solve(geom, conv, split, "loss_volume")
            summary = display.get_summary(design)
            print(f"loss_volume / {summary}")

            design = fct_solve(geom, conv, split, "loss_mass")
            summary = display.get_summary(design)
            print(f"loss_mass / {summary}")
