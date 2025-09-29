"""
Optimize a transformer (given optimization variables and objective).
A global optimizer (differential evolution) is used for the optimization.
"""

__author__ = "Thomas Guillod"
__copyright__ = "Thomas Guillod - Dartmouth College"
__license__ = "Mozilla Public License Version 2.0"

import copy
import numpy as np
import scipy.optimize as optimize
from transformer_scaling import solver
from transformer_scaling import vector


def get_optimal(geom, trg, opt, constant, design, optim):
    """
    Optimize a transformer design with a differential evolution algorithm.
    """

    # extract the optimization parameters
    var_list = optim["var_list"]
    fct_obj = optim["fct_obj"]
    options = optim["options"]

    # get the optimization variables and bounds
    bnd = []
    var = []
    log = []
    for var_tmp, mode_tmp in var_list.items():
        # extract the variable
        bnd_tmp = mode_tmp["bnd"]
        log_tmp = mode_tmp["log"]
        use_tmp = mode_tmp["use"]

        # add the variable
        if use_tmp:
            var.append(var_tmp)
            log.append(log_tmp)
            if log_tmp:
                bnd.append(np.log(bnd_tmp))
            else:
                bnd.append(bnd_tmp)

    # extract a design
    def get_design(x):
        # copy design
        design_tmp = copy.deepcopy(design)

        # assign the optimized variables
        for idx, (var_tmp, log_tmp) in enumerate(zip(var, log, strict=True)):
            if log_tmp:
                design_tmp[var_tmp] = np.exp(x[idx])
            else:
                design_tmp[var_tmp] = x[idx]

        return design_tmp

    # get the objective function
    def get_optim(x):
        # extract a design
        design_tmp = get_design(x)

        # get dimension
        n_var = x.shape[0]
        n_sweep = x.shape[1]

        # assign the optimized variables
        assert n_var == len(var), "invalid length"
        assert n_var == len(log), "invalid length"
        assert n_var == len(bnd), "invalid length"

        # vectorize the design
        design_tmp = vector.get_vectorize(design_tmp, n_sweep)

        # solve the design
        design_tmp = solver.get_solve(geom, trg, opt, constant, design_tmp)

        # get the objective
        obj = fct_obj(design_tmp)

        return obj

    # solve the optimization problem
    result = optimize.differential_evolution(
        get_optim,
        bnd,
        disp=False,
        vectorized=True,
        updating="deferred",
        **options,
    )

    # assign the optimized variable
    design = get_design(result.x)
    design["opt_fun"] = result.fun
    design["opt_nit"] = result.nit
    design["opt_nfev"] = result.nfev
    design["opt_success"] = result.success

    # solve the optimal design
    design = solver.get_solve(geom, trg, opt, constant, design)

    return design
