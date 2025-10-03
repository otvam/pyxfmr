"""
Optimize a transformer (given optimization variables and objective).
A global optimizer (differential evolution) is used for the optimization.
"""

__author__ = "Thomas Guillod"
__copyright__ = "Thomas Guillod - Dartmouth College"
__license__ = "Mozilla Public License Version 2.0"

import numpy as np
import scipy.optimize as optimize


def get_optimal(fct_obj, var_list, options):
    """
    Optimize a transformer design with a differential evolution algorithm.
    """

    # get the optimization variables and bounds
    bnd = []
    tag = []
    log = []
    for tag_tmp, mode_tmp in var_list.items():
        # extract the variable
        bnd_tmp = mode_tmp["bnd"]
        log_tmp = mode_tmp["log"]
        use_tmp = mode_tmp["use"]

        # add the variable
        if use_tmp:
            tag.append(tag_tmp)
            log.append(log_tmp)
            if log_tmp:
                bnd.append(np.log(bnd_tmp))
            else:
                bnd.append(bnd_tmp)

    # extract a design
    def get_var(x):
        # init the variables
        var_tmp = {}

        # assign the optimized variables
        for idx, (tag_tmp, log_tmp) in enumerate(zip(tag, log, strict=True)):
            if log_tmp:
                var_tmp[tag_tmp] = np.exp(x[idx])
            else:
                var_tmp[tag_tmp] = x[idx]

        return var_tmp

    # get the objective function
    def get_optim(x):
        # get dimension
        n_var = x.shape[0]
        n_sweep = x.shape[1]

        # assign the optimized variables
        assert n_var == len(tag), "invalid length"
        assert n_var == len(log), "invalid length"
        assert n_var == len(bnd), "invalid length"

        # extract a design
        var_tmp = get_var(x)

        # get the objective
        obj = fct_obj(var_tmp, n_sweep)

        return obj

    # solve the optimization problem
    res = optimize.differential_evolution(
        get_optim,
        bnd,
        disp=False,
        vectorized=True,
        updating="deferred",
        **options,
    )

    # assign the optimized variable
    var = get_var(res.x)

    return var, res
