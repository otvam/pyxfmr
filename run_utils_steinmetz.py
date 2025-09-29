"""
Extract the Steinmetz parameters from loss data for TDK N97 at 80C.
"""

__author__ = "Thomas Guillod"
__copyright__ = "Thomas Guillod - Dartmouth College"
__license__ = "Mozilla Public License Version 2.0"

import numpy as np
from transformer_utils import steinmetz_extractor


if __name__ == "__main__":
    # loss data for TDK N97 at 80C
    dset = {
        "f_dat": np.array([50e3, 100e3, 200e3, 300e3, 500e3]),
        "B_dat": np.array([25e-3, 50e-3, 100e-3, 200e-3]),
        "p_dat": np.array(
            [
                [0.57e3, 3.91e3, 21.67e3, 125.62e3],
                [1.82e3, 10.12e3, 57.27e3, 339.86e3],
                [5.77e3, 31.04e3, 172.17e3, 994.69e3],
                [13.99e3, 69.55e3, 368.77e3, 2010e3],
                [49.01e3, 219.33e3, 1066e3, 4875e3],
            ]
        ),
    }

    # fitting / evaluation parameters
    grid = {
        "p_min": 5e3,  # minimum loss density to be considered
        "p_max": 500e3,  # maximum loss density to be considered
        "f_eval": np.logspace(np.log10(100e3), np.log10(300e3), 15),  # frequency values
        "B_eval": np.logspace(np.log10(30e-3), np.log10(130e-3), 15),  # flux density values
    }

    # optimization parameters
    optim = {
        "ftol": 1e-6,  # termination tolerance for the optimizer
        "xtol": 1e-6,  # termination tolerance for the optimizer
        "k_bnd": (1e-3, 1e-1),  # bounds for the parameter k
        "alpha_stm_bnd": (1.0, 2.0),  # bounds for the parameter alpha
        "beta_stm_bnd": (2.0, 3.0),  # bounds for the parameter beta
    }

    # get a Steinmetz fit
    (k_stm, alpha_stm, beta_stm) = steinmetz_extractor.get_optimal_steinmetz(dset, grid, optim)

    # evaluate the Steinmetz fit
    steinmetz_extractor.get_eval_steinmetz(dset, grid, k_stm, alpha_stm, beta_stm)
