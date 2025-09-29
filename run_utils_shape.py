"""
Find the optimal aspect ratios for the different transformer types.
"""

__author__ = "Thomas Guillod"
__copyright__ = "Thomas Guillod - Dartmouth College"
__license__ = "Mozilla Public License Version 2.0"

from transformer_utils import transformer_shape


if __name__ == "__main__":
    # define the Steinmetz parameters
    alpha_stm = 1.7215
    beta_stm = 2.4608

    # optimization parameters
    optim = {
        "tol": 1e-6,  # termination tolerance for the optimizer
        "ratio_cw_bnd": (1.0, 10.0),  # bounds for the core / window ratio
        "ratio_c_bnd": (1.0, 5.0),  # bounds for the core aspect ratio
        "ratio_w_bnd": (1.0, 3.0),  # bounds for the winding aspect ratio
    }

    # get the optimal aspect ratios
    transformer_shape.get_optimal_shape("shell_inter", alpha_stm, beta_stm, optim)
    transformer_shape.get_optimal_shape("shell_simple", alpha_stm, beta_stm, optim)
    transformer_shape.get_optimal_shape("core_type", alpha_stm, beta_stm, optim)
    transformer_shape.get_optimal_shape("three_phase", alpha_stm, beta_stm, optim)
