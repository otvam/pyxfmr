"""
Find the optimal aspect ratios for the different transformer types.
"""

__author__ = "Thomas Guillod"
__copyright__ = "Thomas Guillod - Dartmouth College"
__license__ = "Mozilla Public License Version 2.0"

import json
from transformer_utils import transformer_shape


if __name__ == "__main__":
    # define the Steinmetz parameters
    with open("param_steinmetz.json", "r") as fid:
        data = json.load(fid)
        alpha_stm = data["alpha_stm"]
        beta_stm = data["beta_stm"]

    # optimization parameters
    optim = {
        "tol": 1e-6,  # termination tolerance for the optimizer
        "ratio_cw_bnd": (1.0, 10.0),  # bounds for the core / window ratio
        "ratio_c_bnd": (1.0, 5.0),  # bounds for the core aspect ratio
        "ratio_w_bnd": (1.0, 3.0),  # bounds for the winding aspect ratio
    }

    # different transformer configurations
    geom_list = [
        "shell_inter",
        "shell_simple",
        "core_type",
        "three_phase",
    ]

    # get the optimal aspect ratios
    out = {}
    for geom in geom_list:
        out[geom] = transformer_shape.get_optimal_shape(geom, alpha_stm, beta_stm, optim)

    # write the results
    with open("param_shape.json", "w") as fid:
        json.dump(out, fid, indent=4)
