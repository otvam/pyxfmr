"""
Find the optimal aspect ratios for a transformer type:
    - The results are independent of the volume.
    - The insulation distance are neglected.
    - Bounds are used for the optimization.
"""

__author__ = "Thomas Guillod"
__copyright__ = "Thomas Guillod - Dartmouth College"
__license__ = "Mozilla Public License Version 2.0"

import numpy as np
import scipy.optimize as optimize


def _get_geometry(geom, ratio_cw, ratio_c, ratio_w):
    """
    Create the geometry with a given boxed volume and aspect ratios.
    """

    # consider a dummy area product
    A_prd = 1.0

    # consider a dummy boxed volume
    V_def = 1.0

    # compute the window and core dimensions
    A_core = np.sqrt(A_prd * ratio_cw)
    A_window = np.sqrt(A_prd / ratio_cw)
    x_window = np.sqrt(A_window / ratio_w)
    y_window = np.sqrt(A_window * ratio_w)
    t_core = np.sqrt(A_core / ratio_c)
    z_core = np.sqrt(A_core * ratio_c)

    # compute the core and coil volumes
    if (geom == "shell_simple") or (geom == "shell_inter"):
        # core parameters
        x_core = 2 * x_window + 2 * t_core
        y_core = 1 * y_window + 1 * t_core
        V_core = (x_core * y_core - 2 * x_window * y_window) * z_core

        # coil parameters
        V_window = 1 * (2 * x_window * t_core + 2 * x_window * z_core + np.pi * x_window**2) * y_window

        # box parameters
        y_add = 0 * x_window
        x_add = 0 * x_window
        z_add = 2 * x_window
    elif geom == "core_type":
        # core parameters
        x_core = 1 * x_window + 2 * t_core
        y_core = 1 * y_window + 2 * t_core
        V_core = (x_core * y_core - 1 * x_window * y_window) * z_core

        # coil parameters
        V_window = 2 * (x_window * t_core + x_window * z_core + 0.25 * np.pi * x_window**2) * y_window

        # box parameters
        y_add = 0 * x_window
        x_add = 1 * x_window
        z_add = 1 * x_window
    elif geom == "three_phase":
        # core parameters
        x_core = 2 * x_window + 3 * t_core
        y_core = 1 * y_window + 2 * t_core
        V_core = (x_core * y_core - 2 * x_window * y_window) * z_core

        # coil parameters
        V_window = 3 * (x_window * t_core + x_window * z_core + 0.25 * np.pi * x_window**2) * y_window

        # box parameters
        y_add = 0 * x_window
        x_add = 1 * x_window
        z_add = 1 * x_window
    else:
        raise ValueError("invalid geom")

    # compute the boxed volume
    x_box = x_core + x_add
    y_box = y_core + y_add
    z_box = z_core + z_add
    V_box = x_box * y_box * z_box

    # compute the winding geometry
    if geom == "shell_simple":
        n_wdg = 2
        n_block = 1
    elif geom == "shell_inter":
        n_wdg = 4
        n_block = 2
    elif geom == "core_type":
        n_wdg = 4
        n_block = 2
    elif geom == "three_phase":
        n_wdg = 4
        n_block = 1
    else:
        raise ValueError("invalid geom")

    # get the winding size
    d_winding = x_window / n_wdg
    A_winding = n_block * A_window / n_wdg

    # scale the dimension with the boxed volume
    fact = V_def / V_box
    V_core *= fact ** (3 / 3)
    V_window *= fact ** (3 / 3)
    A_winding *= fact ** (2 / 3)
    A_core *= fact ** (2 / 3)
    d_winding *= fact ** (1 / 3)

    return d_winding, A_winding, V_window, A_core, V_core


def _get_loss_factor(geom, d_winding, A_winding, V_window, A_core, V_core, alpha_stm, beta_stm):
    """
    Get the core shape loss factor (objective function).
    """

    # get the current loading for the different geometries
    if geom == "three_phase":
        xi = 2 / 3
    elif geom == "shell_simple":
        xi = 2
    elif geom == "shell_inter":
        xi = 2
    elif geom == "core_type":
        xi = 2
    else:
        raise ValueError("invalid split")

    # get the geometry factor
    fact_geom = 1.0
    fact_geom *= xi ** (2 * beta_stm)
    fact_geom *= (V_core**2) / (A_core ** (2 * beta_stm))
    fact_geom *= (V_window**beta_stm) / (A_winding ** (2 * beta_stm))
    fact_geom *= d_winding ** (2 * beta_stm - 2 * alpha_stm)
    fact_geom = fact_geom ** (1 / (2 + beta_stm))

    return fact_geom


def get_optimal_shape(geom, alpha_stm, beta_stm, optim):
    """
    Get the optimal aspect ratios for a core shape.
    """

    # extract the optimization data
    tol = optim["tol"]
    ratio_cw_bnd = optim["ratio_cw_bnd"]
    ratio_c_bnd = optim["ratio_c_bnd"]
    ratio_w_bnd = optim["ratio_w_bnd"]

    # function for decoding the optimization variable
    def get_extract(x):
        ratio_cw = x[0]
        ratio_c = x[1]
        ratio_w = x[2]

        return ratio_cw, ratio_c, ratio_w

    # optimization function calculating the loss factor
    def get_optim(x):
        (ratio_cw, ratio_c, ratio_w) = get_extract(x)
        (d_winding, A_winding, V_window, A_core, V_core) = _get_geometry(geom, ratio_cw, ratio_c, ratio_w)
        fact_geom = _get_loss_factor(geom, d_winding, A_winding, V_window, A_core, V_core, alpha_stm, beta_stm)

        return fact_geom

    # get the variables bounds
    bnd = [
        ratio_cw_bnd,
        ratio_c_bnd,
        ratio_w_bnd,
    ]

    # get the initial values
    x_init = [
        np.mean(ratio_cw_bnd),
        np.mean(ratio_c_bnd),
        np.mean(ratio_w_bnd),
    ]

    # solve the optimization problem
    result = optimize.minimize(
        get_optim,
        x_init,
        bounds=bnd,
        tol=tol,
    )

    # extraction the results
    (ratio_cw, ratio_c, ratio_w) = get_extract(result.x)

    # show the results
    print(f"optimal shape / {geom}")
    print(f"    optimal ratios")
    print(f"        ratio_cw = {ratio_cw:.4f}")
    print(f"        ratio_c = {ratio_c:.4f}")
    print(f"        ratio_w = {ratio_w:.4f}")
    print(f"    solver data")
    print(f"        status = {result.status}")
    print(f"        success = {result.success}")

    # assign the output
    out = {
        "ratio_cw": float(ratio_cw),
        "ratio_c": float(ratio_c),
        "ratio_w": float(ratio_w),
    }

    return out
