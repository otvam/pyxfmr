"""
Extract the Steinmetz parameters from loss data:
    - Resample the loss data on a grid with interpolation.
    - Filter the loss data with respect to the loss density.
    - Find the optimal Steinmetz parameters.
    - Display the quality of the fit.
"""

__author__ = "Thomas Guillod"
__copyright__ = "Thomas Guillod - Dartmouth College"
__license__ = "Mozilla Public License Version 2.0"

import numpy as np
import scipy.optimize as optimize
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt


def get_interp_losses(dset, grid):
    """
    Interpolate the loss data on a grid.
    """

    # extract the dataset
    f_dat = dset["f_dat"]
    B_dat = dset["B_dat"]
    p_dat = dset["p_dat"]

    # extract the grid data
    p_min = grid["p_min"]
    p_max = grid["p_max"]
    f_eval = grid["f_eval"]
    B_eval = grid["B_eval"]

    # create the interpolant from the loss data
    interp = interpolate.RegularGridInterpolator((np.log10(f_dat), np.log10(B_dat)), np.log10(p_dat))

    # get the interpolation points
    (f_eval, B_eval) = np.meshgrid(f_eval, B_eval)
    f_eval = f_eval.flatten()
    B_eval = B_eval.flatten()

    # interpolate the data on the evaluation grid
    pts = np.array([f_eval, B_eval]).transpose()
    p_eval = 10 ** interp(np.log10(pts))

    # filter the data with respect to the loss density
    idx = (p_eval >= p_min) & (p_eval <= p_max)
    f_eval = f_eval[idx]
    B_eval = B_eval[idx]
    p_eval = p_eval[idx]

    return f_eval, B_eval, p_eval


def get_optimal_steinmetz(dset, grid, optim):
    """
    Get a Steinmetz fit from loss data.
    """

    # extract the optimization data
    ftol = optim["ftol"]
    xtol = optim["xtol"]
    k_bnd = optim["k_bnd"]
    alpha_stm_bnd = optim["alpha_stm_bnd"]
    beta_stm_bnd = optim["beta_stm_bnd"]

    # interpolate the loss data on grid
    (f_fit, B_fit, p_fit) = get_interp_losses(dset, grid)

    # function for decoding the optimization variable
    def get_extract(x):
        k_stm = 10 ** x[0]
        alpha_stm = x[1]
        beta_stm = x[2]

        return k_stm, alpha_stm, beta_stm

    # function computing the relative error
    def get_compute(x):
        (k_stm, alpha_stm, beta_stm) = get_extract(x)
        p_stm = k_stm * (f_fit**alpha_stm) * (B_fit**beta_stm)
        err = np.abs((p_stm - p_fit) / p_fit)

        return err

    # get the variables bounds
    bnd_min = [np.min(np.log10(k_bnd)), np.min(alpha_stm_bnd), np.min(beta_stm_bnd)]
    bnd_max = [np.max(np.log10(k_bnd)), np.max(alpha_stm_bnd), np.max(beta_stm_bnd)]

    # get the initial values
    x_bnd = [np.mean(np.log10(k_bnd)), np.mean(alpha_stm_bnd), np.mean(beta_stm_bnd)]

    # solve the optimization problem
    result = optimize.least_squares(
        get_compute,
        x_bnd,
        bounds=(bnd_min, bnd_max),
        ftol=ftol,
        xtol=xtol,
    )

    # extract the results
    (k_stm, alpha_stm, beta_stm) = get_extract(result.x)

    # show the results
    print(f"optimal steinmetz")
    print(f"    k_stm = {k_stm:.6f}")
    print(f"    alpha_stm = {alpha_stm:.4f}")
    print(f"    beta_stm = {beta_stm:.4f}")
    print(f"solver data")
    print(f"    status = {result.status}")
    print(f"    success = {result.success}")

    return k_stm, alpha_stm, beta_stm


def get_eval_steinmetz(dset, grid, k_stm, alpha_stm, beta_stm):
    """
    Evaluate a Steinmetz fit with loss data.
    """

    # interpolate the loss data on grid
    (f_eval, B_eval, p_eval) = get_interp_losses(dset, grid)

    # get the error
    p_stm = k_stm * (f_eval**alpha_stm) * (B_eval**beta_stm)
    err = (p_stm - p_eval) / p_eval

    # create the figure
    (fig, axes) = plt.subplots(num="error", figsize=(6.4, 7.0))

    # plot the variables
    sc = axes.scatter(1e-3 * f_eval, 1e3 * B_eval, 10, 1e2 * err)

    # add colorbar
    cbar = plt.colorbar(sc, ax=axes)
    cbar.set_label("Relative Error (%)")

    # add cosmetics
    axes.set_xscale("log")
    axes.set_yscale("log")
    axes.set_xlabel("Frequency [kHz]")
    axes.set_ylabel("Flux Density [mT]")
    axes.grid()

    # add cosmetics
    fig.tight_layout()

    # add cosmetics
    plt.tight_layout()

    # get the error metrics
    err = np.abs(err)
    err_rms = np.sqrt(np.mean(err**2))
    err_avg = np.mean(err)
    err_max = np.max(err)

    # print the error metrics
    print(f"error metrics")
    print(f"    err_avg = {1e2 * err_avg:.4f} %")
    print(f"    err_rms = {1e2 * err_rms:.4f} %")
    print(f"    err_max = {1e2 * err_max:.4f} %")
