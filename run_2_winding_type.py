"""
Verify the scaling laws with respect to the winding parameters:
    - Winding conductivity.
    - Winding filling factor.
    - Winding stranding.
    - Winding interleaving.
"""

__author__ = "Thomas Guillod"
__copyright__ = "Thomas Guillod - Dartmouth College"
__license__ = "Mozilla Public License Version 2.0"

import numpy as np
import matplotlib.pyplot as plt
from transformer_scaling import solver
from transformer_scaling import vector
import param


def fct_solve(sigma, k_fill, d_strand, n_inter, n_sweep):
    """
    Solve and return a sweep of transformer designs.
    """

    # transformer geometry
    geom = "shell_simple"

    # geometry target
    trg = "volume"

    # transformer excitation
    conv = "sin"
    split = "1p"

    # optimization type
    opt = "freq"

    # use the simplified design
    simplified = True

    # get the parameters
    constant = param.get_constant()
    design = param.get_design(geom, conv, split, simplified)

    # set the winding values
    design["k_fill"] = k_fill
    design["d_strand"] = d_strand
    design["n_inter"] = n_inter
    design["sigma"] = sigma

    # solve the design
    design = vector.get_vectorize(design, n_sweep)
    design = solver.get_solve(geom, trg, opt, constant, design)

    return design


def fct_fit(x, y):
    """
    Fit a linear curve in a loglog scale (scaling laws).
    The following form is considered: y = scaling * (y ** coeff).
    """

    # fit a linear curve in a loglog
    poly = np.polyfit(np.log10(x), np.log10(y), 1)

    # extract param
    coeff = 1.0 * poly[0]
    scaling = 10.0 ** poly[1]

    return scaling, coeff


def fct_plot(ax, design, var_x, var_y, coeff_ana, label):
    """
    Plot the scaling laws and compute the scaling coefficients.
    """

    # extract the variables
    x_value = design[var_x]
    y_value = design[var_y]

    # find the scaling coefficients
    (scaling, coeff_num) = fct_fit(x_value, y_value)
    x_fit = scaling * (x_value**coeff_num)

    # plot the variables
    ax.plot(label["xscl"] * x_value, label["yscl"] * y_value, "-r")
    ax.plot(label["xscl"] * x_value, label["yscl"] * x_fit, "--b")

    # add cosmetics
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(label["xlabel"])
    ax.set_ylabel(label["ylabel"])
    ax.grid()

    # check the scaling law
    err = np.abs(coeff_ana - coeff_num)
    assert err < 1e-9, "invalid scaling law fit"

    # show the scaling law
    print(f"{var_x} / {var_y} / {coeff_ana:+.2f}")


if __name__ == "__main__":
    # number of sweeps
    n_sweep = 100

    # nominal values for the winding parameters
    k_fill = 0.25
    sigma = 46e6
    d_strand = 71e-6
    n_inter = 1.0

    # define the different sweeps for the winding parameters
    sigma_sweep = np.logspace(np.log10(1e6), np.log10(10e6), n_sweep)
    k_fill_sweep = np.logspace(np.log10(0.1), np.log10(1.0), n_sweep)
    d_strand_sweep = np.logspace(np.log10(10e-6), np.log10(100e-6), n_sweep)
    n_inter_sweep = np.logspace(np.log10(1.0), np.log10(10.0), n_sweep)

    # get the Steinmetz parameters
    (k_stm, alpha_stm, beta_stm) = param.get_steinmetz()

    # run the scaling laws for the conductivity
    coeff_freq = -1
    coeff_loss = -(2 * alpha_stm - beta_stm) / (2 + beta_stm)
    coeff_temp = -(2 * alpha_stm - beta_stm) / (2 + beta_stm)

    design = fct_solve(sigma_sweep, k_fill, d_strand, n_inter, n_sweep)

    print("========================================== conductivity")
    (fig, axes) = plt.subplots(3, num="conductivity", figsize=(6.4, 7.0))
    label = {"xlabel": "sigma (MS/m)", "ylabel": "loss (%)", "xscl": 1e-6, "yscl": 1e2}
    fct_plot(axes[0], design, "sigma", "loss", coeff_loss, label)
    label = {"xlabel": "sigma (MS/m)", "ylabel": "T_diff (C)", "xscl": 1e-6, "yscl": 1e0}
    fct_plot(axes[1], design, "sigma", "T_diff", coeff_temp, label)
    label = {"xlabel": "sigma (MS/m)", "ylabel": "f_sw (kHz)", "xscl": 1e-6, "yscl": 1e-3}
    fct_plot(axes[2], design, "sigma", "f_sw", coeff_freq, label)
    fig.tight_layout()

    # run the scaling laws for the filling factor
    coeff_freq = -1
    coeff_loss = -(2 * alpha_stm - beta_stm) / (2 + beta_stm)
    coeff_temp = -(2 * alpha_stm - beta_stm) / (2 + beta_stm)

    design = fct_solve(sigma, k_fill_sweep, d_strand, n_inter, n_sweep)

    print("========================================== filling")
    (fig, axes) = plt.subplots(3, num="filling", figsize=(6.4, 7.0))
    label = {"xlabel": "k_fill (%)", "ylabel": "loss (%)", "xscl": 1e2, "yscl": 1e2}
    fct_plot(axes[0], design, "k_fill", "loss", coeff_loss, label)
    label = {"xlabel": "k_fill (%)", "ylabel": "T_diff (C)", "xscl": 1e2, "yscl": 1e0}
    fct_plot(axes[1], design, "k_fill", "T_diff", coeff_temp, label)
    label = {"xlabel": "k_fill (%)", "ylabel": "f_sw (kHz)", "xscl": 1e2, "yscl": 1e-3}
    fct_plot(axes[2], design, "k_fill", "f_sw", coeff_freq, label)
    fig.tight_layout()

    # run the scaling laws for the stranding
    coeff_freq = -1
    coeff_loss = +(2 * beta_stm - 2 * alpha_stm) / (2 + beta_stm)
    coeff_temp = +(2 * beta_stm - 2 * alpha_stm) / (2 + beta_stm)

    design = fct_solve(sigma, k_fill, d_strand_sweep, n_inter, n_sweep)

    print("========================================== stranding")
    (fig, axes) = plt.subplots(3, num="stranding", figsize=(6.4, 7.0))
    label = {"xlabel": "d_strand (um)", "ylabel": "loss (%)", "xscl": 1e6, "yscl": 1e2}
    fct_plot(axes[0], design, "d_strand", "loss", coeff_loss, label)
    label = {"xlabel": "d_strand (um)", "ylabel": "T_diff (C)", "xscl": 1e6, "yscl": 1e0}
    fct_plot(axes[1], design, "d_strand", "T_diff", coeff_temp, label)
    label = {"xlabel": "d_strand (um)", "ylabel": "f_sw (kHz)", "xscl": 1e6, "yscl": 1e-3}
    fct_plot(axes[2], design, "d_strand", "f_sw", coeff_freq, label)
    fig.tight_layout()

    # run the scaling laws for the interleaving
    coeff_freq = +1
    coeff_loss = -(2 * beta_stm - 2 * alpha_stm) / (2 + beta_stm)
    coeff_temp = -(2 * beta_stm - 2 * alpha_stm) / (2 + beta_stm)

    design = fct_solve(sigma, k_fill, d_strand, n_inter_sweep, n_sweep)

    print("========================================== interleaving")
    (fig, axes) = plt.subplots(3, num="interleaving", figsize=(6.4, 7.0))
    label = {"xlabel": "n_inter (#)", "ylabel": "loss (%)", "xscl": 1e0, "yscl": 1e2}
    fct_plot(axes[0], design, "n_inter", "loss", coeff_loss, label)
    label = {"xlabel": "n_inter (#)", "ylabel": "T_diff (C)", "xscl": 1e0, "yscl": 1e0}
    fct_plot(axes[1], design, "n_inter", "T_diff", coeff_temp, label)
    label = {"xlabel": "n_inter (#)", "ylabel": "f_sw (kHz)", "xscl": 1e0, "yscl": 1e-3}
    fct_plot(axes[2], design, "n_inter", "f_sw", coeff_freq, label)
    fig.tight_layout()

    # show plots
    plt.show()
