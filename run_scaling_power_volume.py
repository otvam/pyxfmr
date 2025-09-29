"""
Verify the scaling laws with respect to the power density or power rating:
    - Scaling of the power density with a constant power.
    - Scaling of the power with a constant power density.
    - Scaling of the power with a constant efficiency.
    - Scaling of the power with a constant temperature.
"""

__author__ = "Thomas Guillod"
__copyright__ = "Thomas Guillod - Dartmouth College"
__license__ = "Mozilla Public License Version 2.0"

import numpy as np
import matplotlib.pyplot as plt
from transformer_scaling import model
from transformer_scaling import vector
import param


def fct_solve(P_trg, rho_trg, n_sweep):
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

    # get the power and volume variables
    P_trf = design["P_trf"]
    S_trf = design["S_trf"]
    V_rms = design["V_rms"]
    I_rms = design["I_rms"]

    # scale the power
    scl = P_trg / P_trf
    V_rms *= np.sqrt(scl)
    I_rms *= np.sqrt(scl)
    P_trf *= scl
    S_trf *= scl

    # assign the new values
    design["rho_def"] = rho_trg
    design["P_trf"] = P_trf
    design["S_trf"] = S_trf
    design["V_rms"] = V_rms
    design["I_rms"] = I_rms

    # solve the design
    design = vector.get_vectorize(design, n_sweep)
    design = model.get_solve(geom, trg, opt, constant, design)

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

    # nominal power rating of the transformers
    P_cst = 10e3

    # nominal power density of the transformers
    rho_cst = 20e6

    # define the different sweeps for the power density and the power rating
    P_sweep = np.logspace(np.log10(1e3), np.log10(100e3), n_sweep)
    rho_sweep = np.logspace(np.log10(1e6), np.log10(100e6), n_sweep)

    # get the Steinmetz parameters
    (k_stm, alpha_stm, beta_stm) = param.get_steinmetz()

    # run the scaling laws for constant power
    coeff_power = 0
    coeff_density = +1
    coeff_freq = +(1 / 3)
    coeff_loss = +(2 * alpha_stm + 3 * beta_stm - 6) / (3 * beta_stm + 6)
    coeff_temp = +(2 * alpha_stm + 5 * beta_stm - 2) / (3 * beta_stm + 6)

    design = fct_solve(P_cst, rho_sweep, n_sweep)

    print("========================================== power")
    (fig, axes) = plt.subplots(5, num="power", figsize=(6.4, 7.0))
    label = {"xlabel": "rho (kW/dm3)", "ylabel": "P_trf (kW)", "xscl": 1e-6, "yscl": 1e-3}
    fct_plot(axes[0], design, "rho", "P_trf", coeff_power, label)
    label = {"xlabel": "rho (kW/dm3)", "ylabel": "rho (kW/dm3)", "xscl": 1e-6, "yscl": 1e-6}
    fct_plot(axes[1], design, "rho", "rho", coeff_density, label)
    label = {"xlabel": "rho (kW/dm3)", "ylabel": "loss (%)", "xscl": 1e-6, "yscl": 1e2}
    fct_plot(axes[2], design, "rho", "loss", coeff_loss, label)
    label = {"xlabel": "rho (kW/dm3)", "ylabel": "T_diff (C)", "xscl": 1e-6, "yscl": 1e0}
    fct_plot(axes[3], design, "rho", "T_diff", coeff_temp, label)
    label = {"xlabel": "rho (kW/dm3)", "ylabel": "f_sw (kHz)", "xscl": 1e-6, "yscl": 1e-3}
    fct_plot(axes[4], design, "rho", "f_sw", coeff_freq, label)
    fig.tight_layout()

    # run the scaling laws for constant density
    coeff_density = 0
    coeff_power = +1
    coeff_freq = -(1 / 3)
    coeff_loss = -(2 * alpha_stm) / (3 * beta_stm + 6)
    coeff_temp = +(beta_stm - 2 * alpha_stm + 2) / (3 * beta_stm + 6)

    design = fct_solve(P_sweep, rho_cst, n_sweep)

    print("========================================== density")
    (fig, axes) = plt.subplots(5, num="density", figsize=(6.4, 7.0))
    label = {"xlabel": "P_trf (kW)", "ylabel": "P_trf (kW)", "xscl": 1e-3, "yscl": 1e-3}
    fct_plot(axes[0], design, "P_trf", "P_trf", coeff_power, label)
    label = {"xlabel": "P_trf (kW)", "ylabel": "rho (kW/dm3)", "xscl": 1e-3, "yscl": 1e-6}
    fct_plot(axes[1], design, "P_trf", "rho", coeff_density, label)
    label = {"xlabel": "P_trf (kW)", "ylabel": "loss (%)", "xscl": 1e-3, "yscl": 1e2}
    fct_plot(axes[2], design, "P_trf", "loss", coeff_loss, label)
    label = {"xlabel": "P_trf (kW)", "ylabel": "T_diff (C)", "xscl": 1e-3, "yscl": 1e0}
    fct_plot(axes[3], design, "P_trf", "T_diff", coeff_temp, label)
    label = {"xlabel": "P_trf (kW)", "ylabel": "f_sw (kHz)", "xscl": 1e-3, "yscl": 1e-3}
    fct_plot(axes[4], design, "P_trf", "f_sw", coeff_freq, label)
    fig.tight_layout()

    # run the scaling laws for constant efficiency
    coeff_loss = 0
    coeff_power = +1
    coeff_density = +(2 * alpha_stm) / (3 * beta_stm + 2 * alpha_stm - 6)
    coeff_temp = +(beta_stm + 2 * alpha_stm - 2) / (3 * beta_stm + 2 * alpha_stm - 6)
    coeff_freq = -(beta_stm - 2) / (3 * beta_stm + 2 * alpha_stm - 6)

    rho_tmp = rho_cst * (P_sweep / P_cst) ** coeff_density
    design = fct_solve(P_sweep, rho_tmp, n_sweep)

    print("========================================== efficiency")
    (fig, axes) = plt.subplots(5, num="efficiency", figsize=(6.4, 7.0))
    label = {"xlabel": "P_trf (kW)", "ylabel": "P_trf (kW)", "xscl": 1e-3, "yscl": 1e-3}
    fct_plot(axes[0], design, "P_trf", "P_trf", coeff_power, label)
    label = {"xlabel": "P_trf (kW)", "ylabel": "rho (kW/dm3)", "xscl": 1e-3, "yscl": 1e-6}
    fct_plot(axes[1], design, "P_trf", "rho", coeff_density, label)
    label = {"xlabel": "P_trf (kW)", "ylabel": "loss (%)", "xscl": 1e-3, "yscl": 1e2}
    fct_plot(axes[2], design, "P_trf", "loss", coeff_loss, label)
    label = {"xlabel": "P_trf (kW)", "ylabel": "T_diff (C)", "xscl": 1e-3, "yscl": 1e0}
    fct_plot(axes[3], design, "P_trf", "T_diff", coeff_temp, label)
    label = {"xlabel": "P_trf (kW)", "ylabel": "f_sw (kHz)", "xscl": 1e-3, "yscl": 1e-3}
    fct_plot(axes[4], design, "P_trf", "f_sw", coeff_freq, label)
    fig.tight_layout()

    # run the scaling laws for constant temperature
    coeff_temp = 0
    coeff_power = +1
    coeff_density = -(beta_stm - 2 * alpha_stm + 2) / (2 * alpha_stm + 5 * beta_stm - 2)
    coeff_loss = -(beta_stm + 2 * alpha_stm - 2) / (2 * alpha_stm + 5 * beta_stm - 2)
    coeff_freq = -(2 * beta_stm) / (2 * alpha_stm + 5 * beta_stm - 2)

    rho_tmp = rho_cst * (P_sweep / P_cst) ** coeff_density
    design = fct_solve(P_sweep, rho_tmp, n_sweep)

    print("========================================== temperature")
    (fig, axes) = plt.subplots(5, num="temperature", figsize=(6.4, 7.0))
    label = {"xlabel": "P_trf (kW)", "ylabel": "P_trf (kW)", "xscl": 1e-3, "yscl": 1e-3}
    fct_plot(axes[0], design, "P_trf", "P_trf", coeff_power, label)
    label = {"xlabel": "P_trf (kW)", "ylabel": "rho (kW/dm3)", "xscl": 1e-3, "yscl": 1e-6}
    fct_plot(axes[1], design, "P_trf", "rho", coeff_density, label)
    label = {"xlabel": "P_trf (kW)", "ylabel": "loss (%)", "xscl": 1e-3, "yscl": 1e2}
    fct_plot(axes[2], design, "P_trf", "loss", coeff_loss, label)
    label = {"xlabel": "P_trf (kW)", "ylabel": "T_diff (C)", "xscl": 1e-3, "yscl": 1e0}
    fct_plot(axes[3], design, "P_trf", "T_diff", coeff_temp, label)
    label = {"xlabel": "P_trf (kW)", "ylabel": "f_sw (kHz)", "xscl": 1e-3, "yscl": 1e-3}
    fct_plot(axes[4], design, "P_trf", "f_sw", coeff_freq, label)
    fig.tight_layout()

    # show plots
    plt.show()
