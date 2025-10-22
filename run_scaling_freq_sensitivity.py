"""
Verify the frequency diversity property of transformers:
    - The analytical optima are used for the frequency and the number of turns.
    - The best transformer (optimal frequency) is computed.
    - A transformer is computed with a reduced frequency.
"""

__author__ = "Thomas Guillod"
__copyright__ = "Thomas Guillod - Dartmouth College"
__license__ = "Mozilla Public License Version 2.0"

import numpy as np
import matplotlib.pyplot as plt
from transformer_scaling import model
from transformer_scaling import vector
import param


def fct_solve(opt, f_def, n_def, n_sweep):
    """
    Solve the optimal transformer for a given boxed volume and power rating.
    """

    # transformer geometry
    geom = "shell_inter"

    # geometry target
    trg = "volume"

    # transformer excitation
    conv = "sin"
    split = "1x1p"

    # use the simplified design
    simplified = True

    # get the parameters
    constant = param.get_constant()
    design = param.get_design(geom, conv, split, simplified)

    # assign the frequency and the number of turns
    design["f_def"] = f_def
    design["n_def"] = n_def

    # solve the design
    design = vector.get_vectorize(design, n_sweep)
    design = model.get_solve(geom, trg, opt, constant, design)

    # extract the results
    f_sw = design["f_sw"]
    P_loss = design["P_loss"]

    return f_sw, P_loss


def fct_plot(f_sw, P_loss, P_diversity, f_sw_opt, P_loss_opt):
    """
    Plot the frequency diversity of the transformer.
    """

    # create the figure
    (fig, axes) = plt.subplots(1, num="diversity", figsize=(6.4, 4.8))

    # plot the variables
    axes.plot(1e-3 * f_sw, 1e0 * P_loss, "-b")
    axes.plot(1e-3 * f_sw, 1e0 * P_diversity, "--r")
    axes.plot(1e-3 * f_sw_opt, 1e0 * P_loss_opt, "ok")

    # add cosmetics
    axes.set_xscale("linear")
    axes.set_yscale("linear")
    axes.set_xlabel("f_sw (kHz)")
    axes.set_ylabel("P_loss (W)")
    axes.grid()
    fig.tight_layout()


if __name__ == "__main__":
    # number of sweeps
    n_sweep = 100

    # define the frequency sweep
    f_sweep = np.linspace(50e3, 500e3, n_sweep)

    # Steinmetz parameters
    (k_stm, alpha_stm, beta_stm) = param.get_steinmetz()

    # run the scaling laws for the different configurations (frequency sweep with optimal number of turns)
    (f_sw, P_loss) = fct_solve("turn", f_sweep, np.nan, n_sweep)

    # get the best transformer (optimal frequency and optimal number of turns)
    (f_sw_opt, P_loss_opt) = fct_solve("freq_turn", np.nan, np.nan, None)

    # compute the frequency diversity with the analytical expression
    term_1 = ((f_sw**2) / (f_sw_opt**2)) ** (alpha_stm / (2 + beta_stm))
    term_2 = (1 - (alpha_stm / beta_stm) + (alpha_stm / beta_stm) * ((f_sw_opt**2) / (f_sw**2))) ** (beta_stm / (2 + beta_stm))
    P_diversity = P_loss_opt * (term_1 * term_2)

    # plot the frequency diversity
    fct_plot(f_sw, P_loss, P_diversity, f_sw_opt, P_loss_opt)

    # check the frequency diversity
    err = np.max(np.abs(P_loss - P_diversity))
    assert err < 1e-9, "invalid scaling law fit"

    # show plots
    plt.show()
