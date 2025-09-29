"""
Compute single transformer designs with the analytical optima.
"""

__author__ = "Thomas Guillod"
__copyright__ = "Thomas Guillod - Dartmouth College"
__license__ = "Mozilla Public License Version 2.0"

import matplotlib.pyplot as plt
from transformer_scaling import solver
from transformer_scaling import display
import param


if __name__ == "__main__":
    # transformer geometry
    geom = "shell_inter"

    # geometry target
    trg = "volume"

    # optimization type
    opt = "freq"

    # transformer excitation
    conv = "sin"
    split = "1p"

    # use the simplified design
    simplified = True

    # get the parameters
    constant = param.get_constant()
    design = param.get_design(geom, conv, split, simplified)

    # solve the design
    design = solver.get_solve(geom, trg, opt, constant, design)

    # display the solution (with or without details)
    display.get_geom("single", geom, design)
    display.get_disp("single", design)

    # show plots
    plt.show()
