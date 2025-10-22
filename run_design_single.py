"""
Compute single transformer designs with the analytical optima.
"""

__author__ = "Thomas Guillod"
__copyright__ = "Thomas Guillod - Dartmouth College"
__license__ = "Mozilla Public License Version 2.0"

import matplotlib.pyplot as plt
from transformer_scaling import model
from transformer_scaling import display
import param


def fct_select(name, choice, default):
    """
    Get the user choice from the command line.
    """

    # get the options
    print(f"========================================== {name}")
    print(f"select index: {name}")
    for idx, data in enumerate(choice):
        print(f"    {idx} / {data}")

    # get the input
    idx = input(f"enter choice [{default}]? ")

    # parse the input
    try:
        select = choice[int(idx)]
    except (ValueError, IndexError):
        select = default

    # display the selection
    print(f"selected option: {select}")
    print(f"========================================== {name}")

    return select


if __name__ == "__main__":
    # transformer excitation
    trg_list = ["volume", "mass"]
    trg = fct_select("fixed parameter", trg_list, "volume")

    # transformer excitation
    conv_list = ["sin", "dab", "src"]
    conv = fct_select("converter type", conv_list, "sin")

    # transformer configurations
    shape_list = [
        ("shell_inter", "1x1p"),
        ("shell_simple", "1x1p"),
        ("core_type", "1x1p"),
        ("shell_inter", "3x1p_wye"),
        ("shell_inter", "3x1p_delta"),
        ("shell_simple", "3x1p_wye"),
        ("shell_simple", "3x1p_delta"),
        ("core_type", "3x1p_wye"),
        ("core_type", "3x1p_delta"),
        ("three_phase", "1x3p_wye"),
        ("three_phase", "1x3p_delta"),
    ]
    (geom, split) = fct_select("transformer type", shape_list, ("shell_inter", "1x1p"))

    # optimization type
    opt_list = ["none", "freq", "turn", "freq_turn"]
    opt = fct_select("optimization type", opt_list, "freq_turn")

    # use the simplified design
    simplified = True

    # get the parameters
    constant = param.get_constant()
    design = param.get_design(geom, split, conv, simplified)

    # solve the design
    design = model.get_solve(geom, trg, opt, constant, design)

    # display the solution (with or without details)
    display.get_geom("design", geom, design)
    display.get_disp("design", design)

    # show plots
    plt.show()
