"""
Compute single transformer designs with the analytical optima.
"""

__author__ = "Thomas Guillod"
__copyright__ = "Thomas Guillod - Dartmouth College"
__license__ = "Mozilla Public License Version 2.0"

from transformer_scaling import model
from transformer_scaling import display
import param


def fct_solve(geom, conv, split, trg):
    """
    Optimize, solve, and plot a single transformer design (analytical optima).
    """

    # optimization type (optimal frequency and number of turns)
    opt = "freq_turn"

    # use the simplified design
    simplified = True

    # get the parameters
    constant = param.get_constant()
    design = param.get_design(geom, conv, split, simplified)

    # solve the design
    design = model.get_solve(geom, trg, opt, constant, design)

    return design


if __name__ == "__main__":
    # transformer excitation
    conv_list = ["sin", "dab", "src"]

    # different transformer configurations
    shape_list = [
        ("shell_inter", "1p"),
        ("shell_simple", "1p"),
        ("core_type", "1p"),
        ("shell_inter", "sp_wye"),
        ("shell_inter", "sp_delta"),
        ("shell_simple", "sp_wye"),
        ("shell_simple", "sp_delta"),
        ("core_type", "sp_wye"),
        ("core_type", "sp_delta"),
        ("three_phase", "3p_wye"),
        ("three_phase", "3p_delta"),
    ]

    # run the sweeps for the different configurations
    for geom, split in shape_list:
        print(f"========================================== {geom} / {split}")
        for conv in conv_list:
            print(f"========= {conv}")

            design = fct_solve(geom, conv, split, "volume")
            summary = display.get_summary(design)
            print(f"volume / {summary}")

            design = fct_solve(geom, conv, split, "mass")
            summary = display.get_summary(design)
            print(f"mass / {summary}")
