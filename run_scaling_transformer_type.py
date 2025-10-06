"""
Compute the loss factors for the different transformer configurations:
    - Impact of the core shape on the losses.
    - Impact of the waveshapes on the losses.
"""

__author__ = "Thomas Guillod"
__copyright__ = "Thomas Guillod - Dartmouth College"
__license__ = "Mozilla Public License Version 2.0"

import numpy as np
from transformer_scaling import model
import param


def _get_conv_factor(design, alpha_stm, beta_stm):
    """
    Get the loss factor for the waveshapes.
    """

    # extract the solution
    cosphi = design["cosphi"]
    harm_freq = design["harm_freq"]
    igse_loss = design["igse_loss"]
    igse_flux = design["igse_flux"]

    # get the excitation factor
    fact_conv = 1.0
    fact_conv *= harm_freq ** (2 * beta_stm - 2 * alpha_stm)
    fact_conv *= (igse_flux ** (2 * beta_stm)) * (igse_loss**2)
    fact_conv *= (1 / cosphi) ** (2 * beta_stm)
    fact_conv = fact_conv ** (1 / (2 + beta_stm))

    return fact_conv


def _get_shape_factor(design, split, alpha_stm, beta_stm):
    """
    Get the loss factor for the transformer shape.
    """

    # extract the solution
    P_trf = design["P_trf"]
    V_core = design["V_core"]
    A_core = design["A_core"]
    d_winding = design["d_winding"]
    A_winding = design["A_winding"]
    V_conductor = design["V_conductor"]

    # get the current loading for the different geometries
    if split == "1p":
        xi = 2
    elif split in ["3p_wye", "3p_delta"]:
        xi = 2 / 3
    elif split in ["sp_wye", "sp_delta"]:
        xi = 2
    else:
        raise ValueError("invalid split")

    # get the geometry factor
    fact_shape = 1.0
    fact_shape *= (xi * P_trf) ** (2 * beta_stm)
    fact_shape *= (V_core**2) / (A_core ** (2 * beta_stm))
    fact_shape *= (V_conductor**beta_stm) / (A_winding ** (2 * beta_stm))
    fact_shape *= d_winding ** (2 * beta_stm - 2 * alpha_stm)
    fact_shape = fact_shape ** (1 / (2 + beta_stm))

    return fact_shape


def fct_solve(geom, conv, split):
    """
    Solve a transformer design and extract the loss factors.
    """

    # geometry target
    trg = "volume"

    # optimization type (optimal frequency and number of turns)
    opt = "freq_turn"

    # use the simplified design
    simplified = True

    # get the Steinmetz parameters
    (k_stm, alpha_stm, beta_stm) = param.get_steinmetz()

    # get the parameters
    constant = param.get_constant()
    design = param.get_design(geom, conv, split, simplified)

    # solve the design
    design = model.get_solve(geom, trg, opt, constant, design)

    # get the loss factors
    fact_conv = _get_conv_factor(design, alpha_stm, beta_stm)
    fact_shape = _get_shape_factor(design, split, alpha_stm, beta_stm)

    # extract the solution
    P_loss = design["P_loss"]
    penalty = design["penalty"]

    # get the remaining factor (should be a constant)
    fact_other = P_loss / (fact_conv * fact_shape)

    # show the solution
    tag = f"{geom}_{split}_{conv}"
    print(f"{tag} / penalty = {penalty:.2f} # / P_loss = {P_loss:.2f} W")

    return fact_other


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

    # list for collecting the loss factors
    fact_other = []

    # run the sweeps for the different configurations
    for geom, split in shape_list:
        print(f"========================================== {geom} / {split}")
        for conv in conv_list:
            fact_other.append(fct_solve(geom, conv, split))

    # check the loss factors
    err = (np.max(fact_other) - np.min(fact_other)) / np.mean(fact_other)
    assert err < 1e-3, "invalid loss factors"
