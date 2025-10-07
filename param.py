"""
Definition of the parameters of the transformers.
Some parameters are computed in "transformer_utils".
"""

__author__ = "Thomas Guillod"
__copyright__ = "Thomas Guillod - Dartmouth College"
__license__ = "Mozilla Public License Version 2.0"

import json
import os.path
import numpy as np


def get_steinmetz():
    """
    Return the Steinmetz parameters for the core losses.
    The values are computed in "run_utils_steinmetz.py".
    """

    # load the values
    folder = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(folder, "transformer_data", "param_steinmetz.json"), "r") as fid:
        data = json.load(fid)

    # extract the values
    k_stm = data["k_stm"]
    alpha_stm = data["alpha_stm"]
    beta_stm = data["beta_stm"]

    return k_stm, alpha_stm, beta_stm


def get_shape(geom):
    """
    Return the optimal aspect ratios for the geometry.
    The values are computed in "run_utils_shape.py".
    """

    # load the values
    folder = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(folder, "transformer_data", "param_shape.json"), "r") as fid:
        data = json.load(fid)

    # select the values
    data = data[geom]

    # extract the values
    ratio_cw = data["ratio_cw"]
    ratio_c = data["ratio_c"]
    ratio_w = data["ratio_w"]

    return ratio_cw, ratio_c, ratio_w


def get_waveform(conv, phase):
    """
    Return the correction factors for the waveshapes.
    The values are computed in "run_utils_waveform.py".
    """

    # load the values
    folder = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(folder, "transformer_data", "param_waveform.json"), "r") as fid:
        data = json.load(fid)

    # select the values
    data = data[conv][phase]

    # extract the values
    P_trf = data["P_trf"]
    S_trf = data["S_trf"]
    V_rms = data["V_rms"]
    I_rms = data["I_rms"]
    igse_flux = data["igse_flux"]
    igse_loss = data["igse_loss"]
    harm_freq = data["harm_freq"]

    return P_trf, S_trf, V_rms, I_rms, igse_flux, igse_loss, harm_freq


def get_constant():
    """
    Get the constant parameters.
    """

    constant = {
        # ################### area product iterations
        "n_iter": 6,  # number of iterations to find the area product
        "k_error": 1e-3,  # relative tolerance for the area product
        # ################### safety factors
        "fact_core": 1.25,  # safety factor for the core losses
        "fact_winding": 1.10,  # safety factor for the winding losses
        "fact_prox": 1.25,  # safety factor for the proximity losses
        "fact_thermal": 1.25,  # safety factor for the temperature
        # ################### operating ranges
        "f_core_rng": [np.nan, 400e3],  # acceptable frequency for the core material
        "f_winding_rng": [np.nan, 1500e3],  # acceptable frequency for the winding material
        "B_pk_rng": [np.nan, 200e-3],  # acceptable range for the flux density
        "p_core_rng": [np.nan, 500e3],  # acceptable range for the loss density
        "J_rms_rng": [np.nan, 6e6],  # acceptable range for the current density
        "p_winding_rng": [np.nan, 500e3],  # acceptable range for the loss density
        "T_diff_rng": [np.nan, 80.0],  # acceptable range for the temperature
        # ################### optimization ranges
        "f_sw_rng": [10e3, 500e3],  # acceptable range for the switching frequency
        "n_turn_rng": [2.0, 50.0],  # acceptable range for the total number of turns
        "f_sw_ratio": [np.nan, np.nan],  # ratio between the optimal and selected frequency
        "n_turn_ratio": [np.nan, np.nan],  # ratio between the optimal and selected frequency
        # ################### geometry ranges
        "d_winding_rng": [1.5e-3, np.nan],  # range for the size of the windings
        "d_core_rng": [3.0e-3, np.nan],  # range for size of the core limbs
        "d_window_rng": [3.0e-3, np.nan],  # range for size of the window
        "r_winding_rng": [1.0, 12.0],  # range for the winding aspect ratio
        "r_window_rng": [0.5, 6.0],  # range for the window aspect ratio
        "r_core_rng": [0.5, 5.0],  # range for the core aspect ratio
        "r_limb_rng": [0.2, 5.0],  # range for the limb / window aspect ratio
    }

    return constant


def get_design(geom, conv, split, simplified):
    """
    Return the transformer parameters.
    """

    # get the Steinmetz parameters
    (k_stm, alpha_stm, beta_stm) = get_steinmetz()

    # get the optimal aspect ratios
    (ratio_cw, ratio_c, ratio_w) = get_shape(geom)

    # get the targets for the volumetric power density
    rho_def = 20e6

    # get the targets for the gravimetric power density
    gamma_def = 7e3

    # secondary to primary voltage transfer ratio
    n_trf = 1.0

    # get the excitation and the waveshape correction factors
    #   - 1p - single-phase converter / one single-phase transformer
    #   - 3p - three-phase converter / one three-phase transformer (wye or delta connections)
    #   - sp - three-phase converter / three single-phase transformers (wye or delta connections)
    if split == "1p":
        (P_trf, S_trf, V_rms, I_rms, igse_flux, igse_loss, harm_freq) = get_waveform(conv, "1p")
    elif split == "3p_wye":
        (P_trf, S_trf, V_rms, I_rms, igse_flux, igse_loss, harm_freq) = get_waveform(conv, "3p_wye")
    elif split == "3p_delta":
        (P_trf, S_trf, V_rms, I_rms, igse_flux, igse_loss, harm_freq) = get_waveform(conv, "3p_delta")
    elif split == "sp_wye":
        (P_trf, S_trf, V_rms, I_rms, igse_flux, igse_loss, harm_freq) = get_waveform(conv, "3p_wye")
        P_trf = P_trf / 3
        S_trf = S_trf / 3
    elif split == "sp_delta":
        (P_trf, S_trf, V_rms, I_rms, igse_flux, igse_loss, harm_freq) = get_waveform(conv, "3p_delta")
        P_trf = P_trf / 3
        S_trf = S_trf / 3
    else:
        raise ValueError("invalid split")

    # if the simplified model is used, remove the insulation distance
    if simplified:
        d_iso_def = 0.0e-3
        d_wdg_def = 1.0e-3
    else:
        d_iso_def = 1.0e-3
        d_wdg_def = 1.0e-3

    # assign the parameters
    design = {
        # ################### core parameters
        "k_stm": k_stm,  # Steinmetz parameter k
        "alpha_stm": alpha_stm,  # Steinmetz parameter alpha
        "beta_stm": beta_stm,  # Steinmetz parameter beta
        # ################### winding parameters
        "n_inter": 1.0,  # number of interleaving for the winding
        "k_fill": 0.25,  # winding fill factor (packing and litz)
        "d_strand": 71e-6,  # diameter of the litz strands
        "sigma": 46e6,  # conductivity of the conductor
        # ################### thermal parameters
        "h_coeff": 25.0,  # surface convection coefficient
        "A_coeff": 0.8,  # fraction of the area used for cooling
        # ################### density of the materials
        "rho_core": 4920.0,  # density of the core material
        "rho_insulation": 1500.0,  # density of the insulation
        "rho_conductor": 3500.0,  # density of the winding
        # ################### geometry parameters
        "d_iso_def": d_iso_def,  # insulation between the coils
        "d_wdg_def": d_wdg_def,  # minimum size for the coils
        "ratio_cw": ratio_cw,  # aspect ratio between the core and window areas
        "ratio_c": ratio_c,  # aspect ratio of the core limbs
        "ratio_w": ratio_w,  # aspect ratio of the winding windows
        # ################### initial values (only used internally)
        "A_init": 100e-8,  # initial value for the area product
        "f_init": 200e3,  # initial value for the operating frequency
        "n_init": 10.0,  # initial value for the number of turns
        # ################### default values (will be optimized)
        "f_def": 200e3,  # default (non-optimized) operating frequency
        "n_def": 10.0,  # default (non-optimized) number of turns
        # ################### correction factors for non-sinusoidal waveforms
        "igse_flux": igse_flux,  # correction factor for the peak flux
        "igse_loss": igse_loss,  # correction factor for the core losses
        "harm_freq": harm_freq,  # frequency correction factor for the winding losses
        # ################### operating parameters
        "n_trf": n_trf,  # secondary to primary voltage transfer ratio
        "P_trf": P_trf,  # active power rating of the transformer
        "S_trf": S_trf,  # apparent power rating of the transformer
        "I_rms": I_rms,  # RMS current applied to the windings
        "V_rms": V_rms,  # RMS voltage applied to the windings
        # ################### geometry parameters
        "rho_def": rho_def,  # target value for the volumetric power density
        "gamma_def": gamma_def,  # target value for the gravimetric power density
    }

    return design
