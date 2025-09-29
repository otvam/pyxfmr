"""
Definition of the parameters of the transformers.
Some parameters are computed in "transformer_utils".
"""

__author__ = "Thomas Guillod"
__copyright__ = "Thomas Guillod - Dartmouth College"
__license__ = "Mozilla Public License Version 2.0"

import numpy as np


def get_steinmetz():
    """
    Return the Steinmetz parameters for the core losses.
    The values are computed in "run_utils_steinmetz.py".
    """

    k_stm = 0.038971
    alpha_stm = 1.7215
    beta_stm = 2.4608

    return k_stm, alpha_stm, beta_stm


def get_shape(geom):
    """
    Return the optimal aspect ratios for the geometry.
    The values are computed in "run_utils_shape.py".
    """

    if geom == "shell_inter":
        ratio_cw = 5.3403
        ratio_c = 3.5419
        ratio_w = 3.0000
    elif geom == "shell_simple":
        ratio_cw = 5.3404
        ratio_c = 3.5419
        ratio_w = 3.0000
    elif geom == "core_type":
        ratio_cw = 1.7054
        ratio_c = 3.1919
        ratio_w = 3.0000
    elif geom == "three_phase":
        ratio_cw = 1.7665
        ratio_c = 3.1467
        ratio_w = 3.0000
    else:
        raise ValueError("invalid geometry")

    return ratio_cw, ratio_c, ratio_w


def get_converter(conv, phase):
    """
    Return the correction factors for the waveshapes.
    The values are computed in "run_utils_waveform.py".
    """

    if phase == "1p":
        if conv == "sin":
            P_trf = 10000.0000
            S_trf = 10000.0000
            V_rms = 600.0000
            I_rms = 16.6667
            igse_flux = 1.0000
            igse_loss = 1.0000
            harm_freq = 1.0000
        elif conv == "dab":
            P_trf = 10000.0000
            S_trf = 11315.9587
            V_rms = 600.0000
            I_rms = 18.8599
            igse_flux = 1.1105
            igse_loss = 0.8685
            harm_freq = 1.6537
        elif conv == "src":
            P_trf = 10000.0000
            S_trf = 11107.2072
            V_rms = 600.0000
            I_rms = 18.5120
            igse_flux = 1.1105
            igse_loss = 0.8685
            harm_freq = 1.0000
        else:
            raise ValueError("invalid conv")
    elif phase == "3p_wye":
        if conv == "sin":
            P_trf = 10000.0000
            S_trf = 10000.0000
            V_rms = 282.8427
            I_rms = 11.7851
            igse_flux = 1.0000
            igse_loss = 1.0000
            harm_freq = 1.0000
        elif conv == "dab":
            P_trf = 10000.0000
            S_trf = 10943.5812
            V_rms = 282.8427
            I_rms = 12.8971
            igse_flux = 1.0470
            igse_loss = 0.9347
            harm_freq = 1.4103
        elif conv == "src":
            P_trf = 10000.0000
            S_trf = 10471.9755
            V_rms = 282.8427
            I_rms = 12.3413
            igse_flux = 1.0470
            igse_loss = 0.9347
            harm_freq = 1.0000
        else:
            raise ValueError("invalid conv")
    elif phase == "3p_delta":
        if conv == "sin":
            P_trf = 10000.0000
            S_trf = 10000.0000
            V_rms = 489.8979
            I_rms = 6.8041
            igse_flux = 1.0000
            igse_loss = 1.0000
            harm_freq = 1.0000
        elif conv == "dab":
            P_trf = 10000.0000
            S_trf = 10943.5812
            V_rms = 489.8979
            I_rms = 7.4462
            igse_flux = 0.9069
            igse_loss = 1.1633
            harm_freq = 1.4103
        elif conv == "src":
            P_trf = 10000.0000
            S_trf = 10471.9756
            V_rms = 489.8979
            I_rms = 7.1253
            igse_flux = 0.9069
            igse_loss = 1.1633
            harm_freq = 1.0000
        else:
            raise ValueError("invalid conv")
    else:
        raise ValueError("invalid phase")

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
    rho_def = 10e3 / 0.5e-3

    # get the targets for the gravimetric power density
    gamma_def = 10e3 / 1.5

    # secondary to primary voltage transfer ratio
    n_trf = 1.0

    # get the excitation and the waveshape correction factors
    #   - 1p - single-phase converter / one single-phase transformer
    #   - 3p - three-phase converter / one three-phase transformer (wye or delta connections)
    #   - sp - three-phase converter / three single-phase transformers (wye or delta connections)
    if split == "1p":
        (P_trf, S_trf, V_rms, I_rms, igse_flux, igse_loss, harm_freq) = get_converter(conv, "1p")
    elif split == "3p_wye":
        (P_trf, S_trf, V_rms, I_rms, igse_flux, igse_loss, harm_freq) = get_converter(conv, "3p_wye")
    elif split == "3p_delta":
        (P_trf, S_trf, V_rms, I_rms, igse_flux, igse_loss, harm_freq) = get_converter(conv, "3p_delta")
    elif split == "sp_wye":
        (P_trf, S_trf, V_rms, I_rms, igse_flux, igse_loss, harm_freq) = get_converter(conv, "3p_wye")
        P_trf = P_trf / 3
        S_trf = S_trf / 3
    elif split == "sp_delta":
        (P_trf, S_trf, V_rms, I_rms, igse_flux, igse_loss, harm_freq) = get_converter(conv, "3p_delta")
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
        # ################### core parameters (N97 ferrite)
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
        # ################### density of the material
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
        "f_init": 100e3,  # initial value for the operating frequency
        "n_init": 10.0,  # initial value for the number of turns
        # ################### default values (will be optimized)
        "f_def": np.nan,  # default (non-optimized) operating frequency
        "n_def": np.nan,  # default (non-optimized) number of turns
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
