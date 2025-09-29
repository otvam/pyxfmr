"""
Construct, optimize, and solve a transformer design:
    - Create the geometry of the transformer.
    - If asked, compute the optimal operating frequency.
    - If asked, compute the optimal number of turns.
    - Compute the core losses (Steinmetz equation with constant parameters).
    - Compute the winding losses (litz wire with proximity effect).
    - Compute the temperature (with the area and convection coefficient).
"""

__author__ = "Thomas Guillod"
__copyright__ = "Thomas Guillod - Dartmouth College"
__license__ = "Mozilla Public License Version 2.0"

import numpy as np


def _get_divide(d_window, n_block, n_iso, d_wdg_def, d_iso_def):
    """
    Get the insulation distance for the winding window.
    Shrink the insulation distance if the window is too small.
    """

    # get the minimum size of the window (without shrinking)
    d_min = n_block * d_wdg_def + n_iso * d_iso_def

    # apply the shrinking
    d_iso_min = d_iso_def * (d_window / d_min)

    # get the best case
    d_insulation = np.minimum(d_iso_min, d_iso_def)

    return d_insulation


def _get_penalty(value, rng):
    """
    Compute a penalty factor if a variable is outside the bounds.
    """

    # extract the bounds
    (v_min, v_max) = rng

    # compute the penalty
    penalty = 0.0
    if np.isfinite(v_min):
        penalty += np.maximum((v_min - value) / v_min, 0.0)
    if np.isfinite(v_max):
        penalty += np.maximum((value - v_max) / v_max, 0.0)

    return penalty


def _get_loss_core(constant, design):
    """
    Compute the core losses (Steinmetz equation with constant parameters).
    """

    # extract constant
    B_pk_rng = constant["B_pk_rng"]
    p_core_rng = constant["p_core_rng"]
    f_core_rng = constant["f_core_rng"]
    fact_core = constant["fact_core"]

    # extract constant
    f_sw = design["f_sw"]
    n_turn = design["n_turn"]
    V_rms = design["V_rms"]

    k_stm = design["k_stm"]
    alpha_stm = design["alpha_stm"]
    beta_stm = design["beta_stm"]
    igse_flux = design["igse_flux"]
    igse_loss = design["igse_loss"]
    V_core = design["V_core"]
    A_core = design["A_core"]

    # compute the core losses
    B_pk = (igse_flux *  np.sqrt(2) * V_rms) / (2 * np.pi * f_sw * n_turn * A_core)
    p_core = (k_stm * fact_core * igse_loss) * (f_sw**alpha_stm) * (B_pk**beta_stm)
    P_core = V_core * p_core

    # compute validity and set penalty
    penalty = 0.0
    penalty += _get_penalty(B_pk, B_pk_rng)
    penalty += _get_penalty(f_sw, f_core_rng)
    penalty += _get_penalty(p_core, p_core_rng)

    # assign data
    design["B_pk"] = B_pk
    design["P_core"] = P_core
    design["p_core"] = p_core
    design["penalty_core"] = penalty

    return design


def _get_loss_winding(constant, design):
    """
    Compute the winding losses (litz wire with proximity effect).
    """

    # extract constant
    J_rms_rng = constant["J_rms_rng"]
    f_winding_rng = constant["f_winding_rng"]
    p_winding_rng = constant["p_winding_rng"]
    fact_winding = constant["fact_winding"]
    fact_prox = constant["fact_prox"]

    # extract constant
    f_sw = design["f_sw"]
    n_turn = design["n_turn"]
    I_rms = design["I_rms"]
    sigma = design["sigma"]
    k_fill = design["k_fill"]
    d_strand = design["d_strand"]
    harm_freq = design["harm_freq"]
    V_conductor = design["V_conductor"]
    A_winding = design["A_winding"]
    d_winding = design["d_winding"]

    # constant
    mu0 = 4 * np.pi * 1e-7

    # compute the proximity
    f_eff = harm_freq * f_sw
    a_wdg = (fact_prox / 12) * ((np.pi * mu0 * sigma * k_fill * d_strand * d_winding) ** 2)
    r_hf = 1 + a_wdg * (f_eff**2)

    # compute the winding losses
    J_rms = (I_rms * n_turn) / (k_fill * A_winding)
    p_winding = (k_fill * fact_winding) * r_hf * ((J_rms**2) / sigma)
    P_winding = V_conductor * p_winding

    # compute validity and set penalty
    penalty = 0.0
    penalty += _get_penalty(J_rms, J_rms_rng)
    penalty += _get_penalty(f_eff, f_winding_rng)
    penalty += _get_penalty(p_winding, p_winding_rng)

    # assign data
    design["r_hf"] = r_hf
    design["J_rms"] = J_rms
    design["P_winding"] = P_winding
    design["p_winding"] = p_winding
    design["penalty_winding"] = penalty

    return design


def _get_metrics(design):
    """
    Compute and assign the global figures of merit.
    """

    # extract
    P_loss = design["P_loss"]
    P_core = design["P_core"]
    P_winding = design["P_winding"]
    A_thermal = design["A_thermal"]
    P_trf = design["P_trf"]
    S_trf = design["S_trf"]
    V_box = design["V_box"]
    m_tot = design["m_tot"]

    # core and winding loss ratio
    r_cw = P_core / P_winding

    # compute figures of merit
    loss = P_loss / P_trf
    ht = P_loss / A_thermal
    gamma = P_trf / m_tot
    rho = P_trf / V_box
    cosphi = P_trf / S_trf

    # assign
    design["r_cw"] = r_cw
    design["loss"] = loss
    design["rho"] = rho
    design["ht"] = ht
    design["gamma"] = gamma
    design["cosphi"] = cosphi

    return design


def _get_thermal(constant, design):
    """
    Compute the temperature (with the area and convection coefficient).
    """

    # extract constant
    T_diff_rng = constant["T_diff_rng"]
    fact_thermal = constant["fact_thermal"]

    # extract constant
    A_box = design["A_box"]
    h_coeff = design["h_coeff"]
    A_coeff = design["A_coeff"]
    P_winding = design["P_winding"]
    P_core = design["P_core"]

    # get the total losses
    P_loss = P_core + P_winding

    # compute the temperature rise
    A_thermal = A_coeff * A_box
    T_diff = (fact_thermal * P_loss) / (A_thermal * h_coeff)

    # compute validity and set penalty
    penalty = _get_penalty(T_diff, T_diff_rng)

    # assign data
    design["T_diff"] = T_diff
    design["P_loss"] = P_loss
    design["A_thermal"] = A_thermal
    design["penalty_thermal"] = penalty

    return design


def _get_geometry_box(geom, constant, design):
    """
    Create the geometry with from the boxed volume and aspect ratios.
    """

    # extract constant
    d_core_rng = constant["d_core_rng"]
    d_window_rng = constant["d_window_rng"]
    r_window_rng = constant["r_window_rng"]
    r_core_rng = constant["r_core_rng"]
    r_limb_rng = constant["r_limb_rng"]

    # extract constant
    A_prd = design["A_prd"]
    ratio_cw = design["ratio_cw"]
    ratio_c = design["ratio_c"]
    ratio_w = design["ratio_w"]

    # compute the window and core dimensions
    A_core = np.sqrt(A_prd * ratio_cw)
    A_window = np.sqrt(A_prd / ratio_cw)
    x_window = np.sqrt(A_window / ratio_w)
    y_window = np.sqrt(A_window * ratio_w)
    t_core = np.sqrt(A_core / ratio_c)
    z_core = np.sqrt(A_core * ratio_c)

    # compute the core and window volumes
    if (geom == "shell_simple") or (geom == "shell_inter"):
        # core parameters
        x_core = 2 * x_window + 2 * t_core
        y_core = 1 * y_window + 1 * t_core
        V_core = (x_core * y_core - 2 * x_window * y_window) * z_core

        # coil parameters
        V_window = 1 * (2 * x_window * t_core + 2 * x_window * z_core + np.pi * x_window**2) * y_window

        # box parameters
        y_add = 0 * x_window
        x_add = 0 * x_window
        z_add = 2 * x_window
    elif geom == "core_type":
        # core parameters
        x_core = 1 * x_window + 2 * t_core
        y_core = 1 * y_window + 2 * t_core
        V_core = (x_core * y_core - 1 * x_window * y_window) * z_core

        # coil parameters
        V_window = 2 * (x_window * t_core + x_window * z_core + 0.25 * np.pi * x_window**2) * y_window

        # box parameters
        y_add = 0 * x_window
        x_add = 1 * x_window
        z_add = 1 * x_window
    elif geom == "three_phase":
        # core parameters
        x_core = 2 * x_window + 3 * t_core
        y_core = 1 * y_window + 2 * t_core
        V_core = (x_core * y_core - 2 * x_window * y_window) * z_core

        # coil parameters
        V_window = 3 * (x_window * t_core + x_window * z_core + 0.25 * np.pi * x_window**2) * y_window

        # box parameters
        y_add = 0 * x_window
        x_add = 1 * x_window
        z_add = 1 * x_window
    else:
        raise ValueError("invalid geom")

    # compute the boxed volume and area
    x_box = x_core + x_add
    y_box = y_core + y_add
    z_box = z_core + z_add
    V_box = x_box * y_box * z_box
    A_box = 2 * (x_box * y_box) + 2 * (x_box * z_box) + 2 * (y_box * z_box)

    # compute the aspect ratios
    r_core = z_core / t_core
    r_window = y_window / x_window
    r_limb = t_core / x_window

    # compute validity and set penalty for the dimensions
    penalty = 0.0
    penalty += _get_penalty(t_core, d_core_rng)
    penalty += _get_penalty(z_core, d_core_rng)
    penalty += _get_penalty(x_window, d_window_rng)
    penalty += _get_penalty(y_window, d_window_rng)
    penalty += _get_penalty(r_core, r_core_rng)
    penalty += _get_penalty(r_window, r_window_rng)
    penalty += _get_penalty(r_limb, r_limb_rng)

    # add the results
    design["V_box"] = V_box
    design["V_core"] = V_core
    design["V_window"] = V_window
    design["A_window"] = A_window
    design["A_core"] = A_core
    design["A_box"] = A_box
    design["x_window"] = x_window
    design["y_window"] = y_window
    design["t_core"] = t_core
    design["z_core"] = z_core
    design["x_box"] = x_box
    design["y_box"] = y_box
    design["z_box"] = z_box
    design["penalty_box"] = penalty

    return design


def _get_geometry_winding(geom, constant, design):
    """
    Create the geometry of the winding from the winding window.
    """

    # extract constant
    d_winding_rng = constant["d_winding_rng"]
    r_winding_rng = constant["r_winding_rng"]

    # extract constant
    n_inter = design["n_inter"]
    x_window = design["x_window"]
    y_window = design["y_window"]
    A_window = design["A_window"]
    V_window = design["V_window"]
    V_core = design["V_core"]
    d_iso_def = design["d_iso_def"]
    d_wdg_def = design["d_wdg_def"]
    rho_core = design["rho_core"]
    rho_conductor = design["rho_conductor"]
    rho_insulation = design["rho_insulation"]

    # compute the winding geometry
    if geom == "shell_simple":
        nx_div_wdg = 2
        nx_div_iso = 3
        ny_div_wdg = 1
        ny_div_iso = 2
        n_block = 1
    elif geom == "shell_inter":
        nx_div_wdg = 4
        nx_div_iso = 4
        ny_div_wdg = 1
        ny_div_iso = 2
        n_block = 2
    elif geom == "core_type":
        nx_div_wdg = 4
        nx_div_iso = 5
        ny_div_wdg = 1
        ny_div_iso = 2
        n_block = 2
    elif geom == "three_phase":
        nx_div_wdg = 4
        nx_div_iso = 5
        ny_div_wdg = 1
        ny_div_iso = 2
        n_block = 1
    else:
        raise ValueError("invalid geom")

    # get the insulation distance
    x_insulation = _get_divide(x_window, nx_div_wdg, nx_div_iso, d_wdg_def, d_iso_def)
    y_insulation = _get_divide(y_window, ny_div_wdg, ny_div_iso, d_wdg_def, d_iso_def)
    d_insulation = np.minimum(x_insulation, y_insulation)

    # get the winding size
    d_winding = (x_window - nx_div_iso * d_insulation) / nx_div_wdg
    x_winding = (x_window - nx_div_iso * d_insulation) / nx_div_wdg
    y_winding = (y_window - ny_div_iso * d_insulation) / ny_div_wdg
    A_conductor = nx_div_wdg * ny_div_wdg * x_winding * y_winding
    A_winding = n_block * x_winding * y_winding

    # add interleaving
    d_winding = d_winding / n_inter

    # get the area and volumes
    A_insulation = A_window - A_conductor
    V_conductor = (A_conductor / A_window) * V_window
    V_insulation = (A_insulation / A_window) * V_window

    # compute the masses
    m_core = rho_core * V_core
    m_winding = rho_conductor * V_conductor
    m_insulation = rho_insulation * V_insulation
    m_tot = m_core + m_winding + m_insulation

    # compute the aspect ratios
    r_winding = y_winding / x_winding

    # compute validity and set penalty for the dimensions
    penalty = 0.0
    penalty += _get_penalty(x_winding, d_winding_rng)
    penalty += _get_penalty(y_winding, d_winding_rng)
    penalty += _get_penalty(r_winding, r_winding_rng)

    # add the results
    design["m_tot"] = m_tot
    design["m_core"] = m_core
    design["m_winding"] = m_winding
    design["m_insulation"] = m_insulation
    design["V_conductor"] = V_conductor
    design["V_insulation"] = V_insulation
    design["A_conductor"] = A_conductor
    design["A_insulation"] = A_insulation
    design["A_winding"] = A_winding
    design["y_winding"] = y_winding
    design["x_winding"] = x_winding
    design["d_winding"] = d_winding
    design["d_insulation"] = d_insulation
    design["penalty_geom"] = penalty

    return design


def _get_geometry(geom, trg, constant, design):
    """
    Create the geometry of the transformer from the boxed volume or total mass.
    The area product is adapted with an iterative process until the target is met.
    """

    # extract constant
    n_iter = constant["n_iter"]
    k_error = constant["k_error"]

    # iteration to find the correct value
    for _ in range(n_iter):
        # get the geometry
        design = _get_geometry_box(geom, constant, design)
        design = _get_geometry_winding(geom, constant, design)

        # extract data
        P_trf = design["P_trf"]
        V_box = design["V_box"]
        m_tot = design["m_tot"]
        rho_def = design["rho_def"]
        gamma_def = design["gamma_def"]

        # get the values
        if trg == "volume":
            fom_now = V_box
            fom_def = P_trf / rho_def
        elif trg == "mass":
            fom_now = m_tot
            fom_def = P_trf / gamma_def
        else:
            raise ValueError("invalid trg")

        # update factor for the area product
        fact = (fom_def / fom_now) ** (4 / 3)
        error = np.abs(fom_def - fom_now) / fom_def

        # apply the update
        design["A_prd"] = fact * design["A_prd"]

        # check for convergence
        if np.all(error < k_error):
            break

    # get the final geometry
    design = _get_geometry_box(geom, constant, design)
    design = _get_geometry_winding(geom, constant, design)

    return design


def _get_optimal_freq(constant, design, use_opt):
    """
    Compute the optimal frequency such that the AC/DC resistance ratio is "beta/alpha".
    Other constraints (saturation, losses density, thermal, etc.) can be violated.
    """

    # extract constant
    f_sw_rng = constant["f_sw_rng"]
    f_sw_ratio = constant["f_sw_ratio"]

    # solve with the assigned values
    design = _get_loss(constant, design)

    # extract the values
    alpha_stm = design["alpha_stm"]
    beta_stm = design["beta_stm"]
    f_def = design["f_def"]
    f_sw = design["f_sw"]
    r_hf = design["r_hf"]

    # scale the frequency for the optimal ratio
    f_opt = f_sw * np.sqrt(((beta_stm / alpha_stm) - 1) / (r_hf - 1))

    # assign the frequency
    if use_opt:
        f_sw = f_opt
    else:
        f_sw = f_def

    # compute validity and set penalty
    penalty = 0.0
    penalty += _get_penalty(f_sw, f_sw_rng)
    penalty += _get_penalty(f_sw / f_opt, f_sw_ratio)

    # add the results
    design["penalty_freq"] = penalty
    design["f_sw"] = f_sw
    design["f_opt"] = f_opt

    return design


def _get_optimal_turn(constant, design, use_opt):
    """
    Compute the optimal number of turns such that the core / winding loss ratio "2/beta".
    Other constraints (saturation, losses density, thermal, etc.) can be violated.
    """

    # extract constant
    n_turn_rng = constant["n_turn_rng"]
    n_turn_ratio = constant["n_turn_ratio"]

    # solve with the assigned values
    design = _get_loss(constant, design)

    # extract the values
    V_rms = design["V_rms"]
    I_rms = design["I_rms"]
    P_core = design["P_core"]
    P_winding = design["P_winding"]
    beta_stm = design["beta_stm"]
    n_turn = design["n_turn"]
    n_def = design["n_def"]
    n_trf = design["n_trf"]

    # scale the number of turns for the optimal ratio
    n_opt = n_turn * ((beta_stm / 2) * (P_core / P_winding)) ** (1 / (2 + beta_stm))

    # assign the number of turns
    if use_opt:
        n_turn = n_opt
    else:
        n_turn = n_def

    # set the real number of turns
    n_turn_1 = n_turn / np.sqrt(n_trf)
    n_turn_2 = n_turn * np.sqrt(n_trf)

    # set the real voltage and currents
    V_rms_1 = V_rms / np.sqrt(n_trf)
    V_rms_2 = V_rms * np.sqrt(n_trf)
    I_rms_1 = I_rms * np.sqrt(n_trf)
    I_rms_2 = I_rms / np.sqrt(n_trf)

    # compute validity and set penalty
    penalty = 0.0
    penalty += _get_penalty(n_turn / n_opt, n_turn_ratio)
    penalty += _get_penalty(n_turn_1, n_turn_rng)
    penalty += _get_penalty(n_turn_2, n_turn_rng)

    # add the results
    design["penalty_turn"] = penalty
    design["n_turn_1"] = n_turn_1
    design["n_turn_2"] = n_turn_2
    design["V_rms_1"] = V_rms_1
    design["V_rms_2"] = V_rms_2
    design["I_rms_1"] = I_rms_1
    design["I_rms_2"] = I_rms_2
    design["n_turn"] = n_turn
    design["n_opt"] = n_opt

    return design


def _get_loss(constant, design):
    """
    Solve an operating point (core losses, winding losses, and temperature).
    """

    # get the core and winding losses
    design = _get_loss_core(constant, design)
    design = _get_loss_winding(constant, design)

    # compute the temperature
    design = _get_thermal(constant, design)

    # extract the global metrics
    design = _get_metrics(design)

    return design


def get_solve(geom, trg, opt, constant, design):
    """
    Construct, optimize, and solve a transformer design:
        - With the optimal frequency and the optimal number of turns.
        - With a fixed frequency and the optimal number of turns.
        - With a fixed frequency and a fixed number of turns.
    """

    # extract
    f_init = design["f_init"]
    A_init = design["A_init"]
    n_init = design["n_init"]

    # assign dummy values
    design["f_sw"] = f_init
    design["A_prd"] = A_init
    design["n_turn"] = n_init

    # get the geometry
    design = _get_geometry(geom, trg, constant, design)

    # get the optimization type
    if opt == "none":
        design = _get_optimal_freq(constant, design, False)
        design = _get_optimal_turn(constant, design, False)
    elif opt == "freq":
        design = _get_optimal_freq(constant, design, True)
        design = _get_optimal_turn(constant, design, True)
    elif opt == "turn":
        design = _get_optimal_freq(constant, design, False)
        design = _get_optimal_turn(constant, design, True)
    else:
        raise ValueError("invalid opt_type")

    # solve the design
    design = _get_loss(constant, design)

    # get total penalty
    penalty = 0.0
    penalty += design["penalty_box"]
    penalty += design["penalty_geom"]
    penalty += design["penalty_freq"]
    penalty += design["penalty_turn"]
    penalty += design["penalty_core"]
    penalty += design["penalty_winding"]
    penalty += design["penalty_thermal"]
    design["penalty"] = penalty

    return design
