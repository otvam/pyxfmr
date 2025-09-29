"""
Display and plot a transformer design.
"""

__author__ = "Thomas Guillod"
__copyright__ = "Thomas Guillod - Dartmouth College"
__license__ = "Mozilla Public License Version 2.0"

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def _plot_rectangle(ax, x, y, dx, dy, color):
    """
    Add a sharp rectangle to a plot (from center coordinates).
    """

    rectangle = patches.Rectangle(
        (1e3 * (x - dx / 2), 1e3 * (y - dy / 2)),
        1e3 * dx,
        1e3 * dy,
        edgecolor=None,
        facecolor=color,
    )
    ax.add_patch(rectangle)


def _plot_rounded(ax, x, y, dx, dy, r, color):
    """
    Add a rounded rectangle to a plot (from center coordinates).
    """

    phi_1 = np.linspace(0 * np.pi / 2, 1 * np.pi / 2)
    phi_2 = np.linspace(1 * np.pi / 2, 2 * np.pi / 2)
    phi_3 = np.linspace(2 * np.pi / 2, 3 * np.pi / 2)
    phi_4 = np.linspace(3 * np.pi / 2, 4 * np.pi / 2)

    x_bnd = np.concatenate(
        (
            r * np.cos(phi_1) + (dx / 2),
            r * np.cos(phi_2) - (dx / 2),
            r * np.cos(phi_3) - (dx / 2),
            r * np.cos(phi_4) + (dx / 2),
        )
    )
    y_bnd = np.concatenate(
        (
            r * np.sin(phi_1) + (dy / 2),
            r * np.sin(phi_2) + (dy / 2),
            r * np.sin(phi_3) - (dy / 2),
            r * np.sin(phi_4) - (dy / 2),
        )
    )

    xy = np.stack((x + x_bnd, y + y_bnd)).transpose()
    rectangle = patches.Polygon(1e3 * xy, edgecolor=None, facecolor=color)
    ax.add_patch(rectangle)


def _plot_front_core(ax, geom, design):
    """
    Plot the core (front view).
    """

    # extract param
    t_core = design["t_core"]
    x_window = design["x_window"]
    y_window = design["y_window"]

    # plot the core
    if geom == "shell_simple" or geom == "shell_inter":
        # compute the core dimension
        x_core = 2 * x_window + 2 * t_core
        y_core = y_window + t_core
        offset = x_window / 2 + t_core / 2

        # plot the core
        _plot_rectangle(ax, 0.0, 0.0, x_core, y_core, "gray")
        _plot_rectangle(ax, +offset, 0.0, x_window, y_window, "white")
        _plot_rectangle(ax, -offset, 0.0, x_window, y_window, "white")
    elif geom == "core_type":
        # compute the core dimension
        x_core = x_window + 2 * t_core
        y_core = y_window + 2 * t_core

        # plot the core
        _plot_rectangle(ax, 0.0, 0.0, x_core, y_core, "gray")
        _plot_rectangle(ax, 0.0, 0.0, x_window, y_window, "white")
    elif geom == "three_phase":
        # compute the core dimension
        x_core = 2 * x_window + 3 * t_core
        y_core = y_window + 2 * t_core
        offset = x_window / 2 + t_core / 2

        # plot the core
        _plot_rectangle(ax, 0.0, 0.0, x_core, y_core, "gray")
        _plot_rectangle(ax, +offset, 0.0, x_window, y_window, "white")
        _plot_rectangle(ax, -offset, 0.0, x_window, y_window, "white")
    else:
        raise ValueError("invalid geom")


def _plot_top_core(ax, geom, design):
    """
    Plot the core (top view).
    """

    # extract param
    t_core = design["t_core"]
    z_core = design["z_core"]
    x_window = design["x_window"]

    # plot the core
    if geom == "shell_simple" or geom == "shell_inter":
        # compute the core dimension
        x_core = 2 * x_window + 2 * t_core

        # plot the core
        _plot_rectangle(ax, 0.0, 0.0, x_core, z_core, "gray")
    elif geom == "core_type":
        # compute the core dimension
        x_core = x_window + 2 * t_core

        # plot the core
        _plot_rectangle(ax, 0.0, 0.0, x_core, z_core, "gray")
    elif geom == "three_phase":
        # compute the core dimension
        x_core = 2 * x_window + 3 * t_core

        # plot the core
        _plot_rectangle(ax, 0.0, 0.0, x_core, z_core, "gray")
    else:
        raise ValueError("invalid geom")


def _plot_front_winding(ax, geom, design):
    """
    Plot the windings (front view).
    """

    # extract param
    t_core = design["t_core"]
    x_window = design["x_window"]
    y_window = design["y_window"]
    x_winding = design["x_winding"]
    y_winding = design["y_winding"]
    d_insulation = design["d_insulation"]

    # plot the windings
    if geom == "shell_simple":
        offset = x_window + t_core
        shift = x_winding + d_insulation
        for sign in [-1, +1]:
            _plot_rectangle(ax, sign * offset / 2, 0.0, x_window, y_window, "green")
            _plot_rectangle(ax, sign * offset / 2 + sign * shift / 2, 0.0, x_winding, y_winding, "orange")
            _plot_rectangle(ax, sign * offset / 2 - sign * shift / 2, 0.0, x_winding, y_winding, "chocolate")
    elif geom == "shell_inter":
        offset = x_window + t_core
        shift = 3 * x_winding + 2 * d_insulation
        for sign in [-1.0, +1.0]:
            _plot_rectangle(ax, sign * offset / 2, 0.0, x_window, y_window, "green")
            _plot_rectangle(ax, sign * offset / 2, 0.0, 2 * x_winding, y_winding, "orange")
            _plot_rectangle(ax, sign * offset / 2 + shift / 2, 0.0, 1 * x_winding, y_winding, "chocolate")
            _plot_rectangle(ax, sign * offset / 2 - shift / 2, 0.0, 1 * x_winding, y_winding, "chocolate")
    elif geom == "core_type":
        mid = t_core / 2 + x_window / 2
        offset = t_core + 2 * x_winding + 3 * d_insulation
        side = 3 * d_insulation + 2 * x_winding
        shift = x_winding + d_insulation
        for sign in [-1.0, +1.0]:
            for pos in [-mid, +mid]:
                _plot_rectangle(ax, pos + sign * offset / 2, 0.0, side, y_window, "green")
                _plot_rectangle(ax, pos + sign * offset / 2 + sign * shift / 2, 0.0, x_winding, y_winding, "orange")
                _plot_rectangle(ax, pos + sign * offset / 2 - sign * shift / 2, 0.0, x_winding, y_winding, "chocolate")
    elif geom == "three_phase":
        mid = t_core + x_window
        offset = t_core + 2 * x_winding + 3 * d_insulation
        side = 3 * d_insulation + 2 * x_winding
        shift = x_winding + d_insulation
        for sign in [-1.0, +1.0]:
            for pos in [-mid, 0.0, +mid]:
                _plot_rectangle(ax, pos + sign * offset / 2, 0.0, side, y_window, "green")
                _plot_rectangle(ax, pos + sign * offset / 2 + sign * shift / 2, 0.0, x_winding, y_winding, "orange")
                _plot_rectangle(ax, pos + sign * offset / 2 - sign * shift / 2, 0.0, x_winding, y_winding, "chocolate")
    else:
        raise ValueError("invalid geom")


def _plot_top_winding(ax, geom, design):
    """
    Plot the windings (top view).
    """

    # extract param
    t_core = design["t_core"]
    z_core = design["z_core"]
    x_window = design["x_window"]
    x_winding = design["x_winding"]
    d_insulation = design["d_insulation"]

    # plot the windings
    if geom == "shell_simple":
        _plot_rounded(ax, 0.0, 0.0, t_core, z_core, 2 * x_winding + 3 * d_insulation, "green")
        _plot_rounded(ax, 0.0, 0.0, t_core, z_core, 2 * x_winding + 2 * d_insulation, "orange")
        _plot_rounded(ax, 0.0, 0.0, t_core, z_core, 1 * x_winding + 2 * d_insulation, "green")
        _plot_rounded(ax, 0.0, 0.0, t_core, z_core, 1 * x_winding + 1 * d_insulation, "chocolate")
        _plot_rounded(ax, 0.0, 0.0, t_core, z_core, 0 * x_winding + 1 * d_insulation, "green")
    elif geom == "shell_inter":
        _plot_rounded(ax, 0.0, 0.0, t_core, z_core, 4 * x_winding + 4 * d_insulation, "green")
        _plot_rounded(ax, 0.0, 0.0, t_core, z_core, 4 * x_winding + 3 * d_insulation, "chocolate")
        _plot_rounded(ax, 0.0, 0.0, t_core, z_core, 3 * x_winding + 3 * d_insulation, "green")
        _plot_rounded(ax, 0.0, 0.0, t_core, z_core, 3 * x_winding + 2 * d_insulation, "orange")
        _plot_rounded(ax, 0.0, 0.0, t_core, z_core, 1 * x_winding + 2 * d_insulation, "green")
        _plot_rounded(ax, 0.0, 0.0, t_core, z_core, 1 * x_winding + 1 * d_insulation, "chocolate")
        _plot_rounded(ax, 0.0, 0.0, t_core, z_core, 0 * x_winding + 1 * d_insulation, "green")
    elif geom == "core_type":
        mid = t_core / 2 + x_window / 2
        for pos in [-mid, +mid]:
            _plot_rounded(ax, pos, 0.0, t_core, z_core, 2 * x_winding + 3 * d_insulation, "green")
            _plot_rounded(ax, pos, 0.0, t_core, z_core, 2 * x_winding + 2 * d_insulation, "orange")
            _plot_rounded(ax, pos, 0.0, t_core, z_core, 1 * x_winding + 2 * d_insulation, "green")
            _plot_rounded(ax, pos, 0.0, t_core, z_core, 1 * x_winding + 1 * d_insulation, "chocolate")
            _plot_rounded(ax, pos, 0.0, t_core, z_core, 0 * x_winding + 1 * d_insulation, "green")
    elif geom == "three_phase":
        mid = t_core + x_window
        for pos in [-mid, 0.0, +mid]:
            _plot_rounded(ax, pos, 0.0, t_core, z_core, 2 * x_winding + 3 * d_insulation, "green")
            _plot_rounded(ax, pos, 0.0, t_core, z_core, 2 * x_winding + 2 * d_insulation, "orange")
            _plot_rounded(ax, pos, 0.0, t_core, z_core, 1 * x_winding + 2 * d_insulation, "green")
            _plot_rounded(ax, pos, 0.0, t_core, z_core, 1 * x_winding + 1 * d_insulation, "chocolate")
            _plot_rounded(ax, pos, 0.0, t_core, z_core, 0 * x_winding + 1 * d_insulation, "green")
    else:
        raise ValueError("invalid geom")


def _get_geom_front(name, geom, design):
    """
    Plot a transformer geometry (front view).
    """

    # extract param
    x_box = design["x_box"]
    y_box = design["y_box"]

    # create a figure
    (fig, ax) = plt.subplots(num=name)

    # plot the transformer (if valid)
    _plot_front_core(ax, geom, design)
    _plot_front_winding(ax, geom, design)

    # set the axis
    d_add = 0.1 * np.maximum(x_box, y_box)
    ax.set_xlim(-1e3 * (x_box / 2 + d_add), +1e3 * (x_box / 2 + d_add))
    ax.set_ylim(-1e3 * (y_box / 2 + d_add), +1e3 * (y_box / 2 + d_add))
    ax.set_aspect("equal")
    ax.set_xlabel("mm")
    ax.set_ylabel("mm")
    fig.tight_layout()


def _get_geom_top(name, geom, design):
    """
    Plot a transformer geometry (top view).
    """

    # extract param
    x_box = design["x_box"]
    z_box = design["z_box"]

    # create a figure
    (fig, ax) = plt.subplots(num=name)

    # plot the transformer (if valid)
    _plot_top_winding(ax, geom, design)
    _plot_top_core(ax, geom, design)

    # set the axis
    d_add = 0.1 * np.maximum(x_box, z_box)
    ax.set_xlim(-1e3 * (x_box / 2 + d_add), +1e3 * (x_box / 2 + d_add))
    ax.set_ylim(-1e3 * (z_box / 2 + d_add), +1e3 * (z_box / 2 + d_add))
    ax.set_aspect("equal")
    ax.set_xlabel("mm")
    ax.set_ylabel("mm")
    fig.tight_layout()


def get_geom(name, geom, design):
    """
    Plot a transformer geometry).
    """

    _get_geom_front(name + " / front", geom, design)
    _get_geom_top(name + " / top", geom, design)


def get_disp(name, design):
    """
    Display a transformer design (console).
    """

    print(f"========================================== {name}")
    print(f"boxed")
    print(f"    A_prd = {1e8 * design['A_prd']:.2f} cm4")
    print(f"    V_box = {1e6 * design['V_box']:.2f} cm3")
    print(f"    m_tot = {1e0 * design['m_tot']:.2f} kg")
    print(f"    A_box = {1e4 * design['A_box']:.2f} cm2")
    print(f"    x_box = {1e3 * design['x_box']:.2f} mm")
    print(f"    y_box = {1e3 * design['y_box']:.2f} mm")
    print(f"    z_box = {1e3 * design['z_box']:.2f} mm")
    print(f"core")
    print(f"    V_core = {1e6 * design['V_core']:.2f} cm3")
    print(f"    m_core = {1e0 * design['m_core']:.2f} kg")
    print(f"    A_core = {1e4 * design['A_core']:.2f} cm2")
    print(f"    t_core = {1e3 * design['t_core']:.2f} mm")
    print(f"    z_core = {1e3 * design['z_core']:.2f} mm")
    print(f"window")
    print(f"    V_window = {1e6 * design['V_window']:.2f} cm3")
    print(f"    V_conductor = {1e6 * design['V_conductor']:.2f} cm3")
    print(f"    V_insulation = {1e6 * design['V_insulation']:.2f} cm3")
    print(f"    m_winding = {1e0 * design['m_winding']:.2f} kg")
    print(f"    m_insulation = {1e0 * design['m_insulation']:.2f} kg")
    print(f"    A_window = {1e4 * design['A_window']:.2f} cm2")
    print(f"    A_conductor = {1e4 * design['A_conductor']:.2f} cm2")
    print(f"    A_insulation = {1e4 * design['A_insulation']:.2f} cm2")
    print(f"    A_winding = {1e4 * design['A_winding']:.2f} cm2")
    print(f"    x_window = {1e3 * design['x_window']:.2f} mm")
    print(f"    y_window = {1e3 * design['y_window']:.2f} mm")
    print(f"    x_winding = {1e3 * design['x_winding']:.2f} mm")
    print(f"    y_winding = {1e3 * design['y_winding']:.2f} mm")
    print(f"    d_winding = {1e3 * design['d_winding']:.2f} mm")
    print(f"    d_insulation = {1e3 * design['d_insulation']:.2f} mm")
    print(f"operating")
    print(f"    P_trf = {1e-3 * design['P_trf']:.2f} kW")
    print(f"    S_trf = {1e-3 * design['S_trf']:.2f} kVA")
    print(f"    cosphi = {1e2 * design['cosphi']:.2f} %")
    print(f"    f_sw = {1e-3 * design['f_sw']:.2f} kHz")
    print(f"    f_opt = {1e-3 * design['f_opt']:.2f} kHz")
    print(f"    n_trf = {1e0 * design['n_trf']:.2f} #")
    print(f"    I_rms = {1e0 * design['I_rms']:.2f} A")
    print(f"    V_rms = {1e0 * design['V_rms']:.2f} V")
    print(f"    igse_flux = {1e0 * design['igse_flux']:.2f} #")
    print(f"    igse_loss = {1e0 * design['igse_loss']:.2f} #")
    print(f"    harm_freq = {1e0 * design['harm_freq']:.2f} #")
    print(f"    n_turn = {1e0 * design['n_turn']:.2f} #")
    print(f"    n_opt = {1e0 * design['n_opt']:.2f} #")
    print(f"primary")
    print(f"    I_rms_1 = {1e0 * design['I_rms_1']:.2f} A")
    print(f"    V_rms_1 = {1e0 * design['V_rms_1']:.2f} V")
    print(f"    n_turn_1 = {1e0 * design['n_turn_1']:.2f} #")
    print(f"secondary")
    print(f"    I_rms_2 = {1e0 * design['I_rms_2']:.2f} A")
    print(f"    V_rms_2 = {1e0 * design['V_rms_2']:.2f} V")
    print(f"    n_turn_2 = {1e0 * design['n_turn_2']:.2f} #")
    print(f"losses")
    print(f"    P_core = {1e0 * design['P_core']:.2f} W")
    print(f"    P_winding = {1e0 * design['P_winding']:.2f} W")
    print(f"    P_loss = {1e0 * design['P_loss']:.2f} W")
    print(f"stress")
    print(f"    r_hf = {1e0 * design['r_hf']:.2f} #")
    print(f"    r_cw = {1e0 * design['r_cw']:.2f} #")
    print(f"    p_core = {1e-3 * design['p_core']:.2f} mW/cm3")
    print(f"    p_winding = {1e-3 * design['p_winding']:.2f} mW/cm3")
    print(f"    B_pk = {1e3 * design['B_pk']:.2f} mT")
    print(f"    J_rms = {1e-6 * design['J_rms']:.2f} A/mm2")
    print(f"    T_diff = {1e0 * design['T_diff']:.2f} C")
    print(f"    A_thermal = {1e4 * design['A_thermal']:.2f} cm2")
    print(f"final")
    print(f"    ht = {1e-1 * design['ht']:.2f} mW/cm2")
    print(f"    rho = {1e-6 * design['rho']:.2f} kW/dm3")
    print(f"    gamma = {1e-3 * design['gamma']:.2f} kW/kg")
    print(f"    loss = {1e2 * design['loss']:.3f} %")
    print(f"penalty")
    print(f"    penalty_box = {1e0 * design['penalty_box']:.2f} #")
    print(f"    penalty_geom = {1e0 * design['penalty_geom']:.2f} #")
    print(f"    penalty_freq = {1e0 * design['penalty_freq']:.2f} #")
    print(f"    penalty_turn = {1e0 * design['penalty_turn']:.2f} #")
    print(f"    penalty_core = {1e0 * design['penalty_core']:.2f} #")
    print(f"    penalty_winding = {1e0 * design['penalty_winding']:.2f} #")
    print(f"    penalty_thermal = {1e0 * design['penalty_thermal']:.2f} #")
    print(f"    penalty = {1e0 * design['penalty']:.2f} #")
    print(f"========================================== {name}")


def get_summary(name, design):
    """
    Display the main parameters (console).
    """

    penalty = f"penalty = {1e0 * design['penalty']:.2f} #"
    rho = f"rho = {1e-6 * design['rho']:.2f} kW/dm3"
    gamma = f"gamma = {1e-3 * design['gamma']:.2f} kW/kg"
    loss = f"loss = {1e2 * design['loss']:.3f} %"
    f_sw = f"f_sw = {1e-3 * design['f_sw']:.2f} kHz"
    T_diff = f"T_diff = {1e0 * design['T_diff']:.2f} C"
    print(f"{name} / {penalty} / {rho} / {gamma} / {loss} / {f_sw} / {T_diff}")
