"""
Compute the waveforms and the correction factors for a converter type:
    - Single-phase and three-phase excitations (star connection)
    - Sinusoidal, dual-active bridge, or series-resonant converter
    - The magnetizing current of the transformer is neglected.
    - For the converters, an external leakage inductor is used.
"""

__author__ = "Thomas Guillod"
__copyright__ = "Thomas Guillod - Dartmouth College"
__license__ = "Mozilla Public License Version 2.0"

import numpy as np
import scipy.integrate as integrate
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt


def _get_offset(d_sig, s_sig):
    """
    Remove the DC offset.
    """

    s_avg = np.trapezoid(s_sig, d_sig)
    s_sig = s_sig - s_avg

    return s_sig


def _get_rms(d_sig, s_sig):
    """
    Compute the RMS value.
    """

    s_rms = np.sqrt(integrate.trapezoid(s_sig**2, d_sig))

    return s_rms


def _get_cumint(d_sig, s_sig):
    """
    Compute the cumulative integral.
    """

    # get the integral
    s_int = integrate.cumulative_trapezoid(s_sig, d_sig)
    s_int = np.append(0, s_int)

    # remove the offset
    s_avg = np.trapezoid(s_int, d_sig)
    s_int = s_int - s_avg

    return s_int


def _get_flux_factor(d_sig, V_sig):
    """
    Correction factor for the peak flux density (considering the waveshape).
    """

    # remove the offset
    V_sig = _get_offset(d_sig, V_sig)

    # get the RMS value of the voltage
    V_rms = _get_rms(d_sig, V_sig)

    # get the equivalent sinusoidal signal
    V_sin = np.sqrt(2) * V_rms * np.sin(2 * np.pi * d_sig)

    # get the flux density
    B_sig = _get_cumint(d_sig, V_sig)
    B_sin = _get_cumint(d_sig, V_sin)

    # compute the peak to peak value
    k_sig = np.max(B_sig) - np.min(B_sig)
    k_sin = np.max(B_sin) - np.min(B_sin)

    # extract the flux correction factors
    igse_flux = k_sig / k_sin

    return B_sig, B_sin, V_rms, igse_flux


def _get_loss_factor(d_sig, B_sig, B_sin, igse_flux, alpha_stm, beta_stm):
    """
    Correction factor for the core losses (considering the waveshape with iGSE).
    """

    # compute the peak to peak values
    B_pkpk_sig = np.max(B_sig) - np.min(B_sig)
    B_pkpk_sin = np.max(B_sin) - np.min(B_sin)

    # compute the gradient
    dB_sig = np.gradient(B_sig, d_sig)
    dB_sin = np.gradient(B_sin, d_sig)

    # compute the distortion
    k_sig = (B_pkpk_sig ** (beta_stm - alpha_stm)) * integrate.trapezoid(np.abs(dB_sig) ** alpha_stm, d_sig)
    k_sin = (B_pkpk_sin ** (beta_stm - alpha_stm)) * integrate.trapezoid(np.abs(dB_sin) ** alpha_stm, d_sig)

    # extract the loss correction factors
    igse_loss = (1 / (igse_flux ** beta_stm)) * (k_sig / k_sin)

    return igse_loss


def _get_freq_factor(d_sig, I_sig):
    """
    Frequency correction factor for the winding losses (considering the harmonics).
    """

    # remove the offset
    I_sig = _get_offset(d_sig, I_sig)

    # get the RMS value of the voltage
    I_rms = _get_rms(d_sig, I_sig)

    # get the equivalent sinusoidal signal
    I_sin = np.sqrt(2) * I_rms * np.sin(2 * np.pi * d_sig)

    # compute the distortion
    k_sig = integrate.trapezoid(np.gradient(I_sig, d_sig) ** 2, d_sig)
    k_sin = integrate.trapezoid(np.gradient(I_sin, d_sig) ** 2, d_sig)

    # get the equivalent frequency factor
    harm_freq = np.sqrt(k_sig / k_sin)

    return I_rms, harm_freq


def _get_1p_voltage(d_sig):
    """
    Get a single-phase PWM voltage.
    """

    y = np.array([+1.0, -1.0, +1.0])
    x = np.array([0 / 4, 1 / 4, 3 / 4, 4 / 4])
    V_sig = interpolate.PPoly([y], x)(d_sig)

    return V_sig


def _get_3p_voltage_wye(d_sig):
    """
    Get a three-phase PWM voltage (wye connection).
    """

    y = np.array([+1.0, +0.5, -0.5, -1.0, -0.5, +0.5, +1.0])
    x = np.array([0 / 12, 1 / 12, 3 / 12, 5 / 12, 7 / 12, 9 / 12, 11 / 12, 12 / 12])
    V_sig = interpolate.PPoly([y], x)(d_sig)

    return V_sig


def _get_3p_voltage_delta(d_sig):
    """
    Get a three-phase PWM voltage (delta connection).
    """

    y = np.array([+1.0, 0.0, -1.0, 0.0, +1.0])
    x = np.array([0 / 12, 2 / 12, 4 / 12, 8 / 12, 10 / 12, 12 / 12])
    V_sig = interpolate.PPoly([y], x)(d_sig)

    return V_sig


def _get_dab_current(d_sig, V_sig, phi, amp):
    """
    Get the DAB current from the voltage.
    """

    # get the shifted voltage
    V_shift = np.interp(d_sig - phi / (2 * np.pi), d_sig, V_sig, period=1)

    # scale the voltages
    V_shift *= amp

    # get the current with the voltage of the leakage
    I_sig = _get_cumint(d_sig, V_sig - V_shift)

    return I_sig


def _get_plot_waveform(conv, phase, n_coil, d_sig, V_sig, I_sig):
    """
    Plot the current and voltage waveforms.
    """

    # create the figure
    (fig, axes) = plt.subplots(2, num=f"{conv} {phase}", figsize=(6.4, 7.0))

    # plot the variables
    coil_vec = np.linspace(0, 1, n_coil, endpoint=False)
    for coil in coil_vec:
        V_tmp = np.interp(d_sig + coil, d_sig, V_sig, period=1)
        I_tmp = np.interp(d_sig + coil, d_sig, I_sig, period=1)
        axes[0].plot(1e2 * d_sig, 1e0 * V_tmp)
        axes[1].plot(1e2 * d_sig, 1e0 * I_tmp)

    # add cosmetics
    axes[0].set_xlabel("Duty Cycle [%]")
    axes[0].set_ylabel("Voltage [V]")
    axes[0].grid()

    # add cosmetics
    axes[1].set_xlabel("Duty Cycle [%]")
    axes[1].set_ylabel("Current [A]")
    axes[1].grid()

    # add cosmetics
    fig.tight_layout()


def get_converter_waveform(conv, phase, operating, alpha_stm, beta_stm):
    """
    Get the waveform for a single-phase converter.
    """

    # extract the data
    n = operating["n"]
    P_src = operating["P_src"]
    V_src = operating["V_src"]
    phi = operating["phi"]
    amp = operating["amp"]

    # get the time base
    d_sig = np.linspace(0.0, 1.0, n)

    # get the waveforms
    if phase == "1p":
        # set number of phases
        n_coil = 1

        # set the waveshapes
        if conv == "sin":
            V_sig = np.cos(2 * np.pi * d_sig)
            I_sig = np.cos(2 * np.pi * d_sig)
            V_scl = np.sqrt(2)
        elif conv == "dab":
            V_sig = _get_1p_voltage(d_sig)
            I_sig = _get_dab_current(d_sig, V_sig, phi, amp)
            V_scl = 1
        elif conv == "src":
            V_sig = _get_1p_voltage(d_sig)
            I_sig = np.cos(2 * np.pi * d_sig)
            V_scl = 1
        else:
            raise ValueError("invalid conv")
    elif phase == "3p_wye":
        # set number of phases
        n_coil = 3

        # set the waveshapes
        if conv == "sin":
            V_sig = np.cos(2 * np.pi * d_sig)
            I_sig = np.cos(2 * np.pi * d_sig)
            V_scl = np.sqrt(4 / 9)
        elif conv == "dab":
            V_sig = _get_3p_voltage_wye(d_sig)
            I_sig = _get_dab_current(d_sig, V_sig, phi, amp)
            V_scl = 2 / 3
        elif conv == "src":
            V_sig = _get_3p_voltage_wye(d_sig)
            I_sig = np.cos(2 * np.pi * d_sig)
            V_scl = 2 / 3
        else:
            raise ValueError("invalid conv")
    elif phase == "3p_delta":
        # set number of phases
        n_coil = 3

        # set the waveshapes
        if conv == "sin":
            V_sig = np.cos(2 * np.pi * d_sig)
            I_sig = np.cos(2 * np.pi * d_sig)
            V_scl = np.sqrt(4 / 3)
        elif conv == "dab":
            V_sig = _get_3p_voltage_delta(d_sig)
            I_sig = _get_dab_current(d_sig, V_sig, phi, amp)
            V_scl = 1
        elif conv == "src":
            V_sig = _get_3p_voltage_delta(d_sig)
            I_sig = np.cos(2 * np.pi * d_sig)
            V_scl = 1
        else:
            raise ValueError("invalid conv")
    else:
        raise ValueError("invalid phase")

    # scale the voltage
    V_sig *= V_src * V_scl

    # compute the power
    P_tmp = n_coil * integrate.trapezoid(V_sig * I_sig, d_sig)

    # scale the current
    I_sig *= P_src / P_tmp

    # get the harmonic factors
    (I_rms, harm_freq) = _get_freq_factor(d_sig, I_sig)
    (B_sig, B_sin, V_rms, igse_flux) = _get_flux_factor(d_sig, V_sig)
    igse_loss = _get_loss_factor(d_sig, B_sig, B_sin, igse_flux, alpha_stm, beta_stm)

    # get the apparent power
    S_trf = n_coil * V_rms * I_rms

    # get the active power
    P_trf = n_coil * integrate.trapezoid(V_sig * I_sig, d_sig)

    # make the plot
    _get_plot_waveform(conv, phase, n_coil, d_sig, V_sig, I_sig)

    # show the results
    print(f"converter waveform / {conv} {phase}")
    print(f"    P_trf = {P_trf:.4f}")
    print(f"    S_trf = {S_trf:.4f}")
    print(f"    V_rms = {V_rms:.4f}")
    print(f"    I_rms = {I_rms:.4f}")
    print(f"    igse_flux = {igse_flux:.4f}")
    print(f"    igse_loss = {igse_loss:.4f}")
    print(f"    harm_freq = {harm_freq:.4f}")

    # assign the output
    out = {
        "P_trf": float(P_trf),
        "S_trf": float(S_trf),
        "V_rms": float(V_rms),
        "I_rms": float(I_rms),
        "igse_flux": float(igse_flux),
        "igse_loss": float(igse_loss),
        "harm_freq": float(harm_freq),
    }

    return out
