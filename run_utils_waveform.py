"""
Compute the waveforms and the correction factors the different converter types.
"""

__author__ = "Thomas Guillod"
__copyright__ = "Thomas Guillod - Dartmouth College"
__license__ = "Mozilla Public License Version 2.0"

import numpy as np
import matplotlib.pyplot as plt
from transformer_utils import converter_waveform


if __name__ == "__main__":
    # power rating of the converter
    P_src = 10e3

    # voltage rating of the converter
    V_src = 600.0

    # voltage ratio between the bridge
    amp = 1.0

    # phase angle between the bridges
    phi = np.deg2rad(30.0)

    # define the Steinmetz parameters
    alpha_stm = 1.7215

    # number of samples for the waveforms
    n = 25000

    # get the converters (sinusoidal)
    converter_waveform.get_converter_waveform("sin", "1p", P_src, V_src, phi, amp, alpha_stm, n)
    converter_waveform.get_converter_waveform("sin", "3p_wye", P_src, V_src, phi, amp, alpha_stm, n)
    converter_waveform.get_converter_waveform("sin", "3p_delta", P_src, V_src, phi, amp, alpha_stm, n)

    # get the converters (DAB)
    converter_waveform.get_converter_waveform("dab", "1p", P_src, V_src, phi, amp, alpha_stm, n)
    converter_waveform.get_converter_waveform("dab", "3p_wye", P_src, V_src, phi, amp, alpha_stm, n)
    converter_waveform.get_converter_waveform("dab", "3p_delta", P_src, V_src, phi, amp, alpha_stm, n)

    # get the converters (SRC)
    converter_waveform.get_converter_waveform("src", "1p", P_src, V_src, phi, amp, alpha_stm, n)
    converter_waveform.get_converter_waveform("src", "3p_wye", P_src, V_src, phi, amp, alpha_stm, n)
    converter_waveform.get_converter_waveform("src", "3p_delta", P_src, V_src, phi, amp, alpha_stm, n)

    # plot the waveforms
    plt.show()
