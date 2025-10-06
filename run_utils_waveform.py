"""
Compute the waveforms and the correction factors the different converter types.
"""

__author__ = "Thomas Guillod"
__copyright__ = "Thomas Guillod - Dartmouth College"
__license__ = "Mozilla Public License Version 2.0"

import json
import numpy as np
import matplotlib.pyplot as plt
from transformer_utils import converter_waveform


if __name__ == "__main__":
    # define the Steinmetz parameters
    with open("param_steinmetz.json", "r") as fid:
        data = json.load(fid)
        alpha_stm = data["alpha_stm"]
        beta_stm = data["beta_stm"]

    # converter parameters
    operating = {
        "n": 25000,  # number of samples for the waveforms
        "P_src": 10e3,  # power rating of the converter
        "V_src": 600.0,  # voltage rating of the converter
        "phi": np.deg2rad(30.0),  # phase angle between the bridges
        "amp": 1.0,  # voltage ratio between the bridge
    }

    # transformer excitation
    conv_list = ["sin", "dab", "src"]
    phase_list = ["1p", "3p_wye", "3p_delta"]

    # get the converter parameters
    out = {}
    for conv in conv_list:
        out[conv] = {}
        for phase in phase_list:
            out[conv][phase] = converter_waveform.get_converter_waveform(conv, phase, operating, alpha_stm, beta_stm)

    # write the results
    with open("param_waveform.json", "w") as fid:
        json.dump(out, fid, indent=4)

    # plot the waveforms
    plt.show()
