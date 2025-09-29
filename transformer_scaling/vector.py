"""
Vectorize and slice transformer designs.
"""

__author__ = "Thomas Guillod"
__copyright__ = "Thomas Guillod - Dartmouth College"
__license__ = "Mozilla Public License Version 2.0"

import collections
import numpy as np


def get_assemble(design_list):
    """
    Merge a list of design vector into a single design vector.
    """

    # transformer to dict of list
    design = collections.defaultdict(list)
    for design_tmp in design_list:
        for key, value in design_tmp.items():
            design[key].append(value)

    # cast to dict
    design = dict(design)

    # cast to arrays
    for key, value in design.items():
        design[key] = np.array(value)

    return design


def get_slice(design, idx, n_sweep):
    """
    Slice a design vector.
    """

    # guard close
    if idx is None:
        return design
    if n_sweep is None:
        return design

    # slice the design vector
    for var, value in design.items():
        # check
        assert len(value) == n_sweep, "invalid length"

        # assign
        design[var] = value[idx]

    return design


def get_vectorize(design, n_sweep):
    """
    Vectorize a design into a design vector.
    """

    # guard close
    if n_sweep is None:
        return design

    # check and vectorize the design
    for var, value in design.items():
        # check
        assert np.isscalar(value) or (len(value) == n_sweep), "invalid length"

        # assign
        if np.isscalar(n_sweep):
            value = value * np.ones(n_sweep)
            design[var] = value

    return design
