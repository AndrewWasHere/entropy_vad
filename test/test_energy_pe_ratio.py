"""
Copyright 2017, Andrew Lin
All rights reserved.

This software is licensed under the BSD 3-Clause License.
See LICENSE.txt at the root of the project or
https://opensource.org/licenses/BSD-3-Clause
"""
import numpy as np

from entropy_vad import energy_pe_ratio


def test_zeros():
    signal = np.zeros(10)
    eta = energy_pe_ratio(signal, 2)

    assert eta == 0.0


def test_square():
    signal = np.array([1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1])
    eta = energy_pe_ratio(signal, 2)

    assert 17.54 < eta < 17.55
