"""
Copyright 2018, Andrew Lin
All rights reserved.

This software is licensed under the BSD 3-Clause License.
See LICENSE.txt at the root of the project or
https://opensource.org/licenses/BSD-3-Clause
"""
import numpy as np

from entropy_vad import xu_wang_bao_transform


def test_equal():
    """Test all values the same."""
    signal = np.zeros(10)
    new_signal = xu_wang_bao_transform(signal)

    assert (new_signal == np.ones(signal.size - 1, dtype=signal.dtype)).all()


def test_ascending():
    """Test rising time series."""
    signal = np.arange(0, 10, 0.5)
    new_signal = xu_wang_bao_transform(signal)

    assert (new_signal == np.ones(signal.size - 1, dtype=signal.dtype)).all()


def test_descending():
    """Test falling time series."""
    signal = np.arange(10, 0, -0.5)
    new_signal = xu_wang_bao_transform(signal)

    assert (new_signal == np.zeros(signal.size - 1, dtype=signal.dtype)).all()
