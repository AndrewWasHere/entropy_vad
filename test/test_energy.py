"""
Copyright 2018, Andrew Lin
All rights reserved.

This software is licensed under the BSD 3-Clause License.
See LICENSE.txt at the root of the project or
https://opensource.org/licenses/BSD-3-Clause
"""
import numpy as np

from entropy_vad import energy


def test_zero():
    """Zero signal energy."""
    signal = np.zeros(100)
    e = energy(signal)

    assert e == 0.0


def test_nonzero():
    """Nonzero signal energy."""
    signal = np.array([1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0])
    e = energy(signal)

    assert e == 4.0


def test_empty():
    """Test zero-length signal."""
    e = energy(np.array([]))

    assert e == 0.0
