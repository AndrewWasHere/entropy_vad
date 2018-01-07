"""
Copyright 2018, Andrew Lin
All rights reserved.

This software is licensed under the BSD 3-Clause License.
See LICENSE.txt at the root of the project or
https://opensource.org/licenses/BSD-3-Clause
"""
import numpy as np

from entropy_vad import permutation_entropy


def test_dc():
    """Test all signal values the same.

    This checks the edge case of permutations not happening, which would cause
    the log function to break things (log(0) = -inf).
    """
    signal = np.zeros(100)
    pe = permutation_entropy(signal, 2)

    assert pe == 0


def test_bp_example():
    """Test example series from Bandt and Pompe."""
    signal = np.array([4, 7, 9, 10, 6, 11, 3])

    pe = permutation_entropy(signal, 2)

    assert 0.91 < pe < 0.92  # Should be approx 0.918.

    pe = permutation_entropy(signal, 3)

    assert 1.52 < pe < 1.53  # Should be approx 1.522.
