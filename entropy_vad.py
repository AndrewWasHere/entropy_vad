"""
Copyright 2018, Andrew Lin
All rights reserved.

This software is licensed under the BSD 3-Clause License.
See LICENSE.txt at the root of the project or
https://opensource.org/licenses/BSD-3-Clause
"""
import itertools
import typing

import numpy as np


def energy(signal: np.ndarray) -> typing.Union[np.ndarray, np.dtype]:
    """Compute energy of signal.

    Signal energy is the sum of the square of the values in the signal.

    Args:
        signal: input signal.

    Returns:
        signal energy.
    """
    e = np.sum(np.square(signal))
    return e


def permutation_entropy(
    signal: np.ndarray,
    order: int
):
    """Calculate the permutation entropy of signal.

    As described in Bandt and Pompe.

    Args:
        signal: input signal.
        order: number of samples in permutation. Must be >= 2.

    Returns:
        Permutation entropy of signal, a numpy scalar.
    """
    # Permutations of indices.
    permutation_type = [
        list(p)
        for p in itertools.permutations(range(order))
    ]

    relative_freqs = np.zeros(len(permutation_type))
    permutation_samples = signal.size - order + 1

    for idx in range(permutation_samples):
        pi = np.argsort(signal[idx:idx + order]).tolist()
        relative_freqs[permutation_type.index(pi)] += 1

    relative_freqs /= permutation_samples

    # Remove missing permutations.
    relative_freqs = relative_freqs[relative_freqs != 0]

    # Permutation entropy.
    pe = -np.sum(relative_freqs * np.log2(relative_freqs))

    return pe


def xu_wang_bao_transform(signal: np.ndarray) -> np.ndarray:
    """Pre-PE calculation signal transform.

    Signal transform used in Xu, Wang, and Bao before doing PE calculations.

    Args:
        signal: input signal.

    Returns:
        x_new
    """
    x_new = signal[1:] - signal[:-1]
    x_new = (x_new >= 0.0).astype(signal.dtype)

    return x_new


def energy_pe_ratio(
    signal: np.ndarray,
    order: int
) -> typing.Union[np.ndarray, float]:
    """Calculate energy to PE ratio.

    As described in Xu, Wang, and Bao.

    Note: `order` is called `embedding factor` in the paper. We use `order`
    because that is what is used in Bandt and Pompe. Plus it's shorter.

    Args:
        signal: input signal.
        order: number of samples in permutation. Must be >= 2.

    Returns:
        energy:pe ratio.
    """
    # Short time energy.
    ste = energy(signal)

    # Normalized permutation energy.
    npe = permutation_entropy(signal, order) / (order - 1)

    # Energy PE ratio.
    eta = ste / npe if npe != 0.0 else 0.0

    return eta


def entropy_vad_detailed(
    signal: np.ndarray,
    sample_rate: int,
    window_length: float,
    threshold: float,
    order: int
) -> typing.Tuple[
    np.ndarray,
    np.ndarray,
    typing.Union[np.ndarray, float],
    typing.Union[np.ndarray, float]
]:
    """Permutation Entropy-based Voice Activity Detector.

    Args:
        signal: audio signal to VAD.
        sample_rate: sample rate of signal in samples/second.
        window_length: window size in seconds.
        threshold: multiplier of noise energy PE ratio for energy PE threshold.
        order: number of samples in permutation. Must be >= 2.

    Returns:
        Voice activity indicator,
        energy PE ratio,
        energy PE ratio noise (numpy scalar),
        energy PE ratio threshold (numpy scalar)
    """
    # Initialize values.
    n_window_samples = int(window_length * sample_rate)
    signal_eta = np.zeros_like(signal)
    voice_activity = np.zeros_like(signal, dtype=np.bool_)

    # Calculate baseline.
    # First window is assumed to have no voice energy.
    eta_noise = energy_pe_ratio(signal[:n_window_samples], order)
    eta_threshold = threshold * eta_noise

    # VAD signal.
    for start in range(0, signal.size, n_window_samples):
        end = start + n_window_samples
        eta = energy_pe_ratio(signal[start:end], order)

        signal_eta[start:end] = eta
        voice_activity[start:end] = eta > eta_threshold

    return (
        voice_activity,
        signal_eta,
        eta_noise,
        eta_threshold
    )
