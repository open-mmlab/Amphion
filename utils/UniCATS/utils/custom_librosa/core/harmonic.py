#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Harmonic calculations for frequency representations"""

import numpy as np
import scipy.interpolate
import scipy.signal
from ..util.exceptions import ParameterError

__all__ = ["salience", "interp_harmonics"]


def salience(
    S,
    freqs,
    h_range,
    weights=None,
    aggregate=None,
    filter_peaks=True,
    fill_value=np.nan,
    kind="linear",
    axis=0,
):
    """Harmonic salience function.

    Parameters
    ----------
    S : np.ndarray [shape=(d, n)]
        input time frequency magnitude representation (e.g. STFT or CQT magnitudes).
        Must be real-valued and non-negative.

    freqs : np.ndarray, shape=(S.shape[axis])
        The frequency values corresponding to S's elements along the
        chosen axis.

    h_range : list-like, non-negative
        Harmonics to include in salience computation.  The first harmonic (1)
        corresponds to ``S`` itself. Values less than one (e.g., 1/2) correspond
        to sub-harmonics.

    weights : list-like
        The weight to apply to each harmonic in the summation. (default:
        uniform weights). Must be the same length as ``harmonics``.

    aggregate : function
        aggregation function (default: `np.average`)

        If ``aggregate=np.average``, then a weighted average is
        computed per-harmonic according to the specified weights.
        For all other aggregation functions, all harmonics
        are treated equally.

    filter_peaks : bool
        If true, returns harmonic summation only on frequencies of peak
        magnitude. Otherwise returns harmonic summation over the full spectrum.
        Defaults to True.

    fill_value : float
        The value to fill non-peaks in the output representation. (default:
        `np.nan`) Only used if ``filter_peaks == True``.

    kind : str
        Interpolation type for harmonic estimation.
        See `scipy.interpolate.interp1d`.

    axis : int
        The axis along which to compute harmonics

    Returns
    -------
    S_sal : np.ndarray, shape=(len(h_range), [x.shape])
        ``S_sal`` will have the same shape as ``S``, and measure
        the overal harmonic energy at each frequency.

    See Also
    --------
    interp_harmonics

    Examples
    --------
    >>> y, sr = librosa.load(librosa.ex('trumpet'), duration=3)
    >>> S = np.abs(librosa.stft(y))
    >>> freqs = librosa.fft_frequencies(sr)
    >>> harms = [1, 2, 3, 4]
    >>> weights = [1.0, 0.5, 0.33, 0.25]
    >>> S_sal = librosa.salience(S, freqs, harms, weights, fill_value=0)
    >>> print(S_sal.shape)
    (1025, 115)
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
    >>> librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
    ...                          sr=sr, y_axis='log', x_axis='time', ax=ax[0])
    >>> ax[0].set(title='Magnitude spectrogram')
    >>> ax[0].label_outer()
    >>> img = librosa.display.specshow(librosa.amplitude_to_db(S_sal,
    ...                                                        ref=np.max),
    ...                                sr=sr, y_axis='log', x_axis='time', ax=ax[1])
    >>> ax[1].set(title='Salience spectrogram')
    >>> fig.colorbar(img, ax=ax, format="%+2.0f dB")
    """
    if aggregate is None:
        aggregate = np.average

    if weights is None:
        weights = np.ones((len(h_range),))
    else:
        weights = np.array(weights, dtype=float)

    S_harm = interp_harmonics(S, freqs, h_range, kind=kind, axis=axis)

    if aggregate is np.average:
        S_sal = aggregate(S_harm, axis=0, weights=weights)
    else:
        S_sal = aggregate(S_harm, axis=0)

    if filter_peaks:
        S_peaks = scipy.signal.argrelmax(S, axis=0)
        S_out = np.empty(S.shape)
        S_out.fill(fill_value)
        S_out[S_peaks[0], S_peaks[1]] = S_sal[S_peaks[0], S_peaks[1]]

        S_sal = S_out

    return S_sal


def interp_harmonics(x, freqs, h_range, kind="linear", fill_value=0, axis=0):
    """Compute the energy at harmonics of time-frequency representation.

    Given a frequency-based energy representation such as a spectrogram
    or tempogram, this function computes the energy at the chosen harmonics
    of the frequency axis.  (See examples below.)
    The resulting harmonic array can then be used as input to a salience
    computation.

    Parameters
    ----------
    x : np.ndarray
        The input energy

    freqs : np.ndarray, shape=(X.shape[axis])
        The frequency values corresponding to X's elements along the
        chosen axis.

    h_range : list-like, non-negative
        Harmonics to compute.  The first harmonic (1) corresponds to ``x``
        itself.
        Values less than one (e.g., 1/2) correspond to sub-harmonics.

    kind : str
        Interpolation type.  See `scipy.interpolate.interp1d`.

    fill_value : float
        The value to fill when extrapolating beyond the observed
        frequency range.

    axis : int
        The axis along which to compute harmonics

    Returns
    -------
    x_harm : np.ndarray, shape=(len(h_range), [x.shape])
        ``x_harm[i]`` will have the same shape as ``x``, and measure
        the energy at the ``h_range[i]`` harmonic of each frequency.

    See Also
    --------
    scipy.interpolate.interp1d


    Examples
    --------
    Estimate the harmonics of a time-averaged tempogram

    >>> y, sr = librosa.load(librosa.ex('choice'))
    >>> # Compute the time-varying tempogram and average over time
    >>> tempi = np.mean(librosa.feature.tempogram(y=y, sr=sr), axis=1)
    >>> # We'll measure the first five harmonics
    >>> h_range = [1, 2, 3, 4, 5]
    >>> f_tempo = librosa.tempo_frequencies(len(tempi), sr=sr)
    >>> # Build the harmonic tensor
    >>> t_harmonics = librosa.interp_harmonics(tempi, f_tempo, h_range)
    >>> print(t_harmonics.shape)
    (5, 384)

    >>> # And plot the results
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> librosa.display.specshow(t_harmonics, x_axis='tempo', sr=sr, ax=ax)
    >>> ax.set(yticks=0.5 + np.arange(len(h_range)),
    ...        yticklabels=['{:.3g}'.format(_) for _ in h_range],
    ...        ylabel='Harmonic', xlabel='Tempo (BPM)')

    We can also compute frequency harmonics for spectrograms.
    To calculate sub-harmonic energy, use values < 1.

    >>> y, sr = librosa.load(librosa.ex('trumpet'), duration=3)
    >>> h_range = [1./3, 1./2, 1, 2, 3, 4]
    >>> S = np.abs(librosa.stft(y))
    >>> fft_freqs = librosa.fft_frequencies(sr=sr)
    >>> S_harm = librosa.interp_harmonics(S, fft_freqs, h_range, axis=0)
    >>> print(S_harm.shape)
    (6, 1025, 646)

    >>> fig, ax = plt.subplots(nrows=3, ncols=2, sharex=True, sharey=True)
    >>> for i, _sh in enumerate(S_harm):
    ...     img = librosa.display.specshow(librosa.amplitude_to_db(_sh,
    ...                                                      ref=S.max()),
    ...                              sr=sr, y_axis='log', x_axis='time',
    ...                              ax=ax.flat[i])
    ...     ax.flat[i].set(title='h={:.3g}'.format(h_range[i]))
    ...     ax.flat[i].label_outer()
    >>> fig.colorbar(img, ax=ax, format="%+2.f dB")
    """

    # X_out will be the same shape as X, plus a leading
    # axis that has length = len(h_range)
    out_shape = [len(h_range)]
    out_shape.extend(x.shape)

    x_out = np.zeros(out_shape, dtype=x.dtype)

    if freqs.ndim == 1 and len(freqs) == x.shape[axis]:
        harmonics_1d(
            x_out, x, freqs, h_range, kind=kind, fill_value=fill_value, axis=axis
        )

    elif freqs.ndim == 2 and freqs.shape == x.shape:
        harmonics_2d(
            x_out, x, freqs, h_range, kind=kind, fill_value=fill_value, axis=axis
        )
    else:
        raise ParameterError(
            "freqs.shape={} does not match "
            "input shape={}".format(freqs.shape, x.shape)
        )

    return x_out


def harmonics_1d(harmonic_out, x, freqs, h_range, kind="linear", fill_value=0, axis=0):
    """Populate a harmonic tensor from a time-frequency representation.

    Parameters
    ----------
    harmonic_out : np.ndarray, shape=(len(h_range), X.shape)
        The output array to store harmonics

    X : np.ndarray
        The input energy

    freqs : np.ndarray, shape=(x.shape[axis])
        The frequency values corresponding to x's elements along the
        chosen axis.

    h_range : list-like, non-negative
        Harmonics to compute.  The first harmonic (1) corresponds to ``x``
        itself.
        Values less than one (e.g., 1/2) correspond to sub-harmonics.

    kind : str
        Interpolation type.  See `scipy.interpolate.interp1d`.

    fill_value : float
        The value to fill when extrapolating beyond the observed
        frequency range.

    axis : int
        The axis along which to compute harmonics

    See Also
    --------
    harmonics
    scipy.interpolate.interp1d


    Examples
    --------
    Estimate the harmonics of a time-averaged tempogram

    >>> y, sr = librosa.load(librosa.ex('choice'))
    >>> # Compute the time-varying tempogram and average over time
    >>> tempi = np.mean(librosa.feature.tempogram(y=y, sr=sr), axis=1)
    >>> # We'll measure the first five harmonics
    >>> h_range = [1, 2, 3, 4, 5]
    >>> f_tempo = librosa.tempo_frequencies(len(tempi), sr=sr)
    >>> # Build the harmonic tensor
    >>> t_harmonics = librosa.interp_harmonics(tempi, f_tempo, h_range)
    >>> print(t_harmonics.shape)
    (5, 384)

    >>> # And plot the results
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> librosa.display.specshow(t_harmonics, x_axis='tempo', sr=sr, ax=ax)
    >>> ax.set(yticks=0.5 + np.arange(len(h_range)),
    ...        yticklabels=['{:.3g}'.format(_) for _ in h_range],
    ...        ylabel='Harmonic')
    ...        xlabel='Tempo (BPM)')

    We can also compute frequency harmonics for spectrograms.
    To calculate subharmonic energy, use values < 1.

    >>> h_range = [1./3, 1./2, 1, 2, 3, 4]
    >>> S = np.abs(librosa.stft(y))
    >>> fft_freqs = librosa.fft_frequencies(sr=sr)
    >>> S_harm = librosa.interp_harmonics(S, fft_freqs, h_range, axis=0)
    >>> print(S_harm.shape)
    (6, 1025, 646)

    >>> fig, ax = plt.subplots(nrows=3, ncols=2, sharex=True, sharey=True)
    >>> for i, _sh in enumerate(S_harm):
    ...     librosa.display.specshow(librosa.amplitude_to_db(_sh,
    ...                                                      ref=S.max()),
    ...                              sr=sr, y_axis='log', x_axis='time', ax=ax.flat[i])
    ...     ax.flat[i].set(title='h={:.3g}'.format(h_range[i]))
    ...     ax.flat[i].label_outer()
    """

    # Note: this only works for fixed-grid, 1d interpolation
    f_interp = scipy.interpolate.interp1d(
        freqs,
        x,
        kind=kind,
        axis=axis,
        copy=False,
        bounds_error=False,
        fill_value=fill_value,
    )

    idx_out = [slice(None)] * harmonic_out.ndim

    # Compute the output index of the interpolated values
    interp_axis = 1 + (axis % x.ndim)

    # Iterate over the harmonics range
    for h_index, harmonic in enumerate(h_range):
        idx_out[0] = h_index

        # Iterate over frequencies
        for f_index, frequency in enumerate(freqs):
            # Offset the output axis by 1 to account for the harmonic index
            idx_out[interp_axis] = f_index

            # Estimate the harmonic energy at this frequency across time
            harmonic_out[tuple(idx_out)] = f_interp(harmonic * frequency)


def harmonics_2d(harmonic_out, x, freqs, h_range, kind="linear", fill_value=0, axis=0):
    """Populate a harmonic tensor from a time-frequency representation with
    time-varying frequencies.

    Parameters
    ----------
    harmonic_out : np.ndarray
        The output array to store harmonics

    x : np.ndarray
        The input energy

    freqs : np.ndarray, shape=x.shape
        The frequency values corresponding to each element of ``x``

    h_range : list-like, non-negative
        Harmonics to compute.  The first harmonic (1) corresponds to ``x``
        itself.  Values less than one (e.g., 1/2) correspond to
        sub-harmonics.

    kind : str
        Interpolation type.  See `scipy.interpolate.interp1d`.

    fill_value : float
        The value to fill when extrapolating beyond the observed
        frequency range.

    axis : int
        The axis along which to compute harmonics

    See Also
    --------
    harmonics
    harmonics_1d
    """
    idx_in = [slice(None)] * x.ndim
    idx_freq = [slice(None)] * x.ndim
    idx_out = [slice(None)] * harmonic_out.ndim

    # This is the non-interpolation axis
    ni_axis = (1 + axis) % x.ndim

    # For each value in the non-interpolated axis, compute its harmonics
    for i in range(x.shape[ni_axis]):
        idx_in[ni_axis] = slice(i, i + 1)
        idx_freq[ni_axis] = i
        idx_out[1 + ni_axis] = idx_in[ni_axis]

        harmonics_1d(
            harmonic_out[tuple(idx_out)],
            x[tuple(idx_in)],
            freqs[tuple(idx_freq)],
            h_range,
            kind=kind,
            fill_value=fill_value,
            axis=axis,
        )
