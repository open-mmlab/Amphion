# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import scipy.signal as sig
import copy
import librosa


def bandpower(ps, mode="time"):
    """
    estimate bandpower, see https://de.mathworks.com/help/signal/ref/bandpower.html
    """
    if mode == "time":
        x = ps
        l2norm = np.linalg.norm(x) ** 2.0 / len(x)
        return l2norm
    elif mode == "psd":
        return sum(ps)


def getIndizesAroundPeak(arr, peakIndex, searchWidth=1000):
    peakBins = []
    magMax = arr[peakIndex]
    curVal = magMax
    for i in range(searchWidth):
        newBin = peakIndex + i
        if newBin >= len(arr):
            break
        newVal = arr[newBin]
        if newVal > curVal:
            break
        else:
            peakBins.append(int(newBin))
            curVal = newVal
    curVal = magMax
    for i in range(searchWidth):
        newBin = peakIndex - i
        if newBin < 0:
            break
        newVal = arr[newBin]
        if newVal > curVal:
            break
        else:
            peakBins.append(int(newBin))
            curVal = newVal
    return np.array(list(set(peakBins)))


def freqToBin(fAxis, Freq):
    return np.argmin(abs(fAxis - Freq))


def getPeakInArea(psd, faxis, estimation, searchWidthHz=10):
    """
    returns bin and frequency of the maximum in an area
    """
    binLow = freqToBin(faxis, estimation - searchWidthHz)
    binHi = freqToBin(faxis, estimation + searchWidthHz)
    peakbin = binLow + np.argmax(psd[binLow : binHi + 1])
    return peakbin, faxis[peakbin]


def getHarmonics(fund, sr, nHarmonics=6, aliased=False):
    harmonicMultipliers = np.arange(2, nHarmonics + 2)
    harmonicFs = fund * harmonicMultipliers
    if not aliased:
        harmonicFs[harmonicFs > sr / 2] = -1
        harmonicFs = np.delete(harmonicFs, harmonicFs == -1)
    else:
        nyqZone = np.floor(harmonicFs / (sr / 2))
        oddEvenNyq = nyqZone % 2
        harmonicFs = np.mod(harmonicFs, sr / 2)
        harmonicFs[oddEvenNyq == 1] = (sr / 2) - harmonicFs[oddEvenNyq == 1]
    return harmonicFs


def extract_snr(audio, sr=None):
    """Extract Signal-to-Noise Ratio for a given audio."""
    if sr != None:
        audio, _ = librosa.load(audio, sr=sr)
    else:
        audio, sr = librosa.load(audio, sr=sr)
    faxis, ps = sig.periodogram(
        audio, fs=sr, window=("kaiser", 38)
    )  # get periodogram, parametrized like in matlab
    fundBin = np.argmax(
        ps
    )  # estimate fundamental at maximum amplitude, get the bin number
    fundIndizes = getIndizesAroundPeak(
        ps, fundBin
    )  # get bin numbers around fundamental peak
    fundFrequency = faxis[fundBin]  # frequency of fundamental

    nHarmonics = 18
    harmonicFs = getHarmonics(
        fundFrequency, sr, nHarmonics=nHarmonics, aliased=True
    )  # get harmonic frequencies

    harmonicBorders = np.zeros([2, nHarmonics], dtype=np.int16).T
    fullHarmonicBins = np.array([], dtype=np.int16)
    fullHarmonicBinList = []
    harmPeakFreqs = []
    harmPeaks = []
    for i, harmonic in enumerate(harmonicFs):
        searcharea = 0.1 * fundFrequency
        estimation = harmonic

        binNum, freq = getPeakInArea(ps, faxis, estimation, searcharea)
        harmPeakFreqs.append(freq)
        harmPeaks.append(ps[binNum])
        allBins = getIndizesAroundPeak(ps, binNum, searchWidth=1000)
        fullHarmonicBins = np.append(fullHarmonicBins, allBins)
        fullHarmonicBinList.append(allBins)
        harmonicBorders[i, :] = [allBins[0], allBins[-1]]

    fundIndizes.sort()
    pFund = bandpower(ps[fundIndizes[0] : fundIndizes[-1]])  # get power of fundamental

    noisePrepared = copy.copy(ps)
    noisePrepared[fundIndizes] = 0
    noisePrepared[fullHarmonicBins] = 0
    noiseMean = np.median(noisePrepared[noisePrepared != 0])
    noisePrepared[fundIndizes] = noiseMean
    noisePrepared[fullHarmonicBins] = noiseMean

    noisePower = bandpower(noisePrepared)

    r = 10 * np.log10(pFund / noisePower)

    return r, 10 * np.log10(noisePower)
