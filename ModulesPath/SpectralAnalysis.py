#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 09:39:25 2019

@author: bapung
"""

import numpy as np
import scipy.signal as sg
import scipy.fftpack as ft
import scipy.ndimage.filters as smth
import os


def lfpSpectrogram(basePath, sRate, nChans, reqChan, loadfrom=0):

    duration = 3600 * 10
    offsetP = 0
    subname = os.path.basename(os.path.normpath(basePath))

    if loadfrom == 1:
        fileName = basePath + subname + "_BestThetaChan.npy"
        BestThetaInfo = np.load(fileName)
        eegChan = BestThetaInfo

    else:
        fileName = basePath + subname + ".eeg"

        b1 = np.memmap(
            fileName,
            dtype="int16",
            mode="r",
            offset=int(offsetP) * nChans * 2 + 1 * (reqChan - 1) * 2,
        )
        eegChan = b1[0::nChans]

    sos = sg.butter(3, 100, btype="low", fs=sRate, output="sos")
    yf = sg.sosfilt(sos, eegChan)
    f, t, x = sg.spectrogram(yf, fs=sRate, nperseg=5 * 1250, noverlap=3 * 1250)
    sample_data = yf[0 : sRate * 5]
    # yf = ft.fft(yf) / len(eegChan)
    # xf = np.linspace(0.0, SampFreq / 2, len(eegChan) // 2)
    # y1 = 2.0 / (len(xf)) * np.abs(yf[:len(eegChan) // 2])
    # y1 = smth.gaussian_filter(y1, 8)

    return x, f, t, sample_data


def lfpSpectMaze(sub_name, nREMPeriod, RecInfo, channel):

    SampFreq = RecInfo["samplingFrequency"]
    frames = RecInfo["behavFrames"]
    behav = RecInfo["behav"]
    nChans = RecInfo["numChannels"]
    #    ReqChan = RecInfo['SpectralChannel']
    ReqChan = channel
    duration = 5  # chunk of lfp in seconds

    offsetP = ((nREMPeriod - behav[1, 0]) // 1e6) * SampFreq + int(
        np.diff(frames[0, :])
    )
    b1 = np.memmap(
        "/data/EEGData/" + sub_name + ".eeg",
        dtype="int16",
        mode="r",
        offset=int(offsetP) * nChans * 2 + 1 * (ReqChan - 1) * 2,
        shape=(1, nChans * SampFreq * duration),
    )
    eegChan = b1[0, ::nChans]
    sos = sg.butter(3, 100, btype="low", fs=SampFreq, output="sos")
    yf = sg.sosfilt(sos, eegChan)
    yf = ft.fft(yf) / len(eegChan)
    xf = np.linspace(0.0, SampFreq / 2, len(eegChan) // 2)
    y1 = 2.0 / (len(xf)) * np.abs(yf[: len(eegChan) // 2])
    y1 = smth.gaussian_filter(y1, 8)

    return y1, xf


def bestThetaChannel(basePath, sampleRate, nChans, badChannels, saveThetaChan=0):
    """
    fileName: name of the .eeg file
    sampleRate: sampling frequency of eeg;

    """

    badChannels = [x - 1 for x in badChannels]  # zero indexing correction
    duration = 3600  # chunk of lfp in seconds
    nyq = 0.5 * sampleRate
    lowTheta = 5
    highTheta = 10
    subname = os.path.basename(os.path.normpath(basePath))
    fileName = basePath + subname + ".eeg"

    lfpCA1 = np.memmap(
        fileName, dtype="int16", mode="r", shape=(sampleRate * duration, nChans)
    )

    #    goodChannels = [i for i in range(len(a)) if a[i]==1]
    #    lfpCA1[:,goodChannels-1]

    sos = sg.butter(
        3,
        [lowTheta / nyq, highTheta / nyq],
        btype="bandpass",
        output="sos",
        fs=sampleRate,
    )
    yf = sg.sosfiltfilt(sos, lfpCA1, axis=0)

    avgTheta = np.mean(np.square(yf), axis=0)
    idx = np.argsort(avgTheta)

    bestChannels = np.setdiff1d(idx, badChannels, assume_unique=True)[::-1]

    # selecting first three channels

    bestChannels = bestChannels[0:5]

    if saveThetaChan == 1:
        reqChan = bestChannels[0]
        b1 = np.memmap(fileName, dtype="int16", mode="r")
        ThetaExtract = b1[reqChan::nChans]
        ThetaExtract2 = b1[reqChan - 16 :: nChans]

        np.save(basePath + subname + "_BestThetaChan.npy", ThetaExtract)
        np.save(basePath + subname + "_BestThetaChan.npy", ThetaExtract2)

    return bestChannels


def bestRippleChannel(basePath, sampleRate, nChans, badChannels, saveRippleChan=1):
    """
    fileName: name of the .eeg file
    sampleRate: sampling frequency of eeg;

    """

    badChannels = [x - 1 for x in badChannels]  # zero indexing correction
    duration = 60 * 30  # chunk of lfp in seconds
    nyq = 0.5 * sampleRate  # Nyquist frequency for sampling rate
    lowRipple = 150  # ripple lower end frequency in Hz
    highRipple = 250  # ripple higher end frequency in Hz
    subname = os.path.basename(os.path.normpath(basePath))
    fileName = basePath + subname + ".eeg"

    lfpCA1 = np.memmap(
        fileName, dtype="int16", mode="r", shape=(sampleRate * duration, nChans)
    )

    #    goodChannels = [i for i in range(len(a)) if a[i]==1]
    #    lfpCA1[:,goodChannels-1]

    b, a = sg.butter(3, [lowRipple / nyq, highRipple / nyq], btype="bandpass")
    delta = sg.filtfilt(b, a, lfpCA1, axis=0)

    # Hilbert transform for calculating signal's envelope
    analytic_signal = sg.hilbert(delta)
    amplitude_envelope = np.abs(analytic_signal)

    # rms_signal = np.sqrt(np.mean(yf**2))
    avgRipple = np.mean(amplitude_envelope, axis=0)
    idx = np.argsort(avgRipple)
    bestChannels = np.setdiff1d(idx, badChannels, assume_unique=True)[::-1]

    if saveRippleChan == 1:
        bestChan = bestChannels[0]
        best2ndChan = bestChannels[1]

        b1 = np.memmap(fileName, dtype="int16", mode="r")
        RipplelfpExtract = b1[bestChan::nChans]
        Ripple2ndlfpExtract = b1[best2ndChan::nChans]

        Ripplelfps = {"BestChan": RipplelfpExtract, "Best2ndChan": Ripple2ndlfpExtract}

        np.save(basePath + subname + "_BestRippleChans.npy", Ripplelfps)
        # np.save(basePath + subname + "_BestRippleChan.npy", RippleExtract2)

    # selecting first three channels

    bestChannels = bestChannels[0:5]

    return bestChannels
