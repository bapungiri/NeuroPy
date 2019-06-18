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


def lfpSpect(sub_name, nREMPeriod, RecInfo):

    SampFreq = RecInfo['samplingFrequency']
    frames = RecInfo['behavFrames']
    behav = RecInfo['behav']
    nChans = RecInfo['numChannels']
    ReqChan = RecInfo['SpectralChannel']

    offsetP = ((nREMPeriod - behav[2, 0]) // 1e6) * SampFreq + \
        int(np.diff(frames[0, :])) + int(np.diff(frames[1, :]))
    b1 = np.memmap('/data/EEGData/' + sub_name + '.eeg', dtype='int16', mode='r',
                   offset=int(offsetP) * nChans * 2 + 1 * (ReqChan - 1) * 2, shape=(1, nChans * SampFreq * 5))
    eegnrem1 = b1[0, ::nChans]
    sos = sg.butter(3, 100, btype='low', fs=SampFreq, output='sos')
    yf = sg.sosfilt(sos, eegnrem1)
    yf = ft.fft(yf) / len(eegnrem1)
    xf = np.linspace(0.0, SampFreq / 2, len(eegnrem1) // 2)
    y1 = 2.0 / (len(xf)) * np.abs(yf[:len(eegnrem1) // 2])
    y1 = smth.gaussian_filter(y1, 8)

    return y1, xf


def lfpSpectMaze(sub_name, nREMPeriod, RecInfo, channel):

    SampFreq = RecInfo['samplingFrequency']
    frames = RecInfo['behavFrames']
    behav = RecInfo['behav']
    nChans = RecInfo['numChannels']
#    ReqChan = RecInfo['SpectralChannel']
    ReqChan = channel
    duration = 5 # chunk of lfp in seconds

    offsetP = ((nREMPeriod - behav[1, 0]) // 1e6) * \
        SampFreq + int(np.diff(frames[0, :]))
    b1 = np.memmap('/data/EEGData/' + sub_name + '.eeg', dtype='int16', mode='r',
                   offset=int(offsetP) * nChans * 2 + 1 * (ReqChan - 1) * 2, shape=(1, nChans * SampFreq * duration))
    eegnrem1 = b1[0, ::nChans]
    sos = sg.butter(3, 100, btype='low', fs=SampFreq, output='sos')
    yf = sg.sosfilt(sos, eegnrem1)
    yf = ft.fft(yf) / len(eegnrem1)
    xf = np.linspace(0.0, SampFreq / 2, len(eegnrem1) // 2)
    y1 = 2.0 / (len(xf)) * np.abs(yf[:len(eegnrem1) // 2])
    y1 = smth.gaussian_filter(y1, 8)

    return y1, xf


def bestThetaChannel(fileName,sampleRate,nChans,badChannels):

    """
    fileName: name of the .eeg file
    sampleRate: sampling frequency of eeg;

    """

    badChannels= [x-1 for x in badChannels] # zero indexing correction
    duration = 60*10 # chunk of lfp in seconds
    nyq = 0.5 * sampleRate
    lowTheta= 5
    highTheta = 10

    lfpCA1 = np.memmap(fileName, dtype='int16', mode='r', shape=(sampleRate * duration, nChans))

#    goodChannels = [i for i in range(len(a)) if a[i]==1]
#    lfpCA1[:,goodChannels-1]

    sos = sg.butter(3, [lowTheta/nyq, highTheta/nyq], btype='bandpass', fs=sampleRate, output='sos')
    yf = sg.sosfilt(sos, lfpCA1, axis=0)

    avgTheta = np.mean(np.square(yf), axis=0);
    idx = np.argsort(avgTheta)

    bestChannels = np.setdiff1d(idx,badChannels, assume_unique=True)[::-1]

    # selecting first three channels

    bestChannels = bestChannels[0:5]

    return bestChannels
