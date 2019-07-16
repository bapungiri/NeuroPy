#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 13:42:34 2019

@author: bapung
"""

import numpy as np
import scipy.signal as sg
import scipy.fftpack as ft
import scipy.ndimage as smth
import scipy.stats as stat
import numpy.random as rnd


def swr(lfpfile, RippleChannel, samplingFrequency, numChans):

    lfp = lfpfile
    SampFreq = samplingFrequency
    nChans = numChans
    ReqChan = RippleChannel
    nyq = 0.5 * SampFreq
    offsetp = (ReqChan-1)*2
    duration = 3600*14
    lowthresholdFactor = 1
    highThresholdFactor = 2
    # print(duration)

    # loading the required chanel from eeg file for ripple detection
    lfpCA1 = np.memmap(lfp, dtype='int16', mode='r',
                       offset=offsetp, shape=(SampFreq * duration, nChans))
    signal = lfpCA1[:, 0]
    signal = np.array(signal, dtype=np.float)  # convert data to float

    b, a = sg.butter(3, [150/nyq, 240/nyq], btype='bandpass')
    yf = sg.filtfilt(b, a, signal)

    squared_signal = np.square(yf)
    normsquaredsignal = stat.zscore(squared_signal)

    # getting an envelope of the signal
    # analytic_signal = sg.hilbert(yf)
    # amplitude_envelope = stat.zscore(np.abs(analytic_signal))

    windowLength = SampFreq/SampFreq*11
    window = np.ones((int(windowLength),))/windowLength

    smoothSignal = sg.filtfilt(window, 1, squared_signal, axis=0)
    zscoreSignal = stat.zscore(smoothSignal)

    hist_zscoresignal, edges_zscoresignal = np.histogram(
        zscoreSignal, bins=np.linspace(0, 6, 100))

    ThreshSignal = np.diff(np.where(zscoreSignal > lowthresholdFactor, 1, 0))
    start_ripple = np.argwhere(ThreshSignal == 1)
    stop_ripple = np.argwhere(ThreshSignal == -1)

    print(start_ripple.shape, stop_ripple.shape)
    firstPass = np.concatenate((start_ripple, stop_ripple), axis=1)

    # ===== merging close ripples
    minInterRippleSamples = 30/1000*SampFreq
    secondPass = []
    ripple = firstPass[0]
    for i in range(1, len(firstPass)):
        if firstPass[i, 0] - ripple[1] < minInterRippleSamples:
            # Merging ripples
            ripple = [ripple[0], firstPass[i, 1]]
        else:
            secondPass.append(ripple)
            ripple = firstPass[i]

    secondPass.append(ripple)
    secondPass = np.asarray(secondPass)

    # delete ripples with less than threshold power
    thirdPass = []
    peakNormalizedPower = []

    for i in range(0, len(secondPass)):
        maxValue = max(zscoreSignal[secondPass[i, 0]:secondPass[i, 1]])
        if maxValue > highThresholdFactor:
            thirdPass.append(secondPass[i])
            peakNormalizedPower.append(maxValue)

    thirdPass = np.asarray(thirdPass)

    ripple_duration = np.diff(thirdPass, axis=1)/1250*1000

    # delete very short ripples
    shortRipples = np.where(ripple_duration < 20)[0]
    thirdPass = np.delete(thirdPass, shortRipples, 0)

    # delete very short ripples
    shortRipples = np.where(ripple_duration < 20)[0]
    thirdPass = np.delete(thirdPass, shortRipples, 0)

    # selecting some example ripples
    idx = rnd.randint(0, thirdPass.shape[0], 5, dtype='int')
    example_ripples = []
    example_ripples_duration = []  # in frames
    for i in range(5):
        example_ripples.append(
            signal[thirdPass[idx[i], 0]-125:thirdPass[idx[i], 1]+125])
        example_ripples_duration.append(
            thirdPass[idx[i], 1]-thirdPass[idx[i], 0])

    other_measures = dict()
    other_measures['example_ripples'] = [
        example_ripples, example_ripples_duration]
    other_measures['zscore_dist'] = [hist_zscoresignal, edges_zscoresignal]

    return thirdPass, other_measures


def deltawave(sub_name, nREMPeriod, RecInfo):

    SampFreq = RecInfo['samplingFrequency']
    frames = RecInfo['behavFrames']
    behav = RecInfo['behav']
    nChans = RecInfo['numChannels']
    ReqChan = RecInfo['SpectralChannel']

    offsetP = ((nREMPeriod-behav[2, 0])//1e6)*SampFreq + \
        int(np.diff(frames[0, :]))+int(np.diff(frames[1, :]))
    b1 = np.memmap('/data/EEGData/' + sub_name + '.eeg', dtype='int16', mode='r',
                   offset=int(offsetP)*nChans*2 + 1*(ReqChan-1)*2, shape=(1, nChans*SampFreq*5))
    eegnrem1 = b1[0, ::nChans]
    sos = sg.butter(3, 100, btype='low', fs=SampFreq, output='sos')
    yf = sg.sosfilt(sos, eegnrem1)
    yf = ft.fft(yf)/len(eegnrem1)
    xf = np.linspace(0.0, SampFreq/2, len(eegnrem1)//2)
    y1 = 2.0/(len(xf)) * np.abs(yf[:len(eegnrem1)//2])
    y1 = smth.gaussian_filter(y1, 8)

    return y1, xf


def spindle(sub_name, nREMPeriod, RecInfo):

    SampFreq = RecInfo['samplingFrequency']
    frames = RecInfo['behavFrames']
    behav = RecInfo['behav']
    nChans = RecInfo['numChannels']
    ReqChan = RecInfo['SpectralChannel']

    offsetP = ((nREMPeriod-behav[2, 0])//1e6)*SampFreq + \
        int(np.diff(frames[0, :]))+int(np.diff(frames[1, :]))
    b1 = np.memmap('/data/EEGData/' + sub_name + '.eeg', dtype='int16', mode='r',
                   offset=int(offsetP)*nChans*2 + 1*(ReqChan-1)*2, shape=(1, nChans*SampFreq*5))
    eegnrem1 = b1[0, ::nChans]
    sos = sg.butter(3, 100, btype='low', fs=SampFreq, output='sos')
    yf = sg.sosfilt(sos, eegnrem1)
    yf = ft.fft(yf)/len(eegnrem1)
    xf = np.linspace(0.0, SampFreq/2, len(eegnrem1)//2)
    y1 = 2.0/(len(xf)) * np.abs(yf[:len(eegnrem1)//2])
    y1 = smth.gaussian_filter(y1, 8)

    return y1, xf


def sharpWaveOnly(sub_name, nREMPeriod, RecInfo):

    SampFreq = RecInfo['samplingFrequency']
    frames = RecInfo['behavFrames']
    behav = RecInfo['behav']
    nChans = RecInfo['numChannels']
    ReqChan = RecInfo['SpectralChannel']

    offsetP = ((nREMPeriod-behav[2, 0])//1e6)*SampFreq + \
        int(np.diff(frames[0, :]))+int(np.diff(frames[1, :]))
    b1 = np.memmap('/data/EEGData/' + sub_name + '.eeg', dtype='int16', mode='r',
                   offset=int(offsetP)*nChans*2 + 1*(ReqChan-1)*2, shape=(1, nChans*SampFreq*5))
    eegnrem1 = b1[0, ::nChans]
    sos = sg.butter(3, 100, btype='low', fs=SampFreq, output='sos')
    yf = sg.sosfilt(sos, eegnrem1)
    yf = ft.fft(yf)/len(eegnrem1)
    xf = np.linspace(0.0, SampFreq/2, len(eegnrem1)//2)
    y1 = 2.0/(len(xf)) * np.abs(yf[:len(eegnrem1)//2])
    y1 = smth.gaussian_filter(y1, 8)

    return y1, xf
