import numpy as np
import scipy.signal as sg
from dataclasses import dataclass
from typing import Any
import scipy.ndimage as filtSig
from collections import namedtuple


class filter_sig:
    @staticmethod
    def filter_cust(
        signal,
        sampleRate,
        highpass_freq=None,
        lowpass_freq=None,
        order=4,
        filter_function="filtfilt",
        fs=1.0,
        axis=-1,
    ):
        nyq = 0.5 * sampleRate

        b, a = sg.butter(3, [lowpass_freq / nyq, highpass_freq / nyq], btype="bandpass")
        yf = sg.filtfilt(b, a, signal, axis=-1)

        return yf

    @staticmethod
    def filter_ripple(signal, sampleRate=1250):
        lowpass_freq = 150
        highpass_freq = 250
        nyq = 0.5 * sampleRate

        b, a = sg.butter(3, [lowpass_freq / nyq, highpass_freq / nyq], btype="bandpass")
        yf = sg.filtfilt(b, a, signal, axis=-1)

        return yf

    @staticmethod
    def filter_theta(signal, sampleRate=1250):
        lowpass_freq = 4
        highpass_freq = 10
        nyq = 0.5 * sampleRate

        b, a = sg.butter(3, [lowpass_freq / nyq, highpass_freq / nyq], btype="bandpass")
        yf = sg.filtfilt(b, a, signal, axis=-1)

        return yf

    @staticmethod
    def filter_delta(signal, sampleRate=1250):
        lowpass_freq = 0.5
        highpass_freq = 4
        nyq = 0.5 * sampleRate

        b, a = sg.butter(3, [lowpass_freq / nyq, highpass_freq / nyq], btype="bandpass")
        yf = sg.filtfilt(b, a, signal, axis=-1)

        return yf


@dataclass
class spectrogramBands:
    lfp: Any
    sampfreq: float = 1250.0
    window: int = 1250
    overlap: int = int(window / 2)
    smooth: int = 10

    def __post_init__(self):
        f, t, sxx = sg.spectrogram(
            self.lfp, fs=self.sampfreq, nperseg=self.window, noverlap=self.overlap
        )

        smooth = self.smooth

        delta_ind = np.where(((f > 0.5) & (f < 4)) | ((f > 12) & (f < 15)))[0]
        # delta_ind = np.where(((f > 0.5) & (f < 16)))[0]
        delta_sxx = np.mean(sxx[delta_ind, :], axis=0)
        delta_smooth = filtSig.gaussian_filter1d(delta_sxx, smooth, axis=0)

        theta_ind = np.where((f > 6) & (f < 10))[0]  # theta band 0-4 Hz and 12-15 Hz
        theta_sxx = np.mean(sxx[theta_ind, :], axis=0)
        theta_smooth = filtSig.gaussian_filter1d(theta_sxx, smooth, axis=0)

        self.theta = theta_smooth
        self.delta = delta_smooth
        self.freq = f
        self.time = t
        self.sxx = sxx


# def spectrogramBands(lfp, sampfreq, window, sldby, smooth):
#     class Bands:
#         def __init__(self):
#             f, t, sxx = sg.spectrogram(lfp, fs=sampfreq, nperseg=window, noverlap=sldby)

#             delta_ind = np.where(((f > 0.5) & (f < 4)) | ((f > 12) & (f < 15)))[0]
#             # delta_ind = np.where(((f > 0.5) & (f < 16)))[0]
#             delta_sxx = np.mean(sxx[delta_ind, :], axis=0)
#             delta_smooth = filtSig.gaussian_filter1d(delta_sxx, smooth, axis=0)

#             theta_ind = np.where((f > 6) & (f < 10))[0]
#             theta_sxx = np.mean(sxx[theta_ind, :], axis=0)
#             theta_smooth = filtSig.gaussian_filter1d(theta_sxx, smooth, axis=0)

#             self.theta = theta_smooth
#             self.delta = delta_smooth
#             self.freq = f
#             self.time = t
#             self.sxx = sxx

#     bands = Bands()

#     return bands
