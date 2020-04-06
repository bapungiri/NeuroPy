import numpy as np
import scipy.signal as sg
from dataclasses import dataclass
from typing import Any
import scipy.ndimage as filtSig
from collections import namedtuple
import scipy.stats as stats
from scipy.interpolate import interp1d


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


def whiten(strain, interp_psd, dt):
    Nt = len(strain)
    freqs = np.fft.rfftfreq(Nt, dt)
    # print(freqs[:20])
    # freqs1 = np.linspace(0, 2048.0, Nt // 2 + 1)

    # whitening: transform to freq domain, divide by asd, then transform back,
    # taking care to get normalization right.
    hf = np.fft.rfft(strain)
    norm = 1.0 / np.sqrt(1.0 / (dt * 2))
    white_hf = hf / np.sqrt(interp_psd(freqs)) * norm
    white_ht = np.fft.irfft(white_hf, n=Nt)
    return white_ht


@dataclass
class spectrogramBands:
    lfp: Any
    sampfreq: float = 1250.0
    window: int = 1250
    overlap: int = int(window / 8)
    smooth: int = 10

    def __post_init__(self):

        f, t, sxx = sg.spectrogram(
            self.lfp, fs=self.sampfreq, nperseg=self.window, noverlap=self.overlap
        )

        # sxx = stats.zscore(sxx, axis=None)
        smooth = self.smooth

        delta_ind = np.where(((f > 0.5) & (f < 4)) | ((f > 12) & (f < 15)))[0]
        # delta_ind = np.where(((f > 0.5) & (f < 16)))[0]
        delta_sxx = np.mean(sxx[delta_ind, :], axis=0)
        delta_smooth = filtSig.gaussian_filter1d(delta_sxx, smooth, axis=0)

        theta_ind = np.where((f > 6) & (f < 10))[0]
        theta_sxx = np.mean(sxx[theta_ind, :], axis=0)
        theta_smooth = filtSig.gaussian_filter1d(theta_sxx, smooth, axis=0)

        spindle_ind = np.where((f > 10) & (f < 20))[0]
        spindle_sxx = np.mean(sxx[spindle_ind, :], axis=0)
        spindle_smooth = filtSig.gaussian_filter1d(spindle_sxx, smooth, axis=0)

        gamma_ind = np.where((f > 30) & (f < 100))[0]
        gamma_sxx = np.mean(sxx[gamma_ind, :], axis=0)
        gamma_smooth = filtSig.gaussian_filter1d(gamma_sxx, smooth, axis=0)

        ripple_ind = np.where((f > 140) & (f < 250))[0]
        ripple_sxx = np.mean(sxx[ripple_ind, :], axis=0)
        ripple_smooth = filtSig.gaussian_filter1d(ripple_sxx, smooth, axis=0)

        self.delta = delta_smooth
        self.theta = theta_smooth
        self.spindle = spindle_smooth
        self.gamma = gamma_smooth
        self.ripple = ripple_smooth
        self.freq = f
        self.time = t
        self.sxx = sxx
