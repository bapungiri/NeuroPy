import numpy as np
import scipy.signal as sg
from dataclasses import dataclass
from typing import Any
import scipy.ndimage as filtSig
from collections import namedtuple
import scipy.stats as stats
from scipy.interpolate import interp1d
from scipy import fftpack
from scipy.fft import fft


class filter_sig:
    @staticmethod
    def filter_cust(
        signal, sampleRate=1250, hf=None, lf=None, order=3, fs=1.0, axis=-1,
    ):
        nyq = 0.5 * sampleRate

        b, a = sg.butter(order, [lf / nyq, hf / nyq], btype="bandpass")
        yf = sg.filtfilt(b, a, signal, axis=axis)

        return yf

    @staticmethod
    def filter_ripple(signal, sampleRate=1250, ax=0):
        lowpass_freq = 150
        highpass_freq = 250
        nyq = 0.5 * sampleRate

        b, a = sg.butter(3, [lowpass_freq / nyq, highpass_freq / nyq], btype="bandpass")
        yf = sg.filtfilt(b, a, signal, axis=ax)

        return yf

    @staticmethod
    def filter_theta(signal, sampleRate=1250, ax=0):
        lowpass_freq = 4
        highpass_freq = 10
        nyq = 0.5 * sampleRate

        b, a = sg.butter(3, [lowpass_freq / nyq, highpass_freq / nyq], btype="bandpass")
        yf = sg.filtfilt(b, a, signal, axis=ax)

        return yf

    @staticmethod
    def filter_delta(signal, sampleRate=1250, ax=0):
        lowpass_freq = 0.5
        highpass_freq = 4
        nyq = 0.5 * sampleRate

        b, a = sg.butter(3, [lowpass_freq / nyq, highpass_freq / nyq], btype="bandpass")
        yf = sg.filtfilt(b, a, signal, axis=ax)

        return yf

    @staticmethod
    def filter_spindle(signal, sampleRate=1250, ax=0):
        lowpass_freq = 9
        highpass_freq = 18
        nyq = 0.5 * sampleRate

        b, a = sg.butter(3, [lowpass_freq / nyq, highpass_freq / nyq], btype="bandpass")
        yf = sg.filtfilt(b, a, signal, axis=ax)

        return yf

    @staticmethod
    def filter_gamma(signal, sampleRate=1250, ax=0):
        lowpass_freq = 100
        highpass_freq = 150
        nyq = 0.5 * sampleRate

        b, a = sg.butter(3, [lowpass_freq / nyq, highpass_freq / nyq], btype="bandpass")
        yf = sg.filtfilt(b, a, signal, axis=ax)

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

        gamma_ind = np.where((f > 30) & (f < 90))[0]
        gamma_sxx = np.mean(sxx[gamma_ind, :], axis=0)
        gamma_smooth = filtSig.gaussian_filter1d(gamma_sxx, smooth, axis=0)

        ripple_ind = np.where((f > 140) & (f < 250))[0]
        ripple_sxx = np.mean(sxx[ripple_ind, :], axis=0)
        ripple_smooth = filtSig.gaussian_filter1d(ripple_sxx, smooth, axis=0)

        self.delta = delta_smooth
        self.theta = theta_smooth
        self.spindle = spindle_sxx
        self.gamma = gamma_smooth
        self.ripple = ripple_sxx
        self.freq = f
        self.time = t
        self.sxx = sxx
        self.theta_delta_ratio = self.theta / self.delta


@dataclass
class wavelet_decomp:
    lfp: np.array
    freqs: np.array = np.arange(1, 20)
    sampfreq: int = 1250

    def colgin(self):
        """colgin

        Args:
            lowfreq (int, optional): [description]. Defaults to 1.
            highfreq (int, optional): [description]. Defaults to 250.
            nbins (int, optional): [description]. Defaults to 100.

        Returns:
            [type]: [description]
        """
        t_wavelet = np.arange(-4, 4, 1 / self.sampfreq)
        freqs = self.freqs
        signal = self.lfp
        signal = np.tile(np.expand_dims(signal, axis=0), (len(freqs), 1))

        wavelet_at_freqs = np.zeros((len(freqs), len(t_wavelet)))
        for i, freq in enumerate(freqs):
            sigma = freq / (2 * np.pi * 7)
            A = (sigma * np.sqrt(np.pi)) ** -0.5
            wavelet_at_freqs[i, :] = (
                A
                * np.exp(-((t_wavelet) ** 2) / (2 * sigma ** 2))
                * np.exp(2j * np.pi * freq * t_wavelet)
            )

        conv_val = sg.fftconvolve(signal, wavelet_at_freqs, mode="same", axes=-1)

        return np.abs(conv_val)

    def tallonBaudry(self):
        """colgin

        Args:
            lowfreq (int, optional): [description]. Defaults to 1.
            highfreq (int, optional): [description]. Defaults to 250.
            nbins (int, optional): [description]. Defaults to 100.

        Returns:
            [type]: [description]

        Note: square norm is returned instead of norm, which is different from colgin
        """
        signal = self.lfp
        t_wavelet = np.arange(-4, 4, 1 / self.sampfreq)
        freqs = self.freqs

        wave_spec = []
        for freq in freqs:
            sigma = freq / (2 * np.pi * 7)
            A = (sigma * np.sqrt(np.pi)) ** -0.5
            my_wavelet = (
                A
                * np.exp(-((t_wavelet) ** 2) / (2 * sigma ** 2))
                * np.exp(2j * np.pi * freq * t_wavelet)
            )
            # conv_val = np.convolve(signal, my_wavelet, mode="same")
            conv_val = sg.fftconvolve(signal, my_wavelet, mode="same")

            wave_spec.append(conv_val)

        wave_spec = np.abs(np.asarray(wave_spec))
        return wave_spec ** 2

    def bergelCohen(self):
        """colgin

        Args:
            lowfreq (int, optional): [description]. Defaults to 1.
            highfreq (int, optional): [description]. Defaults to 250.
            nbins (int, optional): [description]. Defaults to 100.

        Returns:
            [type]: [description]

        References:
        ---------------
        1) Bergel ivan cohen,
        Note: square norm is returned instead of norm 
        """
        signal = self.lfp
        t_wavelet = np.arange(-4, 4, 1 / self.sampfreq)
        freqs = self.freqs

        wave_spec = []
        for freq in freqs:
            sigma = freq / (2 * np.pi * 7)
            A = (sigma * np.sqrt(np.pi)) ** -0.5
            my_wavelet = (
                A
                * np.exp(-((t_wavelet) ** 2) / (2 * sigma ** 2))
                * np.exp(2j * np.pi * freq * t_wavelet)
            )
            # conv_val = np.convolve(signal, my_wavelet, mode="same")
            conv_val = sg.fftconvolve(signal, my_wavelet, mode="same")

            wave_spec.append(conv_val)

        wave_spec = np.abs(np.asarray(wave_spec))
        return wave_spec * np.linspace(1, 150, 100).reshape(-1, 1)


def hilbertfast(signal):
    """inputs a signal does padding to next power of 2 for faster computation of hilbert transform

    Arguments:
        signal {array} -- [n, dimensional array]

    Returns:
        [type] -- [description]
    """
    hilbertsig = sg.hilbert(signal, fftpack.next_fast_len(len(signal)))

    return hilbertsig[: len(signal)]


def fftnormalized(signal, fs=1250):

    # Number of sample points
    N = len(signal)
    # sample spacing
    T = 1 / fs
    y = signal
    yf = fft(y)
    freq = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)
    pxx = 2.0 / N * np.abs(yf[0 : N // 2])

    return pxx, freq
