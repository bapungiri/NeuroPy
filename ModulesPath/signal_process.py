import numpy as np
import scipy.signal as sg
from dataclasses import dataclass
from typing import Any
from scipy.ndimage import gaussian_filter
import scipy.ndimage as filtSig
from collections import namedtuple
import scipy.stats as stats
from scipy.interpolate import interp1d
from scipy import fftpack
from scipy.fft import fft
from lspopt import spectrogram_lspopt
from waveletFunctions import wavelet
import scipy.interpolate as interp
from scipy.fftpack import next_fast_len
import time
import matplotlib.pyplot as plt


class filter_sig:
    @staticmethod
    def filter_cust(
        signal, sampleRate=1250, hf=None, lf=None, order=3, fs=1.0, ax=-1,
    ):
        nyq = 0.5 * sampleRate

        b, a = sg.butter(order, [lf / nyq, hf / nyq], btype="bandpass")
        yf = sg.filtfilt(b, a, signal, axis=ax)

        return yf

    @staticmethod
    def filter_ripple(signal, sampleRate=1250, ax=0):
        lowpass_freq = 150
        highpass_freq = 240
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
    window: int = 1  # in seconds
    overlap: int = window / 8  # in seconds
    smooth: int = None

    def __post_init__(self):

        window = int(self.window * self.sampfreq)
        overlap = int(self.overlap * self.sampfreq)
        # f, t, sxx = sg.spectrogram(
        #     self.lfp, fs=self.sampfreq, nperseg=window, noverlap=overlap
        # )

        tapers = sg.windows.dpss(M=window, NW=5, Kmax=6)

        sxx_taper = []
        for taper in tapers:
            f, t, sxx = sg.spectrogram(
                self.lfp, window=taper, fs=self.sampfreq, noverlap=overlap
            )
            sxx_taper.append(sxx)
        sxx = np.dstack(sxx_taper).mean(axis=2)

        delta_ind = np.where((f > 0.5) & (f < 4))[0]
        delta_sxx = np.mean(sxx[delta_ind, :], axis=0)

        deltaplus_ind = np.where(((f > 0.5) & (f < 4)) | ((f > 12) & (f < 15)))[0]
        deltaplus_sxx = np.mean(sxx[deltaplus_ind, :], axis=0)
        # delta_ind = np.where(((f > 0.5) & (f < 16)))[0]
        # delta_smooth = filtSig.gaussian_filter1d(delta_sxx, smooth, axis=0)

        theta_ind = np.where((f > 5) & (f < 11))[0]
        theta_sxx = np.mean(sxx[theta_ind, :], axis=0)
        # theta_smooth = filtSig.gaussian_filter1d(theta_sxx, smooth, axis=0)

        spindle_ind = np.where((f > 10) & (f < 20))[0]
        spindle_sxx = np.mean(sxx[spindle_ind, :], axis=0)
        # spindle_smooth = filtSig.gaussian_filter1d(spindle_sxx, smooth, axis=0)

        gamma_ind = np.where((f > 30) & (f < 90))[0]
        gamma_sxx = np.mean(sxx[gamma_ind, :], axis=0)
        # gamma_smooth = filtSig.gaussian_filter1d(gamma_sxx, smooth, axis=0)

        ripple_ind = np.where((f > 140) & (f < 250))[0]
        ripple_sxx = np.mean(sxx[ripple_ind, :], axis=0)

        if self.smooth is not None:
            smooth = self.smooth
            delta_sxx = filtSig.gaussian_filter1d(delta_sxx, smooth, axis=0)
            deltaplus_sxx = filtSig.gaussian_filter1d(deltaplus_sxx, smooth, axis=0)
            theta_sxx = filtSig.gaussian_filter1d(theta_sxx, smooth, axis=0)
            spindle_sxx = filtSig.gaussian_filter1d(spindle_sxx, smooth, axis=0)
            gamma_sxx = filtSig.gaussian_filter1d(gamma_sxx, smooth, axis=0)
            ripple_sxx = filtSig.gaussian_filter1d(ripple_sxx, smooth, axis=0)

        self.delta = delta_sxx
        self.deltaplus = deltaplus_sxx
        self.theta = theta_sxx
        self.spindle = spindle_sxx
        self.gamma = gamma_sxx
        self.ripple = ripple_sxx
        self.freq = f
        self.time = t
        self.sxx = sxx
        self.theta_delta_ratio = self.theta / self.delta
        self.theta_deltaplus_ratio = self.theta / self.deltaplus

    def plotSpect(self, ax=None, freqRange=None):

        if ax is None:
            fig, ax = plt.subplots(1, 1)
        sxx = self.sxx / np.max(self.sxx)
        sxx = gaussian_filter(sxx, sigma=1)
        vmax = np.max(sxx) / 4
        if freqRange is None:
            freq_indx = np.arange(len(self.freq))
        else:
            freq_indx = np.where(
                (self.freq > freqRange[0]) & (self.freq < freqRange[1])
            )[0]
        ax.pcolormesh(
            self.time,
            self.freq[freq_indx],
            sxx[freq_indx, :],
            cmap="Spectral_r",
            vmax=vmax,
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")


@dataclass
class wavelet_decomp:
    lfp: np.array
    freqs: np.array = np.arange(1, 20)
    sampfreq: int = 1250

    def colgin2009(self):
        """colgin


        Returns:
            [type]: [description]
        
        References
        ------------
        1) Colgin, L. L., Denninger, T., Fyhn, M., Hafting, T., Bonnevie, T., Jensen, O., ... & Moser, E. I. (2009). Frequency of gamma oscillations routes flow of information in the hippocampus. Nature, 462(7271), 353-357.
        2) Tallon-Baudry, C., Bertrand, O., Delpuech, C., & Pernier, J. (1997). Oscillatory γ-band (30–70 Hz) activity induced by a visual search task in humans. Journal of Neuroscience, 17(2), 722-734.
        """
        t_wavelet = np.arange(-4, 4, 1 / self.sampfreq)
        freqs = self.freqs
        n = len(self.lfp)
        fastn = next_fast_len(n)
        signal = np.pad(self.lfp, (0, fastn - n), "constant", constant_values=0)
        print(len(signal))
        # signal = np.tile(np.expand_dims(signal, axis=0), (len(freqs), 1))
        # wavelet_at_freqs = np.zeros((len(freqs), len(t_wavelet)), dtype=complex)
        conv_val = np.zeros((len(freqs), n), dtype=complex)
        for i, freq in enumerate(freqs):
            sigma = 7 / (2 * np.pi * freq)
            A = (sigma * np.sqrt(np.pi)) ** -0.5
            wavelet_at_freq = (
                A
                * np.exp(-(t_wavelet ** 2) / (2 * sigma ** 2))
                * np.exp(2j * np.pi * freq * t_wavelet)
            )

            conv_val[i, :] = sg.fftconvolve(
                signal, wavelet_at_freq, mode="same", axes=-1
            )[:n]

        return np.abs(conv_val) ** 2

    def quyen2008(self):
        """colgin


        Returns:
            [type]: [description]
        
        References
        ------------
        1) Le Van Quyen, M., Bragin, A., Staba, R., Crépon, B., Wilson, C. L., & Engel, J. (2008). Cell type-specific firing during ripple oscillations in the hippocampal formation of humans. Journal of Neuroscience, 28(24), 6104-6110.
        """
        t_wavelet = np.arange(-4, 4, 1 / self.sampfreq)
        freqs = self.freqs
        signal = self.lfp
        signal = np.tile(np.expand_dims(signal, axis=0), (len(freqs), 1))

        wavelet_at_freqs = np.zeros((len(freqs), len(t_wavelet)))
        for i, freq in enumerate(freqs):
            sigma = 5 / (6 * freq)
            A = np.sqrt(freq)
            wavelet_at_freqs[i, :] = (
                A
                * np.exp(-((t_wavelet) ** 2) / (sigma ** 2))
                * np.exp(2j * np.pi * freq * t_wavelet)
            )

        conv_val = sg.fftconvolve(signal, wavelet_at_freqs, mode="same", axes=-1)

        return np.abs(conv_val) ** 2

    def bergel2018(self):
        """colgin


        Returns:
            [type]: [description]

        References:
        ---------------
        1) Bergel, A., Deffieux, T., Demené, C., Tanter, M., & Cohen, I. (2018). Local hippocampal fast gamma rhythms precede brain-wide hyperemic patterns during spontaneous rodent REM sleep. Nature communications, 9(1), 1-12.

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

    def torrenceCompo(self):
        wavelet = _check_parameter_wavelet("morlet")
        sj = 1 / (wavelet.flambda() * self.freqs)
        # wave, period, scale, coi = wavelet(
        #     self.lfp, 1 / self.sampfreq, pad=1, dj=0.25, s0, j1, mother
        # )

    def cohen(self, ncycles=3):
        """Implementation of ref. 1 chapter 13


        Returns:
            [type]: [description]

        References:
        ---------------
        1) Cohen, M. X. (2014). Analyzing neural time series data: theory and practice. MIT press.

        """
        signal = self.lfp
        t_wavelet = np.arange(-4, 4, 1 / self.sampfreq)
        freqs = self.freqs

        wave_spec = []
        for freq in freqs:
            s = ncycles / (2 * np.pi * freq)
            A = (s * np.sqrt(np.pi)) ** -0.5
            my_wavelet = (
                A
                * np.exp(-(t_wavelet ** 2) / (2 * s ** 2))
                * np.exp(2j * np.pi * freq * t_wavelet)
            )
            # conv_val = np.convolve(signal, my_wavelet, mode="same")
            conv_val = sg.fftconvolve(signal, my_wavelet, mode="same")

            wave_spec.append(conv_val)

        wave_spec = np.abs(np.asarray(wave_spec))
        return wave_spec ** 2


def hilbertfast(signal, ax=-1):

    """inputs a signal does padding to next power of 2 for faster computation of hilbert transform

    Arguments:
        signal {array} -- [n, dimensional array]

    Returns:
        [type] -- [description]
    """
    signal_length = signal.shape[-1]
    hilbertsig = sg.hilbert(signal, fftpack.next_fast_len(signal_length), axis=ax)

    if np.ndim(signal) > 1:
        hilbertsig = hilbertsig[:, :signal_length]
    else:
        hilbertsig = hilbertsig[:signal_length]

    return hilbertsig


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


def bicoherence(signal, flow=1, fhigh=150, fs=1250, window=4 * 1250, overlap=2 * 1250):

    """Generate bicoherence triangular matrix for signal

    Returns:
        bicoher (freq_req x freq_req, array): bicoherence matrix
        freq_req {array}: frequencies at which bicoherence was calculated

    References:
    -----------------------
    1) Sheremet, A., Burke, S. N., & Maurer, A. P. (2016). Movement enhances the nonlinearity of hippocampal theta. Journal of Neuroscience, 36(15), 4218-4230.
    """
    signal = signal - np.mean(signal)
    f, t, sxx = sg.spectrogram(
        signal, nperseg=window, noverlap=overlap, fs=fs, mode="complex"
    )

    freq_req = f[np.where((f > flow) & (f < fhigh))[0]]
    freq_ind = np.where((f > flow) & (f < fhigh))[0]
    bispec = np.zeros((len(freq_ind), len(freq_ind)), dtype=complex)
    for row, f_ind in enumerate(freq_ind):
        numer = np.mean(
            sxx[f_ind, :] * sxx[freq_ind, :] * np.conj(sxx[freq_ind + f_ind, :]), axis=1
        )
        denom_left = np.mean(np.abs(sxx[f_ind, :] * sxx[freq_ind, :]) ** 2, axis=1)
        denom_right = np.mean(np.abs(sxx[freq_ind + f_ind, :]) ** 2, axis=1)
        bispec[row, :] = numer / np.sqrt(denom_left * denom_right)

    bispec = np.zeros((len(freq_ind), len(freq_ind)), dtype=complex)
    for i in range(sxx.shape[1]):
        numer = np.mean(
            sxx[f_ind, :] * sxx[freq_ind, :] * np.conj(sxx[freq_ind + f_ind, :]), axis=1
        )
        denom_left = np.mean(np.abs(sxx[f_ind, :] * sxx[freq_ind, :]) ** 2, axis=1)
        denom_right = np.mean(np.abs(sxx[freq_ind + f_ind, :]) ** 2, axis=1)
        bispec[row, :] = numer / np.sqrt(denom_left * denom_right)

    bicoher = np.abs(bispec) ** 2
    bicoher = np.fliplr(np.triu(np.fliplr(np.triu(bicoher, k=0)), k=0))
    bispec = np.fliplr(np.triu(np.fliplr(np.triu(bispec, k=0)), k=0))

    return bicoher, freq_req, bispec


def phasePowerCorrelation(signal):
    pass


def csdClassic(lfp, coords):
    time = np.arange(0, lfp.shape[1], 1)
    print(lfp.shape)
    f_pot = interp.interp2d(time, coords, lfp, kind="cubic")

    y_new = np.linspace(min(coords), max(coords), 200)
    lfp_new = f_pot(time, y_new)
    csd = np.diff(np.diff(lfp_new, axis=0), axis=0)

    return csd


def icsd(lfp, coords):
    pass


def mtspect(signal, nperseg, noverlap, fs=1250):
    window = nperseg
    overlap = noverlap

    tapers = sg.windows.dpss(M=window, NW=5, Kmax=6)

    psd_taper = []
    for taper in tapers:
        f, psd = sg.welch(signal, window=taper, fs=fs, noverlap=overlap)
        psd_taper.append(psd)
    psd = np.asarray(psd_taper).mean(axis=0)

    return f, psd

