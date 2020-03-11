import numpy as np
import scipy.signal as sg


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

