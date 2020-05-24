import matplotlib.pyplot as plt
import numpy as np
import pywt
from callfunc import processData
import scipy.stats as stats
import scipy.signal as sg

basePath = [
    # "/data/Clustering/SleepDeprivation/RatJ/Day1/",
    # "/data/Clustering/SleepDeprivation/RatK/Day1/",
    # "/data/Clustering/SleepDeprivation/RatN/Day1/",
    # "/data/Clustering/SleepDeprivation/RatJ/Day2/",
    # "/data/Clustering/SleepDeprivation/RatK/Day2/",
    "/data/Clustering/SleepDeprivation/RatN/Day2/",
    # "/data/Clustering/SleepDeprivation/RatK/Day4/"
]


freq = 250
t_wavelet = np.arange(-4, 4, 1 / 1250)
A = np.sqrt(freq)
sigma = 5 / (6 * freq)
my_wavelet = (
    A * np.exp(-((t_wavelet) ** 2) / sigma ** 2) * np.exp(2j * np.pi * freq * t_wavelet)
)
# B = 0.7
# A = 1 / np.sqrt(np.pi ** 0.5 * B)
# my_wavelet = (
#     A * np.exp(-((t_wavelet) ** 2) / (2 * B ** 2)) * np.exp(2j * np.pi * 1 * t_wavelet)
# )


plt.plot(t_wavelet, my_wavelet)
