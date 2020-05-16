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


sess = processData(basePath[0])
sess.trange = np.array([])
# y1 = (
#     np.sin(2 * np.pi * 4 * t)
#     # * np.cos(2 * np.pi * 3.8 * t)
#     # * np.sin(2 * np.pi * 4.2 * t)
# )
# # y2 = np.cos(2 * np.pi * 3.8 * t)
# # y3 = np.sin(2 * np.pi * 4.2 * t)

# y = y1
y1, _ = sess.ripple.best_chan_lfp

y = y1

# y = np.sin(2 * np.pi * 4 * np.linspace(0, 10, 10 * 1250))

t_wavelet = np.arange(-2, 2, 1 / 1250)
B = 0.1
A = 1 / np.sqrt(np.pi ** 0.5 * B)


frequency = np.logspace(1, 5, 30, base=2)

wave_spec = []
for freq in frequency:
    my_wavelet = (
        A
        * np.exp(-((t_wavelet) ** 2) / (2 * B ** 2))
        * np.exp(2j * np.pi * freq * t_wavelet)
    )
    # conv_val = np.convolve(y, my_wavelet, mode="same")
    conv_val = sg.fftconvolve(y, my_wavelet, mode="same")

    wave_spec.append(conv_val)

wave_spec = np.asarray(wave_spec)
# scale = np.linspace(1000, 9000, 50)
# y_wave, freq = pywt.cwt(y, scales=scale, wavelet="cmor2-600", sampling_period=1 / 1250)


# plt.clf()
# plt.subplot(3, 1, 1)
# plt.plot(y)
# plt.subplot(3, 1, 2)
# plt.imshow(
#     np.abs(wave_spec),
#     extent=[0, 10, frequency[0], frequency[-1]],
#     aspect="auto",
#     origin="lower",
# )
# plt.yscale("log")
# plt.subplot(3, 1, 3)
# plt.plot(t_wavelet, my_wavelet)
