#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.signal as sg
import seaborn as sns
import matplotlib
from collections import namedtuple
from pathlib import Path
import signal_process
import matplotlib.gridspec as gridspec
import colorednoise as cn

cmap = matplotlib.cm.get_cmap("hot_r")


from callfunc import processData

base_freq = 8  # in Hz
N = 60000
beta = 1  # the exponent
samples = N  # number of samples to generate
y_noise = 0.3 * cn.powerlaw_psd_gaussian(beta, samples)
# y_noise = 0.2 * np.random.randn(len(x))

# sample spacing
T = 1.0 / 1250.0
x = np.linspace(0.0, N * T, N)
y1 = np.sin(base_freq * 2.0 * np.pi * x) + y_noise
y2 = (
    np.sin(base_freq * 2.0 * np.pi * x)
    + 0.8 * np.sin(2 * base_freq * 2.0 * np.pi * x)
    + y_noise
)
y3 = (
    np.sin(base_freq * 2.0 * np.pi * x)
    + 0.8 * np.sin(2 * base_freq * 2.0 * np.pi * x)
    + 0.4 * np.sin(4 * base_freq * 2.0 * np.pi * x)
    + y_noise
)
y4 = sg.sawtooth(2 * np.pi * base_freq * x, width=0) + y_noise

y = [y1, y2, y3, y4]


plt.clf()
fig = plt.figure(1, figsize=(10, 15), sharex=True, sharey=True)
gs = gridspec.GridSpec(3, 4, figure=fig)
fig.subplots_adjust(hspace=0.3)


for i in range(len(y)):
    bicoh, freq = signal_process.bicoherence(y[i], fhigh=60)
    f, pxx = sg.welch(y[i], fs=1250, nperseg=4 * 1250, noverlap=2 * 1250)

    axsig = fig.add_subplot(gs[0, i])
    axsig.plot(y[i][:625], "#564d4d")
    axsig.set_ylabel("Amplitude")

    axpxx = fig.add_subplot(gs[1, i])
    axpxx.plot(f, pxx, "#424242")
    axpxx.set_xscale("log")
    axpxx.set_yscale("log")

    axbcoh = fig.add_subplot(gs[2, i])
    axbcoh.pcolormesh(freq, freq, bicoh, cmap="hot", vmax=0.7)
    axbcoh.set_ylim([2, 30])
    axbcoh.set_xlabel("Frequency (Hz)")
    axbcoh.set_ylabel("Frequency (Hz)")


# fig, ax = plt.subplots()
# ax.plot(xf, 2.0 / N * np.abs(yf[: N // 2]))
# plt.show()
