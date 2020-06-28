import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
import scipy.signal as sg
import warnings

warnings.simplefilter(action="default")


#%%bicoh test
# region
# Number of samplepoints
N = 60000
# sample spacing
T = 1.0 / 1250.0
x = np.linspace(0.0, N * T, N)
y = (
    np.sin(10.0 * 2.0 * np.pi * x)
    + 0.8 * np.sin(20.0 * 2.0 * np.pi * x)
    # + 0.4 * np.sin(20.0 * 2.0 * np.pi * x)
    # + sg.sawtooth(2 * 8 * np.pi * x)[::-1]
    + 0.2 * np.random.randn(len(x))
)


yf = scipy.fftpack.fft(y)
xf = np.linspace(0.0, 1.0 / (2.0 * T), int(N / 2))


f, t, sxx = sg.spectrogram(
    y, nperseg=4 * 1250, noverlap=2 * 1250, fs=1250, mode="complex"
)

freq_req = f[np.where(f < 120)[0]]

bispec = np.zeros((len(freq_req), len(freq_req)))
# for row, freq1 in enumerate(freq_req):
#     for col, freq2 in enumerate(freq_req):
#         f1_ind = np.where(f == freq1)[0]
#         f2_ind = np.where(f == freq2)[0]
#         f1f2_ind = np.where(f == (freq1 + freq2))[0]

#         numer = np.mean(sxx[f1_ind, :] * sxx[f2_ind, :] * np.conj(sxx[f1f2_ind, :]))
#         denom_left = np.mean(np.abs(sxx[f1_ind, :] * sxx[f2_ind, :]) ** 2)
#         denom_right = np.mean(np.abs(sxx[f1f2_ind, :]) ** 2)
#         bispec_here = numer / np.sqrt(denom_left * denom_right)

#         bispec[row, col] = np.abs(bispec_here) ** 2

freq_ind = np.where(f < 120)[0]

for row, f_ind in enumerate(freq_ind):
    numer = np.mean(
        sxx[f_ind, :] * sxx[freq_ind, :] * np.conj(sxx[freq_ind + f_ind, :]), axis=1
    )
    denom_left = np.mean(np.abs(sxx[f_ind, :] * sxx[freq_ind, :]) ** 2, axis=1)
    denom_right = np.mean(np.abs(sxx[freq_ind + f_ind, :]) ** 2, axis=1)
    bispec_here = numer / np.sqrt(denom_left * denom_right)
    bispec[row, :] = np.abs(bispec_here) ** 2


# list comprehension style
getfreq = lambda ind: sxx[ind, :] * sxx[freq_ind, :]
getsumf = lambda ind: sxx[freq_ind + ind, :]

bispec = [
    np.abs(
        np.mean(getfreq(f_ind) * np.conj(getsumf(f_ind)), axis=1)
        / np.sqrt(
            np.mean(np.abs(getfreq(f_ind)) ** 2, axis=1)
            * np.mean(np.abs(getsumf(f_ind)) ** 2, axis=1)
        )
    )
    ** 2
    for f_ind in freq_ind
]

bispec = np.triu(bispec, k=0)
bispec = np.fliplr(bispec)
bispec = np.triu(bispec, k=0)
bispec = np.fliplr(bispec)

# bispec = np.flipud(bispec)

plt.pcolormesh(freq_req, freq_req, bispec, cmap="jet", vmax=0.7)
fig, ax = plt.subplots()
ax.plot(xf, 2.0 / N * np.abs(yf[: N // 2]))
plt.show()
# endregion

#%% spectral whitening test
# region

# Number of samplepoints
N = 60000
# sample spacing
T = 1.0 / 1250.0
x = np.linspace(0.0, N * T, N)
y = (
    np.sin(10.0 * 2.0 * np.pi * x)
    + 0.8 * np.sin(20.0 * 2.0 * np.pi * x)
    # + 0.4 * np.sin(20.0 * 2.0 * np.pi * x)
    # + sg.sawtooth(2 * 8 * np.pi * x)[::-1]
    + 0.2 * np.random.randn(len(x))
)


yf = scipy.fftpack.fft(y)
xf = np.linspace(0.0, 1.0 / (2.0 * T), int(N / 2))

spec_amp = np.sqrt(np.abs(np.multiply(yf, np.conjugate(yf))))
yf /= spec_amp
y_whiten = np.real(scipy.fftpack.ifft(yf))[: len(y)]
