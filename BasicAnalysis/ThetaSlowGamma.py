import numpy as np
import matplotlib.pyplot as plt
import pywt
import math
from scipy import signal
import scipy.stats as stats

filename = '/data/DataGen/SleepDeprivation/RatJDay1.npy'

eeg1 = np.load(filename)

start_ind = 1250*60*60*3+1250*60*10
eeg_theta = eeg1[start_ind:start_ind+1250]

t = np.linspace(0, 1, 1250)
# s = np.sin(8 * 2 * np.pi * t) + 0.5*np.sin(16 * 2 *
#                                            np.pi * t) + 0.3*np.sin(30 * 2 * np.pi * t)

# s = np.sin(8 * 2 *
#            np.pi * t) + 0.5*np.sin(16 * 2 * np.pi * t) + 0.1*np.random.randn(len(t))

s = stats.zscore(eeg_theta)
dt = t[1]-t[0]


# fft analysis
fft = np.fft.fft(s)
T = t[1] - t[0]  # sampling interval
N = s.size
# 1/T = frequency
f = np.linspace(0, 1 / T, N)

# == wavelet analysis ==========================
time, sst = t, s
ratio_param = 30

C_range = np.arange(1, 71, 1)
# scales1 = np.arange(1, 100, 1)
# f2 = pywt.scale2frequency(wavelet, scales1)
spect = np.zeros(shape=(len(C_range), 1250))
for wav in range(0, len(C_range)):

    C = C_range[wav]
    B = 2*((7/(2*np.pi*C))**2)
    wavelet = 'cmor' + repr(B) + '-' + repr(C)
    sigma_t = np.sqrt(C/ratio_param)  # ratio_param/(2*np.pi*C)
    # scales = np.arange(0.5, 60, 0.01)

    A = (sigma_t * np.sqrt(np.pi)) ** (-1/2)
    gauss_env = np.exp((-(t-0.5)**2)/(2*sigma_t**2))
    # f2 = pywt.scale2frequency(wavelet, scales)
    mor_wavelet_real = A * np.multiply(gauss_env, np.cos(2*np.pi*C*t))
    mor_wavelet_img = A * np.multiply(gauss_env, np.sin(2*np.pi*C*t))
    cfs1 = np.convolve(s, mor_wavelet_real, mode='same')**2
    cfs2 = np.convolve(s, mor_wavelet_img, mode='same')**2
    cfs = cfs1+cfs2
    # [cfs, frequencies] = pywt.cwt(sst, 600, wavelet)
    # logpower = np.log10((abs(cfs)) ** 2)
    logpower = np.log10(cfs)
    spect[wav] = logpower


spect = np.flip(spect)

# period = frequencies
levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8]

plt.clf()
plt.subplot(411)
plt.ylabel("Amplitude")
plt.xlabel("Time [s]")
plt.title('CA1 Theta (RatJ Day1)', loc='left')
plt.plot(t, s)

plt.subplot(412)
plt.ylabel("Amplitude")
plt.xlabel("Frequency [Hz]")
plt.title('Fourier Transform', loc='left')
# 1 / N is a normalization factor
plt.plot(f[:N // 15], np.abs(fft)[:N // 15] * 1 / N)


plt.subplot(413)
plt.imshow(spect, aspect='auto', vmin=2.5,
           extent=[0, 1, 1, 70], cmap='YlOrRd')
plt.xlabel('Time [s]')
plt.ylabel('Frequency [Hz]')
plt.title('CWT', loc='left')
# plt.contour(t, C_range, spect)

mor_wavelet_real_slow = A * np.multiply(gauss_env, np.cos(2*np.pi*30*t))
plt.subplot(414)
plt.plot(t, mor_wavelet_real_slow)
plt.xlabel('Time [s]')
plt.ylabel('A.U.')
plt.title('Morlet Wavelet for 30 Hz', loc='left')
plt.annotate(r'$\sigma_t = \frac{7}{2 \pi f_o}$', xy=(
    0.1, 2,), fontsize='xx-large')

# f, ax = plt.subplots(figsize=(15, 10))
# ax.contourf(time, np.log2(period), np.log2(power), np.log2(levels),
#             extend='both')

# ax.set_title('%s Wavelet Power Spectrum (%s)' % ('Nino1+2', wavelet))
# ax.set_ylabel('Period (years)')
# Yticks = 2 ** np.arange(np.ceil(np.log2(period.min())),
#                         np.ceil(np.log2(period.max())))
# ax.set_yticks(np.log2(Yticks))
# ax.set_yticklabels(Yticks)
# ax.invert_yaxis()
# ylim = ax.get_ylim()
# ax.set_ylim(ylim[0], -1)
