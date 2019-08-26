import numpy as np
import matplotlib.pyplot as plt
import pywt

t = np.linspace(0, 1, 2000)
s = np.sin(8 * 2 * np.pi * t)+0.5*np.sin(30 * 2 * np.pi * t) + \
    0.3 * np.sin(60 * 2 * np.pi * t)+0.2*np.random.randn(2000)

# fft analysis
fft = np.fft.fft(s)
T = t[1] - t[0]  # sampling interval
N = s.size
# 1/T = frequency
f = np.linspace(0, 1 / T, N)

# == wavelet analysis ==========================
time, sst = t, s
dt = T


C = 30
B = C/7


wavelet = 'cmor' + repr(B) + '-' + repr(C)
scales = np.arange(1, 550)

f = pywt.scale2frequency(wavelet, scales)
mor_wavelet = (1/np.sqrt(np.pi*B))*(np.exp((-t**2)/B))*np.cos(2*np.pi*C*t)

[cfs, frequencies] = pywt.cwt(sst, scales, wavelet)
power = (abs(cfs)) ** 2
period = frequencies
levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8]

plt.clf()
plt.subplot(411)
plt.ylabel("Amplitude")
plt.xlabel("Time [s]")
plt.plot(t, s)

plt.subplot(412)
plt.ylabel("Amplitude")
plt.xlabel("Frequency [Hz]")
# 1 / N is a normalization factor
plt.plot(f[:N // 2], np.abs(fft)[:N // 2] * 1 / N)

plt.subplot(413)
plt.imshow(power)
plt.show


plt.subplot(414)
plt.plot(t, mor_wavelet)

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
0.5
