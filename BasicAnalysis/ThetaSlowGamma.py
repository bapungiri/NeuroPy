import numpy as np
import matplotlib.pyplot as plt
import pywt
import math
from scipy import signal
import scipy.stats as stats
from scipy.fftpack import next_fast_len
from polycoherence import _plot_signal, polycoherence, plot_polycoherence

filename = '/data/DataGen/SleepDeprivation/RatJDay1.npy'

eeg1 = np.load(filename)

start_ind = 1250*60*60*3+1250*60*10
eeg_theta = eeg1[start_ind:start_ind+1250]
t = np.linspace(0, 1, 1250)
s = np.sin(8 * 2 *
           np.pi * t) + 0.5*np.sin(16 * 2 * np.pi * t) + 0.1*np.random.randn(len(t))

# s = stats.zscore(eeg_theta)
dt = t[1]-t[0]


# fft analysis
fft = np.fft.fft(s)
T = t[1] - t[0]  # sampling interval
N = s.size
# 1/T = frequency
f = np.linspace(0, 1 / T, N)

# ======== wavelet analysis ==========================
time, sst = t, s
ratio_param = 30
C_range = np.arange(20, 71, 1)
spect = np.zeros(shape=(len(C_range), 1250))
for wav in range(0, len(C_range)):

    C = C_range[wav]
    B = 2*((7/(2*np.pi*C))**2)
    wavelet = 'cmor' + repr(B) + '-' + repr(C)
    sigma_t = ratio_param/(2*np.pi*C)
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


plt.clf()
plt.subplot(511)
plt.ylabel("Amplitude")
plt.xlabel("Time [s]")
plt.title('CA1 Theta (RatJ Day1)', loc='left')
plt.plot(t, s)
plt.xlim(0, 1)

plt.subplot(512)
plt.ylabel("Amplitude")
plt.xlabel("Frequency [Hz]")
plt.title('Fourier Transform', loc='left')
# 1 / N is a normalization factor
plt.plot(f[2:N // 15], np.log10(np.abs(fft)[2:N // 15] * 1 / N))


plt.subplot(513)
plt.imshow(spect, aspect='auto', vmin=0, vmax=4,
           extent=[0, 1, 20, 70], cmap='jet')
plt.xlabel('Time [s]')
plt.ylabel('Frequency [Hz]')
plt.title('CWT', loc='left')
# plt.colorbar()
# plt.contour(t, C_range, spect)

sigma_t = ratio_param/(2*np.pi*30)
gauss_env = np.exp((-(t-0.5)**2)/(2*sigma_t**2))
mor_wavelet_real_slow = A * np.multiply(gauss_env, np.cos(2*np.pi*30*t))
plt.subplot(514)
plt.plot(t, mor_wavelet_real_slow)
plt.xlabel('Time [s]')
plt.ylabel('A.U.')
plt.title('Morlet Wavelet for 30 Hz', loc='left')
plt.annotate(r'$\sigma_t = \frac{7}{2 \pi f_o}$', xy=(
    0.1, 2,), fontsize='xx-large')

kw = dict(nperseg=N // 8, noverlap=N // 30, nfft=next_fast_len(N // 2))
freq1, freq2, bicoh = polycoherence(
    s, 1250, flim1=(0, 60), flim2=(0, 60), **kw)

plt.subplot(515)
plot_polycoherence(freq1, freq2, bicoh)
