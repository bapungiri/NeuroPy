import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.signal as sg
import scipy.ndimage as smth

signal = np.load('../DataGen/SleepDeprivation/RatJDay1.npy')

SampFreq = 1250
#frames = RecInfo['behavFrames']
#behav = RecInfo['behav']
nChans = 75
ReqChan = 28




nyq = 0.5 * SampFreq

signal = signal[0:1250*3*3600]
signal = np.array(signal, dtype = np.float) # convert data to float



f, t, Sxx = sg.spectrogram(
    signal, SampFreq, nperseg=1250 * 3, noverlap=1250 * 2, nfft=5000)


#zscoreSignal = stat.zscore(signal)


b,a= sg.butter(3, [5/nyq,10/nyq],btype='bandpass')
theta = sg.filtfilt(b,a,signal)

b,a= sg.butter(3, [1/nyq,4/nyq],btype='bandpass')
delta = sg.filtfilt(b,a,signal)


delta_range = np.where((1<f) & (f<4))[0]
delta_power = Sxx[delta_range,:]
delta_mean_power = np.mean(delta_power, axis=0)

theta_range = np.where((5<f) & (f<10))[0]
theta_power = Sxx[theta_range,:]
theta_mean_power = np.mean(theta_power, axis=0)

deltaplus_range = np.where((12<f) & (f<15))[0]
deltaplus_power = Sxx[deltaplus_range,:]
deltaplus_mean_power = np.mean(deltaplus_power, axis=0)

delta_theta_ratio = theta_mean_power/(delta_mean_power+ deltaplus_mean_power)

windowLength = SampFreq/SampFreq*30
window = np.ones((int(windowLength),))/windowLength


delta_theta_ratio = sg.filtfilt(window,1,delta_theta_ratio, axis=0)
# delta_theta_ratio = smth.gaussian_filter1d(delta_theta_ratio, sigma=25)

edges = np.linspace(0,8,300)

hist_states,binedges = np.histogram(delta_theta_ratio,bins=edges) 


plt.clf()

plt.subplot(211)
plt.plot(t/3600,delta_theta_ratio)


plt.subplot(212)

plt.plot(binedges[0:len(binedges)-1],hist_states)

# plt.pcolormesh(t/3600, f, Sxx, cmap='copper', vmax=30)



