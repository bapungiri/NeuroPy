import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.signal as sg
import scipy.stats as stat

basePath = (
    "/home/bapung/Documents/ClusteringHub/EEGAnlaysis/RatJ/RatJ_2019-06-02_03-59-19/"
)
nyq = 0.5 * 1250

sessionName = os.path.basename(os.path.normpath(basePath))
filename = basePath + sessionName + "_BestRippleChans.npy"

data = np.load(filename, allow_pickle=True)


signal = data.item()
signal = signal["BestChan"]

b, a = sg.butter(3, 20 / nyq, btype="lowpass")
yf = sg.filtfilt(b, a, signal)

sig_zscore = stat.zscore(signal)

plt.clf()
plt.subplot(211)
plt.plot(signal)

plt.subplot(212)
plt.plot(sig_zscore)
