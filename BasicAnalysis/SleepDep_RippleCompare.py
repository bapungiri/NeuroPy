
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as smth
import scipy.fftpack as ft
import scipy.signal as sg
import scipy.stats as stat
import lfpDetect as lfpDetect
import SpectralAnalysis as spect


session1 = '/data/Clustering/SleepDeprivation/RatJ/Day1/RatJ_2019-05-31_03-55-36/experiment1/recording1/continuous/Rhythm_FPGA-100.0/continuous.eeg'
session2 = '/data/Clustering/SleepDeprivation/RatJ/Day2/RatJ_2019-06-02_03-59-19/experiment1/recording1/continuous/Rhythm_FPGA-100.0/continuous.eeg'

badchan = [0, 2, 5, 6, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74]
badchan = [x+1 for x in badchan]

bestChanSess1 = spect.bestRippleChannel(session1, 1250, 75, badchan)

RippleSess1, ex1 = lfpDetect.swr(session1, 33, 1250, 75)

thresh_hist = ex1['zscore_dist']
RippleSess2, ex2 = lfpDetect.swr(session2, 32, 1250, 67)

# x1 = np.linspace(125)

ripple_rate1, _ = np.histogram(RippleSess1[:, 0], 140)
ripple_rate2, _ = np.histogram(RippleSess2[:, 0], 140)

trig_loc = (RippleSess1[:, 0]/1250)*1000
# trig_loc = [[x, 'gh'] for x in trig_loc]
# f = open("test.evt", "a")
# np.savetxt('test.evt', trig_loc, newline='\n')

with open('test2.evt', 'w') as file:
    for year in trig_loc:
        file.write("%f gh\n" % year)


# # ex_ripple1 = [item for sublist in ex1 for item in sublist]
# # ex_ripple2 = [item for sublist in ex2 for item in sublist]


# # plt.figure(1)

# plt.subplot(2, 1, 1)
# plt.plot(ripple_rate1, 'r')
# plt.plot(ripple_rate2, 'k')
# # plt.ylabel('# Ripple')
# # plt.xlabel('time')


# # plt.figure(1)
# # plt.subplot(2, 1, 1)
# # plt.plot(ex_ripple1, 'r')

# # plt.subplot(2, 1, 2)
# # plt.plot(ex_ripple2, 'k')
# # plt.ylabel('# Ripple')
# plt.xlabel('time')
