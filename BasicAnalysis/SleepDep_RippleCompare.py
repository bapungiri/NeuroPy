
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as smth
import scipy.fftpack as ft
import scipy.signal as sg
import scipy.stats as stat
import lfpDetect as lfpDetect


session1 = '/data/Clustering/SleepDeprivation/RatJ/Day1/RatJ_2019-05-31_03-55-36/experiment1/recording1/continuous/Rhythm_FPGA-100.0/continuous.eeg'
session2 = '/data/Clustering/SleepDeprivation/RatJ/Day2/RatJ_2019-06-02_03-59-19/experiment1/recording1/continuous/Rhythm_FPGA-100.0/continuous.eeg'


RippleSess1 = lfpDetect.swr(session1, 32, 1250, 75)
# RippleSess2 = lfpDetect.swr(session2, 32, 1250, 67)
