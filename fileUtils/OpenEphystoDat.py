#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 11:12:30 2019

@author: bapung
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.ndimage.filters import gaussian_filter1d
from OsCheck import DataDirPath, figDirPath, RawDataPath
# import scipy.signal as sg
# import scipy.stats as stats
# from scipy.signal import hilbert
from SpectralAnalysis import lfpSpectMaze
import h5py
import seaborn as sns


sourceDir = RawDataPath() + 'SleepDeprivation/Beatrice/MAZE/2019-04-22_07-33-15/'
filename = sourceDir + '100_CH1.continuous'

file = open('filename','r')
