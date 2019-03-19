#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 11:25:34 2019

@author: bapung
"""

import numpy as np
import pandas as pd
#from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
#from scipy.ndimage import gaussian_filter
from OsCheck import DataDirPath, figDirPath
#import scipy.signal as sg
#import scipy.stats as stats
#from scipy.signal import hilbert
import h5py
import seaborn as sns
#sns.set(style="darkgrid")


plt.style.use('seaborn')
        

sourceDir = DataDirPath() + 'sleep/'

arrays = {}
f= h5py.File(sourceDir + 'sleep-basics.mat', 'r') 
for k, v in f.items():
    arrays[k] = np.array(v)

# spks = {}
# fspikes= h5py.File(sourceDir + 'sleep-spikes.mat', 'r') 
fbehav= h5py.File(sourceDir + 'sleep-behavior.mat', 'r') 
fripple = h5py.File(sourceDir + 'sleep-ripple.mat', 'r') 
fspindle= h5py.File(sourceDir + 'sleep-spindle.mat', 'r') 
#fICAStrength = h5py.File('/data/DataGen/ICAStrengthpy.mat', 'r') 

subjects = arrays['basics']
#spikes = spks['spikes']

#figFilename = figDirPath() +'PlaceCells.pdf'
#pdf = PdfPages(figFilename)
k=1
plt.clf()
for sub in range(0,19):
    sub_name = subjects[sub]
    print(sub_name)
    
    area = list((fspindle['spindle'][sub_name]).keys())
    
    if len(area)>1 :
        pfcSpindle = np.transpose(fspindle['spindle'][sub_name]['CTX']['peakTime'][:])
        hpcSpindle = np.transpose(fspindle['spindle'][sub_name]['HPC']['peakTime'][:])
        
        diffTime = ((pfcSpindle.transpose() - hpcSpindle).ravel())/1e6
        
        plt.subplot(2,4,k)
        hist, edge = np.histogram(diffTime, bins= np.arange(-20,20,1))
        plt.plot(edge[0:len(hist)], hist)
        plt.title(sub_name)
        k = k+1

plt.show()
        
    