#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 10:00:47 2019

@author: bapung
"""


import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from OsCheck import DataDirPath, figDirPath
#import scipy.signal as sg
#import scipy.stats as stats
#from scipy.signal import hilbert
import h5py
plt.style.use('seaborn')
        

sourceDir = DataDirPath() + 'wake_new/'

arrays = {}
f= h5py.File(sourceDir + 'wake-basics.mat', 'r') 
for k, v in f.items():
    arrays[k] = np.array(v)

#spks = {}
fspikes= h5py.File(sourceDir + 'testVersion.mat', 'r') 
fbehav= h5py.File(sourceDir + 'wake-behavior.mat', 'r') 
fpos= h5py.File(sourceDir + 'wake-position.mat', 'r') 
fspeed= h5py.File(sourceDir + 'wake-speed.mat', 'r') 
#fICAStrength = h5py.File('/data/DataGen/ICAStrengthpy.mat', 'r') 

subjects = arrays['basics']
#spikes = spks['spikes']

#figFilename = figDirPath() +'PlaceCells.pdf'
#pdf = PdfPages(figFilename)

for sub in [1,2,3,4,5,6]:
    sub_name = subjects[sub]
    print(sub_name)
    
    nUnits = len(fspikes['spikes'][sub_name]['time'])
    celltype={}
    quality={}
    stability={}
    for i in range(0,nUnits):
        celltype[i] = fspikes[fspikes['spikes'][sub_name]['time'][i,0]].value
        quality[i] = fspikes[fspikes['spikes'][sub_name]['quality'][i,0]].value
        stability[i] = fspikes[fspikes['spikes'][sub_name]['StablePrePost'][i,0]].value
    
    
    pyrid = [i for i in range(0,nUnits) if quality[i] < 4 and stability[i] == 1]
    nPyr = len(pyrid)
    cellpyr= [celltype[a] for a in pyrid]

    
    behav = np.transpose(fbehav['behavior'][sub_name]['time'][:])
    states = np.transpose(fbehav['behavior'][sub_name]['list'][:])

    rem_post = states[(states[:,0] > behav[2,0]) & (states[:,2]==2) & (states[:,1]-states[:,0] > 250e3),0:2]
    
    Bins = np.arange(behav[1,0], behav[1,1], 250e3)    
    spkCnt = [np.histogram(cellpyr[x], bins = Bins)[0] for x in range(0,len(cellpyr))]
    corr_mat = np.corrcoef(spkCnt)
    np.fill_diagonal(corr_mat,0)
    
    mean_remcorr = []
    for rem_epoch in range(0,len(rem_post)):
        bin_rem = np.arange(rem_post[rem_epoch,0], rem_post[rem_epoch,1], 250e3)    
        spkCnt_rem = [np.histogram(cellpyr[x], bins = bin_rem)[0] for x in range(0,len(cellpyr))]
        rem_corr = np.corrcoef(spkCnt_rem)
        rem_corr = rem_corr[np.tril_indices(len(cellpyr), -1)]
        mean_remcorr.append(np.nanmean(rem_corr))
        
        
    plt.subplot(3,3,sub+1)
    plt.bar(np.arange(1,len(rem_post)+1,1),height=mean_remcorr)
        
        
#    plt.xticks([])
#    plt.yticks([])

    plt.title(sub_name)
#    plt.tight_layout()
#    pdf.savefig(dpi=300)

#pdf.close()    
    