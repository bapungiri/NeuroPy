# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 10:00:33 2019

@author: Bapun
"""



import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
#import scipy.signal as sg
#import scipy.stats as stats
#from scipy.signal import hilbert
import h5py
import matplotlib as mpl

#
#mpl.rc('axes', linewidth=1.5)
#mpl.rc('font', size = 9)
#mpl.rc('figure', figsize = (10, 14))
#mpl.rc('axes.spines', top=False, right=False)


sourceDir = '/data/DataGen/wake_new/'

arrays = {}
f= h5py.File(sourceDir + 'wake-basics.mat', 'r') 
for k, v in f.items():
    arrays[k] = np.array(v)

#spks = {}
fspikes= h5py.File(sourceDir + 'testVersion.mat', 'r') 
fbehav= h5py.File(sourceDir + 'wake-behavior.mat', 'r') 
fpos= h5py.File(sourceDir + 'wake-position.mat', 'r') 
#fICAStrength = h5py.File('/data/DataGen/ICAStrengthpy.mat', 'r') 

subjects = arrays['basics']
#spikes = spks['spikes']
plt.clf()
for sub in range(0,7):
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
    cellpyr= [celltype[a] for a in pyrid]
    
    behav = np.transpose(fbehav['behavior'][sub_name]['time'][:])
    states = np.transpose(fbehav['behavior'][sub_name]['list'][:])
    posx = (fpos['position'][sub_name]['x'][:])
    posy = (fpos['position'][sub_name]['y'][:])
    post = (fpos['position'][sub_name]['t'][:])
    
    
    
    posx_mz = posx[np.where((post > behav[1,0]) & (post < behav[1,1]))] 
    posy_mz = posy[np.where((post > behav[1,0]) & (post < behav[1,1]))] 
    post_mz = post[np.where((post > behav[1,0]) & (post < behav[1,1]))]
    
    xcoord = np.arange(min(posx_mz),max(posx_mz)+1,2)
    ycoord = np.arange(min(posy_mz),max(posy_mz)+1,2)
    xx,yy = np.meshgrid(xcoord,ycoord)
    
#    flat_list = [item for sublist in cellpyr for item in sublist]
    f2 = np.concatenate(cellpyr, axis =0)
    
    
    for cell in [0]:
        spkt = f2.squeeze()
        spktx = np.interp(spkt,post_mz,posx_mz)
        spkty = np.interp(spkt,post_mz,posy_mz)
            
        pf, xe, ye = np.histogram2d(spktx,spkty,bins = [xcoord,ycoord])
        pft = pf*(1/30)
        
        eps = np.spacing(1)
        pfRate = pf/(pft+eps)
        
        pfRate_smooth= gaussian_filter(pfRate, sigma=2)

#    ICA_strength = np.array(np.transpose(fICAStrength['ActStrength']['subjects'][sub_name]['wake'][:]))

    
        
    #    plt.plot(posx_mz,posy_mz,'.')
    plt.subplot(3,3,sub+1)
    plt.imshow(pfRate_smooth,vmin = 2,vmax = 38, cmap='jet')
    
    plt.xticks([])
    plt.yticks([])
#        plt.subplot(1,2,2)
#        plt.imshow(pfRate_smooth)
    plt.title(sub_name)
    