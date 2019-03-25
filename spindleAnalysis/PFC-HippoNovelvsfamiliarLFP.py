
import numpy as np
import pandas as pd
# rom matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.ndimage.filters import gaussian_filter1d
from OsCheck import DataDirPath, figDirPath
# import scipy.signal as sg
# import scipy.stats as stats
# from scipy.signal import hilbert
from SpectralAnalysis import lfpSpectMaze
import h5py
import seaborn as sns
# sns.set(style="darkgrid")


sourceDir = '/data/DataGen/wake_new/'
sourceDir2 = '/data/DataGen/sleep/'

arrays = {}
f = h5py.File(sourceDir + 'wake-basics.mat', 'r')
for k, v in f.items():
    arrays[k] = np.array(v)

fspikes = h5py.File(sourceDir + 'testVersion.mat', 'r')
fbehav = h5py.File(sourceDir + 'wake-behavior.mat', 'r')
# slpbehav = h5py.File(sourceDir2 + 'wake-behavior.mat')
fpos = h5py.File(sourceDir + 'wake-position.mat')

# savech = np.load(sourceDir2 + 'sleepPy-behavior')


subjects = arrays['basics']

for sub in [5]:
    sub_name = subjects[sub]
    print(sub_name)

    nUnits = len(fspikes['spikes'][sub_name]['time'])
    celltype = {}
    quality = {}
    stability = {}
    for i in range(0, nUnits):
        celltype[i] = fspikes[fspikes['spikes'][sub_name]['time'][i, 0]].value
        quality[i] = fspikes[fspikes['spikes']
                             [sub_name]['quality'][i, 0]].value
        stability[i] = fspikes[fspikes['spikes']
                               [sub_name]['StablePrePost'][i, 0]].value

    behav = np.transpose(fbehav['behavior'][sub_name]['time'][:])
    states = np.transpose(fbehav['behavior'][sub_name]['list'][:])
    frames = np.transpose(fbehav['behavior'][sub_name]['eegFrame'][:])
    posx = (fpos['position'][sub_name]['x'][:])
    posy = (fpos['position'][sub_name]['y'][:])
    post = (fpos['position'][sub_name]['t'][:])


    pyrid = [i for i in range(0, nUnits)
             if quality[i] < 4 and stability[i] == 1]
    cellpyr = [celltype[a] for a in pyrid]

    posx_mz = posx[np.where((post > behav[1,0]) & (post < behav[1,1]))]
    posy_mz = posy[np.where((post > behav[1,0]) & (post < behav[1,1]))]
    post_mz = post[np.where((post > behav[1,0]) & (post < behav[1,1]))]

    y_thresh = 100
    posy_mz = posy_mz - y_thresh
    pos_novel = np.where(posy_mz >0, 1, 0)
    pos_novel = np.diff(pos_novel)
    nov_st = post_mz[np.where(pos_novel ==1)]
    nov_end = post_mz[np.where(pos_novel ==-1)]
    nov_period = np.column_stack((nov_st[0:len(nov_st)-1],nov_end))
    fmlr_period = np.column_stack((nov_end,nov_st[1:len(nov_st)]))

    nov_period = nov_period[nov_period[:,1]-nov_period[:,0] > 5e6, :]
    fmlr_period = fmlr_period[fmlr_period[:,1]-fmlr_period[:,0] > 5e6, :]



    # sleepPeriods = (
    #     (slpbehav['behavior'][sub_name.replace('Maze', 'Sleep')]['list']).value).T
    # slpNrem = np.where((sleepPeriods[:, 2] == 1) & (
    #     sleepPeriods[:, 1] < behav[2, 0] + 10 * 3600e6))[0]
    # lastNrem = sleepPeriods[slpNrem[-1], 0:2]

    BasicInfo = {'samplingFrequency': 1280}
    BasicInfo['behavFrames'] = frames
    BasicInfo['behav'] = behav
    BasicInfo['numChannels'] = 66
    BasicInfo['SpectralChannel'] = 66

    nMazeFrames = int(np.diff(frames[2, :]))
    MAZE = states[(states[:, 0] > behav[1, 0]) & (states[:, 2] == 4), :]

    nov = []
    for i in range(0, len(nov_period)):
        y1, xf = lfpSpectMaze(sub_name, nov_period[i, 0], BasicInfo, channel=66)
        nov.append(y1)
    fam= []
    for i in range(0, len(fmlr_period)):
        y2, xL = lfpSpectMaze(sub_name, fmlr_period[i, 0], BasicInfo, channel=66)
        fam.append(y2)

    nov_hipp = []
    for i in range(0, len(nov_period)):
        y1, xf = lfpSpectMaze(sub_name, nov_period[i, 0], BasicInfo, channel=55)
        nov_hipp.append(y1)
    fam_hipp= []
    for i in range(0, len(fmlr_period)):
        y2, xL = lfpSpectMaze(sub_name, fmlr_period[i, 0], BasicInfo, channel=55)
        fam_hipp.append(y2)


    nov_mean = np.mean(nov,axis=0)
    fam_mean = np.mean(fam,axis=0)
    nov_hipp_mean = np.mean(nov_hipp,axis=0)
    fam_hipp_mean = np.mean(fam_hipp,axis=0)
#    fig, ax = plt.subplots()
    plt.clf()
    ax0 = plt.subplot(1, 1, 1)
    plt.plot(xf, nov_mean, label='pfc-novel')
    plt.plot(xL, fam_mean, 'r', label='pfc-familiar')
    plt.plot(xf, nov_hipp_mean, 'g',label='hpc-novel')
    plt.plot(xL, fam_hipp_mean, 'k', label='hpc-familiar')
#    plt.plot(post_mz, pos_novel)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power (db)')
    plt.yscale('log')
    plt.xlim(0.5, 100)
    plt.legend()
    ax0.spines['right'].set_visible(False)
    ax0.spines['top'].set_visible(False)
