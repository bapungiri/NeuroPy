
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
fpos = h5py.File(sourceDir + 'wake-pos.mat')

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
    pyrid = [i for i in range(0, nUnits)
             if quality[i] < 4 and stability[i] == 1]
    cellpyr = [celltype[a] for a in pyrid]

    # sleepPeriods = (
    #     (slpbehav['behavior'][sub_name.replace('Maze', 'Sleep')]['list']).value).T
    # slpNrem = np.where((sleepPeriods[:, 2] == 1) & (
    #     sleepPeriods[:, 1] < behav[2, 0] + 10 * 3600e6))[0]
    # lastNrem = sleepPeriods[slpNrem[-1], 0:2]

    BasicInfo = {'samplingFrequency': 1250}
    BasicInfo['behavFrames'] = frames
    BasicInfo['behav'] = behav
    BasicInfo['numChannels'] = 66
    BasicInfo['SpectralChannel'] = 66

    nMazeFrames = int(np.diff(frames[2, :]))
    MAZE = states[(states[:, 0] > behav[1, 0]) & (states[:, 2] == 4), :]

    y1, xf = lfpSpectMaze(sub_name, MAZE[3, 0], BasicInfo, channel=66)
    y2, xL = lfpSpectMaze(sub_name, MAZE[8, 0], BasicInfo, channel=50)

#    fig, ax = plt.subplots()
    plt.clf()
    ax0 = plt.subplot(1, 1, 1)
    plt.plot(xf, y1, label='Novel part')
    plt.plot(xL, y2, 'r', label='Familiar part')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power (db)')
    plt.yscale('log')
    plt.xlim(0.5, 100)
    plt.legend()
    ax0.spines['right'].set_visible(False)
    ax0.spines['top'].set_visible(False)
