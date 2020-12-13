#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter
import subjects
import time

#%% Generate _basics.npy files
# region
sessions = subjects.sd([3])
for sub, sess in enumerate(sessions):

    # sess.trange = np.array([])
    # badchans = [0]
    # openephys_settingspath = (
    #     "/data/Clustering/SleepDeprivation/RatS/Day3SD/openEphysSettings/settings.xml"
    # )
    # sess.recinfo.generate_xml(settingsPath=openephys_settingspath)
    sess.recinfo.makerecinfo(nShanks=[6, 8], skulleeg=[51], motion=[192, 193, 194])
# endregion

#%% Generate .prb for spyking circus
# region
sessions = subjects.sd([3])
for sub, sess in enumerate(sessions):

    sess.trange = np.array([])
    # sess.recinfo.probemap.create(xypitch=((16.5, 15), (16, 15)))
    sess.recinfo.probemap.for_spyking_circus(rmv_badchans=True, shanksCombine=1)

# endregion

#%% artifacts file gen
# region
sessions = subjects.sd([3])
for sub, sess in enumerate(sessions):

    zsc = sess.artifact.usingZscore(chans=10, thresh=4.5)
    # sess.artifact.plot(chan=10)
# endregion

#%% Generate position files
# region
for sub, sess in enumerate(sessions):
    sess.trange = np.array([])
    sess.position.getPosition()
    # sess.position.export2Neuroscope()

# endregion

#%% epochs from csv
# region
for sub, sess in enumerate(sessions):
    sess.trange = np.array([])
    sess.epochs.getfromCSV()
# endregion

#%% Gen instantenous firing rate
# region

for sub, sess in enumerate(sessions):
    sess.trange = np.array([])
    sess.spikes.gen_instfiring()
# endregion

#%%Generate bestThetachan
# region

for sub, sess in enumerate(sessions):
    sess.trange = np.array([])
    sess.theta.detectBestChan()
# endregion
