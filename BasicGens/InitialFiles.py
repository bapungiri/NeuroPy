#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter
import subjects


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
    sess.recinfo.makerecinfo(nShanks=8, skulleeg=[51], motion=[192, 193, 194])
# endregion

#%% Generate .prb for spyking circus
# region
for sub, sess in enumerate(sessions):

    sess.trange = np.array([])
    sess.makePrmPrb.makePrbCircus(probetype="diagbio", shanksCombine=1)

# endregion

#%% artifacts file gen
# region
for sub, sess in enumerate(sessions):

    sess.trange = np.array([])
    # sess.artifact.usingZscore(thresh=10)
    sess.artifact.plot(chan=32)
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
