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
sessions = subjects.Sd().ratNday1
for sub, sess in enumerate(sessions):

    sess.trange = np.array([])
    # openephys_settingspath = (
    #     "/data/Clustering/SleepDeprivation/RatS/Day3SD/openEphysSettings/settings.xml"
    # )
    # sess.recinfo.generate_xml(settingsPath=openephys_settingspath)
    sess.recinfo.makerecinfo(nShanks=8)
    # sess.recinfo.makerecinfo(nShanks=8)
# endregion

#%% Generate probemap and .prb for spyking circus
# region
sessions = subjects.Nsd().ratKday2
for sub, sess in enumerate(sessions):

    # sess.trange = np.array([])
    # sess.recinfo.probemap.create(xypitch=((16.5, 15), (16, 15)))
    # sess.recinfo.probemap.plot()
    sess.recinfo.probemap.for_spyking_circus(rmv_badchans=True, shanksCombine=True)

# endregion

#%% artifacts file gen
# region
sessions = subjects.Nsd().ratSday2
for sub, sess in enumerate(sessions):
    # lfp = sess.recinfo.geteeg(chans=64)
    # plt.plot(np.linspace(0, len(lfp) / (1250 * 60), len(lfp)), lfp)
    zsc = sess.artifact.usingZscore(chans=64, thresh=6)
    sess.artifact.plot(chan=64)
# endregion

#%% Generate position files
# region
sessions = subjects.Nsd()
for sub, sess in enumerate(sessions):
    sess.trange = []
    sess.position.getPosition(method="from_metadata", scale=4.0)
    # sess.position.export2Neuroscope()

# endregion

#%% create epochs
# region
sessions = subjects.Nsd().ratSday2
for sub, sess in enumerate(sessions):
    sess.trange = np.array([])
    epochs = {
        "pre": [0, 2358],
        "maze1": [2360, 6031],
        # "sd": [6033, 14818 + 5 * 3600],
        "post": [6033, 36899],
        "maze2": [36902, 40233],
    }
    sess.epochs.make_epochs(new_epochs=epochs)
    # sess.epochs.getfromCSV()

# endregion

#%% create Track
# region
sessions = subjects.sd([3])
for sub, sess in enumerate(sessions):
    sess.trange = np.array([])
    # sess.tracks.create(epoch_names=["maze"])
    # sess.tracks.linearize_position(track_name=["maze2"], sample_sec=3)
    # sess.tracks.estimate_run_laps("maze", speedthresh=(20, 30))
    # sess.epochs.getfromCSV()

# endregion


#%% Gen spikes or instantenous firing rate
# region
sessions = subjects.Sd().ratJday1
for sub, sess in enumerate(sessions):
    # sess.trange = np.array([])
    sess.recinfo.sampfreq = 30000
    sess.spikes.from_Phy(
        folder="/data/Clustering/SleepDeprivation/RatJ/Day1/spykcirc/clus_combined",
        fileformat="diff_folder",
    )
    # sess.spikes.gen_instfiring()
# endregion

#%%Generate bestThetachan
# region

for sub, sess in enumerate(sessions):
    sess.trange = np.array([])
    sess.theta.detectBestChan()
# endregion
