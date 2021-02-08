#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter
import subjects
import time

sessions = subjects.Tn().ratSday5

#%% Generate _basics.npy files
# region
for sub, sess in enumerate(sessions):
    # openephys_settingspath = (
    #     "/data/Clustering/SleepDeprivation/RatS/Day3SD/openEphysSettings/settings.xml"
    # )
    # sess.recinfo.generate_xml(settingsPath=openephys_settingspath)
    sess.recinfo.makerecinfo(nShanks=[6, 8], skulleeg=[51], motion=[192, 193, 194])
    # sess.recinfo.makerecinfo(nShanks=8)
# endregion

#%% Generate probemap and .prb for spyking circus
# region
for sub, sess in enumerate(sessions):

    # sess.trange = np.array([])
    sess.recinfo.probemap.create(xypitch=((16.5, 15), (16, 15)))
    # sess.recinfo.probemap.plot()
    sess.recinfo.probemap.for_spyking_circus(rmv_badchans=True, shanksCombine=True)

# endregion

#%% artifacts file gen
# region
for sub, sess in enumerate(sessions):
    # lfp = sess.recinfo.geteeg(chans=64)
    # plt.plot(np.linspace(0, len(lfp) / (1250 * 60), len(lfp)), lfp)
    zsc = sess.artifact.usingZscore(chans=[28, 64], thresh=7)
    sess.artifact.export2neuroscope()
    sess.artifact.plot()
# endregion

#%% Generate position files
# region
for sub, sess in enumerate(sessions):
    sess.trange = []
    sess.position.getPosition(method="from_metadata", scale=4.0)
    # sess.position.export2Neuroscope()

# endregion

#%% create epochs
# region
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
for sub, sess in enumerate(sessions):
    # sess.trange = np.array([])
    # sess.tracks.create(epoch_names=["maze"])
    # sess.tracks.linearize_position(track_names=["maze"], sample_sec=3)
    sess.tracks.estimate_run_laps("maze", speedthresh=(20, 30))
    # sess.epochs.getfromCSV()

# endregion


#%% Gen spikes or instantenous firing rate
# region
for sub, sess in enumerate(sessions):
    # sess.trange = np.array([])
    sess.recinfo.sampfreq = 30000
    sess.spikes.from_Phy(
        folder="/data/Clustering/SleepDeprivation/RatN/Day4/spykcirc/RatN_Day4_2019-10-15_11-30-06.GUI",
        fileformat="same_folder",
    )
    # sess.spikes.gen_instfiring()
# endregion

#%% label cells
# region
for sub, sess in enumerate(sessions):
    sess.recinfo.sampfreq = 30000
    sess.spikes.label_celltype()


# endregion
#%%Generate bestThetachan
# region

for sub, sess in enumerate(sessions):
    sess.trange = np.array([])
    sess.theta.detectBestChan()
# endregion
