#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import matplotlib
from collections import namedtuple
from pathlib import Path
import matplotlib.gridspec as gridspec
import signal_process
import matplotlib as mpl
from plotUtil import Colormap
import scipy.signal as sg
from ccg import correlograms
import warnings

cmap = matplotlib.cm.get_cmap("hot_r")
warnings.simplefilter(action="default")


from callfunc import processData

#%% Subjects
basePath = [
    "/data/Clustering/SleepDeprivation/RatJ/Day1/",
    "/data/Clustering/SleepDeprivation/RatK/Day1/",
    "/data/Clustering/SleepDeprivation/RatN/Day1/",
    "/data/Clustering/SleepDeprivation/RatJ/Day2/",
    "/data/Clustering/SleepDeprivation/RatK/Day2/",
    "/data/Clustering/SleepDeprivation/RatN/Day2/",
    "/data/Clustering/SleepDeprivation/RatK/Day4/",
    "/data/Clustering/SleepDeprivation/RatN/Day4/",
]


sessions = [processData(_) for _ in basePath]

#%% Plotting cell statistics
# region
plt.clf()
fig = plt.figure(1, figsize=(10, 15))
gs = gridspec.GridSpec(3, 3, figure=fig)
fig.subplots_adjust(hspace=0.3)
for sub, sess in enumerate(sessions):
    sess.trange = np.array([])
    # sess.spikes.stability.firingRate()
    stability = sess.spikes.stability.info
    stability.loc[stability.q < 4, "cellType"] = "pyr"
    stability.loc[stability.q == 6, "cellType"] = "mua"
    stability.loc[stability.q == 8, "cellType"] = "intneur"

    ax = fig.add_subplot(gs[sub])
    # stability.plot.bar(x="cellType", y="stable", stacked=True, ax=ax, rot=0)
    sns.countplot(
        x="cellType",
        hue="stable",
        data=stability,
        order=["pyr", "intneur", "mua"],
        ax=ax,
        palette="Pastel1",
    )

# endregion

#%% Plotting CCG for each cell with data like refractory peiod violation, quality etc
# region

# endregion

#%% Plot spike amplitude over time and CCG
# region
for sub, sess in enumerate(sessions):
    sess.trange = np.array([])

# endregion
