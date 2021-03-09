#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter
from ccg import correlograms
from mathutil import getICA_Assembly
from callfunc import processData
import subjects
from plotUtil import Fig
import networkx as nx

#%% Network analysis of sleep deprivation
# region
figure = Fig()
fig, gs = figure.draw(num=1, grid=(2, 2))
sessions = subjects.Sd().ratSday3
for sub, sess in enumerate(sessions):
    period = sess.epochs.sd
    sd_windows = np.arange(period[0], period[1] + 3600, 3600)
    pyr = sess.spikes.pyr

    for window in sd_windows:
        bins = np.arange(window, window + 3600, 0.25)
        spk_counts = np.asarray([np.histogram(cell, bins=bins)[0] for cell in pyr])
        corr = np.corrcoef(spk_counts)


# endregion