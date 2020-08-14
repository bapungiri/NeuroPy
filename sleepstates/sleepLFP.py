#%%
import warnings

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.ndimage import gaussian_filter

from callfunc import processData
from signal_process import spectrogramBands

warnings.simplefilter(action="default")


plt.clf()

basePath = [
    "/data/Clustering/SleepDeprivation/RatJ/Day1/",
    "/data/Clustering/SleepDeprivation/RatK/Day1/",
    "/data/Clustering/SleepDeprivation/RatN/Day1/",
    # "/data/Clustering/SleepDeprivation/RatJ/Day2/",
    # "/data/Clustering/SleepDeprivation/RatK/Day2/",
    # "/data/Clustering/SleepDeprivation/RatN/Day2/",
    # "/data/Clustering/SleepDeprivation/RatN/Day4/",
]


sessions = [processData(_) for _ in basePath]


#%% Plot spectrogram for different bands separately
# region
for sub, sess in enumerate(sessions):

    sess.trange = np.array([])

# endregion
