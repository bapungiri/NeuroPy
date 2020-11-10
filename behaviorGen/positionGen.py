#%%
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as sg
import scipy.stats as stats
from scipy.stats.stats import pointbiserialr
import seaborn as sns
from scipy.ndimage import gaussian_filter

import signal_process
from callfunc import processData
from plotUtil import Colormap, Fig

cmap = matplotlib.cm.get_cmap("hot_r")
# warnings.simplefilter(action="default")


#%% Subjects
basePath = [
    # "/data/Clustering/SleepDeprivation/RatJ/Day1/",
    # "/data/Clustering/SleepDeprivation/RatK/Day1/",
    "/data/Clustering/SleepDeprivation/RatN/Day1/",
    # "/data/Clustering/SleepDeprivation/RatJ/Day2/",
    # "/data/Clustering/SleepDeprivation/RatK/Day2/",
    # "/data/Clustering/SleepDeprivation/RatN/Day2/",
    # "/data/Clustering/SleepDeprivation/RatK/Day4/"
]


sessions = [processData(_) for _ in basePath]

#%% Running epochs and direction of movement
# region

for sub, sess in enumerate(sessions):

    maze = sess.epochs.maze
    position = sess.position.data
    maze_pos = position[(position.time > maze[0]) & (position.time < maze[1])]


# endregion
