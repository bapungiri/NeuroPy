#%% Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter

from callfunc import processData

basePath = [
    # "/data/Clustering/SleepDeprivation/RatJ/Day1/",
    # "/data/Clustering/SleepDeprivation/RatK/Day1/",
    # "/data/Clustering/SleepDeprivation/RatN/Day1/",
    # "/data/Clustering/SleepDeprivation/RatJ/Day2/",
    # "/data/Clustering/SleepDeprivation/RatK/Day2/",
    # "/data/Clustering/SleepDeprivation/RatN/Day2/",
    # "/data/Clustering/SleepDeprivation/RatK/Day4/",
    "/data/Clustering/SleepDeprivation/RatN/Day4/",
]


sessions = [processData(_) for _ in basePath]

templates = "/data/Clustering/SleepDeprivation/RatK/Day1/spykcirc/RatK_Day1_2019-08-06_03-44-01/RatK_Day1_2019-08-06_03-44-01.GUI/templates.npy"

template_ind = "/data/Clustering/SleepDeprivation/RatK/Day1/spykcirc/RatK_Day1_2019-08-06_03-44-01/RatK_Day1_2019-08-06_03-44-01.GUI/template_ind.npy"
chan_positions = "/data/Clustering/SleepDeprivation/RatK/Day1/spykcirc/RatK_Day1_2019-08-06_03-44-01/RatK_Day1_2019-08-06_03-44-01.GUI/channel_positions.npy"

templates = np.load(templates)
template_ind = np.load(template_ind)
chan_pos = np.load(chan_positions)
