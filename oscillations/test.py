# %%
import os
import random
import warnings
from typing import Dict

# import ipywidgets as widgets
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import scipy.signal as sg
import scipy.stats as stats
import seaborn as sns
import signal_process

from callfunc import processData
from mathutil import threshPeriods
from plotUtil import Colormap, Fig
from scipy import fft, interpolate
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from signal_process import bicoherence
from sklearn import linear_model
from tables.description import Col

# from artifactDetect import findartifact

# warnings.simplefilter(action="default")
#%% Subjects to choose from
basePath = [
    "/data/Clustering/SleepDeprivation/RatJ/Day1/",
    # "/data/Clustering/SleepDeprivation/RatK/Day1/",
    # "/data/Clustering/SleepDeprivation/RatN/Day1/",
    # "/data/Clustering/SleepDeprivation/RatJ/Day2/",
    # "/data/Clustering/SleepDeprivation/RatK/Day2/",
    # "/data/Clustering/SleepDeprivation/RatN/Day2/",
    # "/data/Clustering/SleepDeprivation/RatJ/Day4/",
    # "/data/Clustering/SleepDeprivation/RatK/Day4/",
    # "/data/Clustering/SleepDeprivation/RatN/Day4/",
    # "/data/Clustering/SleepDeprivation/RatA14d1LP/Rollipram/",
]
sessions = [processData(_) for _ in basePath]

