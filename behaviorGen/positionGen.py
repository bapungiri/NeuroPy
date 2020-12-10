#%%
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as sg
import scipy.stats as stats
import seaborn as sns
from scipy.ndimage import gaussian_filter

import signal_process
import subjects
from plotUtil import Colormap, Fig

# warnings.simplefilter(action="default")

#%% Running epochs and direction of movement
# region

for sub, sess in enumerate(sessions):

    maze = sess.epochs.maze
    position = sess.position.data
    maze_pos = position[(position.time > maze[0]) & (position.time < maze[1])]

# endregion

#%% Accerlerometer data visualization
# region
sessions = subjects.sd([3])

for sess in sessions:
    motionChans = sess.recinfo.motionChans
    data = sess.recinfo.geteeg(chans=motionChans)

# endregion