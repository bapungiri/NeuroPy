#%%

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import subjects
from callfunc import processData
from plotUtil import Colormap

#
#%% Ripples
# region
# plt.clf()

# fig, ax = plt.subplots(figsize=(5, 3))
sessions = subjects.sd([3])
for sub, sess in enumerate(sessions):

    # sess.ripple.channels()
    # sess.ripple.detect()
    # ripples = sess.ripple.events
    # sess.ripple.export2Neuroscope()
    sess.ripple.plot()
#
# ax.set_xlim([-5, 10])
# endregion

#%% Spindles
# region
sessions = subjects.sd([3])
for sub, sess in enumerate(sessions):

    sess.trange = np.array([])
    sess.spindle.channels()
    sess.spindle.detect()
    # sess.spindle.plot()

# ax.set_xlim([-5, 10])
# endregion

#%% H-SWA
# region
plt.clf()
for sub, sess in enumerate(sessions):

    sess.trange = np.array([])

    sess.swa.detect()
    # sess.swa.plot()
# endregion

#%% Best Theta channel based on
# region
for sub, sess in enumerate(sessions):

    sess.trange = np.array([])
    # sess.theta.detectBestThetaChan()
# endregion
