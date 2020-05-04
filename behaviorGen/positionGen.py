import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter

from mathutil import getICA_Assembly
from callfunc import processData

basePath = [
    # "/data/Clustering/SleepDeprivation/RatJ/Day1/",
    # "/data/Clustering/SleepDeprivation/RatK/Day1/",
    # "/data/Clustering/SleepDeprivation/RatN/Day1/",
    # "/data/Clustering/SleepDeprivation/RatN/Day1/",
    # "/data/Clustering/SleepDeprivation/RatJ/Day2/",
    # "/data/Clustering/SleepDeprivation/RatK/Day2/",
    # "/data/Clustering/SleepDeprivation/RatN/Day2/",
    "/data/Clustering/SleepDeprivation/RatK/Day4/"
]


sessions = [processData(_) for _ in basePath]

# a = np.arange(16).reshape(4, 4)
# V, W = getICA_Assembly(a)

for sub, sess in enumerate(sessions):
    # sess.makePrmPrb.makePrbCircus("diagbio")
    sess.trange = np.array([])
    # sess.position.getPosition()
    # sess.position.plot()
    # sess.epochs.getfromPosition()
    # sess.spikes.extract()
    # sess.spikes.stability.firingRate()
    # sess.spikes.stability.refPeriodViolation()
    # sess.replay.assemblyICA()


# sess.brainstates.detect()

# violations = sess.spikes.stability.violations
