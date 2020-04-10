import numpy as np
from callfunc import processData
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.cluster import KMeans
from hmmlearn.hmm import GaussianHMM
import scipy.stats as stats
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter
from sklearn import preprocessing
from mpl_toolkits.mplot3d import Axes3D


basePath = [
    # "/data/Clustering/SleepDeprivation/RatJ/Day1/",
    # "/data/Clustering/SleepDeprivation/RatK/Day1/",
    "/data/Clustering/SleepDeprivation/RatN/Day1/",
    # "/data/Clustering/SleepDeprivation/RatJ/Day2/",
    # "/data/Clustering/SleepDeprivation/RatK/Day2/",
    # "/data/Clustering/SleepDeprivation/RatN/Day2/",
]


sessions = [processData(_) for _ in basePath]


for sub, sess in enumerate(sessions):

    # sess.recinfo.makerecinfo()
    sess.trange = np.array([])
    sess.brainstates.detect()


df = sess.brainstates.params
states = sess.brainstates.states
t = df["time"]
a = df.values[:, :8]

min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(a)
# df = pd.DataFrame(x_scaled)

plt.clf()
fig = plt.figure(1, figsize=(6, 10))
gs = GridSpec(2, 1, figure=fig)
fig.subplots_adjust(hspace=0.4)

x = x_scaled[:, 2]
y = x_scaled[:, 1]
z = x_scaled[:, -1]


ax1 = fig.add_subplot(gs[0, 0], projection="3d")
ax1.scatter(x, y, z, s=0.5)


ax2 = fig.add_subplot(gs[1, 0])
ax2.scatter(x, y, s=0.5)
