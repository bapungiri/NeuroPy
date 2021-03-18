#%%
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

#%% Running epochs and direction of movement
# region
sessions = subjects.sd([2])
# se = subjects.sleepDep().allsess
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


#%% Position heatmap before and after sprinkles
# region
figure = Fig()
fig, gs = figure.draw(num=1, grid=(2, 2))
sessions = subjects.Of().ratNday4
for sub, sess in enumerate(sessions):
    maze = sess.epochs.maze
    sprinkle = sess.epochs.sprinkle
    pre_sprinkle = [maze[0], sprinkle[0]]
    mazepos = sess.tracks["maze"]
    pre_pos = mazepos[mazepos.time < sprinkle[0]]
    sprinkle_pos = mazepos[mazepos.time > sprinkle[0]]

    x = mazepos.x
    y = mazepos.y
    xgrid = np.arange(np.min(x), np.max(x), 3)
    ygrid = np.arange(np.min(y), np.max(y), 3)
    x, y = np.meshgrid(xgrid, ygrid)

    posmap_pre = np.histogram2d(pre_pos.x, pre_pos.y, bins=[xgrid, ygrid])[0]
    posmap_post = np.histogram2d(sprinkle_pos.x, sprinkle_pos.y, bins=[xgrid, ygrid])[0]

    ax = plt.subplot(gs[0])
    # ax.plot(pre_pos.x, pre_pos.y, "k")
    ax.pcolormesh(gaussian_filter(posmap_pre, sigma=0.8), cmap="hot", vmin=0, vmax=400)
    ax.axis("off")

    ax = plt.subplot(gs[1])
    # ax.plot(sprinkle_pos.x, sprinkle_pos.y, "k")
    ax.pcolormesh(gaussian_filter(posmap_post, sigma=0.8), cmap="hot", vmin=0, vmax=400)
    ax.axis("off")

    # sess.placefield.pf2d.compute(period=maze)
    sess.placefield.pf2d.plotMap()
    sess.placefield.pf2d.plotRaw()
# endregion