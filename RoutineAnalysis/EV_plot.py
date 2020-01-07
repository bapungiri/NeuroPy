import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

basePath = "/data/Clustering/SleepDeprivation/RatN/Day2/"

for file in os.listdir(basePath):
    if file.endswith(".eeg"):

        subname = file[:-4]
        print(subname)
        fileInitial = basePath + subname
        # print(os.path.join(basePath, file))

expVar = np.load(fileInitial + "_EV.npy")
# rev = np.load(fileInitial + "_REV.npy")
states = np.load(fileInitial + "_behavior.npy")
epochs = np.load(basePath + "epochs.npy", allow_pickle=True)

pre = epochs.item().get("PRE")  # in seconds
maze = epochs.item().get("MAZE")  # in seconds
post = epochs.item().get("POST")  # in seconds

t = np.linspace(post[0], post[1], len(expVar))

# plt.clf()
# plt.plot(t / 3600, expVar)


plt.close("all")
fig, ax = plt.subplots()
patches = []
num_polygons = 5
num_sides = 5

states_post = states[states[:, 0] > post[0], :]

color_states = []

for i in range(len(states_post)):
    a = states_post[i, 0] / 3600
    b = 0.0
    c = states_post[i, 1] / 3600 - states_post[i, 0] / 3600
    polygon = Rectangle((a, b), c, 0.35)
    patches.append(polygon)

    if states_post[i, 2] == 0:
        col = [0.3, 0.3, 0.5]
    if states_post[i, 2] == 1:
        col = [1, 1, 1]
    if states_post[i, 2] == 2:
        col = [1, 1, 1]
    if states_post[i, 2] == 3:
        col = [1, 1, 1]

    color_states.append(col)

p = PatchCollection(patches, facecolor=color_states, alpha=0.4)

# colors = 100 * np.random.rand(len(patches))
# # colors = color_states
# p.set_array(np.array(colors))

ax.add_collection(p)
plt.plot(t / 3600, expVar)

