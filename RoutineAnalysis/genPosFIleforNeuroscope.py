import numpy as np
import os
from time import sleep

basePath = "/data/Clustering/SleepDeprivation/RatJ/Day1/"

for file in os.listdir(basePath):
    if file.endswith(".eeg"):
        subname = file[:-4]
        filename = os.path.join(basePath, file)
        filePrefix = os.path.join(basePath, file[:-4])

nchans = 75
srate = 30000
ripples = np.load(filePrefix + "_ripples.npy", allow_pickle=True)
basics = np.load(filePrefix + "_basics.npy", allow_pickle=True)

data = np.memmap(filePrefix + ".eeg", dtype="int16", mode="r")

nframes = len(data)
position = np.load(filePrefix + "_position.npy", allow_pickle=True)
xcoord = position.item().get("X")
ycoord = position.item().get("Y")
time = position.item().get("time")

# t_eeg = np.linspace(0, nframes / 1250, nframes)

# neuroscope only displays positive values so translating the coordinates
xcoord = xcoord + abs(min(xcoord))
ycoord = ycoord + abs(min(ycoord))

with open(filePrefix + ".pos", "w") as f:
    # f.write(str(5.3) + " " + str(6.3))
    for x, y in zip(xcoord, ycoord):
        f.write(f"{x} {y}/n")
        # f.write("{} {}\n".format(x + 540, y + 650))
        # sleep(0.001)
# if a >= b:
#     a  5
