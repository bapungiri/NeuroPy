import numpy as np
import os
from pathlib import Path as pth
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET


basePath = "/data/Clustering/SleepDeprivation/RatJ/Day1/"
clupath = pth(basePath, "Shank5", "RatJDay1_Shank5.clu.1")
clu = []
with open(clupath) as f:

    for line in f:
        clu.append(int(line))

num_clust = clu[0]
clust_id = np.unique(clu[1:])
spk = clu[1:]

spklist = []
for clus in clust_id:
    spklist.append(np.argwhere(spk == clus) / 30000)
    # spklist.append([i for i in range(len(spk)) if spk[i] == clus])


filename = pth(basePath, "Shank5", "RatJDay1_Shank5.xml")
myroot = ET.parse(filename).getroot()

chan_session = []
clus_pyr = []

for x in myroot.findall("units"):
    for y in x.findall("unit"):
        for z, z1 in zip(y.findall("quality"), y.findall("cluster")):

            if z.text is not None:
                if (z.text).isdigit():

                    clus_pyr.append([int(z.text), int(z1.text)])


# for ind, cell in enumerate(spklist):
#     plt.plot(cell, ind * np.ones(len(cell)), ".")
#     m = "".join(line)
# class ExtractfromClu:
#     def __init__(self, basePath):
#         # self.sessionName = os.path.basename(os.path.normpath(basePath))
#         self.sessionName = basePath.split("/")[-2] + basePath.split("/")[-1]
#         self.basePath = basePath

#         for file in os.listdir(basePath):
#             if file.endswith(".eeg"):

#                 self.subname = file[:-4]
#                 self.filename = os.path.join(basePath, file)
#                 self.filePrefix = os.path.join(basePath, file[:-4])

#     def clu2Spike(self):
#         filepath = pth(self.basePath, "Shank4", "RatJDay1_Shank2.clu.1")
#         with open(filepath) as f:
#             next(f)
#             for i, line in enumerate(f):

#                 m = "".join(line)

#                 if "KeyCount" in m:
#                     track_begin = i + 2
#                     line_frame = linecache.getline(fileName, i + 2).strip().split(" ")
#                     total_frames = int(line_frame[1]) - 1
#                     break


# basePath = [
#     "/data/Clustering/SleepDeprivation/RatJ/Day1/",
#     # "/data/Clustering/SleepDeprivation/RatJ/Day2/",
#     # "/data/Clustering/SleepDeprivation/RatK/Day1/",
#     # "/data/Clustering/SleepDeprivation/RatK/Day2/",
#     # "/data/Clustering/SleepDeprivation/RatN/Day1/",
#     # "/data/Clustering/SleepDeprivation/RatN/Day2/",
# ]

# with open(fileName) as f:
#     next(f)
#     for i, line in enumerate(f):

#         m = "".join(line)

#         if "KeyCount" in m:
#             track_begin = i + 2
#             line_frame = linecache.getline(fileName, i + 2).strip().split(" ")
#             total_frames = int(line_frame[1]) - 1
#             break


# spkgen = [ExtractfromClu(x) for x in basePath]

# for i, sess in enumerate(spkgen):
#     sess.clu2Spike()

