import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import plo

folderPath = "/data/Clustering/SleepDeprivation/RatN/Day1/RatN__2019-10-09_03-52-32/experiment1/recording1/continuous/Rhythm_FPGA-100.0/"

Allneur = []
for i in range(1, 9):
    file = folderPath + "Shank" + str(i) + "/"

    datFile = np.memmap(file + "Shank" + str(i) + ".dat", dtype="int16")
    datFiledur = len(datFile) / (16 * 30000)
    spktime = np.load(file + "spike_times.npy")
    cluID = np.load(file + "spike_clusters.npy")
    cluLabel = pd.read_csv(file + "cluster_KSLabel.tsv", delimiter="\t")
    goodCellsID = cluLabel.cluster_id[cluLabel["KSLabel"] == "good"].tolist()

    spkAll = []
    for i in range(len(goodCellsID)):
        clu_spike_location = spktime[np.where(cluID == goodCellsID[i])[0]]
        spkAll.append(clu_spike_location / 30000)

    tbin = np.arange(0, datFiledur, 0.250)
    spkHist = [np.histogram(i, bins=tbin)[0] for i in spkAll]

    Allneur.append(spkHist)

spk1 = [y for x in Allneur for y in x]
spk2 = [y for x in spk1 for y in x]
spk2 = np.reshape(spk1, (len(spk1), len(tbin) - 1))

# Allneur = np.concatenate((Allneur, spk1), axis=0)

