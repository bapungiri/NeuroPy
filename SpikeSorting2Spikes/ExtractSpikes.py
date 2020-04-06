import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

basepath = Path("/home/bapung/Documents/ClusteringHub/spykcirc/RatN/Day1/")
name = "RatNDay1Shank7"

spkfile = Path(basepath, name, name + ".GUI", "spike_times.npy")
clufile = Path(basepath, name, name + ".GUI", "spike_clusters.npy")
clugrpfile = Path(basepath, name, name + ".GUI", "cluster_group.tsv")

spks = np.load(spkfile)
spkcluid = np.load(clufile)
clus = np.unique(spkcluid)
clugrp = pd.read_csv(clugrpfile, sep="\t")

clugood = np.asarray(clugrp.loc[clugrp["group"] == "good", "cluster_id"])

spkframes = [spks[spkcluid == clu] / 30000 for clu in clugood]
bins = np.arange(0, 54614, 0.1)
binspk = np.asarray([np.histogram(arr, bins=bins)[0] for arr in spkframes])


movcorr = []
for time in np.arange(0, binspk.shape[1] - 100, 8):
    # bins = np.arange(time, time + 1, 0.1)    # binspk = np.asarray([np.histogram(arr, bins=bins)[0] for arr in spkframes])
    temp = binspk[:, range(time, time + 100)]
    spkcorr = np.corrcoef(temp)
    corr_all = spkcorr[np.tril_indices(len(spkcorr), k=-1)]
    movcorr.append(np.nanmean(corr_all))


plt.plot(movcorr)
