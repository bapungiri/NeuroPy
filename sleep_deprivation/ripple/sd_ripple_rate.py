"""_summary_
Question:

Ripple rate first hour **between** SD and NSD (or only restrict to 1h NREM sleep of NSD)
# - One possiblity of faster decay of replay during sleep deprivation could be reduced ripple rate at the begining of POST. This may suggest that in order for cells to replay they need sufficient number/amount of exicitation to last for certain duration of time, so a lower ripple rate could interrupt this requirement.

Results:



"""
import matplotlib.pyplot as plt
import numpy as np
from neuropy import plotting
import pandas as pd
from scipy import stats
import subjects
import seaborn as sns

sessions = (
    subjects.sd.ratJday1
    + subjects.sd.ratKday1
    + subjects.sd.ratNday1
    + subjects.sd.ratSday3
    + subjects.sd.ratRday2
    + subjects.sd.ratUday4
    + subjects.sd.ratVday2
    + subjects.nsd.ratJday2
    + subjects.nsd.ratKday2
    + subjects.nsd.ratNday2
    + subjects.nsd.ratSday2
    + subjects.nsd.ratRday1
    + subjects.nsd.ratUday2
    + subjects.nsd.ratVday1
)

rpl_rate = []
for sub, sess in enumerate(sessions):
    post = sess.paradigm["post"].flatten()
    w = 2 * 3600  # window size
    rpl_rate_sub = sess.ripple.time_slice(post[0], post[0] + w).n_epochs / w
    rpl_rate.append(
        pd.DataFrame(
            {"sub": sess.sub_name, "rpl_rate": [rpl_rate_sub], "grp": sess.tag}
        )
    )


rpl_rate = pd.concat(rpl_rate, ignore_index=True)

_,ax = plt.subplots(nrows=1)
sns.lineplot(data=rpl_rate, x="grp", y="rpl_rate", marker="o")
