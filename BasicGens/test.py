import numpy as np
from pathlib import Path
from datetime import datetime

posFolder = Path("/data/Clustering/SleepDeprivation/RatN/Day4/position")
posfiles = sorted(posFolder.glob("*.csv"))

posfilestimes = [
    datetime.strptime(file.stem, "Take %Y-%m-%d %I.%M.%S %p") for file in posfiles
]


file = "/data/Clustering/SleepDeprivation/RatN/Day4/2019-10-15_11-30-06/experiment1/recording1/events/Message_Center-904.0/TEXT_group_1/timestamps.npy"

msg = np.load(file)
