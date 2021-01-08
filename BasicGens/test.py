import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
import csv
from pathlib import Path

rootdir = Path("/run/media/bapung/mercik/RatS/Day5TwoNovel")
folders = [f for f in rootdir.iterdir() if f.is_dir()]
sub_file = Path(
    "experiment1/recording1/continuous/Intan_Rec._Controller-100.0/continuous.dat"
)

nframes = []
for folder in folders:
    file = folder / sub_file
    nframes.append(len(np.memmap(file, dtype="int16", mode="r")) / 195)

nframes = np.asarray(nframes).astype(int)

print(nframes)
