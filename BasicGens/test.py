import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
import csv
from pathlib import Path

file1 = "/data/Clustering/SleepDeprivation/RatN/Day1/spykcirc/RatNDay1Shank3/RatNDay1Shank3.GUI/templates.npy"

file2 = "/data/Clustering/SleepDeprivation/RatN/Day1/spykcirc/RatNDay1Shank3/RatNDay1Shank3.GUI/template_ind.npy"

file3 = ""
temp = np.load(file1)
temp_ind = np.load(file2)