import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from callfunc import processData

# TODO thoughts on using data class for loading data into function

basePath = [
    # "/data/Clustering/SleepDeprivation/RatJ/Day1/",
    # "/data/Clustering/SleepDeprivation/RatJ/Day2/",
    # "/data/Clustering/SleepDeprivation/RatJ/Day3/",
    # "/data/Clustering/SleepDeprivation/RatJ/Day4/",
    # "/data/Clustering/SleepDeprivation/RatK/Day1/",
    # "/data/Clustering/SleepDeprivation/RatK/Day2/",
    # "/data/Clustering/SleepDeprivation/RatN/Day1/",
    "/data/Clustering/SleepDeprivation/RatN/Day2/"
]

sessions = [processData(_) for _ in basePath]

for sess in sessions:
    sess.eventpsth.hswa_ripple.nQuantiles = 5
    sess.eventpsth.hswa_ripple.plot()

