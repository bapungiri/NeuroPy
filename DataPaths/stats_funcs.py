import numpy as np
from scipy import stats
import pandas as pd


def bootstrap(pairs, data: pd.DataFrame, x, y, hue, hue_order, n_boot=10000, **kwargs):
    rng = np.random.default_rng()
    subsample_mean = lambda arr: np.mean(rng.choice(arr, size=len(arr), replace=True))

    n_pairs = len(pairs)
    for i, p in enumerate(pairs):
        print(p)
        g1 = data[(data[x] == p[0][0]) & (data[hue] == p[0][1])][y].values
        g2 = data[(data[x] == p[1][0]) & (data[hue] == p[1][1])][y].values
        g1_means = np.array([subsample_mean(g1) for _ in range(n_boot)])
        g2_means = np.array([subsample_mean(g2) for _ in range(n_boot)])

        p_val = np.count_nonzero(g1_means >= g2_means) / n_boot
        print(f"{p[0][0]}_{p[0][1]} vs {p[1][0]}_{p[1][1]}: {p_val}")
