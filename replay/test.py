import numpy as np
import math
import pandas as pd
from sklearn.decomposition import FastICA, PCA
from scipy import stats


x = np.random.random(40).reshape(4, 10)
# x = sess.spikes.spks

# x1 = np.sin()

zsc_x = stats.zscore(x, axis=1)

corrmat = np.matmul(zsc_x, zsc_x.T) / x.shape[1]

# lambda_max = (1 + np.sqrt(1 / (x.shape[1] / x.shape[0]))) ** 2
eig_val, eig_mat = np.linalg.eigh(corrmat)
# get_sigeigval = np.where(eig_val > lambda_max)[0]
# n_sigComp = len(get_sigeigval)
pca_fit = PCA(n_components=None, whiten=False).fit_transform(x)

ica_decomp = FastICA(n_components=None, whiten=False).fit(pca_fit)
W = ica_decomp.components_
V = np.matmul(eig_mat[:, :2], W[:2, :])
