import numpy as np
import matplotlib.pyplot as plt
import time
import cupy as cp

x = np.arange(0, 1, 0.02)

mat = np.random.uniform(0, 0.1, size=(43, 10))
mat[0, 0] = 0.6
mat[5, 1] = 0.7
mat[9, 3] = 0.7
mat[10, 4] = 0.6
mat[15, 5] = 0.7
mat[17, 6] = 0.7
mat[19, 7] = 0.7
mat[20, 8] = 0.6
mat[30, 9] = 0.7

mat = np.apply_along_axis(np.convolve, axis=0, arr=mat, v=np.ones(3))

t = np.arange(mat.shape[1])
nt = len(t)
tmid = (nt + 1) / 2
pos = np.arange(mat.shape[0])
npos = len(pos)
pmid = (npos + 1) / 2

nlines = 35000
slope = np.random.uniform(low=-np.pi / 2, high=np.pi / 2, size=nlines)
diag_len = np.sqrt((nt - 1) ** 2 + (npos - 1) ** 2)
intercept = np.random.uniform(low=-diag_len / 2, high=diag_len / 2, size=nlines)

tstart = time.time()
mat_cp = cp.array(mat)
cmat = cp.tile(intercept, (nt, 1)).T
mmat = cp.tile(slope, (nt, 1)).T
tmat = cp.tile(t, (nlines, 1))
cp.cuda.Stream.null.synchronize()
posterior = cp.zeros((nlines, nt))

y_line = (((cmat - (tmat - tmid) * cp.cos(mmat)) / cp.sin(mmat)) + pmid).astype(int)
t_out = cp.where((y_line < 0) | (y_line > npos - 1))
t_in = cp.where((y_line >= 0) & (y_line <= npos - 1))
posterior[t_out] = cp.median(mat_cp[:, t_out[1]], axis=0)
posterior[t_in] = mat_cp[y_line[t_in], t_in[1]]

score = cp.nanmean(posterior, axis=1)
print(time.time() - tstart)


tstart = time.time()
cmat = np.tile(intercept, (nt, 1)).T
mmat = np.tile(slope, (nt, 1)).T
tmat = np.tile(t, (nlines, 1))
posterior = np.zeros((nlines, nt))

y_line = (((cmat - (tmat - tmid) * np.cos(mmat)) / np.sin(mmat)) + pmid).astype(int)
t_out = np.where((y_line < 0) | (y_line > npos - 1))
t_in = np.where((y_line >= 0) & (y_line <= npos - 1))
posterior[t_out] = np.median(mat[:, t_out[1]], axis=0)
posterior[t_in] = mat[y_line[t_in], t_in[1]]

score = np.nanmean(posterior, axis=1)
print(time.time() - tstart)


# plt.ylim([0, npos])
# plt.plot(x, y)

max_line = np.argmax(score)
ymax = (
    (intercept[max_line] - (t - tmid) * np.cos(slope[max_line]))
    / np.sin(slope[max_line])
) + pmid

plt.pcolormesh(mat)
plt.plot(t, ymax)
