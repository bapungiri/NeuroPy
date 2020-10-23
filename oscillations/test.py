import dask.array as da
from scipy import stats

x = da.random.random((10000, 10000), chunks=(1000, 1000))
y = x + x.T
z = stats.zscore(y, axis=1)
a = z.compute()