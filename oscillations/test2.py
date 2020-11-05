import cupyx
import scipy.fft
import numpy as np
import cupy
import time

a = cupy.arange(510000).astype(float)


t = time.time()
b = cupy.convolve(a, a)
print(time.time() - t)


t = time.time()
b1 = np.convolve(a, a)
print(time.time() - t)
