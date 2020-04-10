from joblib import Parallel, delayed
import numpy as np
from numba import njit, prange

# from multiprocessing import Pool
import time

# pool = mp.Pool(10)


# # what are your inputs, and what operation do you want to
# # perform on each input. For example...


@njit(parallel=True)
def chec(num):

    # inputs = np.arange(num)

    def processInput(i):
        # N = len(inputs)
        # time.sleep(5)
        return i * i

    # num_cores = mp.cpu_count()

    # tic = time.time()
    x = np.zeros(num)
    for i in range(num):
        x[i] = processInput(i)

    # toc = time.time()
    # print(toc - tic)

    return x


res = chec(10)


# print(a == res)

# # results = [pool.apply(processInput, args=[i]) for i in inputs]

# with par.ProcessPoolExecutor() as executor:
#     results = [executor.submit(processInput, i) for i in range(10)]
# with Pool(5) as p:
#     results = p.map(processInput, list(inputs))


# a1 = eegdata[: 10 * 1250, 2]
# a2 = eegdata[:1250, 2]
# highfreq = 600
# lowfreq = 300
# sRate = 1250
# nyq = 0.5 * sRate
# b, a = sg.butter(3, [lowfreq / nyq, highfreq / nyq], btype="bandpass")


# yf1 = sg.filtfilt(b, a, a1, axis=0)
# yf2 = sg.filtfilt(b, a, a2, axis=0)

# plt.plot(yf1[: len(yf2)])
# plt.plot(yf2)
