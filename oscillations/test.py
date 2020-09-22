from dataclasses import dataclass
import numpy as np
import typing
import matplotlib.pyplot as plt
from plotUtil import Fig
import sys
from pathlib import Path
from joblib import Parallel, delayed
import multiprocessing
import dask
import scipy.signal as sg

fs = 10e3

N = 1e5

amp = 2 * np.sqrt(2)

freq = 1234.0

noise_power = 0.001 * fs / 2

time = np.arange(N) / fs

x = amp * np.sin(2 * np.pi * freq * time)

x += np.random.normal(scale=np.sqrt(noise_power), size=time.shape)
x[300:1050] = 0
f, Pxx_den = sg.welch(x, fs, nperseg=1024)

plt.clf()
plt.semilogy(f, Pxx_den)

plt.ylim([0.5e-3, 1])

plt.xlabel("frequency [Hz]")

plt.ylabel("PSD [V**2/Hz]")

plt.show()
# what are your inputs, and what operation do you want to
# perform on each input. For example...
# inputs = range(10)


# def processInput(i):
#     return i * i


# num_cores = multiprocessing.cpu_count()

# results = Parallel(n_jobs=num_cores)(delayed(processInput)(i) for i in inputs)

# a = Path(__file__)
# # c = 5
# # k = 7
# print(__file__)
# b = __file__
# @dataclass
# class Params:
#     peak: np.array = c
#     trough: int = 3
#     k: int = k

#     def sanityPlot(self):
#         pass


# def check():
#     a = Params.k

#     return a


# m = Params()

# figure = Fig()

# fig, gs = figure.draw()
# ax = plt.subplot(gs[0])
# ax.plot([1, 2, 3])
# figure.panel_label(ax, "a")
# figure.savefig("hello", scriptname="a.py")


# import inspect


# def hello():
#     previous_frame = inspect.currentframe()
#     (filename, line_number, function_name, lines, index) = inspect.getframeinfo(
#         sys._getframe(1)
#     )
#     return previous_frame


# print(hello())
