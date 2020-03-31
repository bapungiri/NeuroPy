import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sg
from dataclasses import dataclass
from collections import namedtuple


def callcheck(str):

    Point = namedtuple("Point", ["x", "y"])

    p = Point(1, 2)

    return p


b = callcheck("fg")

a = {"sdf": 1}
a["theta"] = [1, 2, 3]
# fs = 10e3
# N = 1e5
# amp = 2 * np.sqrt(2)
# noise_power = 0.01 * fs / 2
# time = np.arange(N) / float(fs)
# mod = 500 * np.cos(2 * np.pi * 0.25 * time)
# carrier = amp * np.sin(2 * np.pi * 3e3 * time + mod)
# noise = np.random.normal(scale=np.sqrt(noise_power), size=time.shape)
# noise *= np.exp(-time / 5)
# x = carrier + noise
# f, t, Sxx = sg.spectrogram(x, fs, nperseg=1250, noverlap=625.0)
# plt.pcolormesh(t, f, Sxx)
# plt.ylabel("Frequency [Hz]")
# plt.xlabel("Time [sec]")
# plt.show()


# import matplotlib as mpl
arr = np.max([1, 2, 3])
# def
# @dataclass


# class session:
#     def __init__(self):
#         self._trange = [4, 5]


# class event(session):
#     def __init__(self):
#         self.ripple = child1()

#     @property
#     def trange(self):
#         return self._trange

#     @trange.setter
#     def trange(self, period):

#         self.ripple.trange = period


# class child1(session):
#     def __init__(self):
#         self.a = [1, 2, 3]
#         super().__init__()

#     @property
#     def trange(self):
#         return self._trange

#     @trange.setter
#     def trange(self, period):

#         self.a = ["d", "b"]


# class child3(event):
#     def __init__(self):
#         self.c = [1, 2, 3]

#         print(self.ripple.trange)


# class child2(event, child3):
#     def __init__(self):
#         self.b = 5

#         super().__init__()


# m = child2()
# m.trange = [5, 6]
