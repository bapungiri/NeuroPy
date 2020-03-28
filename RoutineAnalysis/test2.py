import matplotlib.pyplot as plt
import numpy as np
from bokeh.models import ColumnDataSource

import matplotlib as mpl


# p = figure(plot_width=400, plot_height=400)

# p.line(x, y1, color=col[0])
# p.title = "sdf"
# show(p)

# import matplotlib.pyplot as plt

# plt.close("all")
# fig1, (ax1, ax2) = plt.subplots(2, 1)
# ax1.plot(range(10))
# ax2.plot(range(20))

# allxes = fig1.get_axes()

# # plt.close()
# fig2, ax3 = plt.subplots()
# ax3.add_artist(ax1)


# fig2.show()
# X = [1, 2, 3, 4, 5, 6, 7]
# Y = [1, 3, 4, 2, 5, 8, 6]

# axes1 = fig.add_axes([0.1, 0.1, 0.9, 0.9])  # main axes
# axes2 = fig.add_axes([0.2, 0.6, 0.4, 0.3])  # inset axes

# axes1.plot(ax2)
# # ax.plot()
# plt.axes(ax)

# # fig2, ax3 = plt.figure()
# # ax3.axes(ax)

# plt.show()


# import scipy.signal as sg
# from signal_process import filter_sig as filt

# basepath = "/data/Clustering/SleepDeprivation/RatK/Day1/"

# filename = basepath + "RatN_Day1_2019-10-09_03-52-32_BestRippleChans.npy"

# lfp = np.load(filename, allow_pickle=True).item()
# lfp = lfp["BestChan"]
# lfp_ripple = filt.filter_ripple(lfp)

# analytic_signal = sg.hilbert(lfp_ripple)
# amplitude_envelope = np.abs(analytic_signal)


# class names:
#     def __init__(self):
#         self.string = "gh"


# class session:
#     t = 5

#     def __init__(self):
#         self.obj = names()
#         # obj.trange =
#         self.trange = [4, 5]
#         self.obj.trange = self.trange
#         self.child1 = child1(self.obj)

#         self.obj.child1 = self.child1
#         self.child2 = child2(self.obj)
#         self.string = "mg"


# class child1:
#     def __init__(self, obj):
#         print(hasattr(obj, "child1"))
#         self.time = np.array([1, 2, 3, 5, 6, 7])
#         self.a = 5
#         # self.time = self.time[ind]

#         if self.a > 2:

#             @property
#             def gh(self):
#                 print("gh")


# class child2:
#     def __init__(self, obj):
#         print(obj.child1.time)

#         # self.obj = obj
#         print(hasattr(obj, "child1"))

#         self.time = np.array([11, 2, 3, 8, 1, 9, 0, 3])
#         # self.time = self.time[ind]
#         # self.b = obj.child2.time


# m = session()

# a = None
# a = np.array([3, 5])
# if a.any():
#     print("a is not empty")
