import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

# import seaborn as sns

mpl.style.use("figPublish")

fig1 = plt.figure(figsize=(6, 10))
# gs = GridSpec(3, 2, figure=fig)
ax1 = fig1.add_subplot(1, 1, 1)
ax1.plot([1, 3, 5])

fig2 = plt.figure(figsize=(6, 10))
# gs = GridSpec(3, 2, figure=fig)
ax2 = fig2.add_subplot(2, 1, 1)
ax2.set_figure(fig1)
# ax1 = fig1.add_subplot(2, 1, 1)
# ax1.plot([1, 3, 5])


# class session:
#     def __init__(self):
#         self.a = 4
#         self.b = 5
#         self.ripple = [5, 4]
#         self.child = child(self)


# class child:
#     def __init__(self, sess):
#         self.m = sess.a
#         self.time = sess.ripple


# m = session()

# plt.plot([1, 3, 5])

fig2.show()
