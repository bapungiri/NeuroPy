import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import seaborn as sns

# mpl.interactive(F)
sns.set_style("ticks")
# mpl.use("TkAgg")
"""
Unrecognized backend string 'gtkagg': valid strings are ['GTK3Agg', 'GTK3Cairo', 'MacOSX', 'nbAgg', 'Qt4Agg', 'Qt4Cairo', 'Qt5Agg', 'Qt5Cairo', 'TkAgg', 'TkCairo', 'WebAgg', 'WX', 'WXAgg', 'WXCairo', 'agg', 'cairo', 'pdf', 'pgf', 'ps', 'svg', 'template']
"""

plt.close()

x = np.arange(0, 2, 0.05)
y = np.sin(x)
fig = plt.figure(1, clear=True)
ax = fig.add_axes([0.2, 0.2, 0.5, 0.5])
ax.plot(x, y)
ax.set_title("sine wave")
ax.set_xlabel("angle")
ax.set_ylabel("sine")

sns.despine(ax=ax, right=False)


fig2 = plt.figure(1, clear=True)
ax1 = fig2.add_axes([0.2, 0.2, 0.5, 0.5])
ax1.plot(x, y)
ax1.set_title("sine wave")
ax1.set_xlabel("angle")
ax1.set_ylabel("sine")

sns.despine(ax=ax, right=False)
plt.close("all")

fig3 = plt.figure(figsize=(6, 10))
# gs = GridSpec(2, 2, figure=fig)
ax3 = fig3.add_subplot(2, 1, 1)
fig.show()
fig3.add_subplot(2, 1, 2)
fig2.show()


plt.show()
# plt.
# fig.show
