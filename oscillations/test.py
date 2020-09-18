from dataclasses import dataclass
import numpy as np
import typing
import matplotlib.pyplot as plt
from plotUtil import Fig

c = 5
k = 7


@dataclass
class Params:
    peak: np.array = c
    trough: int = 3
    k: int = k

    def sanityPlot(self):
        pass


def check():
    a = Params.k

    return a


m = Params()

figure = Fig()

fig, gs = figure.draw()
ax = plt.subplot(gs[0])
ax.plot([1, 2, 3])
figure.panel_label(ax, "a")
figure.savefig("hello", scriptname="a.py")

