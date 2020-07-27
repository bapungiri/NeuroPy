import numpy as np
import scipy.signal as sg
from dataclasses import dataclass
from typing import Any
import scipy.ndimage as filtSig
from collections import namedtuple
import scipy.stats as stats
from scipy.interpolate import interp1d
from scipy import fftpack
from scipy.fft import fft
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from datetime import date
import os
from pathlib import Path
import matplotlib.pyplot as plt


class Colormap:
    def dynamicMap(self):
        white = 255 * np.ones(48).reshape(12, 4)
        white = white / 255
        red = mpl.cm.get_cmap("Reds")
        brown = mpl.cm.get_cmap("YlOrBr")
        green = mpl.cm.get_cmap("Greens")
        blue = mpl.cm.get_cmap("Blues")
        purple = mpl.cm.get_cmap("Purples")

        colmap = np.vstack(
            (
                white,
                ListedColormap(red(np.linspace(0.2, 0.8, 16))).colors,
                ListedColormap(brown(np.linspace(0.2, 0.8, 16))).colors,
                ListedColormap(green(np.linspace(0.2, 0.8, 16))).colors,
                ListedColormap(blue(np.linspace(0.2, 0.8, 16))).colors,
                ListedColormap(purple(np.linspace(0.2, 0.8, 16))).colors,
            )
        )

        colmap = ListedColormap(colmap)

        return colmap


def savefig(fig, fname, scriptname, folder=None):

    if folder is None:
        folder = "/home/bapung/Documents/MATLAB/figures/"

    filename = folder + fname + ".pdf"

    today = date.today().strftime("%m/%d/%y")

    fig.text(
        0.95,
        0.01,
        f"{scriptname}\n Date: {today}",
        fontsize=6,
        color="gray",
        ha="right",
        va="bottom",
        alpha=0.5,
    )
    fig.savefig(filename)

