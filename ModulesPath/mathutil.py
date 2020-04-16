import numpy as np
import math
import pandas as pd


def partialcorr(x, y, z):
    """
    correlation between x and y , with controlling for z
    """
    # convert them to pandas series
    x = pd.Series(x)
    y = pd.Series(y)
    z = pd.Series(z)
    # xyz = pd.DataFrame({"x-values": x, "y-values": y, "z-values": z})

    xy = x.corr(y)
    xz = x.corr(z)
    zy = z.corr(y)

    parcorr = (xy - xz * zy) / (np.sqrt(1 - xz ** 2) * np.sqrt(1 - zy ** 2))

    return parcorr


def parcorr_mult(x, y, z):
    """
    correlation between multidimensional x and y , with controlling for multidimensional z

    """

    parcorr = np.zeros((len(z), len(y), len(z)))
    for i, x_ in enumerate(x):
        for j, y_ in enumerate(y):
            for k, z_ in enumerate(z):
                parcorr[k, j, i] = partialcorr(x_, y_, z_)

    revcorr = np.zeros((len(z), len(y), len(z)))
    for i, x_ in enumerate(x):
        for j, y_ in enumerate(y):
            for k, z_ in enumerate(z):
                parcorr[k, j, i] = partialcorr(x_, z_, y_)

    return parcorr, revcorr
