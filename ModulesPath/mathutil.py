import numpy as np
import math
import pandas as pd


def partialcorr(x, y, z):

    # correlation between x and y , with controlling for z

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

    # correlation between x and y , with controlling for z

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

