import time

import numpy as np
import matplotlib.pyplot as plt
from parsePath import name2path


class check(name2path):
    def __init__(self, basePath):
        super().__init__(basePath)

    def mult(self, a, b):
        return a, b

    def catch(self):

        m = 2
        n = 3
        d = self.mult(m, n)
        print(d)


basepath = "/data/Clustering/SleepDeprivation/RatN/Day1/"

sess1 = check(basepath)

sess1.catch()

