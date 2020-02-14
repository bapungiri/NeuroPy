import time

import numpy as np
import matplotlib.pyplot as plt
from parsePath import name2path


class check(name2path):
    def checkint(self):
        print(self.filename)


basepath = "/data/Clustering/SleepDeprivation/RatN/Day1/"

sess1 = check(basepath)

sess1.checkint()

