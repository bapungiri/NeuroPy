import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl


class names:
    def __init__(self):
        self.string = "gh"


class session:
    t = 5

    def __init__(self):
        obj = names()
        # obj.trange =
        self.trange = [4, 5]
        obj.trange = self.trange
        self.child1 = child1(obj)

        obj.child1 = self.child1
        self.child2 = child2(obj)
        self.string = "mg"


class child1:
    def __init__(self, obj):
        print(hasattr(obj, "child1"))
        self.time = np.array([1, 2, 3, 5, 6, 7])
        # self.time = self.time[ind]


class child2:
    def __init__(self, obj):
        print(obj.child1.time)

        # self.obj = obj
        print(hasattr(obj, "child1"))

        self.time = np.array([11, 2, 3, 8, 1, 9, 0, 3])
        # self.time = self.time[ind]
        # self.b = obj.child2.time


m = session()
