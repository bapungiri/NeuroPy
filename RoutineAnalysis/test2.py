import numpy as np
from dataclasses import dataclass


@dataclass
class session(object):
    basepath: str

    # @property
    def __post_init__(self):
        self.ripple = ripple()


class ripple:
    def __init__(self):

        self.a = [1, 2, 3]


class compute(session):
    def printripple(self):
        t = self.ripple
        print(t)


m = compute("fg")

