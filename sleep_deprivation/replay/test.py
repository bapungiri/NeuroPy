import numpy as np
from numpy.typing import NDArray, ArrayLike


class Test:
    def __init__(self, a):
        self.a = a
        self.b: NDArray | ArrayLike | None = None

    def get_abs_b(self):

        c = np.abs(self.b)

        return c
