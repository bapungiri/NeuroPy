import numpy as np


def check(input_arg):
    m = input_arg

    class c:
        @property
        def b(self):
            return m / 2

        @property
        def k(self):
            return self.b / 3

    return c()
