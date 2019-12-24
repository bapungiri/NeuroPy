import numpy as np


class VarCheck:

    k = 3

    def Addtwo(self, a):

        self.b = a + 2
        return self.b

    def Addthree(self):
        self.d = []
        for i in range(10):
            self.d.append(i)
        return self.d
