class hiding:
    # class attribute, "__" befor an attribute will make it to hide
    a = 10

    def __init__(self):
        self.__n = 0

        print(self.a)


class check:
    # class attribute, "__" befor an attribute will make it to hide
    a = 10

    def __init__(self):

        self.m = 10
        # print(self.a)


class check2(check):
    # class attribute, "__" befor an attribute will make it to hide

    def __init__(self):
        super().__init__()

        # self.a = 10

    def gf(self):
        print(self.a)


a = hiding()
# b = check2()
# b.a = 20
# b.gf()

