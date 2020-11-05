from itertools import count


class A:
    _ids = count(0)
    _instance = None

    # def __new__(cls, obj):

    #     if not cls._instance:
    #         print("init did not run")
    #         return obj
    #     else:
    #         print("init ran")
    #         return super(A, cls).__new__(cls)

    def __init__(self, basepath) -> None:
        self.id = next(self._ids)
        self.basepath = basepath
        print("init indeed")


class B:
    def __init__(self, obj) -> None:

        if isinstance(obj, A):
            self._obj = obj

        else:
            self._obj = A(obj)
        self.a = 4

    def thrice(self, k):
        return 3 * k


a = A("hello")
# a.m = 7
# a1 = A("mello")
b = B(a)
c = B("mello")
