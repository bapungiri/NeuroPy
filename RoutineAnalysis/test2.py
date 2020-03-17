from dataclasses import dataclass


# @dataclass
class Car:
    def __init__(self, color, mileage):
        self.color = color
        self.mileage = mileage

        self.__v = 10

    def __str__(self):
        return "a {self.color} car".format(self=self)

    def change_color(self, newcolor):

        self.color = newcolor

    @staticmethod
    def printcolor(ght):

        print(ght)


class child(Car):
    def __init__(self, color, mileage):
        super().__init__(color, mileage)
        self.c = 4


m = Car("red", 54)

# setattr(m, "sdf", "ghy")

