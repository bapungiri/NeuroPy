from dataclasses import dataclass


# @dataclass
class Car:
    def __init__(self, color, mileage):
        self.color = color
        self.mileage = mileage
        setattr(self, "sdf", "ghy")

        self.__v = 10

    def __str__(self):
        return "a {self.color} car".format(self=self)


class child(Car):
    def __init__(self, color, mileage):
        super().__init__(color, mileage)
        self.c = 4


m = Car("red", 54)

# setattr(m, "sdf", "ghy")

