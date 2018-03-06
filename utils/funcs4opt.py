# coding: UTF-8
import numpy as np

LEVIN13 = "LeviN13"
SPHERE = "Sphere"
BANANA = "Banana"       # Officially RosenBrock
GOLDSTEINPRICE = "GoldsteinPrice"
MCCORMICK = "McCormick"
EASOM = "Easom"
ACKLEY = "Ackley"


class absfunc(object):
    name = ""
    best_param = []
    range = []
    minimum = None

    def func(self, x, y):
        pass


class Sphere(absfunc):
    best_param = [0, 0]
    # range = [[float('-inf'), float('inf')], [float('-inf'), float('inf')]]        # Mathematically
    # range = [[float(-10), float(10)], [float(-10), float(10)]]                    # Programmatically
    range = [[float(-5), float(5)], [float(-5), float(5)]]                          # Programmatically
    minimum = 0

    def __init__(self):
        self.name = __class__.__name__

    def func(self, x, y):
        return np.power(x, 2) + np.power(y, 2)


class Banana(absfunc):
    best_param = [1, 1]
    minimum = 0
    range = [[-5, 5],[-5, 5]]

    def __init__(self):
        self.name = __class__.__name__

    def func(self, x, y):
        return 100 * (y - x ** 2) ** 2 + (x - 1) ** 2


class GoldsteinPrice(absfunc):
    best_param = [0, -1]
    range = [[-2, 2], [-2, 2]]
    minimum = 3

    def __init__(self):
        self.name = __class__.__name__

    def func(self, x, y):
        return (1 + ((x + y + 1) ** 2) * (19 - 14 * x + 3 * (x ** 2) - 14 * y + 6 * x * y + 3 * y ** 2)) \
               * (30 + ((2 * x - 3 * y) ** 2) * (18 - 32 * x + 12 * x ** 2 + 48 * y - 36 * x * y + 27 * y ** 2))


class LeviN13(absfunc):
    best_param = [1, 1]
    range = [[-10, 10], [-10, 10]]
    minimum = 0

    def __init__(self):
        self.name = __class__.__name__

    def func(self, x, y):
        return np.power(np.sin(3 * np.pi * x), 2) + np.power(x - 1, 2) \
                * (1 + np.power(np.sin(3 * np.pi * y), 2)) \
                + np.power(y - 1, 2) * (1 + np.power(np.sin(2 * np.pi * y), 2))


class Ackley(absfunc):
    best_param = [0, 0]
    range = [[-32.768, 32.768], [-32.768, 32.768]]
    minimum = 0

    def __init__(self):
        self.name = __class__.__name__

    def func(self, x, y):
        return 20 - 20*np.exp( -0.2 * np.sqrt(1/2 * (x**2 + y**2))) + np.e -1 * np.exp(1/2 * (np.cos(2*np.pi*x) + np.cos(2*np.pi*y)))


class McCormick(absfunc):
    best_param = [-0.54719, -1.54719]
    range = [[-1.5, 4], [-3, 4]]
    minimum = -1.9133

    def func(self, x, y):
        return np.sin(x + y) + (x - y)**2 + -1.5*x + 2.5*y + 1

    def __init__(self):
        self.name = __class__.__name__


class Easom(absfunc):
    best_param = [np.pi, np.pi]
    range = [[-100, 100], [-100, 100]]
    minimum = -1

    def func(self, x, y):
        return -1 * np.cos(x) * np.cos(y) * np.exp(-1 * ((x - np.pi)**2 + (y - np.pi)**2))

    def __init__(self):
        self.name = __class__.__name__


class FuncFactory:
    funcs = {
        GOLDSTEINPRICE: GoldsteinPrice,
        SPHERE: Sphere,
        BANANA: Banana,
        LEVIN13: LeviN13,
        MCCORMICK: McCormick,
        EASOM: Easom,
        ACKLEY: Ackley
    }
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def getFunc(self, name):
        assert name in self.funcs.keys(), "Unknown function name [{}].".format(name)

        return self.funcs[name]()


if __name__ == '__main__':

    opt_func_ackley = FuncFactory().getFunc(EASOM)
    suggest_res = opt_func_ackley.func(*[5.36385984338, 1.08115132361])
    print("best is ", opt_func_ackley.minimum, ". suggest_res is ", suggest_res, " calculated ", opt_func_ackley.func(*opt_func_ackley.best_param))

    # opt_func_ackley = FuncFactory().getFunc(ACKLEY)
    # val = opt_func_ackley.func(*[0.572485461444, -0.567246428474])
    # print("best is ", opt_func_ackley.minimum, ". val is ", val, " calculated ", opt_func_ackley.func(*opt_func_ackley.best_param))

    # for k in FuncFactory().funcs.keys():
    #     opt_func = FuncFactory().getFunc(k)
    #     print("name[{}]".format(opt_func.name))
    #     param = opt_func.best_param
    #     print("  best is {}. [{}]".format(opt_func.minimum, param))
    #     print("  calculated best is ", opt_func.func(*param))
