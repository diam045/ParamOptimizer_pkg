from unittest import TestCase
from nose.tools import ok_, eq_
from utils import funcs4opt
import numpy as np


class FuncFactoryTestCase(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_func_definition(self):
        opt_func = funcs4opt.FuncFactory().getFunc(funcs4opt.SPHERE)
        ok_(isinstance(opt_func, funcs4opt.Sphere))
        opt_func = funcs4opt.FuncFactory().getFunc(funcs4opt.BANANA)
        ok_(isinstance(opt_func, funcs4opt.Banana))
        opt_func = funcs4opt.FuncFactory().getFunc(funcs4opt.EASOM)
        ok_(isinstance(opt_func, funcs4opt.Easom))
        opt_func = funcs4opt.FuncFactory().getFunc(funcs4opt.ACKLEY)
        ok_(isinstance(opt_func, funcs4opt.Ackley))
        opt_func = funcs4opt.FuncFactory().getFunc(funcs4opt.GOLDSTEINPRICE)
        ok_(isinstance(opt_func, funcs4opt.GoldsteinPrice))
        opt_func = funcs4opt.FuncFactory().getFunc(funcs4opt.LEVIN13)
        ok_(isinstance(opt_func, funcs4opt.LeviN13))
        opt_func = funcs4opt.FuncFactory().getFunc(funcs4opt.MCCORMICK)
        ok_(isinstance(opt_func, funcs4opt.McCormick))

    def test_func_calculation(self):
        for k in funcs4opt.FuncFactory().funcs.keys():
            opt_func = funcs4opt.FuncFactory().getFunc(k)

            print(opt_func.name)
            print(k)
            ok_(opt_func.name == k)

            param = opt_func.best_param
            eq_(np.round(opt_func.minimum, 3), np.round(opt_func.func(*param), 3))



