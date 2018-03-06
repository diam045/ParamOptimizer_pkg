# -*- coding: utf-8 -*-

import json
import os
import sys
sys.path.append(".")
import logging
import numpy as np
import shutil
import csv
from datetime import datetime
from opts.optexecutor_factory import OptExecutorFactory
from opts.optexecutor_factory import COMMON_LIB_GPYOPT, COMMON_LIB_HYPEROPT
from opts.optexecutor import COMMON_LIB, COMMON_SEED, COMMON_SCOPE, COMMON_STATUSES, \
    COMMON_LOSSES, COMMON_VALS, COMMON_RESULTS, COMMON_ALGO, COMMON_MAXEVALS
from opts.optexecutor_gpyopt import OptExecutorGpyopt
from utils.request_dict_creator import make_request_fromcsv
LOG_LEVEL = 'DEBUG'

from utils import funcs4opt

class Hyperparam_optimizer(object):

    def main(self):
        logging.basicConfig(
            level=getattr(logging, LOG_LEVEL),
            format='%(asctime)s [%(levelname)s] %(module)s | %(message)s',
            datefmt='%Y/%m/%d %H:%M:%S',
        )

        string_json = '{"seed":0,"lib":"hyperopt","algo":"tpe","scope":{"x":["uniform",-10,10],"y":["uniform",-10,10]},"max_evals":1,"results":{"losses":[3.4683742070179777,3.196277295751884,28.773296779196503,19.62775326741544,20.455256910553278],"statuses":["ok","ok","ok","ok","ok"],"vals":{"y":[-0.13952676978449574,0.3096416495214722,-2.416283380272519,0.28691742075450755,-3.5258018804882827],"x":[1.8571231751102246,1.760795088769135,4.789036584263615,-4.421021563064698,2.832662706729936]}}}'
        # string_json = '{"seed":0,"lib":"hyperopt","algo":"rand","scope":{"x":["uniform",-10,10],"y":["uniform",-10,10]},"max_evals":1,"results":{"losses":[3.4683742070179777,3.196277295751884,28.773296779196503,19.62775326741544,20.455256910553278],"statuses":["ok","ok","ok","ok","ok"],"vals":{"y":[-0.13952676978449574,0.3096416495214722,-2.416283380272519,0.28691742075450755,-3.5258018804882827],"x":[1.8571231751102246,1.760795088769135,4.789036584263615,-4.421021563064698,2.832662706729936]}}}'
        # string_json = '{"seed":0,"lib":"hyperopt","algo":"anneal","scope":{"x":["uniform",-10,10],"y":["uniform",-10,10]},"max_evals":1,"results":{"losses":[3.4683742070179777,3.196277295751884,28.773296779196503,19.62775326741544,20.455256910553278],"statuses":["ok","ok","ok","ok","ok"],"vals":{"y":[-0.13952676978449574,0.3096416495214722,-2.416283380272519,0.28691742075450755,-3.5258018804882827],"x":[1.8571231751102246,1.760795088769135,4.789036584263615,-4.421021563064698,2.832662706729936]}}}'

        # number mismatch
        string_json = '{"seed":0,"lib":"hyperopt","algo":"tpe","scope":{"x":["uniform",-10,10],"y":["uniform",-10,10]},"max_evals":3,"results":{"losses":[3.4777,3.294,28.729,19.62,20.458],"statuses":["ok","ok","ok","ok","ok"],"vals":{"y":[-0.13974,0.3722,-2.419,0.28755,-3.2827],"x":[1.8571,1.765,4.615,-4.068,2.832]}}}'

        string_json = '{"seed":0,"lib":"hyperopt","algo":"tpe","scope":[{"x":["uniform",-10,10]},{"y":["uniform",-10,10]}],"max_evals":1,"results":{"losses":[3.4683742070179777,3.196277295751884,28.773296779196503,19.62775326741544,20.455256910553278],"statuses":["ok","ok","ok","ok","ok"],"vals":{"y":[-0.13952676978449574,0.3096416495214722,-2.416283380272519,0.28691742075450755,-3.5258018804882827],"x":[1.8571231751102246,1.760795088769135,4.789036584263615,-4.421021563064698,2.832662706729936]}}}'
        string_json = '{"seed":0,"lib":"hyperopt","algo":"tpe","scope":[{"x":["quniform",-10,10,0.1]},{"y":["quniform",-10,10,0.1]}],"max_evals":1,"results":{"losses":[3.4683742070179777,3.196277295751884,28.773296779196503],"statuses":["ok","ok","ok"],"vals":{"y":[-0.13952676978449574,0.28691742075450755,-3.5258018804882827],"x":[1.8571231751102246,1.760795088769135,2.832662706729936]}}}'

        executor = OptExecutorFactory.get_executor(string_json)
        rval = executor.suggest()

        print(rval)
        # print(rval["alog"])
        # print(rval["scope"]["x"][0])
        # print(rval["scope"]["y"][0])

    def create_eval_file(self, file_path, newname_base):
        todaydetail = datetime.today()

        fol = os.path.dirname(file_path)
        name, ext = os.path.splitext(os.path.basename(file_path))
        evalfolder="{}/{}".format(fol, name)
        if not os.path.exists(evalfolder):
            os.makedirs(evalfolder)
        shutil.copy(file_path, evalfolder)
        newname = "{}_{}".format(newname_base, todaydetail.strftime("%Y-%m-%d %H%M%S"))
        os.rename("{}/{}.csv".format(evalfolder, name), "{}/{}.csv".format(evalfolder, newname))
        return "{}/{}.csv".format(evalfolder, newname)

    def demo_gpyopt(self):

        opt_func = funcs4opt.FuncFactory().getFunc(funcs4opt.SPHERE)
        # opt_func = funcs4opt.FuncFactory().getFunc(funcs4opt.MCCORMICK)
        # opt_func = funcs4opt.FuncFactory().getFunc(funcs4opt.BANANA)
        # opt_func = funcs4opt.FuncFactory().getFunc(funcs4opt.EASOM)
        func = opt_func.func
        best_param = opt_func.best_param
        test_file="exec_files/sample_01.csv"

        lib = COMMON_LIB_GPYOPT

        algo = "{},{}".format(OptExecutorGpyopt.GPY_MODEL_GP, OptExecutorGpyopt.GPY_AQUISITION_LCB)
        algo = "{},{}".format(OptExecutorGpyopt.GPY_MODEL_GP, OptExecutorGpyopt.GPY_AQUISITION_MPI)
        algo = "{},{}".format(OptExecutorGpyopt.GPY_MODEL_GP, OptExecutorGpyopt.GPY_AQUISITION_EI)

        # algo = "{},{}".format(OptExecutorGpyopt.GPY_MODEL_RF, OptExecutorGpyopt.GPY_AQUISITION_LCB)
        # algo = "{},{}".format(OptExecutorGpyopt.GPY_MODEL_RF, OptExecutorGpyopt.GPY_AQUISITION_MPI)
        # algo = "{},{}".format(OptExecutorGpyopt.GPY_MODEL_RF, OptExecutorGpyopt.GPY_AQUISITION_EI)
        #
        # algo = "{},{}".format(OptExecutorGpyopt.GPY_MODEL_WARPERDGP, OptExecutorGpyopt.GPY_AQUISITION_LCB)
        # algo = "{},{}".format(OptExecutorGpyopt.GPY_MODEL_WARPERDGP, OptExecutorGpyopt.GPY_AQUISITION_MPI)
        # algo = "{},{}".format(OptExecutorGpyopt.GPY_MODEL_WARPERDGP, OptExecutorGpyopt.GPY_AQUISITION_EI)
        #
        # algo = "{},{}".format(OptExecutorGpyopt.GPY_MODEL_GPMCMC, OptExecutorGpyopt.GPY_AQUISITION_LCBMCMC)
        # algo = "{},{}".format(OptExecutorGpyopt.GPY_MODEL_GPMCMC, OptExecutorGpyopt.GPY_AQUISITION_MPIMCMC)
        # algo = "{},{}".format(OptExecutorGpyopt.GPY_MODEL_GPMCMC, OptExecutorGpyopt.GPY_AQUISITION_EIMCMC)


        seed = 0
        # np.random.seed(seed)
        newname_base = "eval_{}_{}_{}".format(algo, opt_func.name, "seed{}".format(seed))
        eval_file = self.create_eval_file(test_file, newname_base)
        num_trial = 20
        for i in range(num_trial):
            request_dict = make_request_fromcsv(eval_file)
            request_dict[COMMON_LIB] = lib
            # print(request_dict[COMMON_SCOPE].keys())

            request_dict[COMMON_ALGO] = algo

            request_dict[COMMON_SEED] = seed
            request_dict[COMMON_SCOPE][0]["x"][1:] = opt_func.range[0]
            request_dict[COMMON_SCOPE][0]["x"][0] = OptExecutorGpyopt.GPY_TYPE_CONTINUOUS
            request_dict[COMMON_SCOPE][1]["y"][1:] = opt_func.range[1]
            request_dict[COMMON_SCOPE][1]["y"][0] = OptExecutorGpyopt.GPY_TYPE_CONTINUOUS

            json_str = json.dumps(request_dict)

            executor = OptExecutorFactory.get_executor(json_str)
            rval, elaps = executor.suggest()
            print(i, " elapsed time:", elaps)
            for j in range(len(rval["statuses"])):
                args = []
                for arg_name in request_dict["results"]["vals"].keys():
                    args.append(rval["vals"][arg_name][j])

                # print(type(args), type(args[0]))
                # print(args)
                loss = func(*args)
                with open(eval_file, 'a') as f:
                    writer = csv.writer(f, lineterminator='\n')
                    writer.writerow([loss]+args)

        keys = [list(x.keys())[0] for x in request_dict[COMMON_SCOPE]]
        print("keys", keys)

        print("seed", request_dict["seed"])
        print("algo", request_dict["algo"])
        print("{} min is ".format(opt_func.name), func(*best_param), "  [{}]".format(best_param))

        losses = json.loads(json_str)["results"]["losses"]
        vals = json.loads(json_str)["results"]["vals"]
        print("  minimum loss is", min(np.array(losses)))
        bestindex = np.argmin(losses)
        print("  bestidx:", bestindex, " x:",vals["x"][bestindex], " y:", vals["y"][bestindex])

        print("keys", vals.keys())
        X_base = vals.values()
        print(" src:", np.array([np.array(line) for line in X_base]).T)




    def demo_hyperopt(self):
        # def f_x(x):
        #     return np.cos(1.5 * x) + 0.1 * x
        # def f_x_y(x, y):
        #     return np.cos(1.5 * x) + 0.1 * x + 0.1 * np.sin(2.0 * y)
        # def f_LeviN13(x, y):
        #     return np.power(np.sin(3*np.pi*x),2) + np.power(x-1,2) * (1+np.power(np.sin(3*np.pi*y),2)) + np.power(y-1,2)*(1+np.power(np.sin(2*np.pi*y),2))
        # def f_Sphere(x, y):
        #     return np.power(x, 2) + np.power(y, 2)
        # def f_Banana(x, y):
        #     return 100 * (y - x**2)**2 + (x - 1)**2
        # def f_Goldstein_Price(x, y):
        #     return (1+((x+y+1)**2)*(19-14*x+3*(x**2)-14*y+6*x*y+3*y**2)) * (30+((2*x-3*y)**2)*(18-32*x+12*x**2+48*y-36*x*y+27*y**2))

        opt_func = funcs4opt.FuncFactory().getFunc(funcs4opt.MCCORMICK)
        opt_func = funcs4opt.FuncFactory().getFunc(funcs4opt.SPHERE)
        # opt_func = funcs4opt.FuncFactory().getFunc(funcs4opt.BANANA)
        # opt_func = funcs4opt.FuncFactory().getFunc(funcs4opt.EASOM)
        func = opt_func.func
        best_param = opt_func.best_param
        test_file="exec_files/sample_01.csv"
        # test_file="exec_files/sample_02.csv"
        # test_file="exec_files/sample_McCormick_tpe.csv"

        algo = "rand"
        algo = "anneal"
        algo = "mix"
        algo = "tpe"
        seed = 10

        # np.random.seed(seed)
        # np.random.RandomState(seed)
        # seed = ''

        newname_base = "eval_{}_{}_{}".format(algo, opt_func.name, "seed{}".format(seed))
        eval_file = self.create_eval_file(test_file, newname_base)
        num_trial = 25
        for i in range(num_trial):
            request_dict = make_request_fromcsv(eval_file)
            # request_dict["scope"].keys()

            request_dict["algo"] = algo

            seed_int = np.random.randint(0, 10E+5)
            request_dict["seed"] = seed_int
            request_dict["seed"] = ''
            # request_dict["seed"] = seed
            # request_dict["seed"] = None

            request_dict["scope"][0]["x"][1:] = opt_func.range[0]
            request_dict["scope"][1]["y"][1:] = opt_func.range[1]


            json_str = json.dumps(request_dict)

            executor = OptExecutorFactory.get_executor(json_str)
            rval, _ = executor.suggest()
            # print(rval)
            for j in range(len(rval["statuses"])):
                # args = [x[j] for x in rval["vals"].values()]
                # args = [x[j] for arg in request_dict["results"]["vals"].keys() for x in rval["vals"][arg]]
                args = []
                for arg_name in request_dict["results"]["vals"].keys():
                    args.append(rval["vals"][arg_name][j])

                # print(type(args), type(args[0]))
                print("next:", args)
                loss = func(*args)
                with open(eval_file, 'a') as f:
                    writer = csv.writer(f, lineterminator='\n')
                    writer.writerow([loss]+args)
            if 0 < len(request_dict["results"][COMMON_LOSSES]):
                print("  min is", min(np.array(request_dict["results"][COMMON_LOSSES])))


        print("seed", request_dict["seed"])
        print("algo", request_dict["algo"])
        print("{} min is ".format(opt_func.name), func(*best_param), "  [{}]".format(best_param))

        losses = json.loads(json_str)["results"]["losses"]
        vals = json.loads(json_str)["results"]["vals"]
        print("  minimum loss is", min(np.array(losses)))
        bestindex = np.argmin(losses)
        print("  bestidx:", bestindex, " x:", vals["x"][bestindex], " y:", vals["y"][bestindex])
        src = np.hstack((np.array(vals["x"]).reshape(-1, 1), np.array(vals["y"]).reshape(-1, 1)))
        print("src.shape", src.shape)
        # print("  src:", src)



    def demo_one_arg_func(self):
        def f_x(x):
            return np.cos(1.5 * x) + 0.1 * x


        func = f_x
        best_param = [2]
        test_file="exec_files/sample_03.csv"

        newname_base = "eval_{}".format("one_arg_func")
        eval_file = self.create_eval_file(test_file, newname_base)

        np.random.seed(0)

        num_trial = 20
        for i in range(num_trial):
            request_dict = make_request_fromcsv(eval_file)
            request_dict["algo"] = "tpe"

            seed_int = np.random.randint(0, 10E+2)
            request_dict["seed"] = seed_int
            request_dict["scope"]["x"][1:] = [0, 10]

            json_str = json.dumps(request_dict)

            executor = OptExecutorFactory.get_executor(json_str)
            rval = executor.suggest()

            for j in range(len(rval["statuses"])):

                # args = [x[j] for x in rval["vals"].values()]
                args = [x[j] for x in rval["vals"][arg] for arg in request_dict["results"]["vals"].keys()]

                loss = func(*args)
                with open(eval_file, 'a') as f:
                    writer = csv.writer(f, lineterminator='\n')
                    writer.writerow([loss]+args)

        # print("f_Banana min is ", f_Banana(1, 1))
        print("{} min is ".format("one_arg_func"), func(*best_param), "  [{}]".format(best_param))

        losses = json.loads(json_str)["results"]["losses"]
        vals = json.loads(json_str)["results"]["vals"]
        print("  minimum loss is", min(np.array(losses)))
        bestindex = np.argmin(losses)
        print("  bestidx:", bestindex, " x:",vals["x"][bestindex])


    def demo_file_exec_w_func(self):

        opt_func = funcs4opt.FuncFactory().getFunc(funcs4opt.SPHERE)
        opt_func = funcs4opt.FuncFactory().getFunc(funcs4opt.BANANA)
        opt_func = funcs4opt.FuncFactory().getFunc(funcs4opt.GOLDSTEINPRICE)

        func = opt_func.func
        best_param = opt_func.best_param
        # test_file="exec_files/sample_Rosenbrock_tpe.csv"
        # test_file="exec_files/sample_Rosenbrock_mix.csv"
        test_file="exec_files/sample_Goldstain_Anneal_1000samples.csv"
        test_file="exec_files/sample_Goldstain_rand_1000samples.csv"
        test_file="exec_files/sample_Goldstain_GPEI_1000samples.csv"
        test_file="exec_files/sample_Goldstain_tpe_1000samples.csv"
        test_file="exec_files/sample_SPHERE_GPEI.csv"
        # test_file="exec_files/sample_Goldstain_GPEI_MCMC_100samples.csv"
        file_name, _ = os.path.splitext(test_file)

        newname_base = "eval_{}".format(opt_func.name)
        eval_file = self.create_eval_file(test_file, newname_base)

        # np.random.seed(0)

        num_trial = 2
        for i in range(num_trial):
            request_dict = make_request_fromcsv(eval_file)
            request_dict["seed"] = request_dict["seed"] + len(request_dict["results"]["losses"])
            json_str = json.dumps(request_dict)

            executor = OptExecutorFactory.get_executor(json_str)
            rval, _ = executor.suggest()

            for j in range(len(rval["statuses"])):

                # args = [x[j] for x in rval["vals"].values()]
                # args = [x[j] for x in rval["vals"][arg] for arg in request_dict["results"]["vals"].keys()]
                keys = [list(x.keys())[0] for x in request_dict[COMMON_SCOPE]]
                args = [rval[COMMON_VALS][key][j] for key in keys]

                loss = func(*args)
                with open(eval_file, 'a') as f:
                    writer = csv.writer(f, lineterminator='\n')
                    writer.writerow([loss]+args)

        print("{} min is ".format("one_arg_func"), func(*best_param), "  [{}]".format(best_param))

        losses = json.loads(json_str)["results"]["losses"]
        vals = json.loads(json_str)["results"]["vals"]
        print("  minimum loss is", min(np.array(losses)))
        bestindex = np.argmin(losses)
        print("  bestidx:", bestindex, " x:",vals["x"][bestindex])

        print("keys", vals.keys())
        X_base = vals.values()
        print(" src:", np.array([np.array(line) for line in X_base]).T)


    def demo_file_exec(self):

        opt_func = funcs4opt.FuncFactory().getFunc(funcs4opt.SPHERE)

        func = opt_func.func
        best_param = opt_func.best_param
        test_file="exec_files/sample_TTS_DOE_2nd.csv"
        test_file="exec_files/sample_TTS_DOE_2nd_GPEI.csv"
        test_file="exec_files/sample_TTS_DOE_2nd_GPEIMCMC.csv"

        newname_base = "eval_{}".format("sample_TTS_DOE_2nd")
        eval_file = self.create_eval_file(test_file, newname_base)
        #
        # np.random.seed(0)

        num_trial = 1
        for i in range(num_trial):
            request_dict = make_request_fromcsv(eval_file)
            json_str = json.dumps(request_dict)

            executor = OptExecutorFactory.get_executor(json_str)
            rval, _ = executor.suggest()

            for j in range(len(rval["statuses"])):

                # args = [x[j] for x in rval["vals"].values()]
                # args = [x[j] for x in rval["vals"][arg] for arg in request_dict["results"]["vals"].keys()]
                keys = [list(x.keys())[0] for x in request_dict[COMMON_SCOPE]]
                args = [rval[COMMON_VALS][key][j] for key in keys]

                # loss = func(*args)
                loss = 100 #"unk"
                with open(eval_file, 'a') as f:
                    writer = csv.writer(f, lineterminator='\n')
                    writer.writerow([loss]+args)

        # print("f_Banana min is ", f_Banana(1, 1))
        print("{} min is ".format("one_arg_func"), func(*best_param), "  [{}]".format(best_param))

        losses = json.loads(json_str)["results"]["losses"]
        vals = json.loads(json_str)["results"]["vals"]
        print("  minimum loss is", min(np.array(losses)))
        bestindex = np.argmin(losses)
        # print("  bestidx:", bestindex, " x:",vals["x"][bestindex])

        print("keys", vals.keys())
        X_base = vals.values()
        print(" src:", np.array([np.array(line) for line in X_base]).T)

if __name__ == '__main__':
    import doctest
    doctest.testmod(optionflags=(doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS))
    hy_op = Hyperparam_optimizer()

    # hy_op.main()
    # hy_op.demo_gpyopt()
    # hy_op.demo_hyperopt()
    # hy_op.demo_one_arg_func()
    hy_op.demo_file_exec_w_func()
    # hy_op.demo_file_exec()
