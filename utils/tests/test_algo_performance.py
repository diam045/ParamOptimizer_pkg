# from unittest import TestCase
# from nose.tools import ok_, eq_
import csv
import json
import shutil
import itertools
import numpy as np
from datetime import datetime
from tqdm import tqdm
from collections import OrderedDict
import os
import sys
sys.path.append("../.")
sys.path.append(".")

from opts import optexecutor
from opts.optexecutor import COMMON_LIB, COMMON_SEED, COMMON_SCOPE, COMMON_STATUSES, \
    COMMON_LOSSES, COMMON_VALS, COMMON_RESULTS, COMMON_ALGO, COMMON_MAXEVALS
from opts.optexecutor_factory import OptExecutorFactory
from opts.optexecutor_factory import COMMON_LIB_HYPEROPT, COMMON_LIB_GPYOPT
from opts.optexecutor_hyperopt import OptExecutorHyperopt as hyp
from opts.optexecutor_gpyopt import OptExecutorGpyopt as gpy
from utils import funcs4opt
from utils.request_dict_creator import make_request_fromcsv


# class AlgorithmPerformTest(TestCase):
class AlgorithmPerformTest(object):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    @staticmethod
    def over_write_condition(request_dict, seed, algo, opt_func, lib):
        request_dict[COMMON_LIB] = lib
        request_dict[COMMON_SEED] = seed
        request_dict[COMMON_ALGO] = algo
        request_dict[COMMON_SCOPE][0]["x"][1:] = opt_func.range[0]
        request_dict[COMMON_SCOPE][1]["y"][1:] = opt_func.range[1]
        if lib == COMMON_LIB_GPYOPT:
            request_dict[COMMON_SCOPE][0]["x"][0] = gpy.GPY_TYPE_CONTINUOUS
            request_dict[COMMON_SCOPE][1]["y"][0] = gpy.GPY_TYPE_CONTINUOUS

    @staticmethod
    def create_eval_env(file_path):
        todaydetail = datetime.today()

        fol = os.path.dirname(file_path)
        name, ext = os.path.splitext(os.path.basename(file_path))
        evalfolder="{}/{}_{}".format(fol, name, todaydetail.strftime("%Y-%m-%d %H%M%S"))
        if not os.path.exists(evalfolder):
            os.makedirs(evalfolder)
        base_file_name = "{}{}".format(name, ext)
        copy_file = "{}/{}".format(evalfolder, base_file_name)
        if not os.path.exists(copy_file):
            print("current:", os.getcwd())
            shutil.copy(file_path, copy_file)

        return evalfolder, copy_file

    def execute_evaluation(self, eval_file, opt_func_name, algo, seed, n_exec, lib):

        opt_func = funcs4opt.FuncFactory().getFunc(opt_func_name)
        func = opt_func.func

        request_dict = make_request_fromcsv(eval_file)
        self.over_write_condition(request_dict, seed, algo, opt_func, lib)

        json_str = json.dumps(request_dict)

        executor = OptExecutorFactory.get_executor(json_str)

        rval, elaps = executor.suggest()

        for j in range(len(rval["statuses"])):
            args = []
            for arg_name in request_dict["results"]["vals"].keys():
                args.append(rval["vals"][arg_name][j])
            loss = func(*args)
            with open(eval_file, 'a') as f:
                writer = csv.writer(f, lineterminator='\n')
                writer.writerow([loss] + args)

        return loss, elaps

    def test_each_performance(self):
        keywords = ["lib", "opt_func_name", "algo", "seed"]
        result_data = []

        test_file="exec_files/sample_01.csv"
        test_file="utils/tests/exec_files/sample_01.csv"

        A = "{},{}".format(gpy.GPY_MODEL_GP, gpy.GPY_AQUISITION_EI)
        B = "{},{}".format(gpy.GPY_MODEL_GP, gpy.GPY_AQUISITION_MPI)
        C = "{},{}".format(gpy.GPY_MODEL_GP, gpy.GPY_AQUISITION_LCB)
        D = "{},{}".format(gpy.GPY_MODEL_GPMCMC, gpy.GPY_AQUISITION_EIMCMC)
        E = "{},{}".format(gpy.GPY_MODEL_GPMCMC, gpy.GPY_AQUISITION_MPIMCMC)
        F = "{},{}".format(gpy.GPY_MODEL_GPMCMC, gpy.GPY_AQUISITION_LCBMCMC)
        G = "{},{}".format(gpy.GPY_MODEL_SPARSEGP, gpy.GPY_AQUISITION_EI)
        H = "{},{}".format(gpy.GPY_MODEL_SPARSEGP, gpy.GPY_AQUISITION_MPI)
        I = "{},{}".format(gpy.GPY_MODEL_SPARSEGP, gpy.GPY_AQUISITION_LCB)

        """ doesn't work very well
        J = "{},{}".format(gpy.GPY_MODEL_WARPERDGP, gpy.GPY_AQUISITION_EI)
        K = "{},{}".format(gpy.GPY_MODEL_WARPERDGP, gpy.GPY_AQUISITION_MPI)
        L = "{},{}".format(gpy.GPY_MODEL_WARPERDGP, gpy.GPY_AQUISITION_LCB)
        M = "{},{}".format(gpy.GPY_MODEL_RF, gpy.GPY_AQUISITION_EI)
        N = "{},{}".format(gpy.GPY_MODEL_RF, gpy.GPY_AQUISITION_MPI)
        O = "{},{}".format(gpy.GPY_MODEL_RF, gpy.GPY_AQUISITION_LCB)
        """

        # evaluate conditions
        # >>>
        libs = [COMMON_LIB_HYPEROPT, COMMON_LIB_GPYOPT]
        algorithms_gpyopt = [A, C]
        algorithms_hyperopt = [hyp.HYP_ALGO_TPE, hyp.HYP_ALGO_ANNEAL, hyp.HYP_ALGO_RAND, hyp.HYP_ALGO_MIX]

        funcs = funcs4opt.FuncFactory().funcs.keys()
        exec_n = 3
        num_trial = 100
        logging_cycle = {5, 10, 20, 25, 50}
        # <<<

        # funcs = [funcs4opt.SPHERE]
        # algorithms_hyperopt = [hyp.HYP_ALGO_TPE, hyp.HYP_ALGO_ANNEAL]


        random_seeds = [x for x in range(exec_n)]
        condition_lists_hyp = (list(itertools.product([COMMON_LIB_HYPEROPT],
                                                      funcs,
                                                      algorithms_hyperopt,
                                                      random_seeds
                                                    )))
        condition_lists_gpy = (list(itertools.product([COMMON_LIB_GPYOPT],
                                                      funcs,
                                                      algorithms_gpyopt,
                                                      random_seeds
                                                    )))
        condition_lists = condition_lists_hyp + condition_lists_gpy

        evalfolder, base_file_path = self.create_eval_env(test_file)

        for condition_list in tqdm(condition_lists):
            # print(condition_list)

            args_dict = dict(zip(keywords, condition_list))

            newname_base = "eval_{}_{}_{}_seed{}".format(*condition_list)
            eval_file = "{}/{}.csv".format(evalfolder, newname_base)
            shutil.copyfile(base_file_path, eval_file)

            args_dict["eval_file"] = eval_file

            losses = []
            elapses = []
            for i in range(num_trial):

                args_dict["n_exec"] = i
                loss, elaps = self.execute_evaluation(**args_dict)
                losses.append(loss)
                elapses.append(elaps)

                if i in logging_cycle:
                    result_data.append(self.make_summ_data(keywords, condition_list, losses, elapses, i))

            result_data.append(self.make_summ_data(keywords, condition_list, losses, elapses, num_trial))

        file_path = "{}/{}".format(evalfolder, "result_summary_{}.csv".format(datetime.today().strftime("%Y-%m-%d %H%M%S")))
        self.write_result(result_data, file_path)

    def make_summ_data(self, keywords, condition_list, losses, elapses, term):
        result_one_data = OrderedDict(zip(keywords, condition_list))
        sum_elaps = sum(np.array(elapses))
        best_loss = min(np.array(losses))
        bestindex = np.argmin(losses)
        result_one_data["term"] = term
        result_one_data["best"] = best_loss
        result_one_data["idx"] = bestindex
        result_one_data["elapsed"] = sum_elaps
        return result_one_data

    def write_result(self, result_data, file_path):
        header = result_data[0].keys()
        with open(file_path, "w", newline='') as f:
            writer = csv.DictWriter(f, header)
            header_row = dict([(val, val) for val in header])
            writer.writerow(header_row)
            for row in result_data:
                writer.writerow(row)


if __name__ == '__main__':

    # print("current:", os.getcwd())
    # os.chdir("../..")
    # print("current:", os.getcwd())

    ins = AlgorithmPerformTest()
    ins.test_each_performance()

