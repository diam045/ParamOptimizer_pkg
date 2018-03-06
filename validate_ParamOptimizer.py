# -*- coding: utf-8 -*-

import argparse
import json
import os
import sys
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
from utils import funcs4opt
LOG_LEVEL = 'DEBUG'
sys.path.append(".")

class Hyperparam_optimizer(object):

    @staticmethod
    def create_eval_file(file_path, newname_base):
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

    def exec_optimization(self, filepath, num_trial, func_name, verbose):
        opt_func = funcs4opt.FuncFactory().getFunc(func_name)

        func = opt_func.func
        best_param = opt_func.best_param
        test_file = filepath
        file_name, _ = os.path.splitext(test_file)

        newname_base = "eval_{}_run{}_".format(opt_func.name, num_trial)
        eval_file = self.create_eval_file(test_file, newname_base)

        for i in range(num_trial):
            request_dict = make_request_fromcsv(eval_file)
            json_str = json.dumps(request_dict)

            executor = OptExecutorFactory.get_executor(json_str)
            rval, _ = executor.suggest()

            for j in range(len(rval[COMMON_STATUSES])):
                keys = [list(x.keys())[0] for x in request_dict[COMMON_SCOPE]]
                vals = [rval[COMMON_VALS][key][j] for key in keys]

                loss = func(*vals)
                with open(eval_file, 'a') as f:
                    writer = csv.writer(f, lineterminator='\n')
                    writer.writerow([loss]+vals)

        losses = json.loads(json_str)[COMMON_RESULTS][COMMON_LOSSES]
        vals = json.loads(json_str)[COMMON_RESULTS][COMMON_VALS]
        bestindex = np.argmin(losses)
        if verbose > 0:
            print("{} min is ".format("one_arg_func"), func(*best_param), "  [{}]".format(best_param))
            print("  minimum loss is", min(np.array(losses)))
            print("  bestidx:", bestindex, " x:",vals["x"][bestindex])

        if verbose > 1:
            X_base = vals.values()
            print("keys", vals.keys())
            print(" src:", np.array([np.array(line) for line in X_base]).T)


if __name__ == '__main__':
    import doctest
    doctest.testmod(optionflags=(doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS))
    hy_op = Hyperparam_optimizer()

    parser = argparse.ArgumentParser(
        description='Parameter Optimizer Execution')
    parser.add_argument('--filepath', '-filepath', default='utils/tests/exec_files/sample_01.csv',
                        help='Path to execution example file')
    parser.add_argument('--num_trial', '-num_trial', default=10, type=int)
    parser.add_argument('--func_name', '-func_name', default='Sphere', choices=(funcs4opt.FuncFactory.funcs.keys()))
    parser.add_argument('--verbose', '-verbose', default=1, type=int)
    args = parser.parse_args()

    hy_op.exec_optimization(args.filepath, args.num_trial, args.func_name, args.verbose)
