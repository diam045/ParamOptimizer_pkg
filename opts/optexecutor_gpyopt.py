# -*- coding: utf-8 -*-

import numpy as np
from collections import defaultdict
from logging import getLogger
from utils.stop_watch import stop_watch_add
from opts.optexecutor import OptExecutor
from opts.optexecutor import COMMON_SEED, COMMON_SCOPE, COMMON_STATUSES, \
    COMMON_LOSSES, COMMON_VALS, COMMON_RESULTS, COMMON_ALGO, COMMON_MAXEVALS

from GPyOpt.methods import BayesianOptimization

class OptExecutorGpyopt(OptExecutor):
    GPY_MODEL_GP = "GP"
    GPY_MODEL_GPMCMC = "GP_MCMC"
    GPY_MODEL_SPARSEGP = "sparseGP"
    GPY_MODEL_WARPERDGP = "warperdGP"
    GPY_MODEL_INPUTWARPEDGP = "InputWarpedGP"
    GPY_MODEL_RF = "RF"
    GPY_AQUISITION_EI = "EI"
    GPY_AQUISITION_EIMCMC = "EI_MCMC"         # (requires GP_MCMC model).
    GPY_AQUISITION_MPI = "MPI"
    GPY_AQUISITION_MPIMCMC = "MPI_MCMC"       # (requires GP_MCMC model).
    GPY_AQUISITION_LCB = "LCB"
    GPY_AQUISITION_LCBMCMC = "LCB_MCMC"       # (requires GP_MCMC model).
    GPY_TYPE_BANDIT = "bandit"
    GPY_TYPE_DISCRETE = "discrete"
    GPY_TYPE_CONTINUOUS = "continuous"
    GPY_TYPE_CATEGORICAL = "categorical"

    LOWER_LIMIT_QNT_3 = 3   # Experimentally, does not work with lower than 3 rows data

    def __init__(self, json_loaded):
        super().__init__(json_loaded)

        if COMMON_SEED in json_loaded.keys():
            self.set_randomseed(json_loaded[COMMON_SEED])
            np.random.seed(self.rand_seed)

        scope_params = json_loaded[COMMON_SCOPE]
        scope_keys = [x for scope_list in scope_params for x in scope_list.keys()]

        domain = []
        for i, key in enumerate(scope_keys):
            try:
                domain.append({'name': key, 'type': scope_params[i][key][0], 'domain': tuple(scope_params[i][key][1:])})
            except AttributeError:
                print('error occurred [{}] on space creation.'.format(key))

        self.domain = domain
        if COMMON_RESULTS in json_loaded:
            self.Y = None
            Y_base = [json_loaded[COMMON_RESULTS][COMMON_LOSSES]]

            X_base = json_loaded[COMMON_RESULTS][COMMON_VALS]
            self.X = None
            if 0 < len(Y_base[0]):
                y_evals = np.empty(shape=[0, 1])
                x_evals = np.empty(shape=[0, len(scope_keys)])
                for i, y in enumerate(Y_base[0]):
                    y_evals = np.vstack([y_evals, y])
                    x = np.array([X_base[key][i] for key in scope_keys])
                    x_evals = np.vstack([x_evals, x])

                self.Y = y_evals
                self.X = x_evals


    @stop_watch_add
    def suggest(self):
        """Return the next parameter suggestion

        >>> import json
        >>> json_str='{"seed":0,"lib":"hyperopt","algo":"tpe","scope":[{"x":["uniform",-10,10]},'
        >>> json_str+='{"y":["uniform",-10,10]}],'
        >>> json_str+='"max_evals":1,'
        >>> json_str+='"results":{"losses":[3.4620,3.192,28.963,19.64,20.458],'
        >>> json_str+='"statuses":["ok","ok","ok","ok","ok"],'
        >>> json_str+='"vals":{"y":[-0.16774,0.3122,-2.416,0.27455,-3.2827],'
        >>> json_str+='"x":[1.857,1.760,4.785,-4.498,2.837]}}}'
        >>> exec=ExecutorFactory.get_executor(json_str)
        >>> reval=json.loads(exec.suggest())

        >>> reval["alog"] == 'tpe'
        True
        >>> reval["scope"]["x"][0] == 4.30378732744839
        True
        >>> reval["scope"]["y"][0] == 0.9762700785464951
        True
        """
        _logger = getLogger(__name__)
        id_qnt = int(self.json_loaded[COMMON_MAXEVALS])
        algorithm_name = "random"

        histo_qnt = 0 if self.Y is None else len(self.Y.ravel())
        if histo_qnt >= self.LOWER_LIMIT_QNT_3:
            algorithm_name = self.json_loaded[COMMON_ALGO]
            if len(algorithm_name.split(",")) > 1:
                acquisition_str = algorithm_name.split(",")[1]
                algorithm_str = algorithm_name.split(",")[0]
            else:
                acquisition_str = self.GPY_AQUISITION_EI
            self.myBopt = BayesianOptimization(f=None, domain=self.domain,
                                               model_type=algorithm_str,
                                               acquisition_type=acquisition_str,
                                               Y=self.Y,
                                               X=self.X,
                                               model_update_interval=2)

            next_x = self.myBopt.suggest_next_locations()
        else:
            _logger.warning("Initial data is not enough. Create random design.")
            from GPyOpt.core.task.space import Design_space
            from GPyOpt.experiment_design.random_design import RandomDesign

            generate_count = self.LOWER_LIMIT_QNT_3 - histo_qnt
            design = RandomDesign(Design_space(self.domain, None))
            next_x = design.get_samples(generate_count)

        arg_names = [space['name'] for space in self.domain]
        statuses = ["new" for x in next_x]
        vals=defaultdict(list)
        for i in range(len(statuses)):
            for j, key in enumerate(arg_names):
                vals[key].append(next_x[i][j])


        results = dict(algo=algorithm_name, statuses=statuses, vals=vals)

        return results
