# -*- coding: utf-8 -*-

from logging import getLogger
from hyperopt import anneal, rand, tpe, hp, Trials, mix, partial
from hyperopt.base import Domain
from utils.stop_watch import stop_watch_add
from opts.optexecutor import OptExecutor

from opts.optexecutor import COMMON_SEED, COMMON_SCOPE, COMMON_STATUSES, \
    COMMON_LOSSES, COMMON_VALS, COMMON_RESULTS, COMMON_ALGO, COMMON_MAXEVALS


class OptExecutorHyperopt(OptExecutor):
    HYP_ALGO_RAND = "rand"
    HYP_ALGO_ANNEAL = "anneal"
    HYP_ALGO_TPE = "tpe"
    HYP_ALGO_MIX = "mix"

    HYP_OUT_RESULT = "result"
    HYP_OUT_STATUS = "status"
    HYP_OUT_MISC = "misc"
    HYP_OUT_VALS = "vals"

    def __init__(self, json_loaded):
        super().__init__(json_loaded)
        self.trials = Trials()

        if COMMON_SEED in json_loaded.keys():
            self.set_randomseed(json_loaded[COMMON_SEED])

        scope_params = json_loaded[COMMON_SCOPE]
        scope_keys = [x for scope_list in scope_params for x in scope_list.keys()]
        space = []
        for i, key in enumerate(scope_keys):
            try:
                #  List先頭を抜いて*listで可変長引数渡し
                # space.append(getattr(hp, v[0])(k, v[1], v[2]))
                #  2018/01 多重層空間定義には対応していない
                method_name = scope_params[i][key][0]
                method_args = scope_params[i][key][1:]
                space.append(getattr(hp, method_name)(key, *method_args))
            except AttributeError:
                print('error occurred [{}] on space creation.'.format(key))

        # space = [
        #             hp.uniform('x', -5, 5),
        #             hp.uniform('y', -5, 5),
        #         ]
        self.domain = Domain(None, space, None, pass_expr_memo_ctrl=False)
        if COMMON_RESULTS in json_loaded:
            losses = json_loaded[COMMON_RESULTS][COMMON_LOSSES]
            statuses = json_loaded[COMMON_RESULTS][COMMON_STATUSES]
            vals = json_loaded[COMMON_RESULTS][COMMON_VALS]
            self.trials = self.create_trials(losses, statuses, vals, scope_keys)

    @staticmethod
    def create_trials(losses, statuses, vals, scope_keys):
        trials = Trials()

        tids = trials.new_trial_ids(len(losses))
        specs = [None for x in range(len(tids))]
        results = []
        miscs = []
        for i in range(len(tids)):
            idxs_content=[[i] for key in scope_keys]
            idxs_vals_content=[]
            for key in scope_keys:
                idxs_vals_content.append([vals[key][i]])

            results.append(dict(loss=losses[i], status=statuses[i]))
            miscs.append(dict(tid=tids[i],
                              cmd=None,
                              idxs=dict(zip(scope_keys, idxs_content)),
                              vals=dict(zip(scope_keys, idxs_vals_content))))

        trials.insert_trial_docs(
            trials.new_trial_docs(
                tids,
                specs,
                results,
                miscs,
            )
        )
        trials.refresh()
        return trials


    @stop_watch_add
    def suggest(self):
        """Return the next parameter suggestion

        >>> import json
        >>> json_str='{"seed":0,"lib":"hyperopt","algo":"tpe","scope":{"x":["uniform",-10,10],"y":["uniform",-10,10]},'
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
        additional_args = []
        executed_alog=self.json_loaded[COMMON_ALGO]
        if executed_alog == self.HYP_ALGO_TPE:
            algo = tpe.suggest
            id_qnt = 1
            additional_args.append(tpe._default_prior_weight)
            additional_args.append(5)   # n_startup_jobs
        elif executed_alog == self.HYP_ALGO_ANNEAL:
            algo = anneal.suggest
        elif executed_alog == self.HYP_ALGO_RAND:
            algo = rand.suggest
        elif executed_alog == self.HYP_ALGO_MIX:
            algo = partial(mix.suggest,
                           p_suggest=[
                               (.1, rand.suggest),
                               (.2, anneal.suggest),
                               (.7, tpe.suggest), ])
        else:
            _logger.warning('unknown algo define. use tpe')
            algo = tpe.suggest

        new_ids = self.trials.new_trial_ids(id_qnt)
        args = [new_ids, self.domain, self.trials, self.rand_seed] + additional_args
        rval_docs = algo(*args)

        statuses = []
        vals={}
        for i in range(len(new_ids)):
            statuses.append(rval_docs[i][self.HYP_OUT_RESULT][self.HYP_OUT_STATUS])
            vals = self.merge_dict_valuelist(vals, rval_docs[i][self.HYP_OUT_MISC][self.HYP_OUT_VALS])

        results = dict(algo=executed_alog, statuses=statuses, vals=vals)

        return results
