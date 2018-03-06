# -*- coding: utf-8 -*-

import json
from utils import checker_jsonstructure as checker
from opts.optexecutor import COMMON_LIB
from opts.optexecutor_hyperopt import OptExecutorHyperopt
from opts.optexecutor_gpyopt import OptExecutorGpyopt

COMMON_LIB_HYPEROPT = "hyperopt"
COMMON_LIB_GPYOPT = "gpyopt"

class OptExecutorFactory(object):

    def get_executor(string_json):
        """
        Algorithmの実行インスタンスを取得するMethod
        >>>
        >>> string_json = '{"seed":0,"lib":"hyperopt","scope":{"x":["uniform",-10,10],"y":["uniform",-10,10]}'
        >>> string_json += '}'
        >>> ret = OptExecutorFactory.get_executor(string_json)
        >>> isinstance(ret, OptExecutorHyperopt)
        True
        """

        json_loaded = json.loads(string_json)
        checker.has_keycontents(json_loaded)
        if json_loaded[COMMON_LIB] == COMMON_LIB_HYPEROPT:
            checker.is_correct_structure_hyperopt(json_loaded)
            return OptExecutorHyperopt(json_loaded)
        elif json_loaded[COMMON_LIB] == COMMON_LIB_GPYOPT:
            return OptExecutorGpyopt(json_loaded)

        else:
            assert "#TODO no implementation {}".format(json_loaded[COMMON_LIB])


