# -*- coding: utf-8 -*-

from collections import defaultdict
from logging import getLogger

COMMON_SEED = 'seed'
COMMON_LIB = 'lib'
COMMON_ALGO = 'algo'
COMMON_SCOPE = 'scope'
COMMON_MAXEVALS = 'max_evals'
COMMON_RESULTS = 'results'
COMMON_LOSSES = "losses"
COMMON_STATUSES = "statuses"
COMMON_VALS = "vals"


class OptExecutor(object):
    """
    Algorithmの実行クラス
    """

    def __init__(self, json_loaded):
        self.json_loaded = json_loaded

    def set_randomseed(self, obj):
        _logger = getLogger(__name__)
        try:
            self.rand_seed = int(obj)
        except:
            # _logger.warning('random seed does not int. set default.')
            _logger.info('random seed does not int. set default.')

    def suggest(self):
        """Return the next parameter suggestion

        >>> exec = OptExecutor(None)
        >>> exec.suggest()

        """
        return None

    @staticmethod
    def merge_dict_valuelist(*dicts):
        r = defaultdict(list)
        for d in dicts:
            for k, v in d.items():
                r[k].extend(v)
        return r


if __name__ == '__main__':
    import doctest
    doctest.testmod(optionflags=(doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS))
