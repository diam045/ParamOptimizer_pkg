# coding: UTF-8

from opts.optexecutor import COMMON_LIB, COMMON_SEED, COMMON_SCOPE, COMMON_STATUSES, \
    COMMON_LOSSES, COMMON_VALS, COMMON_RESULTS, COMMON_ALGO, COMMON_MAXEVALS

class JsonStructureException(Exception):
    pass

class NotEnoughKeysException(JsonStructureException):
    pass

class WrongStructureException(JsonStructureException):
    pass

def has_keycontents(dict_input):
    '''
    dictonary内の不可欠なKeyの存在確認
    :param dict_input:
    :return: bool
    >>> has_keycontents("")
    Traceback (most recent call last):
        ...
    checker_jsonstructure.NotEnoughKeysException: json could not convert to dict correctly.
    >>> has_keycontents({"lib":"hyperopt","algo":"tpe","scope":[{}]})
    True
    '''

    if not isinstance(dict_input, dict):
        raise NotEnoughKeysException("json could not convert to dict correctly.")

    keys = {COMMON_LIB, COMMON_ALGO, COMMON_SCOPE}
    if not set(dict_input) >= keys:
        raise NotEnoughKeysException("Necessary keys [{}] are insuffisient.".format(keys))

    return True

def is_correct_structure_common(dict_input):
    if not isinstance(dict_input[COMMON_SCOPE], list):
        raise WrongStructureException("Contents of scope should be list.")

def is_correct_structure_hyperopt(dict_input):
    '''

    :param dict_input:
    :return: bool
    >>> is_correct_structure_hyperopt({"scope":[{"x":["uniform",-1,1]}],"results":{"losses":[3],"vals":{"x":[0.5]}}})
    True
    >>> is_correct_structure_hyperopt({"scope":[{"x":["uniform",-1,1]}],"results":{"losses":[3],"vals":{"x":[]}}})
    Traceback (most recent call last):
        ...
    checker_jsonstructure.WrongStructureException: variable counts mismatch between 'losses' and 'vals'
    '''

    is_correct_structure_common(dict_input)

    scope_keys = [x for scope_list in dict_input[COMMON_SCOPE] for x in scope_list.keys()]
    scope_funcs = [x[0] for scope_list in dict_input[COMMON_SCOPE] for x in scope_list.values()]

    if not len(scope_keys) > 0:
        raise WrongStructureException("scope args is insuffisient. len() is {}".format(len(scope_keys)))

    for hp_func in scope_funcs:
        if not isinstance(hp_func, str):
            raise WrongStructureException("scope definition is wrong. first element should be hp func name")

    """
    hp.choice(label, options)           :Returns one of the options, which should be a list or tuple.
    hp.randint(label, upper)            :[0, upper)
    hp.uniform(label, low, high)        :Returns a value uniformly between low and high.
    hp.quniform(label, low, high, q)    :round(uniform(low, high) / q) * q
    hp.loguniform(label, low, high)     :exp(uniform(low, high))
    hp.qloguniform(label, low, high, q) :round(exp(uniform(low, high)) / q) * q
    hp.normal(label, mu, sigma)         :Returns a real value that's normally-distributed with mean mu and standard deviation sigma.
    hp.qnormal(label, mu, sigma, q)     :round(normal(mu, sigma) / q) * q
    hp.lognormal(label, mu, sigma)      :exp(normal(mu, sigma))
    hp.qlognormal(label, mu, sigma, q)  :round(exp(normal(mu, sigma)) / q) * q
    """

    if COMMON_RESULTS in dict_input.keys():
        if not len(dict_input[COMMON_RESULTS][COMMON_VALS].keys()) == len(scope_keys):
            raise WrongStructureException("Mismatch between condition and result. {} != {} [{}]".format(
                len(dict_input[COMMON_RESULTS][COMMON_VALS].keys())
                ,len(scope_keys)
                ,dict_input[COMMON_RESULTS][COMMON_VALS].keys()
            ))
        for key_in_scope in scope_keys:
            if not key_in_scope in dict_input[COMMON_RESULTS][COMMON_VALS].keys():
                raise WrongStructureException(
                    "key mismatch between 'scope' and 'results'.[{}]".format(key_in_scope)
                )
            if not len(dict_input[COMMON_RESULTS][COMMON_VALS][key_in_scope]) \
                    == \
                    len(dict_input[COMMON_RESULTS][COMMON_LOSSES]):
                raise WrongStructureException("variable counts mismatch between 'losses' and 'vals'")

    return True


if __name__ == '__main__':
    print(has_keycontents(""))
