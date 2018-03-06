import csv
import sys
from collections import OrderedDict, Callable
from collections import defaultdict
import opts.optexecutor as executor


sys.path.append(".")


class DefaultOrderedDict(OrderedDict):
    # Source: http://stackoverflow.com/a/6190500/562769
    def __init__(self, default_factory=None, *a, **kw):
        if (default_factory is not None and
           not isinstance(default_factory, Callable)):
            raise TypeError('first argument must be callable')
        OrderedDict.__init__(self, *a, **kw)
        self.default_factory = default_factory

    def __getitem__(self, key):
        try:
            return OrderedDict.__getitem__(self, key)
        except KeyError:
            return self.__missing__(key)

    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        self[key] = value = self.default_factory()
        return value

    def __reduce__(self):
        if self.default_factory is None:
            args = tuple()
        else:
            args = self.default_factory,
        return type(self), args, None, None, self.items()

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        return type(self)(self.default_factory, self)

    def __deepcopy__(self, memo):
        import copy
        return type(self)(self.default_factory,
                          copy.deepcopy(self.items()))

    def __repr__(self):
        return 'OrderedDefaultDict(%s, %s)' % (self.default_factory,
                                               OrderedDict.__repr__(self))


KEY_PARAMETERS = [executor.COMMON_SEED,
                  executor.COMMON_LIB,
                  executor.COMMON_ALGO,
                  executor.COMMON_SCOPE,
                  executor.COMMON_MAXEVALS,
                  executor.COMMON_RESULTS]


def _make_request_dict(parameter_dict, scope_list, results_dict, vals_dict, datareader, row):
    if row[0] in KEY_PARAMETERS:
        parameter_dict[row[0]] = None if len(row) < 2 else row[1]
        if parameter_dict[row[0]].isdigit():
            parameter_dict[row[0]] = int(parameter_dict[row[0]])

    if executor.COMMON_SCOPE == row[0]:
        for row_next in datareader:
            if row_next[0] in KEY_PARAMETERS:
                row = row_next
                break

            dict_item = {row_next[0]:[row_next[1]] + ([float(x) for x in row_next[2:] if 0 < len(x)])}
            scope_list.append(dict_item)

        parameter_dict[executor.COMMON_SCOPE] = scope_list
        _make_request_dict(parameter_dict, scope_list, results_dict, vals_dict, datareader, row)

    if executor.COMMON_RESULTS == row[0]:
        args = next(datareader)
        [vals_dict[x] for i, x in enumerate(args) if 0 < i and 0 < len(x)]
        for row_next in datareader:
            results_dict[executor.COMMON_LOSSES].append(float(row_next[0]))
            for i, key in enumerate(vals_dict.keys()):
                vals_dict[key].append(float(row_next[i + 1]))

            if row_next[0] in KEY_PARAMETERS:
                break

        results_dict[executor.COMMON_STATUSES] = ["ok" for x in results_dict[executor.COMMON_LOSSES]]
        results_dict[executor.COMMON_VALS] = vals_dict
        parameter_dict[executor.COMMON_RESULTS] = results_dict


def make_request_fromcsv(csvifle):
    parameter_dict = {}
    scope_list = []
    results_dict = defaultdict(list)
    vals_dict = DefaultOrderedDict(list)

    try:
        with open(csvifle, 'r') as f:
            datareader = csv.reader(f)
            for row in datareader:
                _make_request_dict(parameter_dict, scope_list, results_dict, vals_dict, datareader, row)
    except StopIteration:
        # Normal sequence. End of line.
        pass

    return parameter_dict

if __name__ == '__main__':
    pass


