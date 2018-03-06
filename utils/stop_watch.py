# -- coding: utf-8 --

from functools import wraps
import time
from logging import getLogger

def stop_watch(func) :
    @wraps(func)
    def wrapper(*args, **kargs) :
        _logger = getLogger(__name__)
        start = time.time()
        result = func(*args,**kargs)
        elapsed_time =  time.time() - start
        # print("{} took {} sec.".format(func.__name__, elapsed_time))
        _logger.info("{} took {} sec.".format(func.__name__, elapsed_time))
        return result
    return wrapper


def stop_watch_add(func) :
    @wraps(func)
    def wrapper(*args, **kargs) :
        _logger = getLogger(__name__)
        start = time.time()
        result = func(*args,**kargs)
        elapsed_time =  time.time() - start
        return result, elapsed_time
    return wrapper