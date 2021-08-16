from data_types import SignalsData, SinglePeak
from file_handler import get_file_list_by_ext, read_signals

import timeit
import time
import copy
import random
import numpy as np

from numba import jit, njit, vectorize, guvectorize, float64, float32, int32, int64, cuda


def print_func_time(func):
    import time

    def wrapper(*args, **kwargs):
        start = time.time_ns()
        func(*args, **kwargs)
        end = time.time_ns()
        print('[*] Runtime of the function: {:.15f} seconds.'.format((end - start) / 1.0E+9))

    return wrapper


def speed_test_