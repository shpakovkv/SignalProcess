# Python 3.6
"""
Speed tests for various variation of function.
This set of tests aimed to find the algorithms
with best performance for several method
used in SignalProcess program.

Maintainer: Shpakov Konstantin
Link: https://github.com/shpakovkv/SignalProcess
"""

from matplotlib import pyplot
import os
import sys
import numpy
import bisect
import argparse

import numpy as np
import scipy.integrate as integrate

import arg_parser
import arg_checker
import file_handler
import plotter
from multiprocessing import Pool


from data_types import SignalsData, SinglePeak
from file_handler import get_file_list_by_ext, read_signals

import timeit
import time
import copy
import random

from numba import jit, njit, vectorize, guvectorize, float64, float32, int32, int64, cuda


def print_func_time(func):
    import time

    def wrapper(*args, **kwargs):
        start = time.time_ns()
        func(*args, **kwargs)
        end = time.time_ns()
        print('[*] Runtime of the function: {:.15f} seconds.'.format((end - start) / 1.0E+9))

    return wrapper


def multiplier_and_delay(data, multiplier, delay):
    """Returns the modified data.
        Each column of the data is first multiplied by
        the corresponding multiplier value and
        then the corresponding delay value is subtracted from it.

        data       -- an instance of the SignalsData class
                      OR 2D numpy.ndarray
        multiplier -- the list of multipliers for each columns
                      in the input data.
        delay      -- the list of delays (subtrahend) for each
                      columns in the input data.
        """
    if multiplier is None and delay is None:
        return data

    if isinstance(data, np.ndarray):
        row_number = data.shape[0]
        col_number = data.shape[1]

        # if not multiplier:
        #     multiplier = [1 for _ in range(col_number)]
        # if not delay:
        #     delay = [0 for _ in range(col_number)]

        for col_idx in range(col_number):
            for row_idx in range(row_number):
                data[row_idx][col_idx] = (data[row_idx][col_idx] *
                                          multiplier[col_idx] -
                                          delay[col_idx])
        return data
    elif isinstance(data, SignalsData):
        # if not multiplier:
        #     multiplier = [1 for _ in range(data.count * 2)]
        # if not delay:
        #     delay = [0 for _ in range(data.count * 2)]

        # check_coeffs_number(data.count * 2, ["multiplier", "delay"],
        #                     multiplier, delay)
        for curve_idx in range(data.count):
            col_idx = curve_idx * 2  # index of time-column of current curve
            data.curves[curve_idx].data = \
                multiplier_and_delay(data.curves[curve_idx].data,
                                     multiplier[col_idx:col_idx + 2],
                                     delay[col_idx:col_idx + 2]
                                     )
        return data
    else:
        raise TypeError("Data must be an instance of "
                        "numpy.ndarray or SignalsData.")


@jit(nopython=True)
def multiplier_and_delay_jit(data, multiplier, delay):
    """Returns the modified data.
                Each column of the data is first multiplied by
                the corresponding multiplier value and
                then the corresponding delay value is subtracted from it.

                data       -- an instance of the SignalsData class
                              OR 2D numpy.ndarray
                multiplier -- the list of multipliers for each columns
                              in the input data.
                delay      -- the list of delays (subtrahend) for each
                              columns in the input data.
                """
    # if multiplier is None and delay is None:
    #     return data

    row_number = data.shape[0]
    col_number = data.shape[1]

    # if not multiplier:
    #
    #     multiplier = [1 for _ in range(col_number)]
    # if not delay:
    #     delay = [0 for _ in range(col_number)]
    # check_coeffs_number(col_number, ["multiplier", "delay"],
    #                     multiplier, delay)
    for col_idx in range(col_number):
        for row_idx in range(row_number):
            data[row_idx][col_idx] = (data[row_idx][col_idx] *
                                      multiplier[col_idx] -
                                      delay[col_idx])
    return data


# def multiplier_and_delay_jit2222(data, multiplier, delay):
#     for col_idx in range(col_number):
#         for row_idx in range(row_number):
#             data[row_idx][col_idx] = (data[row_idx][col_idx] *
#                                       multiplier[col_idx] -
#                                       delay[col_idx])
#     return data


@guvectorize([(float64[:], float64, float64, float64[:])], '(n),(),()->(n)')
def mult_del_row_order_vectorized(data, multiplyer, delay, res):
    for i in range(data.shape[0]):
        res[i] = data[i] * multiplyer - delay


@jit(nopython=True, nogil=True)
def mult_del_row_order_jit_nogil(data, multiplier, delay):
    row_number = data.shape[0]
    point_number = data.shape[1]
    for row_idx in np.arange(row_number):
        for point_idx in np.arange(point_number):
            data[row_idx][point_idx] = (data[row_idx][point_idx] *
                                        multiplier[row_idx] -
                                        delay[row_idx])
    return data


@jit(nopython=True)
def mult_del_row_order_jit(data, multiplier, delay):
    row_number = data.shape[0]
    point_number = data.shape[1]
    for row_idx in np.arange(row_number):
        for point_idx in np.arange(point_number):
            data[row_idx][point_idx] = (data[row_idx][point_idx] *
                                        multiplier[row_idx] -
                                        delay[row_idx])
    return data


@jit(nopython=True, nogil=True, parallel=True)
def mult_del_row_order_parallel(data, multiplier, delay):
    row_number = data.shape[0]
    point_number = data.shape[1]
    for row_idx in np.arange(row_number):
        for point_idx in np.arange(point_number):
            data[row_idx][point_idx] = (data[row_idx][point_idx] *
                                        multiplier[row_idx] -
                                        delay[row_idx])
    return data


def mult_del_row_order_std(data, multiplier, delay):
    row_number = data.shape[0]
    point_number = data.shape[1]
    for row_idx in np.arange(row_number):
        for point_idx in np.arange(point_number):
            data[row_idx][point_idx] = (data[row_idx][point_idx] *
                                        multiplier[row_idx] -
                                        delay[row_idx])
    return data


def test_multiplier_and_delay():
    """

    """

    def manual_compare(test_data_folder):

        file_list = get_file_list_by_ext(test_data_folder, "csv")
        data = read_signals([file_list[0]])
        np_data = data.curves[0].data
        print(np_data)
        mult = np.zeros((2,), dtype=np_data.dtype, order='C')
        mult[0] = 1E-9
        mult[1] = 1
        delay = np.zeros((2,), dtype=np_data.dtype, order='C')
        delay[1] = 100

        # average = 0
        number = 1
        std_time = list()
        for idx in range(number):
            tmp = copy.copy(np_data)
            # print(data)
            # print(tmp)
            # print(np_data)
            start_time = time.time()
            # multiplier_and_delay(tmp, mult, delay)
            # print("===============================================")
            # print(data.curves[0].data.shape)
            # print("===============================================")
            multiplier_and_delay(tmp, mult, delay)
            stop_time = time.time()
            std_time.append(stop_time - start_time)
            # average += (stop_time - start_time) / number

        print("Standard Multiplier_and_delay --- {:.10f} seconds for 1 shot / {} loops ---".format(sum(std_time) / len(std_time), number))

        # average = 0
        jit_time = list()
        for idx in range(number):
            tmp = copy.copy(np_data)
            start_time = time.time()
            multiplier_and_delay_jit(tmp, mult, delay)
            stop_time = time.time()
            # average += (stop_time - start_time) / number
            jit_time.append(stop_time - start_time)

        print("jit + Multiplier_and_delay --- {:.10f} seconds for 1 shot / {} loops ---".format(sum(jit_time) / len(jit_time), number))

        print()
        for first, second in zip(std_time, jit_time):
            print("{:.15f}  :  {:.15f}".format(first, second))

    def timeit_compare(folder, timeit_loops):
        env_std = "test_data_folder = '" + folder + "'\n"

        env_std += "file_list = get_file_list_by_ext(test_data_folder, 'csv')\n"
        env_std += "data = read_signals([file_list[0]])\n"
        env_std += "np_data = data.curves[0].data\n"
        env_std += "mult = np.zeros((2,), dtype=np_data.dtype, order='C')\n"
        env_std += "mult[0] = 1E-6\n"
        env_std += "mult[1] = 1.3\n"
        env_std += "delay = np.zeros((2,), dtype=np_data.dtype, order='C')\n"
        env_std += "delay[0] = -100\n"
        env_std += "delay[1] = 100\n"

        std_time = timeit.timeit("multiplier_and_delay(data, mult, delay)",
                                 setup=env_std, number=timeit_loops, globals=globals())

        print("STD - multiplayer_and_delay avg time = {} s  for {} timeit_loops"
              "".format(std_time, timeit_loops))

        jit_time = timeit.timeit("multiplier_and_delay_jit(np_data, mult, delay)",
                                 setup=env_std, number=timeit_loops, globals=globals())
        print("JIT - multiplayer_and_delay avg time = {} s  for {} timeit_loops"
              "".format(jit_time, timeit_loops))

    # manual_compare()
    folder = "F:\\\\PROJECTS\\\\Python\\\\SignalProcess\\\\untracked\\\\test_multiprocessing\\\\multiprocessing_testfiles"
    # timeit_compare(folder, 100)
    manual_compare(folder)


def test_vectorized_mult_del(number=1000):
    np_data = np.zeros(shape=(2, number), dtype=np.float64, order='C')
    print(np_data)
    for idx in range(number):
        np_data[0, idx] = idx
        np_data[1, idx] = random.random() * 1000

    print(np_data)
    print(min(np_data[:, 1]))
    print(max(np_data[:, 1]))
    print()

    mult = np.zeros((2,), dtype=np_data.dtype, order='C')
    mult[0] = 1E+6
    mult[1] = 1.13768
    delay = np.zeros((2,), dtype=np_data.dtype, order='C')
    delay[0] = 3543137
    delay[1] = 437.513

    # print(mult)
    # print(delay)

    print("=======================================================================")
    print("----------   STD   ----------------------------------------------------")
    print(".......................................................................")
    tmp_std = np_data.copy()
    start_time = time.time()
    mult_del_row_order_std(tmp_std, mult, delay)
    stop_time = time.time()
    total_time_std = stop_time - start_time
    print(tmp_std)
    print(min(tmp_std[:, 1]))
    print(max(tmp_std[:, 1]))

    print("=======================================================================")
    print("----------   VECTOR   -------------------------------------------------")
    print(".......................................................................")
    tmp2 = np_data.copy()
    out_v = np.zeros_like(tmp2)
    start_time = time.time()
    mult_del_row_order_vectorized(tmp2, mult, delay, out_v)
    stop_time = time.time()
    total_time_vec = stop_time - start_time
    print(out_v)
    print(min(out_v[:, 1]))
    print(max(out_v[:, 1]))

    print("=======================================================================")
    print("----------   NJIT   ---------------------------------------------------")
    print(".......................................................................")
    tmp_njit = np_data.copy()
    start_time = time.time()
    mult_del_row_order_jit(tmp_njit, mult, delay)
    stop_time = time.time()
    total_time_njit = stop_time - start_time
    print(tmp_njit)
    print(min(tmp_njit[:, 1]))
    print(max(tmp_njit[:, 1]))

    print("=======================================================================")
    print("----------   NJIT nogil  ----------------------------------------------")
    print(".......................................................................")
    tmp_nogil = np_data.copy()
    start_time = time.time()
    mult_del_row_order_jit_nogil(tmp_nogil, mult, delay)
    stop_time = time.time()
    total_time_nogil = stop_time - start_time
    print(tmp_nogil)
    print(min(tmp_nogil[:, 1]))
    print(max(tmp_nogil[:, 1]))

    # print("=======================================================================")
    # print("----------   NJIT nogil parallel  -------------------------------------")
    # print(".......................................................................")
    tmp_parallel = np_data.copy()
    start_time = time.time()
    mult_del_row_order_parallel(tmp_parallel, mult, delay)
    stop_time = time.time()
    total_time_parallel = stop_time - start_time
    print(tmp_parallel)
    print(min(tmp_parallel[:, 1]))
    print(max(tmp_parallel[:, 1]))

    print("STD time        (first call) = {}".format(total_time_std))
    print("Vectorized time (first call) = {}".format(total_time_vec))
    print("NJIT time       (first call) = {}".format(total_time_njit))
    print("nogil time      (first call) = {}".format(total_time_nogil))
    print("parallel time   (first call) = {}".format(total_time_parallel))

    tmp_std = np_data.copy()
    start_time = time.time()
    mult_del_row_order_std(tmp_std, mult, delay)
    stop_time = time.time()
    total_time_std = stop_time - start_time

    tmp2 = np_data.copy()
    out_v = np.zeros_like(tmp2)
    start_time = time.time()
    mult_del_row_order_vectorized(tmp2, mult, delay, out_v)
    stop_time = time.time()
    total_time_vec = stop_time - start_time

    tmp_njit = np_data.copy()
    start_time = time.time()
    mult_del_row_order_jit(tmp_njit, mult, delay)
    stop_time = time.time()
    total_time_njit = stop_time - start_time

    tmp_nogil = np_data.copy()
    start_time = time.time()
    mult_del_row_order_jit_nogil(tmp_nogil, mult, delay)
    stop_time = time.time()
    total_time_nogil = stop_time - start_time

    tmp_parallel = np_data.copy()
    start_time = time.time()
    mult_del_row_order_parallel(tmp_parallel, mult, delay)
    stop_time = time.time()
    total_time_parallel = stop_time - start_time

    print()
    print("STD time        (second call) = {}".format(total_time_std))
    print("Vectorized time (second call) = {}".format(total_time_vec))
    print("NJIT time       (second call) = {}".format(total_time_njit))
    print("nogil time      (second call) = {}".format(total_time_nogil))
    print("parallel time   (second call) = {}".format(total_time_parallel))


    # save_as = "F:\\PROJECTS\\Python\\SignalProcess\\untracked\\test_multiprocessing\\foo.csv"
    # np.savetxt(save_as, out, delimiter=",")

def test_numba(number=1000000):

    # ==========================================================================
    # --------   ROW-ORDERED ARRAY   -------------------------------------------
    # ..........................................................................
    @print_func_time
    @guvectorize([(float64[:], float64, float64, float64[:])], '(n),(),()->(n)', target='cuda')
    def mult_del_row_order_vectorized_cuda(data, multiplyer, delay, res):
        for i in range(data.shape[0]):
            res[i] = data[i] * multiplyer - delay

    @print_func_time
    @guvectorize([(float64[:], float64, float64, float64[:])], '(n),(),()->(n)', target='cpu')
    def mult_del_row_order_vectorized_cpu(data, multiplyer, delay, res):
        for i in range(data.shape[0]):
            res[i] = data[i] * multiplyer - delay

    @print_func_time
    @jit(nopython=True, nogil=True)
    def mult_del_row_order_jit_nogil(data, multiplier, delay):
        row_number = data.shape[0]
        point_number = data.shape[1]
        for row_idx in np.arange(row_number):
            for point_idx in np.arange(point_number):
                data[row_idx][point_idx] = (data[row_idx][point_idx] *
                                            multiplier[row_idx] -
                                            delay[row_idx])

    @print_func_time
    @jit(nopython=True)
    def mult_del_row_order_jit(data, multiplier, delay):
        row_number = data.shape[0]
        point_number = data.shape[1]
        for row_idx in np.arange(row_number):
            for point_idx in np.arange(point_number):
                data[row_idx][point_idx] = (data[row_idx][point_idx] *
                                            multiplier[row_idx] -
                                            delay[row_idx])

    # # @print_func_time
    # @cuda.jit('void(float64[:,:], float64[:], float64[:])')
    # def mult_del_row_order_cuda(data, multiplier, delay):
    #     row_number = data.shape[0]
    #     point_number = data.shape[1]
    #     for row_idx in range(row_number):
    #         for point_idx in range(point_number):
    #             data[row_idx][point_idx] = (data[row_idx][point_idx] *
    #                                         multiplier[row_idx] -
    #                                         delay[row_idx])

    @print_func_time
    @jit(nopython=True, nogil=True, parallel=True)
    def mult_del_row_order_parallel(data, multiplier, delay):
        row_number = data.shape[0]
        point_number = data.shape[1]
        for row_idx in np.arange(row_number):
            for point_idx in np.arange(point_number):
                data[row_idx][point_idx] = (data[row_idx][point_idx] *
                                            multiplier[row_idx] -
                                            delay[row_idx])

    @print_func_time
    def mult_del_row_order_std(data, multiplier, delay):
        row_number = data.shape[0]
        point_number = data.shape[1]
        for row_idx in np.arange(row_number):
            for point_idx in np.arange(point_number):
                data[row_idx][point_idx] = (data[row_idx][point_idx] *
                                            multiplier[row_idx] -
                                            delay[row_idx])

    np_data = np.zeros(shape=(2, number), dtype=np.float64, order='C')
    for idx in range(number):
        np_data[0, idx] = idx
        np_data[1, idx] = random.random() * 1000

    print("==========================================================================")
    print("--------   ROW-ORDERED ARRAY   -------------------------------------------")
    print("..........................................................................")
    print("Data.shape = {}".format(np_data.shape))
    print("Min_y = {}     Max_y = {}".format(min(np_data[:, 1]), max(np_data[:, 1])))
    print()
    print(np_data)
    print()

    mult = np.zeros((2,), dtype=np_data.dtype, order='C')
    mult[0] = 1E+6
    mult[1] = 1.13768
    delay = np.zeros((2,), dtype=np_data.dtype, order='C')
    delay[0] = 3543137
    delay[1] = 437.513

    # WORKING COPY of data
    out_tmp = np.zeros_like(np_data)
    # tmp_vector = np_data.copy()
    # tmp_vector_out = np.zeros_like(tmp_vector)
    # tmp_njit = np_data.copy()
    # tmp_nogil = np_data.copy()

    print("Multiplier = {}".format(mult))
    print("Delay      = {}".format(delay))
    print("============================================")

    print("Standard function:")
    std_output = np_data.copy()
    mult_del_row_order_std(std_output, mult, delay)
    repeats = 10
    print()
    print(std_output)
    print()

    print("Jitted function:")
    for _ in range(repeats):
        tmp_std = np_data.copy()
        mult_del_row_order_jit(tmp_std, mult, delay)
    # print("Data is {}".format("OK" if np.isclose(std_output, tmp_std).all() else "bad!!"))

    # tmp_std = np_data.copy()
    # mult_del_row_order_jit(tmp_std, mult, delay)

    # print("Data is {}".format("OK" if np.isclose(std_output, tmp_std).all() else "bad!!"))
    print()

    print("NO-GIL function:")
    for _ in range(repeats):
        tmp_std = np_data.copy()
        mult_del_row_order_jit_nogil(tmp_std, mult, delay)
    # print("Data is {}".format("OK" if np.isclose(std_output, tmp_std).all() else "bad!!"))

    # tmp_std = np_data.copy()
    # mult_del_row_order_jit_nogil(tmp_std, mult, delay)

    # print("Data is {}".format("OK" if np.isclose(std_output, tmp_std).all() else "bad!!"))
    print()

    print("Parallel no-gil function:")
    for _ in range(repeats):
        tmp_std = np_data.copy()
        mult_del_row_order_parallel(tmp_std, mult, delay)
    # print("Data is {}".format("OK" if np.isclose(std_output, tmp_std).all() else "bad!!"))

    # tmp_std = np_data.copy()
    # mult_del_row_order_parallel(tmp_std, mult, delay)

    # print("Data is {}".format("OK" if np.isclose(std_output, tmp_std).all() else "bad!!"))

    # print()
    # print("Pause 5 seconds ~~~~~~~~")
    # print()
    # time.sleep(5)

    print("Guvectorized (CPU) function:")
    for _ in range(repeats):
        tmp_std = np_data.copy()
        out_tmp = np.zeros_like(np_data)
        mult_del_row_order_vectorized_cpu(tmp_std, mult, delay, out_tmp)
        # print("Data is {}".format("OK" if np.isclose(out_tmp, std_output).all() else "bad!!"))

    print("Guvectorized (CPU) no out buffer function:")
    for _ in range(repeats):
        tmp_std = np_data.copy()
        mult_del_row_order_vectorized_cpu(tmp_std, mult, delay, tmp_std)
        # print("Data is {}".format("OK" if np.isclose(tmp_std, std_output).all() else "bad!!"))

    print("Guvectorized (CUDA) no out buffer function:")
    for _ in range(repeats):
        tmp_std = np_data.copy()
        mult_del_row_order_vectorized_cuda(tmp_std, mult, delay, tmp_std)
        # print("Data is {}".format("OK" if np.isclose(tmp_std, std_output).all() else "bad!!"))

    # print("CUDA function:")
    # threadsperblock = 32
    # blockspergrid = (tmp_std.size + (threadsperblock - 1)) // threadsperblock
    # for _ in range(2):
    #     tmp_std = np_data.copy()
    #     # mult_del_row_order_jit_nogil(tmp_std, mult, delay)
    #     mult_del_row_order_cuda[threadsperblock, blockspergrid](tmp_std, mult, delay)
    #     print("Data is {}".format("OK" if np.isclose(tmp_std, std_output).all() else "bad!!"))
    # print(tmp_std)
    # print()
    # print(std_output)

    # print("Data is close for {} from {} elements".format(np.isclose(out_tmp, std_output).sum(),
    #                                                      std_output.shape[0] * std_output.shape[1]))
    # print(out_tmp)
    # print()
    # print(tmp_std)

    # tmp_std = np_data.copy()
    # out_tmp = np.zeros_like(np_data)
    # mult_del_row_order_vectorized(tmp_std, mult, delay, out_tmp)

    # print("Data is {}".format("OK" if np.isclose(out_tmp, std_output).all() else "bad!!"))
    # print("Data is close for {} from {} elements".format(np.isclose(out_tmp, std_output).sum(),
    #                                                      std_output.shape[0] * std_output.shape[1]))
    print()

    # for idx in range(30):
    #     print("{x1}\t{y1}\n{x2}\t{y2}\n".format(x1=std_output[0, idx], y1=std_output[1, idx],
    #                                             x2=out_tmp[0, idx], y2=out_tmp[1, idx]))
    # ==========================================================================
    # --------   CLASSIC COLUMN-ORDERED ARRAY   --------------------------------
    # ..........................................................................

    @print_func_time
    @jit(nopython=True)
    def mult_del_col_order_jit(data, multiplier, delay):
        col_number = data.shape[1]
        row_number = data.shape[0]
        for col_idx in np.arange(col_number):
            for row_idx in np.arange(row_number):
                data[col_idx][row_idx] = (data[col_idx][row_idx] *
                                          multiplier[col_idx] -
                                          delay[col_idx])

    @print_func_time
    def mult_del_trans_and_vec(data, multiplier, delay, res):
        @guvectorize([(float64[:], float64, float64, float64[:])], '(n),(),()->(n)')
        def mult_del_row_order_vectorized(data, multiplyer, delay, res):
            for i in range(data.shape[0]):
                res[i] = data[i] * multiplyer - delay

        out_data = np.transpose(data)
        res = np.transpose(res)
        mult_del_row_order_vectorized(out_data, multiplier, delay, res)
        return np.transpose(res)

    classic_data = np.zeros(shape=(number, 2), dtype=np.float64, order='C')
    for idx in range(number):
        classic_data[idx, 0] = idx
        classic_data[idx, 1] = random.random() * 1000

    # print("Data.shape = {}".format(classic_data.shape))
    # print("Min_y = {}     Max_y = {}".format(min(classic_data[:, 1]), max(classic_data[:, 1])))
    # print()
    # print(classic_data)
    # print()
    # cx = classic_data.copy()


if __name__ == "__main__":
    # test_multiplier_and_delay()
    # test_vectorized_mult_del(1000000)
    test_numba(10000000)


    @print_func_time
    def rotate_array(arr):
        np.transpose(arr)


    # np_data = np.zeros(shape=(2, 1000000000), dtype=np.float64, order='C')
    # print(np_data.shape)
    # start = time.time_ns()
    # np_data = np.transpose(np_data)
    # np_data = np.transpose(np_data)
    # np_data = np.transpose(np_data)
    # np_data = np.transpose(np_data)
    # np_data = np.transpose(np_data)
    # end = time.time_ns()
    # print(np_data.shape)
    # print()
    # print(end - start)
