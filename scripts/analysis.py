# Python 3.6
"""
Data analysis functions.

Maintainer: Shpakov Konstantin
Link: https://github.com/shpakovkv/SignalProcess
"""

from matplotlib import pyplot as plt
import timeit
import numpy as np
from numba import njit
import time
from data_types import SignalsData, SinglePeak, SingleCurve


def correlation_func_2d(curve1, curve2):
    """ Returns the correlation of a signal (curve1) with
    another signal (curve2) as a function of delay (2D ndarray).
    The input array structure:
    [type][point]
    where type is the type of column - time (0) or value (1)
    point is the index of time-value pair in the array.

    The signals shape must be equal.
    The time step of both signals must be the same.

    The length of output signal is curve1.shape[1]

    :param curve1: 2D ndarray with time column and value column
    :type curve1: np.ndarray
    :param curve2: 2D ndarray with time column and value column
    :type curve2: np.ndarray
    :return: 2D ndarray with time column and value column
    :rtype: np.ndarray
    """
    assert curve1.ndim == curve2.ndim, \
        "Curves have different number of dimensions: {} and {}" \
        "".format(curve1.ndim, curve2.ndim)
    assert curve1.ndim == 2, \
        "The curve1 has the number of dimensions ({}) not as expected ({})." \
        "".format(curve1.ndim, 2)
    assert curve2.ndim == 2, \
        "The curve2 has the number of dimensions ({}) not as expected ({})." \
        "".format(curve2.ndim, 2)
    assert curve1.shape[1] == curve2.shape[1], \
        "The curves have different number of points: {} and {}" \
        "".format(curve1.shape[1], curve2.shape[1])

    time_step = np.ndarray(shape=(2,), dtype=np.float64)
    time_step[0] = (curve1[0, -1] - curve1[0, 0]) / (curve1.shape[1] - 1)
    time_step[1] = (curve2[0, -1] - curve2[0, 0]) / (curve2.shape[1] - 1)
    tolerance = time_step.min() / 1e6
    assert np.isclose(time_step[0], time_step[1], atol=tolerance), \
        "Curve1 and curve2 have different time step: {} and {}. " \
        "The difference exceeds tolerance {}" \
        "".format(time_step[0], time_step[1], tolerance)

    # get correlation
    res = np.correlate(curve1[1], curve2[1], mode='full')

    # add time column
    time_col = np.arange(- curve1.shape[1] + 1, curve1.shape[1], dtype=np.float64)
    time_col *= time_step[0]

    # make 2D array [time/val][point]
    res = np.stack((time_col, res), axis=0)
    return res


def correlation_func_2d_jit(curve1, curve2):
    """

    :param curve1:
    :type curve1: np.ndarray
    :param curve2:
    :type curve2: np.ndarray
    :return:
    :rtype:
    """
    assert curve1.ndim == curve2.ndim, \
        "Curves have different number of dimensions: {} and {}" \
        "".format(curve1.ndim, curve2.ndim)
    assert curve1.ndim == 2, \
        "The curve1 has the number of dimensions ({}) not as expected ({})." \
        "".format(curve1.ndim, 2)
    assert curve2.ndim == 2, \
        "The curve2 has the number of dimensions ({}) not as expected ({})." \
        "".format(curve2.ndim, 2)
    assert curve1.shape[1] == curve2.shape[1], \
        "The curves have different number of points: {} and {}" \
        "".format(curve1.shape[1], curve2.shape[1])

    time_step = np.ndarray(shape=(2,), dtype=np.float64)
    time_step[0] = (curve1[0, -1] - curve1[0, 0]) / (curve1.shape[1] - 1)
    time_step[1] = (curve2[0, -1] - curve2[0, 0]) / (curve2.shape[1] - 1)
    tolerance = time_step.min() / 1e6
    assert np.isclose(time_step[0], time_step[1], atol=tolerance), \
        "Curve1 and curve2 have different time step: {} and {}. " \
        "The difference exceeds tolerance {}" \
        "".format(time_step[0], time_step[1], tolerance)

    # get correlation
    corr_len = curve1.shape[1] * 2 - 1
    res = np.zeros(shape=(2, corr_len), dtype=np.float64)
    fill_correlation_arr(curve1[1], curve2[1], res)

    # add time column
    time_col = np.arange(- curve1.shape[1] + 1, curve1.shape[1], dtype=np.float64)
    time_col *= time_step[0]

    # make 2D array [time/val][point]
    res = np.stack((time_col, res), axis=0)
    return res


def correlation_func_1d(curve1, curve2):
    """

    :param curve1:
    :type curve1: np.ndarray
    :param curve2:
    :type curve2: np.ndarray
    :return:
    :rtype:
    """
    assert curve1.ndim == curve2.ndim, \
        "Curves have different number of dimensions: {} and {}" \
        "".format(curve1.ndim, curve2.ndim)
    assert curve1.ndim == 1, \
        "The curve1 has the number of dimensions ({}) not as expected ({})." \
        "".format(curve1.ndim, 1)
    assert curve2.ndim == 1, \
        "The curve2 has the number of dimensions ({}) not as expected ({})." \
        "".format(curve2.ndim, 1)
    assert curve1.shape[0] == curve2.shape[0], \
        "The curves have different number of points: {} and {}" \
        "".format(curve1.shape[0], curve2.shape[0])

    # get correlation
    res = np.correlate(curve1, curve2, mode='full')

    # add time column
    time_col = np.arange(- curve1.shape[0] + 1, curve1.shape[0])

    # make 2D array [time/val][point]
    res = np.stack((time_col, res), axis=0)
    return res


@njit(parallel=True)
def fill_correlation_arr(curve1, curve2, res):
    corr_len = curve1.shape[0] * 2 - 1
    for idx in range(corr_len):
        shift = idx - curve1.shape[0] + 1
        # res[shift] = get_correlation_chunk(curve1, curve2, shift)
        chunk = 0
        for j in range(curve1.shape[0]):
            if 0 <= j - shift < curve2.shape[0]:
                chunk += curve1[j] * curve2[j - shift]
        res[idx] = chunk


@njit()
def get_correlation_chunk(curve1, curve2, shift):
    chunk = 0
    for j in range(curve1.shape[0]):
        if 0 <= j - shift < curve2.shape[0]:
            chunk += curve1[j] * curve2[j - shift]
    return chunk


def test_correlation_2d():
    t = 1.5
    fs = 441
    f = 7.3
    samples = np.linspace(0, t, int(fs * t), endpoint=False, dtype=np.float64)
    signal = np.sin(2 * np.pi * f * samples)
    data = np.stack((samples, signal), axis=0)
    correlation = correlation_func_2d(data, data)


def test_correlation_2d_jit():
    t = 1.5
    fs = 441
    f = 7.3
    samples = np.linspace(0, t, int(fs * t), endpoint=False, dtype=np.float64)
    signal = np.sin(2 * np.pi * f * samples)
    data = np.stack((samples, signal), axis=0)
    correlation = correlation_func_2d_jit(data, data)


if __name__ == "__main__":
    print("Start")
    t = 1.5
    fs = 44100
    f = 7.3
    samples = np.linspace(0, t, int(fs*t), endpoint=False, dtype=np.float64)
    signal = np.sin(2 * np.pi * f * samples)
    data = np.stack((samples, signal), axis=0)

    f = 8.12
    samples2 = np.linspace(0, t, int(fs * t), endpoint=False, dtype=np.float64)
    signal2 = np.sin(2 * np.pi * f * samples2)
    data2 = np.stack((samples2, signal2), axis=0)

    # plt.plot(samples, signal)
    # plt.show()

    print("Calculating...")

    correlation = correlation_func_2d(data, data2)
    # print(correlation[0])
    # print("================================================================")
    # print(correlation[1])
    time.sleep(1)

    print("Plotting...")
    fig, axes = plt.subplots(2, 1)
    axes[0].plot(samples, signal)
    axes[0].plot(samples2, signal2)
    axes[1].plot(correlation[0], correlation[1])
    plt.show()

    # print(timeit.timeit("test_correlation_2d()", setup="from __main__ import test_correlation_2d", number=100))
    # print(timeit.timeit("test_correlation_2d_jit()", setup="from __main__ import test_correlation_2d_jit", number=100))

    print("Done!")
