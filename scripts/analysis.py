# Python 3.6
"""
Data analysis functions.

Maintainer: Shpakov Konstantin
Link: https://github.com/shpakovkv/SignalProcess
"""
import numpy
from matplotlib import pyplot as plt
import timeit
import numpy as np
from numba import njit
import time
from data_types import SignalsData, SinglePeak, SingleCurve
from arg_checker import check_plot_param
from plotter import find_nearest_idx
from scipy.signal import correlate as scipy_correlate
from scipy import interpolate


# =======================================================================
# ------   CORRELATION   ------------------------------------------------
# =======================================================================


def print_curve_base(arr, name=""):
    print(f"[{name}] Start = {arr[0]},  Stop = {arr[-1]},  Step = {arr[1] - arr[0]},   length = {len(arr)}")


def correlation_func_2d(curve1, curve2):
    """ Returns the correlation of a signal (curve1) with
    another signal (curve2) as a function of delay (2D ndarray).

    Output values are normalized to the autocorrelation value
    of the smallest of the input curves at t=0.

    The input array structure:
    [type][point]
    where type is the type of column - time (0) or amplitude (1)
    point is the index of time/amplitude value in the array.

    The number of signal points may not match.
    The time step of both signals must be the same !!

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
    # assert curve1.shape[1] == curve2.shape[1], \
    #     "The curves have different number of points: {} and {}" \
    #     "".format(curve1.shape[1], curve2.shape[1])

    time_step = np.ndarray(shape=(2,), dtype=np.float64)
    time_step[0] = (curve1[0, -1] - curve1[0, 0]) / (curve1.shape[1] - 1)
    time_step[1] = (curve2[0, -1] - curve2[0, 0]) / (curve2.shape[1] - 1)
    tolerance = time_step.min() / 1e6

    # assert np.isclose(time_step[0], time_step[1], atol=tolerance), \
    #     "Curve1 and curve2 have different time step: {} and {}. " \
    #     "The difference exceeds tolerance {}" \
    #     "".format(time_step[0], time_step[1], tolerance)

    if not np.isclose(time_step[0], time_step[1], atol=tolerance):
        if time_step[1] < time_step[0]:
            # curve_2 is more precise, need to interpolate curve_1
            time_step[0] = time_step[1]
            interp_funk = interpolate.interp1d(curve1[0], curve1[1])
            new_curve_x = numpy.arange(curve1[0, 0], stop=curve1[0, -1], step=time_step[0])
            new_curve_y = interp_funk(new_curve_x)
            curve1 = numpy.stack((new_curve_x, new_curve_y))
        else:
            # curve_1 is more precise, need to interpolate curve_2
            time_step[1] = time_step[0]
            interp_funk = interpolate.interp1d(curve2[0], curve2[1])
            new_curve_x = numpy.arange(curve2[0, 0], stop=curve2[0, -1], step=time_step[1])
            new_curve_y = interp_funk(new_curve_x)
            curve2 = numpy.stack((new_curve_x, new_curve_y))

    # get correlation
    # res = np.correlate(curve1[1], curve2[1], mode='full')
    # print_curve_base(curve1[1], "CURVE_1")
    # print_curve_base(curve2[1], "CURVE_2")
    res = scipy_correlate(curve1[1], curve2[1], mode='full')

    # add time column
    # np.arange(first, last, step)  == [first, last)  last point is not included
    # print(f"start = {curve1[0, 0] - curve2[0, -1]},    stop = {curve1[0, -1] - curve2[0, 0] + time_step[0]},   "
    #       f"step = {time_step[0]}")
    time_col = np.arange(curve1[0, 0] - curve2[0, -1],
                         curve1[0, -1] - curve2[0, 0] + time_step[0],
                         time_step[0],
                         dtype=np.float64)

    # even/odd length correction
    if len(time_col) > len(res):
        time_col = time_col[:-1]

    # take the autocorrelation of the smallest curve as a reference
    smaller_curve = curve1
    if curve1.shape[1] > curve2.shape[1]:
        smaller_curve = curve2

    auto_corr = scipy_correlate(smaller_curve[1], smaller_curve[1], mode='full')
    auto_corr_center = smaller_curve.shape[1] - 1

    # auto_corr_0 is the auto-correlation of curve_1 at shift=0
    auto_corr_0 = auto_corr[auto_corr_center]

    # normalization
    res /= auto_corr_0

    # make 2D array [time/val][point]
    res = np.stack((time_col, res), axis=0)
    return res


def correlate_single(curve1, curve2):
    """ Returns the correlation function of two input curves.

    :param curve1: first SingleCurve
    :type curve1: SingleCurve
    :param curve2: second SingleCurve
    :type curve2: SingleCurve
    :return: new SingleCurve with correlation function
    :rtype: SingleCurve
    """
    corr = correlation_func_2d(curve1.data, curve2.data)

    label = "Corr_{}_to_{}".format(curve1.label, curve2.label)
    return SingleCurve(corr,
                       label=label,
                       units="a.u.",
                       t_units=curve1.t_units
                       )


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
    # np.arange(first, last, step)  == [first, last)  last point is not included
    time_col = np.arange(curve1[0, 0] - curve2[0, -1],
                         curve1[0, -1] - curve2[0, 0] + time_step[0],
                         time_step[0],
                         dtype=np.float64)

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


def correlate_multiple(signals_data, list_of_parameter_sets):
    """ Calculates correlation function for one or more
    curve pairs.
    Return a list of SingleCurves with correlation functions.

    :param signals_data:
    :type signals_data: SignalsData
    :param list_of_parameter_sets:
    :type list_of_parameter_sets:
    :return: list of SingleCurves with correlation functions
    :rtype: list
    """
    corr_data = list()

    for idx, correlate_set in enumerate(list_of_parameter_sets):
        curve1_idx, curve2_idx, add_to_signals = correlate_set

        corr_curve = correlate_single(signals_data.get_single_curve(curve1_idx),
                                      signals_data.get_single_curve(curve2_idx)
                                      )
        corr_data.append(corr_curve)
        if add_to_signals:
            signals_data.add_from_array(corr_curve.data,
                                        labels=[corr_curve.label],
                                        units=["a.u."],
                                        time_units=signals_data.time_units
                                        )
    return corr_data


def do_correlate(signals_data, cl_args):
    """ Calculates correlation function for one or more
    curve pairs. Curve indices are taken from cl_args namespace.

    cl_args.correlate is a list, each element of which contains
    the parameters necessary for the correlation analysis
    of two curves from sigals_data : (Curve1_idx, Curve2_idx, AddToSignals?)
    Where the 'AddToSignals?' parameter indicates whether
    the resulting correlation curve should be added
    to the signals_data.

    Return a list of SingleCurves with correlation functions.

    :param signals_data: signals data
    :type signals_data: SignalsData
    :param cl_args: namespace with command line arguments
    :type cl_args: argparse.Namespace
    :return: list of correlation curves data (SignalsData)
    :rtype: list
    """

    for correlate_set in cl_args.correlate:
        check_plot_param(correlate_set[:2], signals_data.cnt_curves, param_name="correlate")

    correlate_data = correlate_multiple(signals_data, cl_args.correlate)

    return correlate_data


def do_correlate_part(signals_data, cl_args):
    """ Calculates correlation function for one or more
    curve segment pairs. Curve indices and segment borders
    are taken from cl_args namespace.

    cl_args.correlate_part is a list, each element of which contains
    the parameters necessary for the correlation analysis
    of two curve segments from sigals_data :
    (Curve1_idx, left1, right1, Curve2_idx, left2, right2, AddToSignals?)

    Where:

    left and right define the boundaries of the section
    of the corresponding curve to be processed;

    the 'AddToSignals?' parameter indicates whether
    the resulting correlation curve should be added
    to the signals_data.

    Return a list of SingleCurves with correlation functions.
    :param signals_data:
    :type signals_data:
    :param cl_args:
    :type cl_args:
    :return: list of correlation curves data (SignalsData)
    :rtype: list
    """
    for correlate_set in cl_args.correlate_part:
        check_plot_param(correlate_set[0:4:3], signals_data.cnt_curves, param_name="correlate_part")

    correlate_data = correlate_part_multiple(signals_data, cl_args.correlate_part)
    return correlate_data


def correlate_part_multiple(signals_data, list_of_parameter_sets):
    """ Calculates correlation function for one or more
    curve pairs.
    Return a list of SingleCurves with correlation functions.

    :param signals_data:
    :type signals_data: SignalsData
    :param list_of_parameter_sets:
    :type list_of_parameter_sets:
    :return: list of SingleCurves with correlation functions
    :rtype: list
    """
    corr_data = list()

    for idx, correlate_set in enumerate(list_of_parameter_sets):
        curve1_idx, left1, right1, curve2_idx, left2, right2, add_to_signals = correlate_set
        curve1 = signals_data.get_single_curve(curve1_idx)
        if left1 != right1:
            start = find_nearest_idx(curve1.get_x(), left1, side='right')
            stop = find_nearest_idx(curve1.get_x(), right1, side='left')
            curve1 = SingleCurve(curve1.data[:, start: stop],
                                 label=curve1.label,
                                 units=curve1.units,
                                 t_units=curve1.t_units
                                 )

        curve2 = signals_data.get_single_curve(curve2_idx)
        if left2 != right2:
            start = find_nearest_idx(curve2.get_x(), left2, side='right')
            stop = find_nearest_idx(curve2.get_x(), right2, side='left')
            curve2 = SingleCurve(curve2.data[:, start: stop],
                                 label=curve2.label,
                                 units=curve2.units,
                                 t_units=curve2.t_units)

        corr_curve = correlate_single(curve1, curve2)
        corr_data.append(corr_curve)

        if add_to_signals:
            signals_data.add_from_array(corr_curve.data,
                                        labels=[corr_curve.label],
                                        units=["a.u."],
                                        time_units=signals_data.time_units
                                        )
    return corr_data


# =======================================================================
# ------   STATISTICS   -------------------------------------------------
# =======================================================================

def get_2d_array_stat_by_columns(data):
    """ Calculates the statistics (mean value, standard deviation,
    maximum deviation and number of non-nan values)
    of a two-dimensional array by columns.
    Ignores nan values.

    Input data structure: data[row][column]

    Example:
              col1  col2  col3   col4
      data([
    row1     [ 3,    10,   22,   104],
    row2     [ 2,    15,   24,   108],
    row3     [ 1,    20,   26,   107]])

    result([ 2.0,  15.0,  24.0, 106,3333])

    :param delay_list: list with delay values delay_list[shot_num][delay_num]
    :type delay_list: list or np.ndarray
    :return: two-dimensional ndarray: result[column][stat_type]
    where stat_type are: mean, std deviation, max deviation, number of non-nan samples
    :rtype: np.ndarray
    """
    stats_num = 4
    if not isinstance(data, np.ndarray):
        data = np.array(data, dtype=np.float64)

    assert data.ndim == 2, \
        "Data must be two-dimensional ndarray/list. " \
        "Found {} dimension(s) instead.".format(data.ndim)

    # number of columns
    # shape[0] -> rows;   shape[1] -> columns
    col_num = data.shape[1]
    res = np.zeros(shape=(col_num, stats_num), dtype=np.float64)
    for col in range(col_num):
        col_mean, col_std, col_maxerr, col_sample_num = get_1d_array_stat(data[:, col])
        res[col, 0] = col_mean
        res[col, 1] = col_std
        res[col, 2] = col_maxerr
        res[col, 3] = col_sample_num

    # # count non-nan values for all rows (along axis=0) separately for each column
    # number_of_samples = np.count_nonzero(~np.isnan(data), axis=0)

    return res


def get_1d_array_stat(data):
    """ Calculates statistics (mean value, standard deviation,
     maximum deviation and number of non-nan values) for single-axis ndarray or list.
     Ignores nan values.

    :param data: list or 1D ndarray with delay values data[delay_num]
    :type data: list or np.ndarray
    :return: tuple of 4 values: mean, std deviation, max deviation, number of samples
    :rtype: tuple
    """
    if not isinstance(data, np.ndarray):
        data = np.array(data, dtype=np.float64)
    assert data.ndim == 1, \
        "Data must be one-dimensional ndarray/list. " \
        "Found {} dimension(s) instead.".format(data.ndim)

    arr_mean = np.nanmean(data)
    arr_std = np.nanstd(data)
    arr_maxerr = np.nanmax(np.abs(arr_mean - data))
    arr_samples_num = np.count_nonzero(~np.isnan(data))
    return arr_mean, arr_std, arr_maxerr, arr_samples_num


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
