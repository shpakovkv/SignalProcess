# Python 3.6
"""
Maintainer: Shpakov Konstantin
Link: https://github.com/shpakovkv/SignalProcess
"""

import numpy as np
from numba import jit

from data_types import SignalsData, SinglePeak


@jit(nopython=True)
def _multiplier_and_delay_jit(data, multiplier, delay):
    """Modifies data by formula:

    data[ i ][ j ][ : ] = (data[ i ][ j ][ : ] * multiplier[ i ][ j ]) - delay[ i ][ j ]

    Returns modified data.

    The data must be 3-dim ndarray.
    The multiplier and the delay must be 2-dim ndarrays.
    the first two dimensions of data, multiplier and delay must be the same.
    (data.shape[0] == multiplier.shape[0] == delay.shape[0]
     data.shape[1] == multiplier.shape[1] == delay.shape[1])

    The data's 3rd dimension may be any integer > 0.

    :param data: 3-dim ndarray
    :type data: np.ndarray

    :param multiplier: 2-dim ndarray
    :type multiplier: np.ndarray

    :param delay: 2-dim ndarray
    :type delay: np.ndarray

    :return: modified data
    :rtype: np.ndarray
    """
    curves = data.shape[0]
    points = data.shape[2]
    for cur in range(curves):
        for axis in range(2):
            for point_idx in range(points):
                data[cur][axis][point_idx] = (data[cur][axis][point_idx] *
                                              multiplier[cur][axis] -
                                              delay[cur][axis])


def multiplier_and_delay_ndarray(data, multiplier, delay):
    """Checks input arguments for compatibility (throws an exception if any check falls),
    then modifies the data using the formula:

    data[ i ][ j ][ : ] = (data[ i ][ j ][ : ] * multiplier[ i ][ j ]) - delay[ i ][ j ]

    Returns modified data.

    The data must be 3-dim ndarray.
    The multiplier and the delay must be 2-dim ndarrays.
    the first two dimensions of data, multiplier and delay must be the same.
    (data.shape[0] == multiplier.shape[0] == delay.shape[0]
     data.shape[1] == multiplier.shape[1] == delay.shape[1])

    The data's 3rd dimension may be any integer > 0.

    :param data: 3-dim ndarray
    :type data: np.ndarray

    :param multiplier: 2-dim ndarray
    :type multiplier: np.ndarray

    :param delay: 2-dim ndarray
    :type delay: np.ndarray

    :return: modified data
    :rtype: np.ndarray
    """
    fixed_ndim = 3

    # if nothing to do
    if multiplier is None and delay is None:
        return data

    # check data type
    if isinstance(data, SignalsData):
        # get ndarray
        data = data.data

    assert isinstance(data, np.ndarray), \
        "Data must be of type numpy.ndarray or SignalsData. " \
        "Got {} instead.".format(type(data))

    assert isinstance(multiplier, np.ndarray), \
        "Multiplier must be of type numpy.ndarray " \
        "Got {} instead.".format(type(multiplier))

    assert isinstance(delay, np.ndarray), \
        "Delay must be of type numpy.ndarray " \
        "Got {} instead.".format(type(delay))

    # check ndim, shape
    assert data.ndim == fixed_ndim, \
        "Data must be {}-dimensional ndarray. " \
        "Got {}-dimensional instead".format(fixed_ndim, data.ndim)

    assert multiplier.ndim == fixed_ndim - 1, \
        "Multiplier must be {}-dimensional ndarray. " \
        "Got {}-dimensional instead".format(fixed_ndim - 1, multiplier.ndim)

    assert delay.ndim == fixed_ndim - 1, \
        "Delay must be {}-dimensional ndarray. " \
        "Got {}-dimensional instead".format(fixed_ndim - 1, delay.ndim)

    assert multiplier.shape == data.shape[: -1], \
        "The number of rows and columns in multiplier ({}) " \
        "must be equal to the number of curves and axes in " \
        "data ({})".format(multiplier.shape, data.shape[:-1])

    assert delay.shape == data.shape[: -1], \
        "The number of rows and columns in delay ({}) " \
        "must be equal to the number of curves and axes in " \
        "data ({})".format(delay.shape, data.shape[:-1])

    # process
    return _multiplier_and_delay_jit(data, multiplier, delay)


def multiplier_and_delay(signals, multiplier, delay):
    """Wrapper for multiplier_and_delay_ndarray function.

    Takes SignalsdData or ndarray instance.
    Returns modified data

    :param signals: signal's data to be modified
    :type signals: SignalsData or np.ndarray

    :param multiplier: 2-dim ndarray
    :type multiplier: np.ndarray

    :param delay: 2-dim ndarray
    :type delay: np.ndarray

    :return: signals with modified data
    :rtype: SignalsData or np.ndarray
    """
    if isinstance(signals, SignalsData):
        signals.data = multiplier_and_delay_ndarray(signals.data, multiplier, delay)
    elif isinstance(signals, np.ndarray):
        signals = multiplier_and_delay_ndarray(signals, multiplier, delay)
    else:
        raise TypeError("Expected SignalsData or ndarray instance as first argument, "
                        "got {} instead.".format(type(signals)))
    return signals


# def multiplier_and_delay_old(data, multiplier, delay):
#     """Returns the modified data.
#     Each column of the data is first multiplied by
#     the corresponding multiplier value and
#     then the corresponding delay value is subtracted from it.
#
#     data       -- an instance of the SignalsData class
#                   OR 2D numpy.ndarray
#     multiplier -- the list of multipliers for each columns
#                   in the input data.
#     delay      -- the list of delays (subtrahend) for each
#                   columns in the input data.
#     """
#     if multiplier is None and delay is None:
#         return data
#
#     if isinstance(data, np.ndarray):
#         row_number = data.shape[0]
#         col_number = data.shape[1]
#
#         if not multiplier:
#             multiplier = [1 for _ in range(col_number)]
#         if not delay:
#             delay = [0 for _ in range(col_number)]
#         # check_coeffs_number(col_number, ["multiplier", "delay"],
#         #                     multiplier, delay)
#         for col_idx in range(col_number):
#             for row_idx in range(row_number):
#                 data[row_idx][col_idx] = (data[row_idx][col_idx] *
#                                           multiplier[col_idx] -
#                                           delay[col_idx])
#         return data
#     elif isinstance(data, SignalsData):
#         if not multiplier:
#             multiplier = [1 for _ in range(data.count * 2)]
#         if not delay:
#             delay = [0 for _ in range(data.count * 2)]
#
#         # check_coeffs_number(data.count * 2, ["multiplier", "delay"],
#         #                     multiplier, delay)
#         for curve_idx in range(data.count):
#             col_idx = curve_idx * 2  # index of time-column of current curve
#             data.curves[curve_idx].data = \
#                 multiplier_and_delay_old(data.curves[curve_idx].data,
#                                      multiplier[col_idx:col_idx + 2],
#                                      delay[col_idx:col_idx + 2]
#                                      )
#         return data
#     else:
#         raise TypeError("Data must be an instance of "
#                         "numpy.ndarray or SignalsData.")


def multiplier_and_delay_peak(peaks, multiplier, delay, curve_idx):
    """Returns the modified peaks data.
    Each time and amplitude of each peak is first multiplied by
    the corresponding multiplier value and
    then the corresponding delay value is subtracted from it.

    peaks       -- the list of the SinglePeak instance
    multiplier  -- the list of multipliers for each columns
                   in the SignalsData.
    delay       -- the list of delays (subtrahend)
                   for each columns in the SignalsData.
    curve_idx   -- the index of the curve to which the peaks belong
    """
    for peak in peaks:
        if not isinstance(peak, SinglePeak):
            raise ValueError("'peaks' must be a list of "
                             "the SinglePeak instance.")
    if not multiplier and not delay:
        raise ValueError("Specify multipliers or delays.")
    if not multiplier:
        multiplier = [1 for _ in range(len(delay))]
    if not delay:
        delay = [0 for _ in range(multiplier)]
    if curve_idx >= len(multiplier):
        raise IndexError("The curve index({}) is greater than the length "
                         "({}) of multiplier/delay."
                         "".format(curve_idx, len(multiplier)))
    corr_peaks = []
    time_mult = multiplier[curve_idx * 2]
    time_del = delay[curve_idx * 2]
    amp_mult = multiplier[curve_idx * 2 + 1]
    amp_del = delay[curve_idx * 2 + 1]
    for peak in peaks:
        corr_peaks.append(SinglePeak(peak.time * time_mult - time_del,
                                     peak.val * amp_mult - amp_del,
                                     peak.idx, peak.sqr_l, peak.sqr_r))
    return corr_peaks
