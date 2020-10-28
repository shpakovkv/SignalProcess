# Python 3.6
"""
Maintainer: Shpakov Konstantin
Link: https://github.com/shpakovkv/SignalProcess
"""

import numpy as np

from data_types import SignalsData, SinglePeak


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

        if not multiplier:
            multiplier = [1 for _ in range(col_number)]
        if not delay:
            delay = [0 for _ in range(col_number)]
        # check_coeffs_number(col_number, ["multiplier", "delay"],
        #                     multiplier, delay)
        for col_idx in range(col_number):
            for row_idx in range(row_number):
                data[row_idx][col_idx] = (data[row_idx][col_idx] *
                                          multiplier[col_idx] -
                                          delay[col_idx])
        return data
    elif isinstance(data, SignalsData):
        if not multiplier:
            multiplier = [1 for _ in range(data.count * 2)]
        if not delay:
            delay = [0 for _ in range(data.count * 2)]

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