import os
import bisect
import numpy as np
from matplotlib import pyplot as plt
import scipy.integrate as integrate


pos_polarity_labels = {'pos', 'positive', '+'}
neg_polarity_labels = {'neg', 'negative', '-'}


def find_nearest_idx(sorted_arr, value, side='auto'):
    """
    Returns the index of the 'sorted_arr' element closest to 'value'

    :param sorted_arr: sorted array/list of ints or floats
    :param value: the int/float number to which the
                  closest value should be found
    :param side: 'left': search among values that are lower then X
                 'right': search among values that are greater then X
                 'auto': handle all values (default)
    :type sorted_arr: array-like
    :type value: int, float
    :type side: str ('left', 'right', 'auto')
    :return: the index of the value closest to 'value'
    :rtype: int

    .. note:: if two numbers are equally close and side='auto',
           returns the index of the smaller one.
    """

    idx = bisect.bisect_left(sorted_arr, value)
    # print("IDX = {};  SIDE = {}; max = {};   min = {}".format(idx, side, np.nanmax(sorted_arr), np.nanmin(sorted_arr)))
    if idx == 0:
        return idx if side == 'auto' or side == 'right' else None

    if idx == len(sorted_arr):
        return idx if side == 'auto' or side == 'left' else None

    after = sorted_arr[idx]
    before = sorted_arr[idx - 1]
    if side == 'auto':
        return idx if after - value < value - before else idx - 1
    else:
        if after == value:
            return idx
        return idx if side == 'right' else idx - 1


def level_excess(x,
                 y,
                 level,
                 start=0,
                 step=1,
                 window=0,
                 rising_front=True):
    """Checks if 'Y' values excess 'level' value
    for 'X' in range from X(start) to X(start) + window
    OR for x in range from X(start) - window to X(start)

    :param x:           array with X data
    :param y:           array with Y data
    :param level:       level value
    :param start:       start index of data
    :param step:        step with which elements of the array are traversed
                        step > 0: checks elements to the RIGHT from start idx
                        step < 0: checks elements to the LEFT from start idx
    :param window:      check window width
    :param rising_front:front type, rising (True) or falling (False)


    :type x:            array-like
    :type y:            array-like
    :type level:        float, int ''same as param y''
    :type start:        int
    :type step:         int
    :type window:       float, int ''same as param x''
    :type rising_front: bool
    :return:            True and an index of first Y element that are
                        bigger/lower 'level' OR returns False and
                        an index of the last checked element
    :rtype:             tuple ''(bool, int)''
    """
    # TODO: speed up level_excess() with numba
    if window == 0:  # window default value
        datalen = x.shape[0] - 1
        while (start < datalen) and (np.isnan(x[start])):
            start += 1
        stop = -1
        if isinstance(x, np.ndarray):
            stop = x.shape[0] - 1
        else:
            stop = len(x) - 1
        while (stop > start) and (np.isnan(x[stop])):
            stop -= 1
        window = x[stop] - x[start]

    idx = start  # zero-based index
    while ((idx >= 0) and (idx < len(y)) and
           (abs(x[idx] - x[start]) <= window)):
        if rising_front:
            if y[idx] > level:
                return True, idx
        else:
            if y[idx] < level:
                return True, idx
        idx += step

    if idx == len(y):
        idx -= 1
    return False, idx


def is_pos(polarity):
    """Checks if the polarity (str flag) is positive.

    :param polarity:    word denoting polarity
    :type polarity:     str
    :return:            True if the polarity is positive,
                        otherwise returns False
    :rtype:             bool
    """
    global pos_polarity_labels
    global neg_polarity_labels
    if polarity.lower() in pos_polarity_labels:
        return True
    if polarity.lower() in neg_polarity_labels:
        return False
    else:
        raise ValueError("Wrong polarity value ({})".format(polarity))


def is_neg(polarity):
    """Checks if the polarity (str flag) is negative.

    :param polarity:    word denoting polarity
    :type polarity:     str
    :return:            True if the polarity is negative,
                        otherwise returns False
    :rtype:             bool
    """
    return not is_pos(polarity)


def check_polarity(curve, time_bounds=(None, None)):
    """Checks whether the curve is mostly positive or negative
    on a certain interval.

    :param curve:       curve data
    :param time_bounds: the left and the right boundaries of
                        the specified interval
    :type curve:        SignalProcess.SingleCurve
    :type time_bounds:  tuple, list ''(float, float)''

    :return:            the word denoting polarity
    :rtype:             str
    """
    if time_bounds[0] is None:
        time_bounds = (0, time_bounds[1])
    if time_bounds[1] is None:
        time_bounds = (time_bounds[0], curve.points)
    integr = integrate.trapz(curve.val[time_bounds[0]:time_bounds[1]],
                             curve.time[time_bounds[0]:time_bounds[1]])
    # print("Voltage_INTEGRAL = {}".format(integr))
    if integr >= 0:
        return 'positive'
    return 'negative'


def find_curve_front(curve,
                     level,
                     front="auto",
                     bounds=None,
                     interpolate=False,
                     save_plot=False,
                     plot_name="voltage_front.png"):
    """Find time point (x) of voltage curve edge at specific level
    within specified time bounds.
    Default: Negative polarity, -0.2 MV level

    "auto" front type selection means the first front of the signal:
    rising front for positive signals
    and falling front for negative signals

    :param curve: curve data
    :param level: amplitude value to find
    :param front: the polarity of the curve "rise"/"fall"/"auto"
    :param bounds: left and right borders to search
    :param interpolate: if false finds the nearest curve point,
                        if true finds the exact time using a linear approximation
    :param save_plot: bool flag
    :param plot_name: plot full file name to save as

    :type curve: SingleCurve
    :type level: float
    :type front: str
    :type bounds: list or None
    :type interpolate: bool
    :type save_plot: bool
    :type plot_name: str

    :return:  (time, amplitude) or (None, None)
    :rtype: tuple(float, float)
    """

    assert front == "auto" or front == "rise" or front == "fall", \
        "Error! Wrong front type entered.\n" \
        "Expected \"auto\" or \"rise\" or \"fall\". Got \"{}\" instead." \
        "".format(front)
    if front == 'auto':
        # auto selection:
        # rising front for positive signals
        # and falling front for negative signals
        polarity = check_polarity(curve)
        front = "rise" if is_pos(polarity) else "fall"

        # if front == "rise":
        #     level = abs(level)
        # else:
        #     level = -abs(level)

    is_rising = True if front == "rise" else False

    front_checked = False
    idx = 0

    # bounds check
    start = 0
    window = 0
    if bounds is not None:
        if bounds[0] is not None:
            if bounds[0] > curve.time[-1]:
                return None, None
            start = find_nearest_idx(curve.time, bounds[0], side='right')

        if bounds[1] is not None:
            if bounds[1] < curve.time[0]:
                return None, None
            stop = find_nearest_idx(curve.time, bounds[1], side='right')
            window = stop - start + 1

    # search for rising front and the signal starts above level value
    if is_rising and curve.val[start] >= level:
        dropped_below, below_idx = level_excess(curve.time, curve.val, level,
                                                start=start,
                                                window=window,
                                                rising_front=False)
        if dropped_below:
            sub_window = window - below_idx + start - 1
            front_checked, idx = level_excess(curve.time, curve.val, level,
                                              start=below_idx + 1,
                                              window=sub_window,
                                              rising_front=is_rising)

    # search for falling front and the signal starts below level value
    elif not is_rising and curve.val[start] <= level:
        rose_above, above_idx = level_excess(curve.time, curve.val, level,
                                             start=start,
                                             window=window,
                                             rising_front=True)
        if rose_above:
            sub_window = window - above_idx + start - 1
            front_checked, idx = level_excess(curve.time, curve.val, level,
                                              start=above_idx + 1,
                                              window=sub_window,
                                              rising_front=is_rising)

    # normal condition
    else:
        front_checked, idx = level_excess(curve.time, curve.val, level,
                                          start=start,
                                          rising_front=is_rising)

    front_time = curve.time[idx]
    front_val = curve.val[idx]

    if front_checked and interpolate:
        if 0 < idx < len(curve.time) - 1:
            front_val = level
            front_time = get_front_time_with_aprox(curve.time[idx - 1],
                                                   curve.val[idx - 1],
                                                   curve.time[idx],
                                                   curve.val[idx],
                                                   level)

    if front_checked:
        if save_plot:
            plt.close('all')
            plt.plot(curve.time, curve.val, '-b')
            plt.plot([front_time], [front_val], '*r')
            # plt.show()
            folder = os.path.dirname(plot_name)
            if folder != "" and not os.path.isdir(folder):
                os.makedirs(folder)
            plt.savefig(plot_name)
            plt.close('all')
        return front_time, front_val
    return None, None


def get_front_time_with_aprox(time1, amp1, time2, amp2, target_amp):
    assert time2 > time1, \
        "Time1 must be less than time2."
    assert max(amp1, amp2) >= target_amp >= min(amp1, amp2), \
        "Target amplitude ({}) is out of range for given two points [{}, {}]" \
        "".format(target_amp, amp1, amp2)

    if amp1 == target_amp:
        return time1
    if amp2 == target_amp:
        return time2
    if amp2 == amp1:
        return time2

    # rising edge
    # if amp2 > amp1:
    proportion = (target_amp - amp1) / (amp2 - amp1)
    target_time = time1 + proportion * (time2 - time1)

    return target_time

