# Python 3.6
"""
Peak handle functions.


Maintainer: Shpakov Konstantin
Link: https://github.com/shpakovkv/SignalProcess
"""

from __future__ import print_function

import matplotlib
import matplotlib.pyplot as plt

import os
import sys
import numpy
import bisect
import argparse
import time

import numpy as np
import scipy.integrate as integrate

# import SignalProcess as sp
import arg_parser
import arg_checker
import file_handler
import plotter
from multiprocessing import Pool
from itertools import repeat

from data_manipulation import multiplier_and_delay
from data_types import SinglePeak, SignalsData, SingleCurve
from analysis import get_2d_array_stat_by_columns

from file_handler import check_files_for_duplicates
from file_handler import parse_csv_for_peaks


pos_polarity_labels = {'pos', 'positive', '+'}
neg_polarity_labels = {'neg', 'negative', '-'}

NOISEATTENUATION = 0.75
SAVETODIR = 'Peaks'
SINGLEPLOTDIR = 'SinglePlot'
MULTIPLOTDIR = 'MultiPlot'
PEAKDATADIR = 'PeakData'
DEBUG = False
PEAKFINDERDEBUG = False
STAT_NAME = {"front_delay": "Front delay"}


def get_parser():
    """Returns final CL args parser.

    :return: argparse.parser
    """
    p_use = ('python %(prog)s [options]\n'
             '       python %(prog)s @file_with_options')
    p_desc = ('')
    p_ep = ('')

    parser = argparse.ArgumentParser(
        parents=[arg_parser.get_input_files_args_parser(),
                 arg_parser.get_mult_del_args_parser(),
                 arg_parser.get_plot_args_parser(),
                 arg_parser.get_peak_args_parser(),
                 arg_parser.get_output_args_parser(),
                 arg_parser.get_utility_args_parser(),
                 arg_parser.get_front_args_parser()],
        prog='PeakProcess.py',
        description=p_desc, epilog=p_ep, usage=p_use,
        fromfile_prefix_chars='@',
        formatter_class=argparse.RawTextHelpFormatter
    )
    return parser


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
    if idx == 0:
        return idx if side == 'auto' or side == 'right' else None

    if idx == len(sorted_arr):
        return idx if side == 'auto' or side == 'left' else None

    after = sorted_arr[idx]
    before = sorted_arr[idx - 1]
    if side == 'auto':
        return idx if after - value < value - before else idx - 1
    else:
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
    if window == 0:      # window default value
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
    integr =  integrate.trapz(curve.val[time_bounds[0]:time_bounds[1]],
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
    assert time2 >= time1, \
        "Time1 must be less or equal than Time2."
    assert max(amp1, amp2) >= target_amp >= min(amp1, amp2), \
        "Target amplitude ({}) is out of range for given two points [{}, {}]" \
        "".format(target_amp, amp1, amp2)

    if time1 == time2:
        return time1

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


def peak_finder(x, y, level, diff_time, time_bounds=(None, None),
                tnoise=None, is_negative=True, graph=False,
                noise_attenuation=0.5):
    """Finds peaks on the curve (x, y). 
    Searchs for negative peaks by default.
    
    :param x: array of time values
    :param y: array of amplitude values
    :param level: peak threshold (all amplitude values 
                  below this level will be ignored)
    :param diff_time: the minimum difference between two neighboring peaks.
                      If the next peak is at the front (fall or rise) 
                      of the previous peak, and the "distance" from 
                      its maximum to that front (at the same level) is less 
                      than the diff_time, this second peak will be ignored.
    :param time_bounds: tuple with the left and the right search boundaries
    :param tnoise: maximum half-period of noise fluctuation.
    :param is_negative: specify False for positive curve.
                        for negative curve (by default) the 'y' array will be 
                        inverted before process and inverted again at the end.
    :param graph: specify True to display a graph with found peaks
    :param noise_attenuation: Attenuation of the second half-wave 
                              with a polarity reversal (noise). If too many 
                              noise maxima are defined as real peaks, 
                              reduce this value.
     
    :type x: numpy.ndarray
    :type y: numpy.ndarray
    :type level: float, int ''same as values of param y''
    :type diff_time: float, int ''same as values of param x''
    :type time_bounds: tuple, list ''(float, float)''
    :type tnoise: float, int ''same as values of param x''
    :type is_negative: bool
    :type graph: bool
    :type noise_attenuation: float

    :return: (peaks_list, log) - the list of peaks (SinglePeak instances) 
             and the process log
    :rtype: (list, str)
    """

    # print("============ peak_finder ================")
    # print("level : {}\ndiff_time : {}\ntime_bounds : {}\ntnoise : {}\n"
    #       "is_negative : {}\ngraph : {}\nnoise_attenuation : {}\n"
    #       "start_idx : {}\nstop_idx : {}"
    #       "".format(level, diff_time, time_bounds, tnoise,
    #                 is_negative, graph, noise_attenuation,
    #                 start_idx, stop_idx))
    # print("-------------------")

    # Checkout the inputs
    peak_log = ""
    assert level != 0, 'Invalid level value!'

    if is_negative:
        y = -y
        level = -level

    if not tnoise:
        tnoise = (x[1] - x[0]) * 4
        peak_log += 'Set "tnoise" to default 4 stops = ' + str(tnoise) + "\n"

    assert len(time_bounds) == 2, ("time_bounds has incorrect number of "
                                   "values. 2 expected, " +
                                   str(len(time_bounds)) + " given.")
    assert len(x) == len(y), ("The length of X ({}) is not equal to the "
                              "length of Y ({}).".format(len(x), len(y)))

    if time_bounds[0] is None:
        time_bounds = (x[0], time_bounds[1])
    if time_bounds[1] is None:
        time_bounds = (time_bounds[0], x[-1])
    start_idx = find_nearest_idx(x, time_bounds[0], side='right')
    stop_idx = find_nearest_idx(x, time_bounds[1], side='left')

    peak_list = []

    if start_idx is None or stop_idx is None:
        # the interval is [start_idx, stop_idx)
        # start_idx is included; stop_idx is excluded
        peak_log += "Time bounds is out of range.\n"
        return peak_list, peak_log

    time_delta = 0.0
    if x[0] != x[1]:
        time_delta = x[1] - x[0]
    else:
        time_part = x[stop_idx] - x[start_idx]
        time_delta = time_part / (stop_idx - start_idx - 1)
    diff_idx = int(diff_time / time_delta)

    if PEAKFINDERDEBUG:
        print("Diff_time = {}, Diff_idx = {}".format(diff_time, diff_idx))

    i = start_idx
    while i < stop_idx :
        if y[i] > level:
            max_y = y[i]  # local max (may be real peak or not)
            max_idx = i

            # search for a bigger local max within the diff_time from
            # the found one
            # y[i] == max_y condition is needed
            # for flat-top peaks (wider than diff_time)
            while (i <= stop_idx and
                   (x[i] - x[max_idx] <= diff_time or
                    y[i] == max_y)):
                if y[i] > max_y:
                    max_y = y[i]
                    max_idx = i
                i += 1
            if PEAKFINDERDEBUG:
                print("local_max = [{:.3f}, {:.3f}] i={}"
                      "".format(x[max_idx], max_y, max_idx))

            # search for a bigger value within the diff_time
            # to the left from the found local maximum
            # if found: this is a little signal fluctuation on the fall edge
            # (not a real peak)
            [is_noise, _] = level_excess(x, y, max_y, start=max_idx,
                                         step=-1, window=diff_time,
                                         rising_front=True)
            if PEAKFINDERDEBUG and is_noise:
                print('Left Excess at x({:.2f}, {:.2f}) '
                      '== Not a peak at fall edge!'.format(x[i], y[i]))

            # search for a polarity reversal within tnose from this local max
            # if found: this is a noise (not a real peak)
            if not is_noise:
                # search to the right from the local max
                [is_noise, j] = level_excess(x, y,
                                             -max_y * noise_attenuation,
                                             start=max_idx, step=1,
                                             window=tnoise,
                                             rising_front=False)

                if PEAKFINDERDEBUG and is_noise:
                    print('Noise to the right x({:.2f}, {:.2f})'
                          ''.format(x[j], y[j]))
                else:
                    # search to the left from the local max
                    [is_noise, j] = level_excess(x, y,
                                                 -max_y * noise_attenuation,
                                                 start=max_idx, step=-1,
                                                 window=tnoise,
                                                 rising_front=False)
                    if PEAKFINDERDEBUG and is_noise:
                        print('Noise to the left x({:.2f}, {:.2f})'
                              ''.format(x[j], y[j]))

            if not is_noise:
                # all checks passed, the local max is the real peak
                peak_list.append(SinglePeak(x[max_idx], max_y, max_idx))
                continue
        i += 1

    peak_log += 'Number of peaks: ' + str(len(peak_list)) + "\n"

    # LOCAL INTEGRAL CHECK
    # needed for error probability estimation
    di = int(diff_time * 2 // time_delta)    # diff window in index units

    if di > 3:
        for idx in range(len(peak_list)):
            pk = peak_list[idx]
            # square = pk.val * time_delta * di
            square = pk.val * di
            
            intgr_l = 0
            intgr_r = 0
            peak_log += ("Peak[{:3d}] = [{:7.2f},   {:4.1f}]   "
                         "Square factor [".format(idx, pk.time, pk.val))
            if pk.idx - di >= 0:
                intgr_l = integrate.trapz(y[pk.idx-di : pk.idx+1])
                peak_list[idx].sqr_l = intgr_l / square
                peak_log += "{:.3f}".format(intgr_l / square)
            peak_log += " | "
            if pk.idx + di < len(y):  # stop_idx
                intgr_r = integrate.trapz(y[pk.idx: pk.idx + di + 1])
                peak_list[idx].sqr_r = intgr_r / square
                peak_log += "{:.3f}".format(intgr_r / square)
            peak_log += "]"
            peak_log += "  ({:.3f})".format((intgr_r + intgr_l) / square)
            peak_log += "\n"
        if peak_list:
            peak_log += "\n"
    # integr_l, integr_r: The closer the value to unity,
    # the greater the probability that the peak is imaginary (erroneous)

    if is_negative:
        y = -y
        level = -level
        for i in range(len(peak_list)):
            peak_list[i].invert()
    if graph:
        # plotting curve
        plt.plot(x[start_idx:stop_idx], y[start_idx:stop_idx], '-',
                    color='#8888bb')
        plt.xlim(time_bounds)
        # plotting level line
        plt.plot([x[0], x[len(x) - 1]], [level, level], ':',
                    color='#80ff80')
        # marking overall peaks
        peaks_x = [p.time for p in peak_list]
        peaks_y = [p.val for p in peak_list]
        plt.scatter(peaks_x, peaks_y, s=50, edgecolors='#ff7f0e',
                       facecolors='none', linewidths=2)
        plt.scatter(peaks_x, peaks_y, s=80, edgecolors='#dd3328',
                       facecolors='none', linewidths=2)
        plt.show()

    return peak_list, peak_log


def group_peaks(data, window):
    """Groups the peaks from different curves.
    Each group corresponds to one single event (for example: 
    one act of X-Ray emission, registered by several detectors).
    
    :param data: three-dimensional array containing data 
                 on all the peaks of all curves
                 The array structure:
                 data[curve_idx][peak_idx] == SinglePeak instance
                 If a curve with curve_idx index has no peaks 
                 the data[curve_idx] contains an empty list. 
    :param window: peaks coincide when their X values are within
                   +/-window interval from average X (time) position 
                   of peak (event). "Average" because X (time) value 
                   of a peak (event) may differ from curve to curve.
    :return: peak_data - the three-dimensional array containing data 
             on all the peaks (grouped by time) of all curves
             The array structure:
             peak_data[curve_idx][group_idx] == SinglePeak instance if 
             this curve has a peak related to this event (group), else None
    """

    def insert_group(peak, peak_data, groups_time,
                     num_peak_in_gr, wf, gr):
        """Inserts new group of peaks to the peak_data array 
        at a specific index.
        
        :param peak: new peak to add
        :param peak_data: the 3-dimensional array with peaks data
        :param groups_time: the list with groups average time
        :param num_peak_in_gr: the list contains 
                               the number of peaks in each group
        :param wf: waveform (curve) index
        :param gr: new group index (insert on this index)
        :return: None
        """
        groups_time.insert(gr, peak.time)
        num_peak_in_gr.insert(gr, 1)
        for curve_i in range(len(peak_data)):
            if curve_i == wf:
                peak_data[curve_i].insert(gr, SinglePeak(*peak.data_full))
            else:
                peak_data[curve_i].insert(gr, None)


    def add_pk_to_gr(peak, peak_data, groups_time,
                     num_peak_in_gr, wf, gr):
        """Adds new peak (from another curve) to existing group.
        It is assumed that the group contains None value
        on the place of this peak.
        
        :param peak: new peak to add
        :param peak_data: the 3-dimensional array with peaks data
        :param groups_time: the list with groups average time
        :param num_peak_in_gr: the list contains 
                               the number of peaks in each group
        :param wf: waveform (curve) index
        :param gr: new group index (insert on this index)
        :return: None
        """
        groups_time[gr] = ((groups_time[gr] * num_peak_in_gr[gr] +
                            peak.time) /
                           (num_peak_in_gr[gr] + 1))
        num_peak_in_gr[gr] += 1
        peak_data[wf][gr] = SinglePeak(*peak.data_full)

    if len(data) == 1 and len(data[0]) == 0:
        return [[]]

    # wf == waveform == curve
    start_wf = 0
    # skip first curves if they have no peaks
    while not data[start_wf] and start_wf < len(data):
        start_wf += 1

    # 1D array with average time value of peak group
    groups_time = [peak.time for peak in data[start_wf]]

    # 1D array with numbers of peaks in each group
    num_peak_in_gr = [1] * len(groups_time)

    dt = abs(window)
    curves_count = len(data)

    # the 3-dimensional array will contain data
    # on all the peaks (grouped by time)
    peak_data = [[]]
    for peak in data[start_wf]:
        peak_data[0].append(SinglePeak(*peak.data_full))
    for curve_idx in range(0, start_wf):
        peak_data.insert(0, [None] * len(groups_time))
    for curve_idx in range(start_wf + 1, curves_count):
        peak_data.append([None] * len(groups_time))

    if curves_count <= 1:
        return peak_data

    '''---------- making groups of peaks ------------------------------
    two peaks make group when they are close enough
    ('X' of a peak is within +/- dt interval from 'X' of the group)
    with adding new peak to a group,
    the 'X' parameter of the group changes to (X1 + X2 + ... + Xn)/n
    where n - number of peaks in group
    '''

    for wf in range(start_wf + 1, curves_count):
        '''
        wf == waveform index = curve index
        gr == group index (zero-based index of current group)
        pk == peak index (zero-based index of current peak
              in the peak list of current waveform)
        '''
        gr = 0
        pk = 0

        while data[wf] is not None and pk < len(data[wf]):  # and len(data[wf]) > 0:
            '''ADD PEAK TO GROUP
            when curve[i]'s peak[j] is in
            +/-dt interval from peaks of group[gr]
            '''
            if gr < len(groups_time) \
                    and abs(groups_time[gr] - data[wf][pk].time) <= dt:
                if (len(data[wf]) > pk + 1 and
                        (abs(groups_time[gr] - data[wf][pk].time) >
                         abs(groups_time[gr] - data[wf][pk + 1].time))):
                    # next peak of data[wf] matches better
                    # insert new group for current data[wf]'s peak
                    insert_group(data[wf][pk], peak_data, groups_time,
                                 num_peak_in_gr, wf, gr)
                    pk += 1
                elif (len(groups_time) > gr + 1 and
                          (abs(groups_time[gr] - data[wf][pk].time) >
                               abs(groups_time[gr + 1] - data[wf][pk].time))):
                    # current peak matches next group better
                    pass
                else:
                    add_pk_to_gr(data[wf][pk], peak_data, groups_time,
                                 num_peak_in_gr, wf, gr)
                    pk += 1
                if gr == len(groups_time) - 1 and pk < len(data[wf]):
                    # Last peak_data column was filled but there are
                    # more peaks in the data[wf], so adds new group
                    gr += 1

            elif (gr < len(groups_time) and
                          data[wf][pk].time < groups_time[gr] - dt):
                '''INSERT NEW GROUP
                when X-position of current peak of curve[wf] is 
                to the left of current group by more than dt
                '''
                insert_group(data[wf][pk], peak_data, groups_time,
                             num_peak_in_gr, wf, gr)
                pk += 1

            elif gr >= len(groups_time) - 1:
                '''APPEND NEW GROUP
                when X-position of current peak of curve[wf] is to the right 
                of the last group
                '''
                insert_group(data[wf][pk], peak_data, groups_time,
                             num_peak_in_gr, wf, len(groups_time))
                pk += 1
                gr += 1

            if gr < len(groups_time) - 1:
                gr += 1

    return peak_data


def get_peaks(data, args, verbose):
    """Searches for peaks using parameters from args namespace.
    
    :param data:    SignalsData instance
    :param args:    argparse.namespace with arguments
    :param verbose: shows more info during the process
    
    :return:        three-dimensional array containing data 
                    on all the peaks of curves with index in args.curves list
                    The array structure:
                    data[curve_idx][peak_idx] == SinglePeak instance
                    For the curves not in the args.curves list:
                    data[curve_idx] == None
    """
    unsorted_peaks = [None] * data.cnt_curves
    for idx in args.curves:
        if verbose:
            print("Curve #" + str(idx))
        new_peaks, peak_log = peak_finder(
            data.get_x(idx), data.get_y(idx),
            level=args.level, diff_time=args.pk_diff,
            time_bounds=args.t_bounds, tnoise=args.t_noise,
            is_negative=args.level < 0,
            noise_attenuation=args.noise_att,
            graph=False
        )
        unsorted_peaks[idx] = new_peaks
        if verbose:
            print(peak_log)
    return unsorted_peaks


def check_curves_list(curves, signals_data):
    """Checks the indexes of the curves to process.
    Raises the exception if the index is out of range.
    
    :param curves: the list of indexes of curves to find peaks for
    :param signals_data: SignalsData instance
    :return: None
    """
    for curve_idx in curves:
        assert curve_idx < signals_data.cnt_curves, \
            ("The curve index {} is out of range. The total number "
             "of curves: {}.".format(curve_idx, signals_data.cnt_curves))


def global_check(options):
    """Input options global check.

    Returns changed options with converted values.

    options -- namespace with options
    """
    # file import args check
    options = arg_checker.file_arg_check(options)

    # partial import args check
    options = arg_checker.check_partial_args(options)

    # multiplier and delay args check
    arg_checker.check_and_prepare_multiplier_and_delay(options,
                                                       data_axes=2,
                                                       dtype=np.float64)
    # plot args check
    options = arg_checker.plot_arg_check(options)

    # curve labels check
    arg_checker.label_check(options.labels)

    # # data manipulation args check
    # options = arg_checker.data_corr_arg_check(options)

    # save data args check
    options = arg_checker.save_arg_check(options)

    # convert_only arg check
    options = arg_checker.convert_only_arg_check(options)

    # other args check
    options = arg_checker.check_utility_args(options)

    # # analysis args check
    # options = arg_checker.check_analysis_args(options)

    # peak search args check
    options = arg_checker.peak_param_check(options)

    # front delay args check
    options = arg_checker.front_delay_check(options)

    return options


def get_pk_filename(data_files, save_to, shot_name):
    """Compiles the full path to the files with peaks data.
    
    :param data_files:  the list of files with signals data
    :param save_to:     the folder to save peaks data to
    :param shot_name:   the name of current shot
    :return:            full path + prefix for file name with peak data
    """
    return os.path.join(os.path.dirname(data_files[0]),
                                        save_to,
                                        PEAKDATADIR,
                                        shot_name)


def get_peak_files(pk_filename):
    """Returns the list of the peak files.
    If peak files are not found or the folder containing 
    peak data is not found, returns [].

    :param pk_filename: full path + prefix of file names with peak data
    :return: list of full paths
    """
    peak_folder = os.path.dirname(pk_filename)
    file_prefix = os.path.basename(pk_filename)
    if os.path.isdir(peak_folder):
        peak_file_list = []
        for name in file_handler.get_file_list_by_ext(peak_folder, '.csv', sort=True):
            if os.path.basename(name).startswith(file_prefix):
                peak_file_list.append(name)
        return peak_file_list
    return []


def read_single_peak(filename):
    """Reads one file containing the data of the peaks.

    :param   filename:  file with peak (one group of peaks) data
    :return:            grouped peaks data with one peak (group)
                        peaks[curve_idx][0] == SinglePeak instance if 
                        this curve has a peak related to this event (group), 
                        else peaks[curve_idx][0] == None.
    """
    data = numpy.genfromtxt(filename, delimiter=',')

    if data.ndim == 1:
        data = np.expand_dims(data, axis=1)

    peaks = []
    curves_count = data.shape[1]
    for idx in range(curves_count):
        # new_peak = SinglePeak(time=data[idx, 1], value=data[idx, 2],
        #                       sqr_l=data[idx, 3], sqr_r=data[idx, 4])
        new_peak = SinglePeak(time=data[1, idx], value=data[2, idx],
                              sqr_l=data[3, idx], sqr_r=data[4, idx])
        if new_peak.time != 0 or new_peak.val != 0:
            peaks.append([new_peak])
            # peaks[idx].append(new_peak)
        else:
            peaks.append([None])
            # peaks[idx].append(None)
    return peaks


def read_peaks(file_list):
    """Reads all the files containing the data of the peaks.

    :param file_list:   list of files with peak (one group of peaks) data
    :return:            grouped peaks data
                        peaks[curve_idx][group_idx] == SinglePeak instance if
                        this curve has a peak related to this event (group), 
                        else peaks[curve_idx][group_idx] == None.
    """
    if file_list is None or len(file_list) == 0:
        return None
    else:
        groups = read_single_peak(file_list[0])
        curves_count = len(groups)
        for file_idx in range(1, len(file_list)):
            new_group = read_single_peak(file_list[file_idx])
            for wf in range(curves_count):  # wavefrorm number
                groups[wf].append(new_group[wf][0])
        return groups


def renumber_peak_files(file_list, start=1):
    """Checks the file numbering, if the numbering is not continuous 
    or does not start from the specified value, 
    then renames the files and changes the file_list.
    
    :param file_list: the list of files names
    :param start: the numbering must begin with this value
    :return: None
    """
    n1, n2 = file_handler.numbering_parser(file_list)
    digits = n2 - n1
    short_names = [os.path.basename(name) for name in file_list]
    file_nums = [int(name[n1: n2]) for name in short_names]
    dir = os.path.dirname(file_list[0])
    name_format = '{prefix}{num:0' + str(digits) + 'd}{postfix}'
    for i in range(len(file_nums)):
        if file_nums[i] != i + start:
            new_name = (name_format.format(prefix=short_names[i][0: n1],
                                           num=i + start,
                                           postfix=short_names[i][n2:]))
            new_name = os.path.join(dir, new_name)
            os.rename(file_list[i], new_name)
            file_list[i] = new_name


def do_job(args, shot_idx):
    """Process one shot according to the input arguments:
        - applies multiplier, delay
        - finds peaks
        - groups peaks from different curves by time
        - saves peaks and peak plots
        - re-read peak files after peak plot closed
          (user may delete false-positive peak files
          while peak plot window is not closed)
        - plots and saves user specified plots and multiplots


    :param args: namespace with all input args
    :param shot_idx: the number of shot to process

    :type args: argparse.Namespace
    :type shot_idx: int

    :return: None
    """
    number_of_shots = len(args.gr_files)
    if shot_idx < 0 or shot_idx > number_of_shots:
        raise IndexError("Error! The shot_index ({}) is out of range ({} shots given)."
                         "".format(shot_idx, number_of_shots))

    file_list = args.gr_files[shot_idx]
    verbose = not args.silent
    shot_name = file_handler.get_shot_number_str(file_list[0], args.num_mask,
                                                 args.ext_list)
    # get SignalsData
    data = file_handler.read_signals(file_list,
                                     start=args.partial[0],
                                     step=args.partial[1],
                                     points=args.partial[2],
                                     labels=args.labels,
                                     units=args.units,
                                     time_units=args.time_units,
                                     # verbose=verbose
                                     )

    # if verbose:
    #     print("The number of curves = {}".format(data.cnt_curves))

    # checks the number of columns with data,
    # and the number of multipliers, delays, labels
    args.multiplier = arg_checker.check_multiplier(args.multiplier,
                                                   curves_count=data.cnt_curves)
    args.delay = arg_checker.check_delay(args.delay,
                                         curves_count=data.cnt_curves)
    arg_checker.check_coeffs_number(data.cnt_curves, ["label", "unit"],
                                    args.labels, args.units)

    # multiplier and delay
    data = multiplier_and_delay(data,
                                args.multiplier,
                                args.delay)

    # find peaks
    peaks_data = None
    if args.level:
        if verbose:
            print('LEVEL = {}'.format(args.level))

        check_curves_list(args.curves, data)

        if verbose:
            print("Searching for peaks...")

        unsorted_peaks = get_peaks(data, args, verbose)

        # step 7 - group peaks [and plot all curves with peaks]
        peaks_data = group_peaks(unsorted_peaks, args.gr_width)

        # step 8 - save peaks data
        if verbose:
            print("Saving peak data...")

        # full path without peak number and extension:
        pk_filename = get_pk_filename(file_list,
                                      args.save_to,
                                      shot_name)

        file_handler.save_peaks_csv(pk_filename, peaks_data, args.labels)

        # step 9 - save multicurve plot
        multiplot_name = pk_filename + ".plot.png"

        if verbose:
            print("Saving all peaks as " + multiplot_name)
        fig = plotter.plot_multiplot(data, peaks_data, args.curves,
                                     xlim=args.t_bounds, hide=args.peak_hide)
        plt.savefig(multiplot_name, dpi=300)
        if args.peak_hide:
            plt.close(fig)

        else:
            plt.show()

    delay_values = None
    if args.front_delay:
        delay_values = do_front_delay_all(data, args, shot_name, verbose=True)

    if args.read:
        if verbose:
            print("Reading peak data...")

        pk_filename = get_pk_filename(file_list,
                                      args.save_to,
                                      shot_name)
        peak_files = get_peak_files(pk_filename)
        peaks_data = read_peaks(peak_files)
        renumber_peak_files(peak_files)

    # plot preview and save
    if args.plot:
        plotter.do_plots(data, args, shot_name,
                         peaks=peaks_data, verbose=verbose,
                         hide=args.p_hide)

    # filename = "D:\\Experiments\\2023\\2023-09-07-PMT-Linear-colimator20mm-foil\\PMT-peaks\\PEAKS\\peaks_all.csv"
    # shot_col = 5
    # curve_col = 7
    # time_col = 2
    # amp_col = 3
    # curves_count = 24
    #
    # all_shot_peaks = parse_csv_for_peaks(filename, shot_col, curve_col, time_col, amp_col, curves_count,
    #                                      transposed=False)
    #
    # peaks_data = [None] * curves_count
    #
    # if int(shot_name) in all_shot_peaks.keys():
    #     peaks_data = all_shot_peaks[int(shot_name)]

    # plot and save multi-plots
    if args.multiplot:
        plotter.do_multiplots(data, args, shot_name,
                              peaks=peaks_data, verbose=verbose,
                              hide=args.mp_hide)
    return delay_values


def print_front_delay_stats(args, outputs):
    delay_stats = get_2d_array_stat_by_columns(outputs)
    print("----------------------------------------------------------------")
    for idx in range(delay_stats.shape[0]):
        cur1 = args.front_delay[idx]["cur1"]
        cur2 = args.front_delay[idx]["cur2"]
        slope1 = args.front_delay[idx]["slope1"]
        slope2 = args.front_delay[idx]["slope2"]
        level1 = args.front_delay[idx]["level1"]
        level2 = args.front_delay[idx]["level2"]
        units1 = "a.u."
        units2 = "a.u."
        if args.units is not None:
            units1 = args.units[cur1]
            units2 = args.units[cur2]
        label1 = "Curve{}".format(cur1)
        label2 = "Curve{}".format(cur2)
        if args.labels is not None:
            label1 = args.labels[cur1]
            label2 = args.labels[cur2]
        print("DELAY of the {}'s {} front at {} [{}] "
              "relative to the {}'s {} front at {} [{}] statistics:"
              "".format(label2,
                        slope2,
                        level2,
                        units2,
                        label1,
                        slope1,
                        level1,
                        units1))
        print("Mean = {:.6f} {};   Std. Dev. = {:.6f} {};    "
              "Max. Dev. = {:.6f} {};   Number of samples = {}"
              "".format(delay_stats[idx, 0], args.time_units,
                        delay_stats[idx, 1], args.time_units,
                        delay_stats[idx, 2], args.time_units,
                        int(delay_stats[idx, 3])
                        )
              )
        print()


def print_process_time(start_time, stop_time, args):
    print()
    print("--------- Finished ---------")
    spent = stop_time - start_time
    units = "seconds"
    if spent > 3600:
        spent /= 3600
        units = "hours"
    elif spent > 60:
        spent /= 60
        units = "minutes"

    print("--- Time spent: {:.2f} {units} for {shots} shots, {files} files. ---"
          "".format(spent, units=units, shots=len(args.gr_files),
                    files=sum(len(file_list) for file_list in args.gr_files)
                    )
          )


def get_two_fronts_delay(curve1, level1, front1,
                         curve2, level2, front2,
                         bounds1=None,
                         bounds2=None,
                         interpolate=False,
                         save=False, plot_name="voltage_front", verbose=True):
    """Finds the falling or rising edge for the first and second curve,
    calculates the delay between the first edge and the second at the appropriate levels.
    May prints information during the process.
    May saves plots with front points.

    :param curve1: first curve, to get the front point
    :type curve1: SingleCurve

    :param level1: first front trigger level
    :type level1: float

    :param front1: first front type: "rise" | "fall" | "auto"
    :type front1: str

    :param curve2: second curve, to get the front point
    :type curve2: SingleCurve

    :param level2: second front trigger level
    :type level2: float

    :param front2: second front type: "rise" | "fall" | "auto"
    :type front2: str

    :param bounds1: left and right borders to search for curve1
    :type bounds1: list or None

    :param bounds2: left and right borders to search for curve2
    :type bounds2: list or None

    :param interpolate: if false finds the nearest curve point,
                        if true finds the exact time using a linear approximation
    :type interpolate: bool

    :param save: save two plots with first front point and second one
    :type save: bool

    :param plot_name: plot name without extension
    :type plot_name: str

    :param verbose: print fronts delay information or not
    :type verbose: bool

    :return: list of lists with SinglePeaks,
    return[front_idx][peak_idx]
    where front_idx == 0 or 1 and peak_idx == 0

    :rtype: list
    """
    save_as = plot_name + "_1st.png"
    x1, y1 = find_curve_front(curve1,
                              level=level1,
                              front=front1,
                              bounds=bounds1,
                              interpolate=interpolate,
                              save_plot=save,
                              plot_name=save_as)

    save_as = plot_name + "_2nd.png"
    x2, y2 = find_curve_front(curve2,
                              level=level2,
                              front=front2,
                              bounds=bounds2,
                              interpolate=interpolate,
                              save_plot=save,
                              plot_name=save_as)
    front_points = list()
    if x1 is not None:
        front_points.append([SinglePeak(x1, y1, 0)])
    else:
        front_points.append([None])
    if x2 is not None:
        front_points.append([SinglePeak(x2, y2, 0)])
    else:
        front_points.append([None])
    if verbose:
        print("First curve front time = {}".format(x1))
        print("Second curve front time = {}".format(x2))
        if x1 is not None and x2 is not None:
            print("Delay between {} {} front (at {}) and {} {} front (at {}) == {:.4f}"
                  "".format(curve1.label, "negative" if level1 < 0 else "positive", level1,
                            curve2.label, "negative" if level2 < 0 else "positive", level2,
                            x2 - x1))
        else:
            print("Delay between {} {} front (at {}) and {} {} front (at {}) == {:.4f}"
                  "".format(curve1.label, "negative" if level1 < 0 else "positive", level1,
                            curve2.label, "negative" if level2 < 0 else "positive", level2,
                            0))
    return front_points


def print_pulse_duration(curve1, level1, front1, save=False, prefix="pulse_"):
    save_as = prefix + ".png"
    front_points = list()
    x1, y1 = find_curve_front(curve1,
                              level=level1,
                              front=front1,
                              save_plot=save,
                              plot_name=save_as)

    if front1 == 'auto':
        polarity = check_polarity(curve1)
        front1 = "rise" if is_pos(polarity) else "fall"
    is_rising1 = True if front1 == "rise" else False
    is_rising1 = not is_rising1

    start_idx = 0
    for idx in range(curve1.data.shape[1]):
        if is_rising1:
            if curve1.data[1, idx] >= level1:
                start_idx += 1
            else:
                break
        else:
            if curve1.data[1, idx] <= level1:
                start_idx += 1
            else:
                break
    start_idx += 3

    # find pulse end
    fall_found, idx = level_excess(curve1.data[0], curve1.data[1], level1, start=start_idx, rising_front=is_rising1)
    x2 = curve1.data[0, idx]
    y2 = curve1.data[1, idx]
    front_points.append([SinglePeak(x1, y1, 0), SinglePeak(x2, y2, 1)])
    print("First curve front time = {}".format(x1))
    print("Second curve front time = {}".format(x2))
    print("Delay between {}ing front (at {}) and {}ing front (at {}) == "
          "".format(front1, level1, "ris" if front1 == "fall" else "fall",
                    level1,
                    ), end="")
    if fall_found:
        print("{:.4f}".format(x2 - x1))
    else:
        print("OVF")
    return front_points


def do_front_delay_all(data, args, shot_idx, verbose):
    """ For all --front-delay flags:
          1. Calculates delay between fronts of two signals.
          2. Prints the value to console.
          3. Saves graph of two curves with front points.

        :param data: SignalsData instance
        :type data: SignalsData

        :param args: command line arguments, entered by the user
        :type args: argparse.Namespace

        :param shot_idx: the name of current shot
        :type shot_idx: str or int

        :param verbose: shows more info during the process
        :type verbose: bool

        :return: the list of delay values
        :rtype: float
        """
    front_delay_data = list()
    for idx, front_param in enumerate(args.front_delay):
        new_delay = do_front_delay_single(data, front_param, shot_idx, args.t_bounds, verbose, args.unixtime)
        front_delay_data.append(new_delay)
    return front_delay_data


def do_front_delay_single(data, front_param, shot_idx, xlim=(None, None), verbose=False, unixtime=False):
    """ Calculates delay between fronts of two signals.
    Prints the value to console.
    Saves graph of two curves with front points.

    :param data: SignalsData instance
    :type data: SignalsData

    :param front_param: the front delay parameters dict
    :type front_param: dict

    :param shot_idx: the name of current shot
    :type shot_idx: str or int

    :param xlim: the tuple/list with the left and the right X bounds in X units.
    :type xlim: tuple | list

    :param verbose: shows more info during the process
    :type verbose: bool

    :param unixtime: specifies the type of the time column: time or unixtime
    :type unixtime: bool

    :return: the delay value
    :rtype: float
    """

    print_format = "{:.6f}"

    peaks = [None] * data.cnt_curves
    cur1 = front_param["cur1"]
    level1 = front_param["level1"]
    slope1 = front_param["slope1"]
    cur2 = front_param["cur2"]
    level2 = front_param["level2"]
    slope2 = front_param["slope2"]
    bounds1 = front_param["bounds1"]
    bounds2 = front_param["bounds2"]
    front_points = get_two_fronts_delay(data.get_single_curve(cur1),
                                        level1,
                                        slope1,
                                        data.get_single_curve(cur2),
                                        level2,
                                        slope2,
                                        bounds1=bounds1,
                                        bounds2=bounds2,
                                        interpolate=True,
                                        save=False,
                                        verbose=False)
    # front_points is the list of list of SinglePeaks
    # front_points[curve_idx][peak_idx]
    if cur1 == cur2:
        peaks[front_param["cur1"]] = [front_points[0][0], front_points[1][0]]
    else:
        peaks[front_param["cur1"]] = front_points[0]
        peaks[front_param["cur2"]] = front_points[1]

    the_delay = None
    if all(val is not None for val in front_points):
        if front_points[1][0] is not None and front_points[0][0] is not None:
            the_delay = front_points[1][0].time - front_points[0][0].time

    if verbose:
        # print()
        print("{} DELAY of the {}'s {} front at {} [{}] "
              "relative to the {}'s {} front at {} [{}] = {} {}"
              "".format(shot_idx,
                        data.get_curve_label(front_param["cur2"]),
                        front_param["slope2"],
                        front_param["level2"],
                        data.get_curve_units(front_param["cur2"]),
                        data.get_curve_label(front_param["cur1"]),
                        front_param["slope1"],
                        front_param["level1"],
                        data.get_curve_units(front_param["cur1"]),
                        print_format.format(the_delay) if the_delay is not None else None,
                        data.time_units))

    if front_param["save_to"] is not None:
        plot_name = "{shot}_cur{idx1}_{slope1}_and_cur{idx2}_{slope2}_fronts.png" \
                    "".format(shot=shot_idx,
                              idx1=front_param["cur1"],
                              slope1=front_param["slope1"],
                              idx2=front_param["cur2"],
                              slope2=front_param["slope2"]
                              )
        save_as = os.path.join(front_param["save_to"], plot_name)

        if cur1 == cur2:
            plotter.plot_multiple_curve(data,
                                        (front_param["cur1"]),
                                        peaks=peaks,
                                        unixtime=unixtime,
                                        hide=True)
        else:
            plotter.plot_multiplot(data,
                                   peaks,
                                   (front_param["cur1"], front_param["cur2"]),
                                   xlim=xlim,
                                   unixtime=unixtime,
                                   hide=True)

        if not os.path.isdir(front_param["save_to"]):
            try:
                os.makedirs(front_param["save_to"])
            except FileExistsError as e:
                assert os.path.isdir(front_param["save_to"]), e.strerror

        plt.savefig(save_as, dpi=400)
        plt.close('all')

    return the_delay


def main():
    parser = get_parser()

    # # for debugging
    # file_name = '/home/shpakovkv/Projects/PythonSignalProcess/untracked/args/peak_20150515N99.arg'
    # with open(file_name) as fid:
    #     file_lines = [line.strip() for line in fid.readlines()]
    # args = parser.parse_args(file_lines)

    args = parser.parse_args()

    verbose = not args.silent

    # try:
    args = global_check(args)
    # print("\n\n".join("\n".join(str(val) for val in sublist) for sublist in args.gr_files))

    '''
    num_mask (tuple) - contains the first and last index
    of substring of filenameplt.show
    That substring contains the shot number.
    The last idx is excluded: [first, last).
    Read numbering_parser docstring for more info.
    '''

    num_mask = file_handler.numbering_parser([files[0] for
                                              files in args.gr_files])
    args_dict = vars(args)
    args_dict["num_mask"] = num_mask

    if args.hide_all:
        # by default backend == Qt5Agg
        # savefig() time for Qt5Agg == 0.926 s
        #                for Agg == 0.561 s
        # for single curve with 10000 points and one peak
        # run on Intel Core i5-4460 (average for 100 runs)
        # measured by cProfile
        matplotlib.use("Agg")

    # MAIN LOOP
    start_time = time.time()
    if args.level or args.read or args.front_delay or args.multiplot or args.plot:

        shot_list = [shot_idx for shot_idx in range(len(args.gr_files))]
        with Pool(args.threads) as p:
            outputs = p.starmap(do_job, zip(repeat(args), shot_list))
            if args.front_delay is not None:
                print_front_delay_stats(args, outputs)

    # arg_checker.print_duplicates(args.gr_files)
    stop_time = time.time()
    print_process_time(start_time, stop_time, args)

    check_files_for_duplicates(args)


if __name__ == '__main__':
    main()

    # TODO: cl description
    # TODO: test refactored PeakProcess
    # TODO: refactor verbose mode
