# Python 3.6
"""
Peak handle functions.


Maintainer: Shpakov Konstantin
Link: https://github.com/shpakovkv/SignalProcess
"""

from __future__ import print_function

import matplotlib
import matplotlib.pyplot as pyplot

import os
import sys
import numpy
import bisect
import argparse

import numpy as np
import scipy.integrate as integrate

# import SignalProcess as sp
import arg_parser
import arg_checker
import file_handler
import plotter
from multiprocessing import Pool

from multiplier_and_delay import multiplier_and_delay
from data_types import SinglePeak


pos_polarity_labels = {'pos', 'positive', '+'}
neg_polarity_labels = {'neg', 'negative', '-'}

NOISEATTENUATION = 0.75
SAVETODIR = 'Peaks'
SINGLEPLOTDIR = 'SinglePlot'
MULTIPLOTDIR = 'MultiPlot'
PEAKDATADIR = 'PeakData'
DEBUG = False
PEAKFINDERDEBUG = False


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
                 arg_parser.get_utility_args_parser()],
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


def level_excess(x, y, level, start=0, step=1,
                 window=0, is_positive=True):
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
    :param is_positive: the direction of the check
                        True: checks whether the 'Y' value rises above 'level'
                        False: checks whether the 'Y' value 
                        comes down below 'level'
    :type x:            array-like
    :type y:            array-like
    :type level:        float, int ''same as param y''
    :type start:        int
    :type step:         int
    :type window:       float, int ''same as param x''
    :type is_positive:  bool
    :return:            True and an index of first Y element that are 
                        bigger/lower 'level' OR returns False and 
                        an index of the last checked element
    :rtype:             tuple ''(bool, int)''
    """

    idx = start          # zero-based index
    if window == 0:      # window default value
        window = x[-1] - x[start]

    while ((idx >= 0) and (idx < len(y)) and
               (abs(x[idx] - x[start]) <= window)):
        if not is_positive and (y[idx] < level):
            # downward
            return True, idx
        elif is_positive and (y[idx] > level):
            # upward
            return True, idx
        idx += step
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
                     level=-0.2,
                     polarity='auto',
                     save_plot=False,
                     plot_name="voltage_front.png"):
    """Find time point (x) of voltage curve edge at specific level
    Default: Negative polarity, -0.2 MV level
    
    :param curve: curve data
    :param level: amplitude value to find
    :param polarity: the polarity of the curve
    :param save_plot: bool flag
    :param plot_name: plot full file name to save as
    
    :type curve: SingleCurve
    :type level: float
    :type polarity: str '+'/'pos'/'-'/'neg'/'auto'
    :type save_plot: bool
    :type plot_name: str
    
    :return:  (time, amplitude) or (None, None)
    :rtype: tuple(float, float)
    """

    if polarity=='auto':
        polarity = check_polarity(curve)
        if is_pos(polarity):
            level = abs(level)
        else:
            level = -abs(level)

    front_checked, idx = level_excess(curve.time, curve.val, level,
                                      is_positive=is_pos(polarity))
    if front_checked:
        if save_plot:
            pyplot.close('all')
            pyplot.plot(curve.time, curve.val, '-b')
            pyplot.plot([curve.time[idx]], [curve.val[idx]], '*r')
            # pyplot.show()
            folder = os.path.dirname(plot_name)
            if folder != "" and not os.path.isdir(folder):
                os.makedirs(folder)
            pyplot.savefig(plot_name)
            pyplot.close('all')
        return curve.time[idx], curve.val[idx]
    return None, None


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
                                         is_positive=True)
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
                                             is_positive=False)

                if PEAKFINDERDEBUG and is_noise:
                    print('Noise to the right x({:.2f}, {:.2f})'
                          ''.format(x[j], y[j]))
                else:
                    # search to the left from the local max
                    [is_noise, j] = level_excess(x, y,
                                                 -max_y * noise_attenuation,
                                                 start=max_idx, step=-1,
                                                 window=tnoise,
                                                 is_positive=False)
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
        pyplot.plot(x[start_idx:stop_idx], y[start_idx:stop_idx], '-',
                    color='#8888bb')
        pyplot.xlim(time_bounds)
        # plotting level line
        pyplot.plot([x[0], x[len(x) - 1]], [level, level], ':',
                    color='#80ff80')
        # marking overall peaks
        peaks_x = [p.time for p in peak_list]
        peaks_y = [p.val for p in peak_list]
        pyplot.scatter(peaks_x, peaks_y, s=50, edgecolors='#ff7f0e',
                       facecolors='none', linewidths=2)
        pyplot.scatter(peaks_x, peaks_y, s=80, edgecolors='#dd3328',
                       facecolors='none', linewidths=2)
        pyplot.show()

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
    unsorted_peaks = [None] * data.count
    for idx in args.curves:
        if verbose:
            print("Curve #" + str(idx))
        new_peaks, peak_log = peak_finder(
            data.time(idx), data.value(idx),
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
        assert curve_idx < signals_data.count, \
            ("The curve index {} is out of range. The total number "
             "of curves: {}.".format(curve_idx, signals_data.count))


def global_check(options):
    """Input options global check.

    Returns changed options with converted values.

    options -- namespace with options
    """
    # file import args check
    options = arg_checker.file_arg_check(options)

    # partial import args check
    options = arg_checker.check_partial_args(options)

    # plot args check
    options = arg_checker.plot_arg_check(options)

    # curve labels check
    arg_checker.label_check(options.labels)

    # # data manipulation args check
    # options = arg_checker.data_corr_arg_check(options)
    #
    # # save data args check
    # options = arg_checker.save_arg_check(options)
    #
    # # convert_only arg check
    # options = arg_checker.convert_only_arg_check(options)

    # peak search args check
    options = arg_checker.peak_param_check(options)

    options = arg_checker.check_utility_args(options)

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
                                     time_unit=args.time_unit)
    if verbose:
        print("The number of curves = {}".format(data.count))

    # checks the number of columns with data,
    # and the number of multipliers, delays, labels
    args.multiplier = arg_checker.check_multiplier(args.multiplier,
                                                   count=data.count)
    args.delay = arg_checker.check_delay(args.delay,
                                         count=data.count)
    arg_checker.check_coeffs_number(data.count, ["label", "unit"],
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
        pyplot.savefig(multiplot_name, dpi=300)
        if args.peak_hide:
            pyplot.close(fig)

        else:
            pyplot.show()

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

    # plot and save multi-plots
    if args.multiplot:
        plotter.do_multiplots(data, args, shot_name,
                              peaks=peaks_data, verbose=verbose,
                              hide=args.mp_hide)


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

    '''
    num_mask (tuple) - contains the first and last index
    of substring of filenamepyplot.show
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
    import time
    start_time = time.time()
    if (args.level or
            args.read):
        # for shot_idx in range(len(args.gr_files)):
        #     do_job(args, shot_idx)
        with Pool(args.threads) as p:
            p.starmap(do_job, [(args, shot_idx) for shot_idx in range(len(args.gr_files))])
    stop_time = time.time()

    # arg_checker.print_duplicates(args.gr_files)

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

    print("--- Time spent: {:.2f} {units} for {n} shots ---".format(spent, units=units, n=len(args.gr_files)))


if __name__ == '__main__':
    main()

    # TODO: cl description
    # TODO: test refactored PeakProcess
    # TODO: refactor verbose mode
