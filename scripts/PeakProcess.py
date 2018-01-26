#!/usr/bin/python
# -*- coding: Windows-1251 -*-
from __future__ import print_function

from matplotlib import pyplot
import os
import sys
import bisect
import argparse
import scipy.integrate as integrate

import SignalProcess as sp


pos_polarity_labels = {'pos', 'positive', '+'}
neg_polarity_labels = {'neg', 'negative', '-'}


def get_parser():
    """Returns final CL args parser.

    :return: argparse.parser
    """
    base_parser = sp.get_base_parser()
    p_use = ('python %(prog)s [options]\n'
             '       python %(prog)s @file_with_options')
    p_desc = ('')
    p_ep = ('')

    parser = argparse.ArgumentParser(parents=[base_parser],
                                     prog='PeakProcess.py', usage=p_use,
                                     description=p_desc, epilog=p_ep)
    parser.add_argument(
        '--peak',
        action='store',
        dest='save',
        metavar=('LEVEL', 'DIFF_TIME'),
        nargs='+',
        type=float,
        help='description in development\n\n')

    parser.add_argument(
        '--time-bounds',
        action='store',
        dest='save',
        metavar=('LEFT', 'RIGHT'),
        nargs=2,
        type=float,
        help='description in development\n\n')

    parser.add_argument(
        '--t-noise',
        action='store',
        dest='save',
        metavar='T',
        type=float,
        help='description in development\n\n')

    parser.add_argument(
        '--noise-attenuation',
        action='store',
        dest='save',
        type=float,
        help='description in development\n\n')

    parser.add_argument(
        '-s', '--save',
        action='store_true',
        dest='save',
        help='description in development\n\n')

    parser.add_argument(
        '-t', '--save-to', '--target-dir',
        action='store',
        metavar='DIR',
        dest='save_to',
        default='',
        help='specify the output directory.\n\n')

    parser.add_argument(
        '--prefix',
        action='store',
        metavar='PREFIX',
        dest='prefix',
        default='',
        help='specify the file name prefix. This prefix will be added\n'
             'to the output file names during the automatic\n'
             'generation of file names.\n'
             'Default=\'\'.\n\n')

    parser.add_argument(
        '--postfix',
        action='store',
        metavar='POSTFIX',
        dest='postfix',
        default='',
        help='specify the file name postfix. This postfix will be\n'
             'added to the output file names during the automatic\n'
             'generation of file names.\n'
             'Default=\'\'.\n\n')

    parser.add_argument(
        '-o', '--output',
        action='store',
        nargs='+',
        metavar='FILE',
        dest='out_names',
        help='specify the list of file names after the flag.\n'
             'The output files with data will be save with the names\n'
             'from this list. This will override the automatic\n'
             'generation of file names.\n'
             'NOTE: you must enter file names for \n'
             '      all the input shots.\n\n')
    return parser


class SinglePeak:
    """Peak object.
    Contains fields:
        time - time value
        val - amplitude value
        idx - index in the SignalsData
        sqr_l - 
        sqr_r - 
    """
    # TODO sqr_l sqr_r description
    def __init__(self, time=None, value=None, index=None,
                 sqr_l=0, sqr_r=0):
        self.time = time
        self.val = value
        self.idx = index
        self.sqr_l = sqr_l
        self.sqr_r = sqr_r

    def invert(self):
        if self.val is not None:
            self.val = -self.val

    def get_time_val(self):
        return [self.time, self.val]

    def set_time_val_idx(self, data):
        if len(data) != 3:
            raise ValueError("Wrong number of values to unpack. "
                             "3 expected, " + str(len(data)) +
                             " given.")
        self.time = data[0]
        self.val = data[1]
        self.idx = data[2]

    def set_data_full(self, data):
        count = 5
        if len(data) != count:
            raise ValueError("Wrong number of values to unpack. "
                             "{} expected, {} given."
                             "".format(count, len(data)))
        self.time = data[0]
        self.val = data[1]
        self.idx = data[2]
        self.sqr_l = data[3]
        self.sqr_r = data[4]

    def get_time_val_idx(self):
        return [self.time, self.val, self.idx]

    def get_data_full(self):
        return [self.time, self.val, self.idx, self.sqr_l, self.sqr_r]

    xy = property(get_time_val, doc="Get [time, value] of peak.")
    data = property(get_time_val_idx, set_time_val_idx,
                    doc="Get/set [time, value, index] of peak.")
    data_full = property(get_data_full, set_data_full,
                         doc="Get/set [time, value, index, "
                             "sqr_l, sqr_r] of peak.")


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
    """Checks if the (str) 'polarity' is positive.
    
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
    """Checks if the (str) 'polarity' is negative.

    :param polarity:    word denoting polarity
    :type polarity:     str
    :return:            True if the polarity is negative, 
                        otherwise returns False
    :rtype:             bool
    """
    global pos_polarity_labels
    global neg_polarity_labels
    if polarity.lower() in pos_polarity_labels:
        return False
    if polarity.lower() in neg_polarity_labels:
        return True
    else:
        raise ValueError("Wrong polarity value ({})".format(polarity))


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
    """Find x (time) of voltage front on specific level
    Default: Negative polarity, -0.2 MV level
    PeakProcess.level_excess(x, y, level, start=0, step=1,
    window=0, is_positive=True):
    
    :param curve: 
    :param level: 
    :param polarity: 
    :param save_plot: 
    :param plot_name: 
    :type curve: 
    :type level: 
    :type polarity: 
    :type save_plot: 
    :type plot_name: 
    :return: 
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
                noise_attenuation=0.5, debug=False):
    """Peaks search (negative by default)
    
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
    :param debug: debug mode (outputs additional information while searching for peaks)
     
    :type x: numpy.ndarray
    :type y: numpy.ndarray
    :type level: float, int ''same as values of param y''
    :type diff_time: float, int ''same as values of param x''
    :type time_bounds: tuple, list ''(float, float)''
    :type tnoise: float, int ''same as values of param x''
    :type is_negative: bool
    :type graph: bool
    :type noise_attenuation: float
    :type debug: bool

    :return: a list 
    """

    # Checkout the inputs
    peak_log = ""
    assert level != 0, 'Invalid level value!'

    if is_negative:
        y = -y
        level = abs(level)

    if not tnoise:
        # print('tnoise parameter is empty. ')
        tnoise = x[3] - x[1]
        peak_log += 'Set "tnoise" to default 2 stops = ' + str(tnoise) + "\n"

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
    diff_idx = int(diff_time // (x[1] - x[0]))
    if debug:
        print("Diff_time = {}, Diff_idx = {}".format(diff_time, diff_idx))

    peak_list = []

    if start_idx is None or stop_idx is None:
        peak_log += "Time bounds is out of range.\n"
        return peak_list, peak_log
    # ==========================================================================
    # print('Starting peaks search...')

    i = start_idx
    while i < stop_idx :
        if y[i] > level:
            max_y = y[i]
            max_idx = i

            # search for a bigger peak within the diff_time from the found one
            # y[i] == max_y condition is needed for flat-top peaks
            while (i <= stop_idx and
                   (x[i] - x[max_idx] <= diff_time or
                    y[i] == max_y)):
                if y[i] > max_y:
                    max_y = y[i]
                    max_idx = i
                i += 1
            if debug:
                print("local_max = [{:.3f}, {:.3f}] i={}"
                      "".format(x[max_idx], max_y, max_idx))

            # search for a bigger value within the diff_time
            # to the left from the found local maximum
            # if found: this is a little signal fluctuation on the fall front
            # (not a real peak)
            [is_noise, _] = level_excess(x, y, max_y, start=max_idx,
                                         step=-1, window=diff_time,
                                         is_positive=True)
            if debug and is_noise:
                print('Left Excess at x({:.2f}, {:.2f}) '
                      '== Not a peak at front fall!'.format(x[i], y[i]))

            # search for a polarity reversal within tnose from this local max
            # if found: this is a noise (not a real peak)
            if not is_noise:
                # search to the right from this local max
                [is_noise, j] = level_excess(x, y,
                                             -max_y * noise_attenuation,
                                             start=max_idx, step=1,
                                             window=tnoise,
                                             is_positive=False)

                if debug and is_noise:
                    print('Noise to the right x({:.2f}, {:.2f})'
                          ''.format(x[j], y[j]))
                else:
                    # search to the left from this local max
                    [is_noise, j] = level_excess(x, y,
                                                 -max_y * noise_attenuation,
                                                 start=max_idx, step=-1,
                                                 window=tnoise,
                                                 is_positive=False)
                    if debug and is_noise:
                        print('Noise to the left x({:.2f}, {:.2f})'
                              ''.format(x[j], y[j]))

            if not is_noise:
                peak_list.append(SinglePeak(x[max_idx], max_y, max_idx))
                continue
        i += 1

    peak_log += 'Number of peaks: ' + str(len(peak_list)) + "\n"

    # LOCAL INTEGRAL CHECK
    dt = x[1] - x[0]
    di = int(diff_time * 2 // dt)    # diff window in index units

    if di > 3:
        for idx in range(len(peak_list)):
            pk = peak_list[idx]
            # square = pk.val * dt * di
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
    act of X-Ray emission, registered by several detectors).
    
    If 
    
    :param data: three-dimensional array containing data 
                 on all the peaks of all curves
                 The array structure:
                 data[curve_idx][peak_idx] == SinglePeak instance
    :param window: peaks coincide when their X values are within
                   +/-window interval from average X (time) position 
                   of peak (event). "Average" because X (time) value 
                   of a peak (event) may differ from curve to curve.
    :return: (peak_data, peak_map)
             where peak_data is three-dimensional array containing data 
             on all the peaks (grouped by time) of all curves
             The array structure:
             peak_data[curve_idx][group_idx] == SinglePeak instance if 
             this curve has a peak related to this event (group), else None
             
             and peak_map is similar array:
             peak_map[curve_idx][group_idx] == True if this curve has a 
             peak related to this event (group), else False
    """

    def insert_group(peak, peak_data, peak_map, group_time,
                     num_peak_in_gr, wf, gr):
        group_time.insert(gr, peak.time)
        num_peak_in_gr.insert(gr, 1)
        for curve_i in range(len(peak_data)):
            if curve_i == wf:
                peak_map[curve_i].insert(gr, True)
                peak_data[curve_i].insert(gr, SinglePeak(*peak.data_full))
            else:
                peak_map[curve_i].insert(gr, False)
                peak_data[curve_i].insert(gr, None)


    def add_pk_to_gr(peak, peak_data, peak_map, peak_time,
                     num_peak_in_gr, wf, gr):
        peak_time[gr] = ((peak_time[gr] * num_peak_in_gr[gr] +
                          peak.time) /
                         (num_peak_in_gr[gr] + 1))
        num_peak_in_gr[gr] = num_peak_in_gr[gr] + 1
        peak_map[wf][gr] = True
        peak_data[wf][gr] = SinglePeak(*peak.data_full)

    # wf == waveform == curve
    start_wf = 0

    # skip first curves if they have no peaks
    for wf in range(len(data)):
        if data[wf]:
            start_wf = wf
            break

    # 1D array with average X (time) data of peak group
    peak_time = []
    for peak in data[start_wf]:
        peak_time.append(peak.time)

    # 1D array with numbers of peaks in each group
    num_peak_in_gr = [1] * len(peak_time)

    dt = abs(window)
    curves_count = len(data)

    # at the beginning we have only start_wf's peaks
    peak_map = [[True] * len(peak_time)]
    for curve in range(1, curves_count):
        peak_map.append([False] * len(peak_time))

    peak_data = [[]]
    for peak in data[start_wf]:
        peak_data[0].append(SinglePeak(*peak.data_full))
    for curve_idx in range(0, start_wf):
        peak_data.insert(0, [None] * len(peak_time))
    for curve_idx in range(start_wf + 1, curves_count):
        peak_data.append([None] * len(peak_time))

    if curves_count <= 1:
        return peak_data, peak_map

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

        while pk < len(data[wf]) and len(data[wf]) > 0:
            '''ADD PEAK TO GROUP
            when curve[i]'s peak[j] is in
            +/-dt interval from peaks of group[gr]
            '''
            if gr < len(peak_time) and abs(peak_time[gr] - data[wf][pk].time) <= dt:
                if (len(data[wf]) > pk + 1 and
                        (abs(peak_time[gr] - data[wf][pk].time) >
                         abs(peak_time[gr] - data[wf][pk + 1].time))):
                    # next peak of data[wf] matches better
                    # insert new group for current data[wf]'s peak
                    insert_group(data[wf][pk], peak_data, peak_map, peak_time,
                                 num_peak_in_gr, wf, gr)
                    pk += 1
                elif (len(peak_time) > gr + 1 and
                          (abs(peak_time[gr] - data[wf][pk].time) >
                               abs(peak_time[gr + 1] - data[wf][pk].time))):
                    # current peak matches next group better
                    pass
                else:
                    add_pk_to_gr(data[wf][pk], peak_data, peak_map, peak_time,
                                 num_peak_in_gr, wf, gr)
                    pk += 1
                if gr == len(peak_time) - 1 and pk < len(data[wf]):
                    # Last peak_data column was filled but there are
                    # more peaks in the data[wf], so adds new group
                    gr += 1

            elif (gr < len(peak_time) and
                          data[wf][pk].time < peak_time[gr] - dt):
                '''INSERT NEW GROUP
                when X-position of current peak of curve[wf] is 
                to the left of current group by more than dt
                '''
                insert_group(data[wf][pk], peak_data, peak_map, peak_time,
                             num_peak_in_gr, wf, gr)
                pk += 1

            elif gr >= len(peak_time) - 1:
                '''APPEND NEW GROUP
                when X-position of current peak of curve[wf] is to the right 
                of the last group
                '''
                insert_group(data[wf][pk], peak_data, peak_map, peak_time,
                             num_peak_in_gr, wf, len(peak_data))
                pk += 1
                gr += 1

            if gr < len(peak_time) - 1:
                gr += 1

    return peak_data, peak_map



if __name__ == '__main__':
    parser = get_parser()

    args = parser.parse_args()
    verbose = not args.silent

    try:
        args = sp.global_check(args)
    except Exception as e:
        print()
        sys.exit(e)

    # TODO: reformat lines PEP8
    # TODO: English comments
    # TODO: cl args original
    # TODO: cl read/save/plot (copy from SP)
    # TODO: cl peak finder
    # TODO: cl peak reader
    # TODO: cl replot peak multiplots
    # TODO: cl description
    # TODO: cl args discription
    print('Done!!!')