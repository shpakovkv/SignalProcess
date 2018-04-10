"""Peak handle functions."""
from __future__ import print_function

from matplotlib import pyplot
import os
import numpy
import bisect
import argparse
import scipy.integrate as integrate

import SignalProcess as sp


pos_polarity_labels = {'pos', 'positive', '+'}
neg_polarity_labels = {'neg', 'negative', '-'}

NOISEATTENUATION = 0.75
SAVETODIR = 'Peaks'
SINGLEPLOTDIR = 'SinglePlot'
MULTIPLOTDIR = 'MultiPlot'
PEAKDATADIR = 'PeakData'


def get_parser():
    """Returns final CL args parser.

    :return: argparse.parser
    """
    p_use = ('python %(prog)s [options]\n'
             '       python %(prog)s @file_with_options')
    p_desc = ('')
    p_ep = ('')

    parser = argparse.ArgumentParser(
        parents=[sp.get_input_files_args_parser(),
                 sp.get_mult_del_args_parser(),
                 sp.get_plot_args_parser(),
                 sp.get_data_corr_args_parser(),
                 get_peak_args_parser()],
        prog='PeakProcess.py',
        description=p_desc, epilog=p_ep, usage=p_use,
        fromfile_prefix_chars='@',
        formatter_class=argparse.RawTextHelpFormatter)
    return parser


def get_peak_args_parser():
    """Returns peak search options parser.
    """
    peak_args_parser = argparse.ArgumentParser(add_help=False)

    peak_args_parser.add_argument(
        '--level',
        action='store',
        dest='level',
        metavar='LEVEL',
        type=float,
        help='description in development\n\n')

    peak_args_parser.add_argument(
        '--diff', '--diff-time',
        action='store',
        dest='pk_diff',
        metavar='DIFF_TIME',
        type=float,
        help='description in development\n\n')

    peak_args_parser.add_argument(
        '--curves',
        action='store',
        dest='curves',
        metavar='CURVE',
        nargs='+',
        type=int,
        help='description in development\n\n')

    peak_args_parser.add_argument(
        '-s', '--save',
        action='store_true',
        dest='save',
        help='description in development\n\n')

    peak_args_parser.add_argument(
        '--bounds', '--time-bounds',
        action='store',
        dest='t_bounds',
        metavar=('LEFT', 'RIGHT'),
        nargs=2,
        type=float,
        default=(None, None),
        help='description in development\n\n')

    peak_args_parser.add_argument(
        '--noise-half-period', '--t-noise',
        action='store',
        dest='t_noise',
        metavar='T',
        type=float,
        help='description in development\n\n')

    peak_args_parser.add_argument(
        '--noise-attenuation',
        action='store',
        dest='noise_att',
        type=float,
        default=NOISEATTENUATION,
        help='description in development\n\n')

    peak_args_parser.add_argument(
        '--group-diff',
        action='store',
        dest='gr_diff',
        metavar='GR_DIFF',
        type=float,
        help='description in development\n\n')

    peak_args_parser.add_argument(
        '--hide-found-peaks',
        action='store_true',
        dest='peak_hide',
        help='description in development\n\n')

    peak_args_parser.add_argument(
        '--hide-all',
        action='store_true',
        dest='hide_all',
        help='description in development\n\n')

    peak_args_parser.add_argument(
        '--read',
        action='store_true',
        dest='read',
        help='description in development\n\n')

    return peak_args_parser


class SinglePeak:
    """Peak object. Contains information on one peak point.
    """
    # TODO sqr_l sqr_r description
    def __init__(self, time=None, value=None, index=None,
                 sqr_l=0, sqr_r=0):
        """        
        :param time: time (X) value of peak point
        :param value: amplitude (Y) value of peak point
        :param index: index of peak point (in SingleCurve data)
        :param sqr_l: 'square factor' of the left edge of the peak:
                      the closer the both sqr_l and sqr_r values 
                      to '1', the higher the probability 
                      that this is an erroneous peak
        :param sqr_r: same as sqr_r
        """
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


def save_peaks_csv(filename, peaks):
    """Saves peaks data.

    :param filename: the peak data file name prefix
    :param peaks: the list of list of peaks [curve_idx][peak_idx]
    :return: None
    """
    folder_path = os.path.dirname(filename)
    if folder_path and not os.path.isdir(folder_path):
        os.makedirs(folder_path)

    if len(filename) > 4 and filename[-4:].upper() == ".CSV":
        filename = filename[0:-4]

    for gr in range(len(peaks[0])):
        content = ""
        for wf in range(len(peaks)):
            pk = peaks[wf][gr]
            if pk is None:
                pk = SinglePeak(0, 0, 0)
            # TODO add curves labels to the peaks files
            content = (content +
                       "{:3d},{:0.18e},{:0.18e},"
                       "{:0.3f},{:0.3f},{:0.3f}\n".format(
                           wf, pk.time, pk.val,
                           pk.sqr_l, pk.sqr_r, pk.sqr_l + pk.sqr_r
                       )
                       )
        postfix = "_peak{:03d}.csv".format(gr + 1)
        # print("Saving " + filename + postfix)
        with open(filename + postfix, 'w') as fid:
            fid.writelines(content)
            # print("Done!")


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
    PeakProcess.level_excess(x, y, level, start=0, step=1,
    window=0, is_positive=True):
    
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
    :rtupe: tuple(float, float)
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
    :param debug: debug mode (outputs additional information)
     
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

    :return: list 
    """

    # Checkout the inputs
    peak_log = ""
    assert level != 0, 'Invalid level value!'

    if is_negative:
        y = -y
        level = abs(level)

    if not tnoise:
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
        level = -abs(level)
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
    :return: peak_data - the three-dimensional array containing data 
             on all the peaks (grouped by time) of all curves
             The array structure:
             peak_data[curve_idx][group_idx] == SinglePeak instance if 
             this curve has a peak related to this event (group), else None
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

        while data[wf] is not None and pk < len(data[wf]):  # and len(data[wf]) > 0:
            '''ADD PEAK TO GROUP
            when curve[i]'s peak[j] is in
            +/-dt interval from peaks of group[gr]
            '''
            if gr < len(peak_time) \
                    and abs(peak_time[gr] - data[wf][pk].time) <= dt:
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

    # return peak_data, peak_map
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
                    For curves not in the args.curves list:
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
        # TODO: autosave single curve peaks preview
        unsorted_peaks[idx] = new_peaks
        if verbose:
            print(peak_log)
    return unsorted_peaks


def check_curves_list(curves, signals_data):
    """Checks the indexes of the curves to process.
    Raises the exception if the index is out of range.
    
    :param curves: the list of indexes of curves to find peaks for
    :param signals_data: SIgnalsData instance
    :return: None
    """
    for curve_idx in curves:
        assert curve_idx < signals_data.count, \
            ("The curve index {} is out of range. The total number "
             "of curves: {}.".format(curve_idx, signals_data.count))


def global_check(options):
    """Input options global check.
    
    Checks the input arguments and converts them to the required format.

    :param options: namespace with options
    :type options:   argparse.namespace
    
    :return: changed namespace with converted values
    """
    # input directory and files check
    if options.src_dir:
        options.src_dir = options.src_dir.strip()
        assert os.path.isdir(options.src_dir), \
            "Can not find directory {}".format(options.src_dir)
    if options.files:
        gr_files = sp.check_file_list(options.src_dir, options.files)
        if not options.src_dir:
            options.src_dir = os.path.dirname(gr_files[0][0])
    else:
        gr_files = sp.get_grouped_file_list(options.src_dir,
                                         options.ext_list,
                                         options.group_size,
                                         options.sorted_by_ch)
    options.gr_files = gr_files

    # Now we have the list of files, grouped by shots:
    # gr_files == [
    #               ['shot001_osc01.wfm', 'shot001_osc02.csv', ...],
    #               ['shot002_osc01.wfm', 'shot002_osc02.csv', ...],
    #               ...etc.
    #             ]

    # check partial import options
    options.partial = sp.check_partial_args(options.partial)

    # # raw check labels not used
    # # instead: the forbidden symbols are replaced during CSV saving
    # if options.labels:
    #     assert global_check_labels(options.labels), \
    #         "Label value error! Only latin letters, " \
    #         "numbers and underscore are allowed."

    def path_constructor(path, arg_name, default_path, default_dir):
        """Checks path, makes dir if not exists.
        
        :param path: user entered path
        :param arg_name: the name of the command line argument 
                         through which the path was entered
        :param default_path: the default output path 
        :param default_dir: the destination folder inside the default path
        :return: path
        """
        if not path:
            path = os.path.join(default_path, default_dir)
        path = sp.check_param_path(path, arg_name)
        return path

    options.save_to = path_constructor(None, '--save-to',
                                       os.path.dirname(gr_files[0][0]),
                                       SAVETODIR)

    options.plot_dir = path_constructor(options.plot_dir, '--p-save',
                                        options.save_to, SINGLEPLOTDIR)

    options.multiplot_dir = path_constructor(options.multiplot_dir,
                                             '--mp-save', options.save_to,
                                             MULTIPLOTDIR)

    # check and convert plot and multiplot options
    if options.plot:
        options.plot = sp.global_check_idx_list(options.plot, '--plot',
                                             allow_all=True)
    if options.multiplot:
        for idx, m_param in enumerate(options.multiplot):
            options.multiplot[idx] = sp.global_check_idx_list(m_param,
                                                           '--multiplot')
    # raw check offset_by_voltage parameters (types)
    options.it_offset = False  # interactive offset process
    if options.offset_by_front:
        assert len(options.offset_by_front) in [2, 4], \
            ("error: argument {arg_name}: expected 2 or 4 arguments.\n"
             "[IDX LEVEL] or [IDX LEVEL WINDOW POLYORDER]."
             "".format(arg_name="--offset-by-curve_level"))
        if len(options.offset_by_front) < 4:
            options.it_offset = True
        options.offset_by_front = \
            sp.global_check_front_params(options.offset_by_front)

    # raw check y_auto_zero parameters (types)
    if options.y_auto_zero:
        options.y_auto_zero = sp.global_check_y_auto_zero_params(options.y_auto_zero)


    # raw check multiplier and delay
    if options.multiplier is not None and options.delay is not None:
        assert len(options.multiplier) == len(options.delay), \
            ("The number of multipliers ({}) is not equal"
             " to the number of delays ({})."
             "".format(len(options.multiplier), len(options.delay)))

    # ==============================================================
    # original PeakProcess args
    if any([options.level, options.pk_diff, options.gr_diff, options.curves]):
        assert all([options.level, options.pk_diff, options.gr_diff, options.curves]), \
            "To start the process of finding peaks, '--level', " \
            "'--diff-time', '--group-diff', '--curves' arguments are needed."
        assert options.pk_diff >=0, \
            "'--diff-time' value must be non negative real number."
        assert options.gr_diff >=0, \
            "'--group-diff' must be non negative real number."
        assert all(idx >= 0 for idx in options.curves), \
            "Curve index must be non negative integer"

    if options.t_noise:
        assert options.t_noise >= 0, \
            "'--noise-half-period' must be non negative real number."
    assert options.noise_att > 0, \
        "'--noise-attenuation' must be real number > 0."

    if all(bound is not None for bound in options.t_bounds):
        assert options.t_bounds[0] < options.t_bounds[1], \
            "The left time bound must be less then the right one."

    if options.hide_all:
        options.p_hide = options.mp_hide = options.peak_hide = True

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
        for name in sp.get_file_list_by_ext(peak_folder, '.csv', sort=True):
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
    peaks = []
    curves_count = data.shape[0]
    for idx in range(curves_count):
        new_peak = SinglePeak(time=data[idx, 1], value=data[idx, 2])
        if new_peak.time != 0 and new_peak.val != 0:
            peaks.append([new_peak])
            # peaks[idx].append(new_peak)
        else:
            peaks.append([None])
            # peaks[idx].append(None)
    return peaks


def read_peaks(file_list):
    """Reads all the files containing the data of the peaks.

    :param file_list:   file with peak (one group of peaks) data
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
    n1, n2 = sp.numbering_parser(file_list)
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




if __name__ == '__main__':
    parser = get_parser()

    file_name = '/home/shpakovkv/Projects/PythonSignalProcess/untracked/args/peak_20150515N99.arg'
    with open(file_name) as fid:
        file_lines = [line.strip() for line in fid.readlines()]
    args = parser.parse_args(file_lines)
    verbose = not args.silent

    # try:
    args = global_check(args)

    '''
    num_mask (tuple) - contains the first and last index
    of substring of filename
    That substring contains the shot number.
    The last idx is excluded: [first, last).
    Read numbering_parser docstring for more info.
    '''

    num_mask = sp.numbering_parser([files[0] for
                                   files in args.gr_files])
    # MAIN LOOP
    print("Check Loop in")  # debugging
    if (args.level or
            args.plot or
            args.multiplot or
            args.read):
        print("==> In loop")  # debugging
        for shot_idx, file_list in enumerate(args.gr_files):
            shot_name = sp.get_shot_number_str(file_list[0], num_mask,
                                               args.ext_list)

            # get SignalsData
            data = sp.read_signals(file_list,
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
            args.multiplier = sp.check_multiplier(args.multiplier,
                                                  count=data.count)
            args.delay = sp.check_delay(args.delay,
                                        count=data.count)
            sp.check_coeffs_number(data.count, ["label", "unit"],
                                   args.labels, args.units)

            # check y_zero_offset parameters (if idx is out of range)
            if args.y_auto_zero:
                args = sp.do_y_zero_offset(data, args)

            # check offset_by_voltage parameters (if idx is out of range)
            if args.offset_by_front:
                args = sp.do_offset_by_front(data, args, shot_name)

            # reset to zero
            if args.zero:
                data = sp.do_reset_to_zero(data, args, verbose)

            # multiplier and delay
            data = sp.multiplier_and_delay(data,
                                           args.multiplier,
                                           args.delay)

            # find peaks
            peaks_data = None
            if args.level:
                check_curves_list(args.curves, data)
                if verbose:
                    print("Searching for peaks...")

                unsorted_peaks = get_peaks(data, args, verbose)

                # step 7 - group peaks [and plot all curves with peaks]
                peaks_data = group_peaks(unsorted_peaks, args.gr_diff)

                # step 8 - save peaks data
                if verbose:
                    print("Saving peak data...")

                # full path without peak number and extension:
                pk_filename = get_pk_filename(file_list,
                                              args.save_to,
                                              shot_name)

                save_peaks_csv(pk_filename, peaks_data)

                # step 9 - save multicurve plot
                multiplot_name = pk_filename + ".plot.png"

                if verbose:
                    print("Saving all peaks as " + multiplot_name)
                sp.plot_multiplot(data, peaks_data, args.curves,
                                  xlim=args.t_bounds)
                pyplot.savefig(multiplot_name, dpi=400)
                if args.peak_hide:
                    pyplot.show(block=False)
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
                sp.do_plots(data, args, shot_name,
                            peaks=peaks_data, verbose=verbose)

            # plot and save multi-plots
            if args.multiplot:
                sp.do_multiplots(data, args, shot_name,
                                 peaks=peaks_data, verbose=verbose)

            # save data
            if args.save:
                saved_as = sp.do_save(data, args, shot_name, verbose)
                labels = [data.label(cr) for cr in data.idx_to_label.keys()]
                sp.save_m_log(file_list, saved_as, labels, args.multiplier,
                              args.delay, args.offset_by_front,
                              args.y_auto_zero, args.partial)

    sp.print_duplicates(args.gr_files, 30)
    # except Exception as e:
    #     print()
    #     sys.exit(e)
    # TODO: final peak table
    # TODO: cl description
    # TODO: cl args description
    # TODO exception handle (via sys.exit(e))

    print('Done!!!')  # debugging
