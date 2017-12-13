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
        help='saves the shot data to a CSV file after all the changes\n'
             'have been applied.\n'
             'NOTE: if one shot corresponds to one CSV file, and\n'
             '      the output directory is not specified, the input\n'
             '      files will be overwritten.\n\n')

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
    """
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
    Returns the index of the value closest to 'value'
    
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
           returns the index of the smallest one.
    """

    idx = bisect.bisect_left(sorted_arr, value)
    if idx == 0:
        if side == 'auto' or side == 'right':
            return idx
        else:
            return None

    if idx == len(sorted_arr):
        if side == 'auto' or side == 'left':
            return idx
        else:
            return None

    after = sorted_arr[idx]
    before = sorted_arr[idx - 1]
    if side == 'auto':
        if after - value < value - before:
            return idx
        else:
            return idx - 1
    elif side == 'left':
        return idx - 1
    elif side == 'right':
        return idx



def level_excess_check(x, y, level, start=0, step=1,
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
    PeakProcess.level_excess_check(x, y, level, start=0, step=1,
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

    front_checked, idx = level_excess_check(curve.time, curve.val, level,
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
    # Поиск пиков (положительных или отрицательных)
    # Пример:
    # Peaks = PeakFinder_v4( x, y, -1, 5E-9, 0.8, 5E-9, 1);
    # Peaks = PeakFinder_v4( x, y, -1, 5, 0.8, 5, 1);

    # x - время (массив непрерывно увеличивающихся значений)
    # y - значения (массив)

    # level - уровень при превышении которого (по модулю) сигнал
    # считается пиком. При level > 0 идет поиск положительных пиков,
    # при level < 0 идет поиск отрицательных пиков.

    # tnoise - максимальный полупериод колебания наводки (время между максимумом и минимумом волны).

    # fflevel - (FrontFallLevel) критерий конца текущего пика. Если модуль значения Y
    # опустится ниже определенной доли от максимума (fflevel), то за пределами
    # этого значения начинается поиск следующего максимума

    # diff_time - окно различия. Если на фронте спада максимума растет
    # следующий максимум, то он должен отстоять от фронта спада предыдущего
    # пика на это значение. В единицах измерения "x".
    # Предполагается diff_time >= tnoise (но не обязательно)

    # graph = 0 если строить проверочный график не надо, напрмер, если функция
    # выполняется внутри цикла

    # noise_attenuation (default 0.5)
    # Ослабление второй полуволны при переполюсовке (наводке). Если программа
    # пропускает (не распознает) много сигналов-наводок, понизьте значение.


    # Проверка введенных значений
    peak_log = ""
    if level == 0:
        raise ValueError('Invalid level value!')

    if is_negative:
        y = -y
        level = -level

    if not tnoise:
        # print('tnoise parameter is empty. ')
        tnoise = x(3) - x(1)
        peak_log += 'Set "tnoise" to default 2 stops = ' + str(tnoise) + "\n"

    if len(time_bounds) != 2:
        raise ValueError("time_bounds has incorrect number of values. "
                         "2 expected, " + str(len(time_bounds)) +
                         " given.")
    # проверка длины массивов
    if len(x) > len(y):
        raise IndexError('Length(X) > Length(Y) by ' + str(len(x) - len(y)))
    elif len(x) < len(y):
        raise IndexError('Warning! Length(X) < Length(Y) by ' + str(len(y) - len(x)))

    if time_bounds[0] is None:
        time_bounds = (x[0], time_bounds[1])
    if time_bounds[1] is None:
        time_bounds = (time_bounds[0], x[-1])
    start_idx = find_nearest_idx(x, time_bounds[0], side='right')
    stop_idx = find_nearest_idx(x, time_bounds[1], side='left')
    if start_idx is None or stop_idx is None:
        peak_log += "Time bounds is out of range.\n"
        return [], peak_log
    diff_idx = int(diff_time // (x[1] - x[0]))
    if debug:
        print("Diff_time = {}, Diff_idx = {}".format(diff_time, diff_idx))

    peak_list = []
    # ==========================================================================
    # print('Starting peaks search...')

    i = start_idx
    while i < stop_idx :
        # Событие превышение уровня
        if y[i] > level:
            # print('Overlevel occurance!')
            # сохранение текущих максимальных значений
            max_y = y[i]
            max_idx = i

            # перебираем все точки внутри diff_time или до конца данных
            while (i <= stop_idx and
                   (x[i] - x[max_idx] <= diff_time or
                    y[i] == max_y)):
                if x[i] > 82 and x[i] < 83:
                    pass
                temp01 = y[i]
                temp02 = x[i]
                if y[i] > max_y:
                    # сохранение текущих максимальных значений
                    max_y = y[i]
                    max_idx = i
                i += 1
            if debug:
                print("local_max = [{:.3f}, {:.3f}] i={}".format(x[max_idx], max_y, max_idx))

            # print('Found max element')
            # перебираем точки слева от пика в пределах diff_time
            # если находим точку повыше - то это "взбрык" на фронте спада
            # а не настоящий пик
            [is_noise, _] = level_excess_check(x,
                                               y,
                                               max_y,
                                               start=max_idx,
                                               step=-1,
                                               window=diff_time,
                                               is_positive=True)
            # print('Right window check completed.')
            if debug and is_noise:
                print('Left Excess at x({:.2f}, {:.2f}) '
                      '== Not a peak at front fall!'.format(x[i], y[i]))

            # проверка пика (наводка или нет)
            # перебираем от max_idx справа все точки в пределах tnoise
            # или до конца данных

            if not is_noise:
                # проверка на переполюсовку выход за level с учетом ослабления
                # второй полуволны NoiseAttenuation
                [is_noise, j] = level_excess_check(x,
                                                   y,
                                                   -max_y * noise_attenuation,
                                                   start=max_idx,
                                                   step=1,
                                                   window=tnoise,
                                                   is_positive=False)

                if debug and is_noise:
                    print('Noise to the right x({:.2f}, {:.2f})'.format(x[j], y[j]))
                else:
                    # проверка на наводку в другую сторону от max_idx
                    [is_noise, j] = level_excess_check(x,
                                                       y,
                                                       -max_y * noise_attenuation,
                                                       start=max_idx,
                                                       step=-1,
                                                       window=tnoise,
                                                       is_positive=False)
                    if debug and is_noise:
                        print('Noise to the left x({:.2f}, {:.2f})'.format(x[j], y[j]))

            # если не наводка, то записываем
            if not is_noise:
                peak_list.append(SinglePeak(x[max_idx], max_y, max_idx))
                # print('Found peak!')
                continue
        i += 1

    peak_log += 'Number of peaks: ' + str(len(peak_list)) + "\n"
    # for pk in peak_list:
    #     print("[{:.3f}, {:.3f}]    ".format(pk.time, pk.val), end="")
    # print()

    # LOCAL INTEGRAL CHECK
    dt = x[1] - x[0]
    di = int(diff_time * 2 // dt)    # diff window in index units

    if di > 3:
        for idx in range(len(peak_list)):
            pk = peak_list[idx]
            # square = pk.val * dt * di
            square = pk.val * di
            integral_left = 0
            integral_right = 0
            peak_log += ("Peak[{:3d}] = [{:7.2f},   {:4.1f}]   "
                         "Square factor [".format(idx, pk.time, pk.val))
            if pk.idx - di >= 0:
                integral_left = integrate.trapz(y[pk.idx-di : pk.idx+1])
                peak_list[idx].sqr_l = integral_left / square
                peak_log += "{:.3f}".format(integral_left / square)
            peak_log += " | "
            if pk.idx + di < len(y):  # stop_idx
                integral_right = integrate.trapz(y[pk.idx: pk.idx + di + 1])
                peak_list[idx].sqr_r = integral_right / square
                peak_log += "{:.3f}".format(integral_right / square)
            peak_log += "]"
            peak_log += "  ({:.3f})".format((integral_right + integral_left) / square)
            peak_log += "\n"
        if peak_list:
            peak_log += "\n"


    if is_negative:
        y = -y
        level = -level
        for i in range(len(peak_list)):
            peak_list[i].invert()
    # строим проверочные графики, если это необходимо
    if graph:
        # plotting curve
        pyplot.plot(x[start_idx:stop_idx], y[start_idx:stop_idx], '-', color='#8888bb')
        pyplot.xlim(time_bounds)
        # plotting level line
        pyplot.plot([x[0], x[len(x) - 1]], [level, level], ':', color='#80ff80')
        # marking overall peaks
        peaks_x = [p.time for p in peak_list]
        peaks_y = [p.val for p in peak_list]
        pyplot.scatter(peaks_x, peaks_y, s=50, edgecolors='#ff7f0e', facecolors='none', linewidths=2)
        pyplot.scatter(peaks_x, peaks_y, s=80, edgecolors='#dd3328', facecolors='none', linewidths=2)
        # pyplot.plot(peaks_x, peaks_y, 'o', color='#ff550e', facecolors='none')
        # pyplot.plot(peaks_x, peaks_y, 'xb', markersize=9)
        pyplot.show()

    return peak_list, peak_log


def group_peaks(data, window):
    # Groups the peaks from different X-Ray detectors
    # each group corresponds to one single act of X-Ray emission
    #
    # data - list with N elements, each element represents one curve
    # curve - list witn M elements, each element represents one peak
    # peak - list with 2 float numbers: [time, value]
    #
    # data[curve_idx][peak_idx][0] = X-value of peak with peak_idx index of curve with curve_idx index
    # data[curve_idx][peak_idx][1] = Y-value of peak with peak_idx index of curve with curve_idx index
    # where curve_idx and peak_idx - zero-based index of curve and peak
    #
    # window - peaks coincide when their X values are within...
    # ... +/-window interval from average X (time) position of peak ()
    # "Average" because X (time) value of a peak may differ from curve to curve

    start_wf = 0
    for wf in range(len(data)):
        if data[wf]:
            start_wf = wf
            break

    peak_time = []                  # 1D array with average X (time) data of peak group
    for peak in data[start_wf]:
        peak_time.append(peak.time)

    dt = abs(window)                            # changes variable name for shortness
    curves_count = len(data)                    # number of waveforms
    num_peak_in_gr = [1] * len(peak_time)       # 1D array with numbers of peaks in each group

    peak_map = [[True] * len(peak_time)]            # (peak_map[curve_idx][group_idx] == True) means "there IS a peak"
    for curve in range(1, curves_count):          # False value means...
        peak_map.append([False] * len(peak_time))   # ..."this curve have not peak at this time position"

    peak_data = [[]]
    for peak in data[start_wf]:    # peaks of first curve
        peak_data[0].append(SinglePeak(*peak.data_full))

    for curve_idx in range(0, start_wf):
        peak_data.insert(0, [None] * len(peak_time))
    for curve_idx in range(start_wf + 1, curves_count):
        peak_data.append([None] * len(peak_time))

    if curves_count <= 1:  # if less than 2 elements = no comparison
        return peak_data, peak_map

    # ---------- making groups of peaks ------------------------------
    # makes groups of peaks
    # two peaks make group when they are close enought
    # ('X' of a peak is within +/- dt interval from 'X' of the group)
    # with adding new peak to a group,
    # the 'X' parameter of the group changes to (X1 + X2 + ... + Xn)/n
    # where n - number of peaks in group
    #
    # for all waveforms exept first
    for wf in range(start_wf + 1, curves_count):
                # wf == 'waveform index'
        gr = 0  # gr == 'group index', zero-based index of current group
        pk = 0  # pk == 'peak index', zero-based index of current peak
                # (in peak list of current waveform)

        while pk < len(data[wf]) and len(data[wf]) > 0:
            # ================================================================
            # ADD PEAK TO GROUP
            # check if curve[i]'s peak[j] is in
            # +/-dt interval from peaks of group[gr]
            # print "Checking Group[" + str(gr) + "]"

            if gr < len(peak_time) and abs(peak_time[gr] - data[wf][pk].time) <= dt:
                # check if X-position of current peak of curve[wf] matches group gr
                if (len(data[wf]) > pk + 1 and
                        (abs(peak_time[gr] - data[wf][pk].time) >
                         abs(peak_time[gr] - data[wf][pk + 1].time))):
                    # next peak of data[wf] matches better
                    # insert new column for current data[wf]'s peak
                    peak_time.insert(gr, data[wf][pk].time)
                    num_peak_in_gr.insert(gr, 1)
                    peak_map[wf].insert(gr, True)
                    peak_data[wf].insert(gr, SinglePeak(
                        *data[wf][pk].data_full))
                    for curve_i in range(curves_count):
                        if curve_i != wf:
                            peak_map[curve_i].insert(gr, False)  # new row to other columns of peak map table
                            peak_data[curve_i].insert(gr, None)  # new row to other columns of peak data table
                    pk += 1
                elif (len(peak_time) > gr + 1 and
                          (abs(peak_time[gr] - data[wf][pk].time) >
                               abs(peak_time[gr + 1] - data[wf][pk].time))):
                    # current peak matches next group better
                    pass
                else:
                    # print "Waveform[" + str(wf) + "] Peak[" + str(pk) + "]   action:    " + "group match"
                    peak_time[gr] = ((peak_time[gr] * num_peak_in_gr[gr] +
                                     data[wf][pk].time) /
                                     (num_peak_in_gr[gr] + 1))        # recalculate average X-position of group
                    num_peak_in_gr[gr] = num_peak_in_gr[gr] + 1         # update count of peaks in current group
                    peak_map[wf][gr] = True                         # update peak_map
                    peak_data[wf][gr] = SinglePeak(*data[wf][pk].data_full)          # add peak to output peak data array
                    pk += 1                                         # go to the next peak
                if gr == len(peak_time) - 1 and pk < len(data[wf]):
                    # Last peak_data column was filled but there are more peaks in the data[wf]
                    # adds new group
                    gr += 1


            # ===============================================================================================
            # INSERT NEW GROUP
            # check if X-position of current peak of curve[wf] is to the left of current group by more than dt
            elif gr < len(peak_time) and data[wf][pk].time < peak_time[gr] - dt:
                # print "Waveform[" + str(wf) + "] Peak[" + str(pk) + "]   action:    " + "left insert"
                peak_time.insert(gr, data[wf][pk].time)       # insert new group of peaks into the groups table
                num_peak_in_gr.insert(gr, 1)                # update the number of peaks in current group
                peak_map[wf].insert(gr, True)               # insert row to current wf column of peak map table
                peak_data[wf].insert(gr, SinglePeak(*data[wf][pk].data_full))   # insert row to current wf column of peak data table
                for curve_i in range(curves_count):
                    if curve_i != wf:
                        peak_map[curve_i].insert(gr, False)  # new row to other columns of peak map table
                        peak_data[curve_i].insert(gr, None)  # new row to other columns of peak data table
                pk += 1                                      # go to the next peak of current curve

            # ===============================================================================================
            # APPEND NEW GROUP
            # here X-position of current peak of curve[wf] is to the right of current group by more than dt
            # checks if current group is the latest in the groups table

            # elif (data[wf][pk].time > peak_time[gr] + dt) and (gr >= len(peak_time) - 1):
            elif gr >= len(peak_time) - 1:
                # print "Waveform[" + str(wf) + "] Peak[" + str(pk) + "]   action:    " + "insert at the end"
                peak_time.append(data[wf][pk].time)           # add new group at the end of the group table
                num_peak_in_gr.append(1)                    # update count of peaks and groups
                peak_map[wf].append(True)                   # add row to current wf column of peak map table
                peak_data[wf].append(SinglePeak(*data[wf][pk].data_full))           # add row to current wf column of peak data table
                for curve_i in range(curves_count):
                    if curve_i != wf:
                        peak_map[curve_i].append(False)  # add row to other columns of peak map table
                        peak_data[curve_i].append(None)  # add row to other columns of peak data table
                pk += 1                                  # go to the next peak...
                gr += 1                                  # ... and the next group

            # ===============================================================================================
            # APPEND NEW GROUP
            # if we are here then the X-position of current peak of curve[curve_i]
            # is to the right of current group by more than dt
            # so go to the next group
            '''
            print()
            for cr_dat in peak_data:
                tmp_list = ["[{} : {}]".format(*local_peak.xy) 
                            if local_peak is not None else "[.....]" for local_peak in cr_dat]
                print("  |\t".join(tmp_list))
            '''
            if gr < len(peak_time) - 1:
                gr += 1
    # END OF GROUPING
    # =======================================================================================================
    # print peak_time
    # print num_peak_in_gr
    return peak_data, peak_map
# ===============================================================================================================



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
    print('Done!!!')