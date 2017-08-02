#!/usr/bin/python
# -*- coding: Windows-1251 -*-
from __future__ import print_function
from matplotlib import pyplot


class SinglePeak:
    def __init__(self, time=None, value=None, index=None):
        self.time = time
        self.val = value
        self.idx = index

    def invert(self):
        if self.val is not None:
            self.val = -self.val

    def get_time_val(self):
        return [self.time, self.val]

    def set_time_val_idx(self, data):
        if len(data) > 3:
            raise ValueError("Too many values to unpack. "
                             "3 expected, " + str(len(data)) +
                             " given.")
        self.time = data[0]
        self.val = data[1]
        self.idx = data[2]

    def get_time_val_idx(self):
        return [self.time, self.val, self.idx]

    xy = property(get_time_val, doc="Get [time, value] of peak.")
    data = property(get_time_val_idx, set_time_val_idx, doc="Get/set [time, value, index] of peak.")


def level_excess_check(x, y, level, start=0, step=1, window=0, is_positive=True):
    # функция проверяет, выходят ли значение по оси Y за величину уровня level
    # проверяются элементы от x(start) до x(start) +/- window
    #
    # step > 0 проверяются значения СПРАВА от стартового элемента (i = start)
    # step < 0 проверяются значения СЛЕВА от стартового элемента (i = start)
    #
    # is_positive == True: функция проверяет, поднимается ли значение 'y' выше значения 'level'
    # is_positive == False: функция проверяет, опускается ли значение 'y' ниже значения 'level'
    # Если да, то ВЫВОДИТ True и индекс первого элемента массива выходящего за 'level'
    # Если нет, то ВЫВОДИТ False и индекс последнего проверенного элемента

    idx = start          # zero-based index
    if window == 0:      # window default value
        window = x[-1] - x[start]

    while (idx >= 0) and (idx < len(y)) and (abs(x[idx] - x[start]) <= window):
        if not is_positive and (y[idx] < level):
            # проверка на выход величины за пределы уровня level (в сторону уменьшения)
            return True, idx
        elif is_positive and (y[idx] > level):
            # проверка на выход величины за пределы уровня level (в сторону уменьшения)
            return True, idx
        idx += step
    return False, idx


def peak_finder(x, y, level, diff_time, tnoise=None, is_negative=True, graph=False, noise_attenuation=0.5):
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

    if level == 0:
        raise ValueError('Invalid level value!')

    if is_negative:
        y = -y
        level = -level

    if not tnoise:
        # print('tnoise parameter is empty. ')
        tnoise = x(3) - x(1)
        print('Set "tnoise" to default 2 stops = ' + str(tnoise))

    # проверка длины массивов
    if len(x) > len(y):
        print('Warning! Length(X) > Length(Y) by ' + str(len(x) - len(y)))
    elif len(x) < len(y):
        print('Warning! Length(Y) > Length(X) by ' + str(len(y) - len(x)))

    peak_list = []
    # ==========================================================================
    # print('Starting peaks search...')

    i = 1
    while i < len(y):
        # Событие превышение уровня
        if y[i] > level:
            # print('Overlevel occurance!')
            # сохранение текущих максимальных значений
            max_y = y[i]
            max_idx = i

            # перебираем все точки внутри diff_time или до конца данных
            while (i <= len(y) and
                   (x[i] - x[max_idx] <= diff_time or
                    y[i] == max_y)):
                if y[i] > max_y:
                    # сохранение текущих максимальных значений
                    max_y = y[i]
                    max_idx = i
                i += 1
            print("local_max = [{}, {}]".format(x[max_idx], max_y))
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
            # if is_noise:
            #     print('Left Excess at x(' + str([x(j), y(j)]) + ')')

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

                if is_noise:
                    i = j
                else:
                    # проверка на наводку в другую сторону от max_idx
                    [is_noise, _] = level_excess_check(x,
                                                       y,
                                                       -max_y * noise_attenuation,
                                                       start=max_idx,
                                                       step=-1,
                                                       window=tnoise,
                                                       is_positive=False)
                    # if is_noise:
                    #     print('Noise at x(' + str([x(j), y(j)]) + ')')

            # если не наводка, то записываем
            if not is_noise:
                peak_list.append(SinglePeak(x[max_idx], max_y, max_idx))
                # print('Found peak!')
                continue
        i += 1

    print('Number of peaks: ' + str(len(peak_list)), end=':    ')
    for pk in peak_list:
        print("[{:.3f}, {:.3f}]    ".format(pk.time, pk.val), end="")
    print()

    if is_negative:
        y = -y
        level = -level
        for i in range(len(peak_list)):
            peak_list[i].invert()
    # строим проверочные графики, если это необходимо
    if graph:
        # plotting curve
        pyplot.plot(x, y, '-k')
        # plotting level line
        pyplot.plot([x[0], x[len(x) - 1]], [level, level], ':g')
        # marking overall peaks
        peaks_x = [p.time for p in peak_list]
        peaks_y = [p.val for p in peak_list]
        pyplot.plot(peaks_x, peaks_y, '*g')
        pyplot.show()
    return peak_list


def find_voltage_front(x,
                       y,
                       level=-0.2,
                       is_positive=False,
                       visualize=False):
    # Find x (time) of voltage front on specific level
    # Default: Negative polarity, -0.2 MV level

    # PeakProcess.level_excess_check(x, y, level, start=0, step=1, window=0, is_positive=True):
    front_checked, idx = level_excess_check(x, y, level, is_positive=is_positive)
    if front_checked:
        if visualize:
            pyplot.plot(x, y, '-b')
            pyplot.plot([x[idx]], [y[idx]], '*r')
            pyplot.show()
        return x[idx], y[idx]
    return None, None


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
        peak_data[0].append(SinglePeak(*peak.data))

    for curve_idx in range(0, start_wf):
        peak_data.insert(0, [None] * len(peak_time))
    for curve_idx in range(start_wf + 1, curves_count):
        peak_data.append([None] * len(peak_time))

    if curves_count <= 1:                           # if less than 2 elements = no comparison
        return peak_data, peak_map

    # ---------- making groups of peaks ------------------------------
    # makes groups of peaks
    # two peaks make group when they are close enought ('X' of a peak is within +/- dt interval from 'X' of the group)
    # with adding new peak to a group, the 'X' parameter of the group changes to (X1 + X2 + ... + Xn)/n
    # where n - number of peaks in group
    #
    # for all waveforms exept first
    for wf in range(start_wf + 1, curves_count):
                # wf == 'waveform index'
        gr = 0  # gr == 'group index', zero-based index of current group
        pk = 0  # pk == 'peak index', zero-based index of current peak (in peak list of current waveform)

        while pk < len(data[wf]) and len(data[wf]) > 0:                         # for all peaks in input curve's peaks data
            # ===============================================================================================
            # ADD PEAK TO GROUP
            # check if curve[i]'s peak[j] is in +/-dt interval from peaks of group[gr]
            # print "Checking Group[" + str(gr) + "]"

            if abs(peak_time[gr] - data[wf][pk].time) <= dt:
                # print "Waveform[" + str(wf) + "] Peak[" + str(pk) + "]   action:    " + "group match"
                peak_time[gr] = ((peak_time[gr] * num_peak_in_gr[gr] + data[wf][pk].time) /
                                 (num_peak_in_gr[gr] + 1))        # recalculate average X-position of group
                num_peak_in_gr[gr] = num_peak_in_gr[gr] + 1         # update count of peaks in current group
                peak_map[wf][gr] = True                         # update peak_map
                peak_data[wf][gr] = SinglePeak(*data[wf][pk].data)          # add peak to output peak data array
                pk += 1                                         # go to the next peak

            # ===============================================================================================
            # INSERT NEW GROUP
            # check if X-position of current peak of curve[wf] is to the left of current group by more than dt
            elif data[wf][pk].time < peak_time[gr] - dt:
                # print "Waveform[" + str(wf) + "] Peak[" + str(pk) + "]   action:    " + "left insert"
                peak_time.insert(gr, data[wf][pk].time)       # insert new group of peaks into the groups table
                num_peak_in_gr.insert(gr, 1)                # update the number of peaks in current group
                peak_map[wf].insert(gr, True)               # insert row to current wf column of peak map table
                peak_data[wf].insert(gr, SinglePeak(*data[wf][pk].data))   # insert row to current wf column of peak data table
                for curve_i in range(curves_count):
                    if curve_i != wf:
                        peak_map[curve_i].insert(gr, False)  # new row to other columns of peak map table
                        peak_data[curve_i].insert(gr, None)  # new row to other columns of peak data table
                pk += 1                                      # go to the next peak of current curve

            # ===============================================================================================
            # APPEND NEW GROUP
            # check if X-position of current peak of curve[wf] is to the right of current group by more than dt
            # and current group is the latest in the groups table
            elif (data[wf][pk].time > peak_time[gr] + dt) and (gr >= len(peak_time) - 1):
                # print "Waveform[" + str(wf) + "] Peak[" + str(pk) + "]   action:    " + "insert at the end"
                peak_time.append(data[wf][pk].time)           # add new group at the end of the group table
                num_peak_in_gr.append(1)                    # update count of peaks and groups
                peak_map[wf].append(True)                   # add row to current wf column of peak map table
                peak_data[wf].append(SinglePeak(*data[wf][pk].data))           # add row to current wf column of peak data table
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
            if gr < len(peak_time) - 1:
                gr += 1
    # END OF GROUPING
    # =======================================================================================================
    # print peak_time
    # print num_peak_in_gr
    return peak_data, peak_map
