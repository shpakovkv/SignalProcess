#!/usr/bin/python
# -*- coding: Windows-1251 -*-
from matplotlib import pyplot


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


    idx = start                       # zero-based index
    if window == 0:                 # window default value
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


def peak_finder(x, y, level, diffwindow, tnoise=None, is_negative=True, graph=False):
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

    # diffwindow - окно различия. Если на фронте спада максимума растет
    # следующий максимум, то он должен отстоять от фронта спада предыдущего
    # пика на это значение. В единицах измерения "x".
    # Предполагается diffwindow >= tnoise (но не обязательно)

    # graph = 0 если строить проверочный график не надо, напрмер, если функция
    # выполняется внутри цикла
    # Ослабление второй полуволны при переполюсовке (наводке). Если программа
    # пропускает (не распознает) много сигналов-наводок, понизьте значение.
    noise_attenuation = 0.7

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

    peak_x = []
    peak_y = []
    peak_idx = []
    # ==========================================================================
    # print('Starting peaks search...')

    max_y = 0
    max_idx = 0

    i = 1
    while i < len(y):
        # Событие превышение уровня
        if y[i] > level:
            # print('Overlevel occurance!')
            # сохранение текущих максимальных значений
            max_y = y[i]
            max_idx = i

            is_noise = False
            # перебираем все точки внутри diffwindow или до конца данных
            while i <= len(y) and x[i] - x[max_idx] <= diffwindow:
                if y[i] > max_y:
                    # сохранение текущих максимальных значений
                    max_y = y[i]
                    max_idx = i
                i += 1
            # print('Found max element')
            # перебираем точки слева от пика в пределах diffwindow
            # если находим точку повыше - то это "взбрык" на фронте спада
            # а не настоящий пик
            [is_noise, j] = level_excess_check(x,
                                               y,
                                               max_y,
                                               start=max_idx,
                                               step=-1,
                                               window=diffwindow,
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
                                                   -level * noise_attenuation,
                                                   start=max_idx,
                                                   step=1,
                                                   window=tnoise,
                                                   is_positive=False)

                if is_noise:
                    i = j
                else:
                    # проверка на наводку в другую сторону от max_idx
                    [is_noise, j] = level_excess_check(x,
                                                       y,
                                                       -level * noise_attenuation,
                                                       start=max_idx,
                                                       step=-1,
                                                       window=tnoise,
                                                       is_positive=False)
                    # if is_noise:
                    #     print('Noise at x(' + str([x(j), y(j)]) + ')')

            # если не наводка, то записываем
            if not is_noise:
                peak_y.append(max_y)
                peak_x.append(x[max_idx])
                peak_idx.append(max_idx)
                # print('Found peak!')
                continue
        i += 1

    print('Number of peaks: ' + str(len(peak_y)))

    if is_negative:
        y = -y
        level = -level
        for i in range(len(peak_y)):
            peak_y[i] = -peak_y[i]
    print('Peaks searching: done.')
    print('--------------------------------------------------------')
    # строим проверочные графики, если это необходимо
    if graph:
        # plotting curve
        pyplot.plot(x, y, '-k')
        # plotting level line
        pyplot.plot([x[0], x[len(x) - 1]], [level, level], ':g')
        # marking overall peaks
        pyplot.plot(peak_x, peak_y, '*g')
        pyplot.show()
    return [peak_x, peak_y]


def group_peaks():
    pass