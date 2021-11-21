# Python 3.6
"""

Maintainer: Shpakov Konstantin
Link: https://github.com/shpakovkv/SignalProcess
"""

import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import dates as md
import bisect
import colorsys
import argparse

from data_types import SignalsData


class ColorRange:
    """Color code iterator. Generates contrast colors.
    Returns the hexadecimal RGB color code (for example '#ffaa00')
    """
    def __init__(self, start_hue=0, hue_step=140, min_hue_diff=20,
                 saturation=(90, 50, 90),
                 luminosity=(40, 75, 45)):
        self.start = start_hue
        self.step = hue_step
        self.window = min_hue_diff
        self.hue_range = 360
        self.s_list = saturation
        self.l_list = luminosity

    def too_close(self, val_list, val, window=None):
        if window is None:
            window = self.window
        for item in val_list:
            if item - window < val < item + window:
                return True
        return False

    def calc_count(self):
        start_list = [self.start]
        count = 0
        for i in range(0, 361):
            count += (self.hue_range - start_list[-1]) // self.window
            remainder = (self.hue_range - start_list[-1]) % self.window
            new_start = self.step - remainder + 1 if remainder > 0 else 0
            if self.too_close(start_list, new_start):
                break
            else:
                start_list.append(new_start)
        return count

    @staticmethod
    def hsl_to_rgb_code(hue, saturation, luminosity):
        hue = float(hue) / 360.0
        saturation = saturation / 100.0
        luminosity = luminosity / 100.0
        # print([hue, saturation, luminosity])
        rgb_float = colorsys.hls_to_rgb(hue, luminosity, saturation)
        # print(rgb_float)
        rgb_int = [int(round(val * 255)) for val in rgb_float]
        # print("{:.2f}, {:.2f}, {:.2f}".format(*rgb), end=" == ")
        rgb_code = "#{:02x}{:02x}{:02x}".format(*rgb_int)
        return rgb_code

    def __iter__(self):
        while True:
            offset = 0
            for sat, lumi in zip(self.s_list, self.l_list):
                last = self.start + offset
                offset += 20
                yield self.hsl_to_rgb_code(last, sat, lumi)
                for i in range(0, self.calc_count()):
                    new_hue = last + self.step
                    if new_hue > 360:
                        new_hue -= 360
                    last = new_hue
                    yield self.hsl_to_rgb_code(new_hue, sat, lumi)


def do_multiplots(signals_data, cl_args, plot_name,
                  peaks=None, verbose=False, hide=False):
    """Plots all the multiplot graphs specified by the user.
    Saves the graphs that the user specified to save.

    :param signals_data: SignalsData instance
    :param cl_args: user-entered arguments (namespace from parser)
    :param plot_name: shot number, needed for saving
    :param peaks: the list of list of peaks (SinglePeak instance)
                    peak_data[0] == list of peaks for data.curves[curves_list[0]]
                    peak_data[1] == list of peaks for data.curves[curves_list[1]]
                    etc.
    :param verbose: show additional information or no

    :type signals_data: SignalsData
    :type cl_args: argparse.Namespace
    :type plot_name: str
    :type peaks: list of lists of SinglePeak
    :type verbose: bool

    :return: None

    """

    for curve_list in cl_args.multiplot:
        check_plot_param(curve_list, signals_data.cnt_curves)

    for curve_list in cl_args.multiplot:
        plot_multiplot(signals_data, peaks, curve_list,
                       xlim=cl_args.t_bounds,
                       unixtime=cl_args.unixtime,
                       hide=hide)
        if cl_args.multiplot_dir is not None:
            idx_list = "_".join(str(i) for
                                i in sorted(curve_list))
            mplot_name = ("{shot}_curves_"
                          "{idx_list}.multiplot.png"
                          "".format(shot=plot_name,
                                    idx_list=idx_list))
            mplot_path = os.path.join(cl_args.multiplot_dir,
                                      mplot_name)
            plt.savefig(mplot_path, dpi=400)
            if verbose:
                print("Multiplot is saved {}"
                      "".format(mplot_path))
        if not cl_args.mp_hide:
            plt.show()
        else:
            # draw plot, but don't pause the process
            # the plot will be closed as soon as drawn
            plt.show(block=False)
        plt.close('all')


def check_plot_param(idx_list, curves_count):
    """Checks if any index from args list is greater than the curves count.

    args            -- the list of curve indexes
    curves_count    -- the count of curves
    """
    error_text = ("The curve index ({idx}) is out of range "
                  "[0:{count}].")
    for idx in idx_list:
        assert (idx < curves_count) & (idx >= 0), \
            (error_text.format(idx=idx, count=curves_count))


def plot_multiplot(data, peak_data, curves_list,
                   xlim=None, amp_unit=None,
                   time_units=None, title=None,
                   unixtime=False, hide=False):
    """Plots subplots for all curves with index in curve_list.
    Optional: plots peaks.
    Subplots are located one under the other.

    data -- the SignalsData instance
    peak_data -- the list of list of peaks (SinglePeak instance)
                 of curves with index in curve_list
                 peak_data[0] == list of peaks for first curve data.curves[curves_list[0]]
                 peak_data[1] == list of peaks for second curve data.curves[curves_list[1]]
                 etc.
    curves_list -- the list of curve indexes in data to be plotted
    xlim        -- the tuple/list with the left and
                   the right X bounds in X units.
    show        -- (bool) show/hide the graph window
    save_as     -- filename (full path) to save the plot as .png
                   Does not save by default.
    amp_unit    -- the unit of Y scale for all subplots.
                   If not specified, the curve.unit parameter will be used
    time_units   -- the unit of time scale for all subplots.
                   If not specified, the time_units parameter of
                   the first curve in curves_list will be used
    title       -- the main title of the figure.
    hide        -- truth turns interactive graph mode off (time save option)
    """
    # TODO: new args description

    # print("data = {}\n"
    #       "peak_data = {}\n"
    #       "curves_list = {}\n"
    #       "xlim = {}\n"
    #       "amp_unit = {}\n"
    #       "time_units = {}\n"
    #       "title = {}\n"
    #       "unixtime = {}"
    #       "".format(data,
    #                 str([peak.get_time_val() for peak in peak_data[0]]),
    #                 curves_list,
    #                 xlim, amp_unit,
    #                 time_units, title,
    #                 unixtime))

    plt.close('all')
    if hide is True:
        plt.ioff()
    else:
        plt.ion()

    fig, axes = plt.subplots(len(curves_list), 1, sharex='all', squeeze=False)
    # # an old color scheme
    # colors = ['#1f22dd', '#ff7f0e', '#9467bd', '#d62728', '#2ca02c',
    #           '#8c564b', '#17becf', '#bcbd22', '#e377c2']
    if title is not None:
        fig.suptitle(title)

    for wf in range(len(curves_list)):
        # plot curve

        if unixtime:
            axes[wf, 0].plot(md.epoch2num(data.time(curves_list[wf])),
                             data.value(curves_list[wf]),
                             '-', color='#9999aa', linewidth=0.5)
        else:
            axes[wf, 0].plot(data.get_x(curves_list[wf]),
                             data.get_y(curves_list[wf]),
                             '-', color='#9999aa', linewidth=0.5)
        axes[wf, 0].tick_params(direction='in', top=True, right=True)

        # set bounds
        if xlim is not None and xlim[0] is not None and xlim[1] is not None:
            axes[wf, 0].set_xlim(xlim)
            axes[wf, 0].set_ylim(calc_y_lim(data.get_x(curves_list[wf]),
                                            data.get_y(curves_list[wf]),
                                            xlim, reserve=0.1))
        # y label (units only)
        if amp_unit is None:
            axes[wf, 0].set_ylabel(data.get_curve_units(curves_list[wf]),
                                   size=10, rotation='horizontal')
        else:
            axes[wf, 0].set_ylabel(amp_unit, size=10, rotation='horizontal')

        # subplot title
        amp_label = data.get_curve_label(curves_list[wf])
        # if data.curves[curves_list[wf]].unit:
        #     amp_label += ", " + data.curves[curves_list[wf]].unit
        axes[wf, 0].text(0.99, 0.01, amp_label, verticalalignment='bottom',
                         horizontalalignment='right',
                         transform=axes[wf, 0].transAxes, size=8)

        # Time axis label
        if wf == len(curves_list) - 1:
            time_label = "Time"
            if time_units:
                time_label += ", " + time_units
            else:
                time_label += ", " + data.time_units
            axes[wf, 0].set_xlabel(time_label, size=10)
            if unixtime:
                dt_fmt = md.DateFormatter('%Y-%m-%d %H:%M:%S')
                axes[wf, 0].xaxis.set_major_formatter(dt_fmt)  # set format
                plt.subplots_adjust(bottom=0.22)  # make more space for datetime values
                plt.xticks(rotation=25)  # rotate long datetime values to avoid overlapping
        axes[wf, 0].tick_params(labelsize=8)

        # plot peaks scatter
        if peak_data is not None:
            color_iter = iter(ColorRange())
            for pk in peak_data[curves_list[wf]]:
                color = next(color_iter)
                if pk is not None:
                    axes[wf, 0].scatter([pk.time], [pk.val], s=20,
                                        edgecolors=color, facecolors='none',
                                        linewidths=1.5)
                    # axes[wf, 0].scatter([pk.time], [pk.val], s=50,
                    #                     edgecolors='none', facecolors=color,
                    #                     linewidths=1, marker='x')
    fig.subplots_adjust(hspace=0)
    return fig


def do_plots(signals_data, cl_args, shot_name, peaks=None, verbose=False, hide=False):
    """Plots all the single curve graphs specified by the user.
    Saves the graphs that the user specified to save.

    :param signals_data: SignalsData instance
    :param cl_args: user-entered arguments (namespace from parser)
    :param shot_name: shot number, needed for saving
    :param peaks: peaks: the list of list of peaks (SinglePeak instance)
                  peak_data[0] == list of peaks for data.curves[curves_list[0]]
                  peak_data[1] == list of peaks for data.curves[curves_list[1]]
                  etc.
    :param verbose: show additional information or no
    :param hide: truth turns interactive graph mode off (time save option)

    :type signals_data: SignalsData
    :type cl_args: argparse.Namespace
    :type shot_name: str
    :type peaks: list
    :type verbose: bool
    :type hide: bool

    :return: None
    """
    if cl_args.plot[0] == -1:  # 'all'
        cl_args.plot = list(range(0, signals_data.cnt_curves))
    else:
        check_plot_param(cl_args.plot, signals_data.cnt_curves)
    for curve_idx in cl_args.plot:
        curve_peaks = peaks[curve_idx] if peaks is not None else None
        fig = plot_multiple_curve(signals_data, curve_idx, curve_peaks,
                                  xlim=cl_args.t_bounds,
                                  unixtime=cl_args.unixtime, hide=hide)
        if cl_args.plot_dir is not None:
            plot_name = (
                "{shot}_curve_{idx}_{label}.plot.png"
                "".format(shot=shot_name, idx=curve_idx,
                          label=signals_data.get_curve_label(curve_idx)))
            plot_path = os.path.join(cl_args.plot_dir, plot_name)
            plt.savefig(plot_path, dpi=400)
            if verbose:
                print("Plot is saved as {}".format(plot_path))
        if not hide:
            plt.show(block=True)
        else:
            plt.close(fig)


def plot_multiple_curve(signals, curve_list, peaks=None,
                        xlim=None, amp_unit=None,
                        time_units=None, title=None,
                        unixtime=False, hide=False):
    """Draws one or more curves on one graph.
    Additionally draws peaks on the underlying layer
    of the same graph, if the peaks exists.
    The color of the curves iterates though the ColorRange iterator.

    NOTE: after the function execution you need to show() or save() pyplot
          otherwise the figure will not be saved or shown

    signals     -- the SignalsData instance with curves data
    curve_list  -- the list of curve indexes
    peaks       -- the list or SinglePeak instances
    title       -- the title for plot
    amp_unit    -- the units for curves Y scale
    time_units   -- the unit for time scale
    xlim        -- the tuple with the left and the right X bounds
    hide        -- truth turns interactive graph mode off (time save option)
    """
    # index of corresponding row
    x_row = 0
    y_row = 1

    plt.close('all')

    # turn on/off interactive mode
    if hide is True:
        plt.ioff()
    else:
        plt.ion()

    fig = plt.figure()

    if xlim is not None and xlim[0] is not None and xlim[1] is not None:
        plt.xlim(xlim)
    color_iter = iter(ColorRange())

    if isinstance(curve_list, int):
        curve_list = [curve_list]

    for curve_idx in curve_list:
        if len(curve_list) > 1:
            color = next(color_iter)
        else:
            color = '#9999aa'
        # print("||  COLOR == {} ===================".format(color))
        if unixtime:
            plt.plot(md.epoch2num(signals.get_x(curve_idx)),
                     signals.get_y(curve_idx),
                     '-', color=color, linewidth=1)
        else:
            plt.plot(signals.get_x(curve_idx),
                     signals.get_y(curve_idx),
                     '-', color=color, linewidth=1)
        axes_obj = plt.gca()
        axes_obj.tick_params(direction='in', top=True, right=True)
        if unixtime:
            dt_fmt = md.DateFormatter('%Y-%m-%d %H:%M:%S')
            axes_obj.xaxis.set_major_formatter(dt_fmt)      # set format
            plt.subplots_adjust(bottom=0.22)                # make more space for datetime values
            plt.xticks(rotation=25)                         # rotate long datetime values to avoid overlapping

    time_label = "Time"
    amp_label = "Amplitude"
    if time_units is not None:
        time_label += ", " + time_units
    elif signals.time_units is not None:
        time_label += ", " + signals.time_units

    if amp_unit is not None:
        amp_label += ", " + amp_unit
    elif all(signals.get_curve_units(curve_list[0]) ==
             signals.get_curve_units(idx) for idx in curve_list):
        amp_label += ", " + signals.get_curve_units(curve_list[0])

    if title is not None:
        plt.title(title)
    elif len(curve_list) > 1:
        # TODO: LEGEND
        pass
    else:
        plt.title(signals.get_curve_label(curve_list[0]))
    plt.xlabel(time_label)
    plt.ylabel(amp_label)

    if peaks is not None:
        peak_x = [peak.time for peak in peaks if peak is not None]
        peak_y = [peak.val for peak in peaks if peak is not None]
        plt.scatter(peak_x, peak_y, s=50, edgecolors='#ff7f0e',
                    facecolors='none', linewidths=2)
        plt.scatter(peak_x, peak_y, s=90, edgecolors='#dd3328',
                    facecolors='none', linewidths=2)
        # plt.scatter(peak_x, peak_y, s=150, edgecolors='none',
        #             facecolors='#133cac', linewidths=1.5, marker='x')
    return fig


def calc_y_lim(time, y, time_bounds=None, reserve=0.1):
    """Returns (min_y, max_y) tuple with y axis bounds.
    The axis boundaries are calculated in such a way
    as to show all points of the curve with an indent
    (default = 10% of the span of the curve) from
    top and from bottom.

    :param time:       the array of time points
    :param y:           the array of amplitude points
    :param time_bounds: the tuple/list with the left and the right X bounds in X units.
    :param reserve:     the indent size (the fraction of the curve's range)

    :return: tuple with 'y' boundaries (min_y, max_y)
    :rtype: tuple
    """
    if time_bounds is None:
        time_bounds = (None, None)
    if time_bounds[0] is None:
        time_bounds = (time[0], time_bounds[1])
    if time_bounds[1] is None:
        time_bounds = (time_bounds[0], time[-1])
    start = find_nearest_idx(time, time_bounds[0], side='right')
    stop = find_nearest_idx(time, time_bounds[1], side='left')
    y_max = np.amax(y[start:stop])
    y_min = np.amin(y[start:stop])
    y_range = y_max - y_min
    reserve *= y_range
    if y_max == 0 and y_min == 0:
        y_max = 1.4
        y_min = -1.4
    return y_min - reserve, y_max + reserve


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

# TODO: add save option to plot_multiplot()
# TODO: unify do_plots() and plot_multiplot()
