# Python 3.6
"""
Maintainer: Shpakov Konstantin
Link: https://github.com/shpakovkv/SignalProcess
"""

import os
import sys
import argparse

import numpy as np
from scipy.signal import savgol_filter
from matplotlib import pyplot as plt
import matplotlib

import data_types
import arg_parser
import arg_checker
import file_handler
import plotter

from numba import vectorize, float64

from multiplier_and_delay import *

from PeakProcess import find_nearest_idx, check_polarity, is_pos, find_curve_front, print_front_delay, print_pulse_duration

verbose = True
global_log = ""
DEBUG = True


# ========================================
# -----     CL INTERFACE     -------------
# ========================================
def get_parser():
    """Returns final parser.
    """
    p_disc = ""

    p_ep = ""

    p_use = ("python %(prog)s [options]\n"
             "       python %(prog)s @file_with_options")

    final_parser = argparse.ArgumentParser(
        parents=[arg_parser.get_input_files_args_parser(),
                 arg_parser.get_mult_del_args_parser(),
                 arg_parser.get_data_corr_args_parser(),
                 arg_parser.get_plot_args_parser(),
                 arg_parser.get_output_args_parser(),
                 arg_parser.get_utility_args_parser()],
        prog='SignalProcess.py',
        description=p_disc, epilog=p_ep, usage=p_use,
        fromfile_prefix_chars='@',
        formatter_class=argparse.RawTextHelpFormatter)
    return final_parser


# ========================================
# -----    WORKFLOW     ------------------
# ========================================
def add_to_log(s, print_to_console=True):
    # not used
    global global_log
    global_log += s
    if print_to_console:
        print(s, end="")


def smooth_voltage(y_data, window=101, poly_order=3):
    """This function returns smoothed copy of 'y_data'.

    y_data      -- 1D numpy.ndarray value points
    window      -- The length of the filter window
                   (i.e. the number of coefficients).
                   window_length must be a positive
                   odd integer >= 5.
    poly_order  -- The order of the polynomial used to fit
                   the samples. polyorder must be less
                   than window_length.
    The values below are optimal for 1 ns resolution of
    voltage waveform of ERG installation:
    poly_order = 3
    window = 101
    """

    # calc time_step and converts to nanoseconds
    assert isinstance(window, int), \
        "The length of the filter window must be positive integer >= 5."
    assert isinstance(poly_order, int), \
        "The polynomial order of the filter must be positive integer."
    if len(y_data) < window:
        window = len(y_data) - 1
    if window % 2 == 0:
        # window must be even number
        window += 1
    if window < 5:
        # lowest possible value
        window = 5

    if len(y_data) >= 5:
        # print("WINDOW LEN = {}  |  POLY ORDER = {}"
        #       "".format(window, poly_order))
        y_smoothed = savgol_filter(y_data, window, poly_order, mode='nearest')
        return y_smoothed

    # too short array to be processed
    return y_data


def align_and_append_ndarray(*args):
    """Returns 2D numpy.ndarray containing all input 2D numpy.ndarrays.
    If input arrays have different number of rows,
    fills missing values with 'nan'.

    :param args: 2d ndarrays
    :type args: np.ndarray

    :return: united 2D ndarray
    :rtype: np.ndarray
    """
    # CHECK TYPE & LENGTH
    for arr in args:
        if not isinstance(arr, np.ndarray):
            raise TypeError("Input arrays must be instances "
                            "of numpy.ndarray class.")
        if np.ndim(arr) != 2:
            raise ValueError("Input arrays must have 2 dimensions.")

    # ALIGN & APPEND
    col_len_list = [arr.shape[0] for arr in args]
    max_rows = max(col_len_list)
    data = np.empty(shape=(max_rows, 0), dtype=float, order='F')
    for arr in args:
        miss_rows = max_rows - arr.shape[0]
        nan_arr = np.empty(shape=(miss_rows, arr.shape[1]),
                           dtype=float, order='F')
        nan_arr[:] = np.nan
        # nan_arr[:] = np.NAN

        aligned_arr = np.append(arr, nan_arr, axis=0)
        data = np.append(data, aligned_arr, axis=1)
    if DEBUG:
        print("aligned array shape = {}".format(data.shape))
    return data


def y_zero_offset(curve, start_x, stop_x):
    """
    Returns the Y zero level offset value.
    Use it for zero level correction before PeakProcess.

    curve               -- SingleCurve instance
    start_x and stop_x  -- define the limits of the
                           X interval where Y is filled with noise only.
    """
    if start_x < curve.time[0]:
        start_x = curve.time[0]
    if stop_x > curve.time[-1]:
        stop_x = curve.time[-1]
    assert stop_x > start_x, \
        ("Error! start_x value ({}) must be lower than stop_x({}) value."
         "".format(start_x, stop_x))
    if start_x > curve.time[-1] or stop_x < curve.time[0]:
        return 0
    start_idx = find_nearest_idx(curve.time, start_x, side='right')
    stop_idx = find_nearest_idx(curve.time, stop_x, side='left')
    amp_sum = 0.0
    for val in curve.val[start_idx:stop_idx + 1]:
        amp_sum += val
    return amp_sum / (stop_idx - start_idx + 1)


def y_zero_offset_all(signals_data, curves_list, start_stop_tuples):
    """Return delays list for all columns in SignalsData stored
    in filename_or_list.
    Delays for Y-columns will be filled with
    the Y zero level offset values for only specified curves.
    For all other Y columns and for all X columns delay will be 0.

    Use it for zero level correction before PeakProcess.

    filename_or_list    -- file with data or list of files,
                           if the data stored in several files
    curves_list         -- zero-based indices of curves for which
                           you want to find the zero level offset
    start_stop_tuples   -- list of (start_x, stop_x) tuples for each
                           curves in curves list.
                           You can specify one tuple or list and
                           it will be applied to all the curves.
    file_type           -- file type (Default = 'CSV')
    delimiter           -- delimiter for csv files
    """

    assert len(curves_list) == len(start_stop_tuples), \
        "Error! The number of (start_x, stop_x) tuples ({}) " \
        "does not match the number of specified curves " \
        "({}).".format(len(start_stop_tuples), len(curves_list))

    delays = [0 for _ in range(2 * signals_data.count)]
    for tuple_idx, curve_idx in enumerate(curves_list):
        y_data_idx = curve_idx * 2 + 1
        delays[y_data_idx] = y_zero_offset(signals_data.curves[curve_idx],
                                           *start_stop_tuples[tuple_idx])
    # print("Delays = ")
    # for i in range(0, len(delays), 2):
    #     print("{}, {},".format(delays[i], delays[i + 1]))
    return delays


def pretty_print_nums(nums, prefix=None, postfix=None,
                      s=u'{pref}{val:.2f}{postf}', show=True):
    """Prints template 's' filled with values from 'nums',
    'prefix' and 'postfix' arrays for all numbers in 'nums'.

    nums    -- array of float or int
    prefix  -- array of prefixes for all values in 'nums'
               or single prefix string for all elements.
    postfix -- array of postfixes for all values in 'nums'
               or single postfix string for all elements.
    s       -- template string.
    """
    if not prefix:
        prefix = ("" for _ in nums)
    elif isinstance(prefix, str):
        prefix = [prefix for _ in nums]
    if not postfix:
        postfix = ("" for _ in nums)
    elif isinstance(postfix, str):
        postfix = [postfix for _ in nums]
    message = ""
    for pref, val, postf in zip(prefix, nums, postfix):
        if val > 0:
            pref += "+"
        message += s.format(pref=pref, val=val, postf=postf) + "\n"
    if show:
        print(message)
    return message


def raw_y_auto_zero(params, multiplier, delay):
    """Returns raw values for y_auto_zero parameters.
    'Raw values' are values before applying the multiplier.

    params      -- y auto zero parameters (see --y-auto-zero
                   argument's hep for more detail)
    multiplier  -- the list of multipliers for all the curves
                   in the SignalsData
    delay       -- the list of delays for all the curves
                   in the SignalsData
    """
    raw_params = []
    for item_idx, item in enumerate(params):
        curve_idx = item[0]
        start_x = (item[1] + delay[curve_idx * 2]) / multiplier[curve_idx * 2]
        stop_x = (item[2] + delay[curve_idx * 2]) / multiplier[curve_idx * 2]
        raw_params.append([curve_idx, start_x, stop_x])
    return raw_params


def update_by_y_auto_zero(data, y_auto_zero_params,
                          multiplier, delay, verbose=True):
    """The function analyzes the mean amplitude in
    the selected X range of the selected curves.
    The Y zero offset values obtained are added to
    the corresponding items of the original list of delays.
    The new list of delays is returned.

    data                -- SignalsData instance
    y_auto_zero_params  -- y auto zero parameters (see --y-auto-zero
                           argument's hep for more detail)
    multiplier          -- the list of multipliers for all the curves
                           in the SignalsData
    delay               -- the list of delays for all the curves
                           in the SignalsData
    verbose             -- show/hide information during function execution
    """
    new_delay = delay[:]
    curves_list = [item[0] for item in y_auto_zero_params]
    start_stop_tuples = [(item[1], item[2]) for item in
                         raw_y_auto_zero(y_auto_zero_params,
                                         multiplier,
                                         delay)]
    for idx, new_val in \
            enumerate(y_zero_offset_all(data,
                                        curves_list,
                                        start_stop_tuples)):
        new_delay[idx] += new_val * multiplier[idx]

    if verbose:
        print()
        print("Y auto zero offset results:")
        curves_labels = ["Curve[{}]: ".format(curve_idx) for
                         curve_idx in curves_list]
        # print(pretty_print_nums([new_delay[idx] for idx in
        #                               range(0, len(new_delay), 2)],
        #                               curves_labels_DEBUG, " ns",
        #                               show=False))
        print(pretty_print_nums([new_delay[idx] for idx in
                                 range(1, len(new_delay), 2)],
                                curves_labels, show=False))
    return new_delay


def do_y_zero_offset(signals_data, cl_args):
    """Analyzes the signal fragment in the selected time interval and
    finds the value of the DC component.
    Subtracts the constant component of the signal.

    NOTE: this function only changes the delay list. Apply it to
    the signals data via multiplier_and_delay function.

    :param signals_data: SignalsData instance with curves data
    :param cl_args: command line arguments, entered by the user

    :type signals_data: SignalsData
    :type cl_args: argparse.Namespace

    :return: changed cl_args
    :rtype: argparse.Namespace
    """
    arg_checker.check_y_auto_zero_params(signals_data, cl_args.y_auto_zero)

    # updates delays with accordance to Y zero offset
    cl_args.delay = update_by_y_auto_zero(signals_data, cl_args.y_auto_zero,
                                          cl_args.multiplier,
                                          cl_args.delay,
                                          verbose=True)
    return cl_args


def get_front_point(signals_data, args, multiplier, delay,
                    front_plot_name, interactive=False):
    """Finds the most left front point where the curve
    amplitude is greater (lower - for negative curve) than
    the level (args[1]) value.
    To improve accuracy, the signal is smoothed by
    the savgol filter.

    :param signals_data: the SignalsData instance
    :param args: the list of 4 values [idx, level, window, order]:
                 idx    - the index of the curve
                 level  - the amplitude level
                 window - length of the filter window (must be
                          an odd integer greater than 4)
                 order  - (int) the order of the polynomial used
                          to fit the samples (must be < window)
    :param multiplier: list of multipliers not yet applied to data
    :param delay: list of delays not yet applied to data
    :param front_plot_name: the full path to save the graph with the curve
                            and the marked point on the rise front
                            (fall front - for negative curve)
    :param interactive: turns on interactive mode of the smooth filter
                        parameters selection

    :type signals_data: SignalsData
    :type args: tuple/list
    :type multiplier: tuple/list
    :type delay: tuple/list
    :type front_plot_name: str
    :type interactive: bool

    :return: front point
    :rtype: SinglePeak
    """
    # TODO: fix for new SignalsData class
    if not os.path.isdir(os.path.dirname(front_plot_name)):
        os.makedirs(os.path.dirname(front_plot_name))

    curve_idx = args[0]
    level = args[1]
    window = args[2]
    poly_order = args[3]

    # curve = data_types.SingleCurve(signals_data.get_single_curve(curve_idx))
    # curve = signals_data.get_single_curve(curve_idx).copy()

    # make local copy of target curve
    curve = SignalsData(np.copy(signals_data.get_curve_2d_arr(curve_idx)),
                        labels=[signals_data.get_curve_label(0)],
                        units=[signals_data.get_curve_units(0)],
                        time_units=signals_data.time_units)

    # polarity = check_polarity(curve.get_single_curve(0))
    front = "fall" if level < 0 else "rise"

    # TODO: user enter front type

    # if is_pos(polarity):
    #     level = abs(level)
    # else:
    #     level = -abs(level)

    cur_mult = multiplier[curve_idx: curve_idx + 1]
    cur_del = delay[curve_idx: curve_idx + 1]

    # apply multiplier and delay to local copy
    curve.data = multiplier_and_delay(curve.data, cur_mult, cur_del)
    # get x and y columns
    data_x = curve.get_x(0)
    data_y = curve.get_y(0)

    first_num, last_num = file_handler.get_real_num_bounds_1d(data_y)

    data_y = data_y[first_num:last_num + 1]
    data_x = data_x[first_num:last_num + 1]

    print("Time offset by curve front process.")
    print("Searching curve[{idx}] \"{label}\" front at level = {level}"
          "".format(idx=curve_idx, label=curve.get_curve_label(0), level=level))

    plot_title = None
    smoothed_curve = None
    front_point = None
    if interactive:
        print("\n------ Interactive offset_by_curve_front process ------\n"
              "Enter two space separated positive integer value\n"
              "for WINDOW and POLYORDER parameters.\n"
              "\n"
              "WINDOW must be an odd integer value >= 5,\n"
              "POLYORDER must be 0, 1, 2, 3, 4 or 5 and less than WINDOW.\n"
              "\n"
              "The larger the WINDOW value, the greater the smooth effect.\n"
              "\n"
              "The larger the POLYORDER value, the more accurate \n"
              "the result of smoothing.\n"
              "\n"
              "Close graph window to continue.")
    cancel = False
    # interactive cycle is performed at least once
    while not cancel:
        # smooth curve
        data_y_smooth = smooth_voltage(data_y, window, poly_order)
        smooth_data = np.stack((data_x, data_y_smooth))

        smoothed_curve = data_types.SingleCurve(smooth_data,
                                                curve.get_curve_label(0),
                                                curve.get_curve_units(0),
                                                curve.time_units)
        # find front
        front_x, front_y = find_curve_front(smoothed_curve,
                                            level, front)

        plot_title = ("Curve[{idx}] \"{label}\"\n"
                      "".format(idx=curve_idx, label=curve.get_curve_label(0)))
        if front_x is not None:
            front_point = data_types.SinglePeak(front_x, front_y, 0)
            plot_title += "Found front at [{},  {}]".format(front_x, front_y)
        else:
            plot_title += "No front found."
            front_point = None

        if interactive:
            # visualize for user
            # curve.append_2d_array(smoothed_curve.data,
            #                       labels=[curve.get_curve_label(0)],
            #                       units=[curve.get_curve_units(0)])
            curve.add_from_array(smoothed_curve.data,
                                 labels=[curve.get_curve_label(0) + "_sm"],
                                 units=[curve.get_curve_units(0)],
                                 time_units=curve.time_units)
            plotter.plot_multiple_curve(curve,
                                        curve_list=[0, 1],
                                        peaks=[front_point],
                                        title=plot_title)

            print("\nPrevious values:  {win}  {ord}\n"
                  "".format(win=window, ord=poly_order))
            print("Close plot and press enter, without entering a value "
                  "to exit the interactive mode. "
                  "The previously entered values will be used.")
            plt.show(block=True)

            while True:
                # get user input
                try:
                    print("Enter WINDOW POLYORDER >>>", end="")
                    user_input = sys.stdin.readline().strip()
                    if user_input == "":
                        # exit interactive mode,
                        # and use previously entered parameters
                        cancel = True
                        break
                    user_input = user_input.split()
                    assert len(user_input) == 2, ""
                    new_win = int(user_input[0])
                    assert new_win > 4, ""
                    new_ord = int(user_input[1])
                    assert new_win > new_ord, ""
                    assert 0 <= new_ord <= 5, ""
                except (ValueError, AssertionError):
                    print("Wrong values!")
                else:
                    window = new_win
                    poly_order = new_ord
                    break
            curve.delete_curve(1)
        else:
            cancel = True


    # save final version of the plot
    curve.add_from_array(smoothed_curve.data,
                         labels=[curve.get_curve_label(0) + "_sm"],
                         units=[curve.get_curve_units(0)],
                         time_units=curve.time_units)

    plotter.plot_multiple_curve(curve,
                                curve_list=[0, 1],
                                peaks=[front_point],
                                title=plot_title)

    plt.savefig(front_plot_name, dpi=400)
    # show plot to avoid tkinter warning "can't invoke "event" command..."
    plt.show(block=False)
    plt.close('all')

    return front_point


def do_offset_by_front(signals_data, cl_args, shot_name):
    """Finds the most left front point where the curve
    amplitude is greater (lower - for negative curve) than
    the 'level' value (cl_args.offset_by_front[1]).
    To improve accuracy, the signal is smoothed by
    the savgol filter.

    Saves a curve plot with the point on the front.

    Makes the point the origin of the time axis for
    all signals (changes the list of the delays, but not applies it).

    :param signals_data: SignalsData instance with curves data
    :param cl_args: command line arguments, entered by the user
    :param shot_name: the current shot number (needed for graph save)

    :type signals_data: SignalsData
    :type cl_args: argparse.Namespace
    :type shot_name: str

    :return: new delay structure
    :rtype: np.ndarray
    """

    arg_checker.check_idx_list(cl_args.offset_by_front[0], signals_data.cnt_curves - 1,
                               "--offset-by-curve-front")

    front_plot_name = file_handler.get_front_plot_name(cl_args.offset_by_front,
                                                       cl_args.save_to, shot_name)
    front_point = get_front_point(signals_data, cl_args.offset_by_front,
                                  cl_args.multiplier, cl_args.delay,
                                  front_plot_name,
                                  interactive=cl_args.it_offset)
    # update delays
    new_delay = cl_args.delay[:]
    if front_point is not None:
        for idx in range(0, cl_args.delay.shape[0]):
            new_delay[idx, 0] += front_point.time
        return new_delay
    return None


def zero_curves(signals, curve_indexes, verbose=False):
    """Resets the y values to 0, for all curves with index in
    curve_indexes.

    :param signals: SignalsData instance with curves data
    :param curve_indexes: command line arguments, entered by the user
    :param verbose: shows more info during the process

    :type signals: SignalsData
    :type curve_indexes: list
    :type verbose: bool

    :return: changed SignalsData
    :rtype: SignalsData
    """
    arg_checker.check_idx_list(curve_indexes,
                               signals.cnt_curves - 1,
                               "--set-to-zero")
    if verbose:
        print("Resetting to zero the values of the curves "
              "with indexes: {}".format(curve_indexes))

    if curve_indexes[0] == -1:
        curve_indexes = list(range(0, signals.cnt_curves))
        # TODO: check that args did not change
    for idx in curve_indexes:
        signals.data[idx, 1:, :].fill(0)
    return data


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

    # data manipulation args check
    options = arg_checker.data_corr_arg_check(options)

    # save data args check
    options = arg_checker.save_arg_check(options)

    # convert_only arg check
    options = arg_checker.convert_only_arg_check(options)

    # other args
    options = arg_checker.check_utility_args(options)

    return options


# ============================================================================
# --------------   MAIN    ----------------------------------------
# ======================================================
if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()
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
    if args.hide_all:
        # by default backend == Qt5Agg
        # savefig() time for Qt5Agg == 0.926 s
        #                for Agg == 0.561 s
        # for single curve with 10000 points and one peak
        # run on Intel Core i5-4460 (average for 100 runs)
        # measured by cProfile
        matplotlib.use("Agg")
    else:
        matplotlib.use("Qt5Agg")

    if args.convert_only:
        for shot_idx, file_list in enumerate(args.gr_files):
            shot_name = file_list[0]

            # get SignalsData
            data = file_handler.read_signals(file_list, start=args.partial[0],
                                             step=args.partial[1], points=args.partial[2],
                                             labels=args.labels, units=args.units,
                                             time_units=args.time_units)

            # save data
            if args.save:
                saved_as = file_handler.do_save(data, args, shot_name,
                                                save_as=args.out_names[shot_idx],
                                                verbose=verbose,
                                                separate_files=args.separate_save)
                labels = [data.label(cr) for cr in data.idx_to_label.keys()]
                file_handler.save_m_log(file_list, saved_as, labels, args.multiplier,
                                        args.delay, args.offset_by_front,
                                        args.y_auto_zero, args.partial)
    elif (args.save or
            args.plot or
            args.multiplot or
            args.multicurve or
            args.offset_by_front):

        num_mask = file_handler.numbering_parser([files[0] for
                                                 files in args.gr_files])

        # print("Groups: {}".format(args.gr_files))

        for shot_idx, file_list in enumerate(args.gr_files):
            shot_name = file_handler.get_shot_number_str(file_list[0], num_mask,
                                                         args.ext_list)

            # get SignalsData
            data = file_handler.read_signals(file_list, start=args.partial[0],
                                             step=args.partial[1], points=args.partial[2],
                                             labels=args.labels, units=args.units,
                                             time_units=args.time_units)

            # checks the number of columns with data,
            # as well as the number of multipliers, delays, labels
            arg_checker.check_coeffs_number(data.cnt_curves * 2, ["multiplier", "delay"],
                                            args.multiplier, args.delay)
            arg_checker.check_coeffs_number(data.cnt_curves, ["label", "unit"],
                                            args.labels, args.units)

            # check y_zero_offset parameters (if idx is out of range)
            if args.y_auto_zero is not None:
                args = do_y_zero_offset(data, args)

            # reset to zero
            if args.zero is not None:
                data = zero_curves(data, args.zero, verbose)

            # check offset_by_voltage parameters (if idx is out of range)
            new_delay = None
            if args.offset_by_front is not None:
                new_delay = do_offset_by_front(data, args, shot_name)

            # mu
            # ltiplier and delay
            if new_delay is None:
                data = multiplier_and_delay(data, args.multiplier, args.delay)
            else:
                data = multiplier_and_delay(data, args.multiplier, new_delay)

            # plot preview and save
            if args.plot is not None:
                plotter.do_plots(data, args, shot_name, verbose=verbose, hide=args.p_hide)

            # plot and save multi-plots
            if args.multiplot is not None:
                plotter.do_multiplots(data, args, shot_name, verbose=verbose)

            # plot and save multicurve plots
            if args.multicurve is not None:
                plotter.do_multicurve_plots(data, args, shot_name, verbose=verbose)

            # ========================================================================
            # ------   GET FRONT DELAY   ---------------------------------------------
            # ========================================================================
            # peaks = [None] * data.cnt_curves
            # front_points = print_front_delay(data.get_single_curve(2), -0.2, "fall",
            #                                  data.get_single_curve(1), 1.5, "rise",
            #                                  save=True, prefix="fronts_"+shot_name)
            # # front_points = [None] * 3
            # # pulse_front = print_pulse_duration(data.get_single_curve(1), 1.0, "rise",
            # #                                    save=True, prefix="pulse_")
            # # front_points[1] = pulse_front[0]
            # # pulse_front = print_pulse_duration(data.get_single_curve(2), 1.0, "rise",
            # #                                    save=True, prefix="pulse_")
            # # front_points[2] = pulse_front[0]
            #
            # peaks[2] = front_points[0]
            #
            # peaks[1] = front_points[1]
            #
            # if args.multiplot is not None:
            #     plotter.do_multiplots(data, args, shot_name,
            #                           peaks=peaks,
            #                           hide=False,
            #                           verbose=verbose)

            # ========================================================================

            # save data
            if args.save:
                # print("Saving as {}".format(args.out_names[shot_idx]))
                saved_as = file_handler.do_save(data, args, shot_name,
                                                save_as=args.out_names[shot_idx],
                                                verbose=verbose,
                                                separate_files=args.separate_save)
                # labels = [data.labels(cr) for cr in data.idx_to_label.keys()]

                file_handler.save_m_log(file_list, saved_as, data.get_labels(), args.multiplier,
                                        args.delay, args.offset_by_front,
                                        args.y_auto_zero, args.partial)

    # if verbose:
    #     arg_checker.print_duplicates(args.gr_files, 30)

    # except Exception as e:
    #     print()
    #     sys.exit(e)
    # ========================================================================
    # TODO: description
    # TODO: file names validity check
    # TODO: test plot and multiplot save
    # TODO: test refactored SignalProcess
    # TODO: add support for ISF files
    # TODO: --as-log-sequence file loading
    # TODO: refactor docstrings
