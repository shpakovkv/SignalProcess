# Python 3.6
"""
Maintainer: Shpakov Konstantin
Link: https://github.com/shpakovkv/SignalProcess
"""

import os
import sys
import argparse
import psutil
import signal

import numpy as np
from scipy.signal import savgol_filter
from matplotlib import pyplot as plt
import matplotlib

import data_types
import arg_parser
import arg_checker
import file_handler
import plotter
import itertools

from numba import vectorize, float64
from multiprocessing import Pool

from data_manipulation import *

from PeakProcess import find_nearest_idx, check_polarity, is_pos, find_curve_front, get_two_fronts_delay, print_pulse_duration
from analysis import do_correlate, do_correlate_part

from file_handler import check_files_for_duplicates

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
                 arg_parser.get_analysis_args_parser(),
                 arg_parser.get_plot_args_parser(),
                 arg_parser.get_output_args_parser(),
                 arg_parser.get_utility_args_parser(),
                 arg_parser.get_front_args_parser()],
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


def get_y_zero_offset(curve, start_x, stop_x):
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

    amp_sum = np.nansum(curve.val[start_idx:stop_idx + 1])
    return amp_sum / (stop_idx - start_idx + 1)


def get_delays_with_y_zero_offsets(signals_data, multiplier, delay, curves_list, start_stop_tuples):
    """Calculates zero level correction for specified curves.
    Returns delays structure for all columns in SignalsData.
    Delays for Y-columns of specified curves will be filled with
    the Y zero level offset values.
    For all other Y columns and for all X columns delay will be 0.

    Use it for zero level correction before PeakProcess.

    :param signals_data: SignalsData structure with all curve data
    :type signals_data: SignalsData
    :param multiplier: np.ndarray(ndim=2) with multiplier[curve, axis] values
                       for X-columns (multiplier[:, 0])
                       and for Y-columns (multiplier[:, 1]) of signals_data array
    :type multiplier: np.ndarray
    :param delay: np.ndarray(ndim=2) with delay[curve, axis] values
                  for X-columns (delay[:, 0])
                  and for Y-columns (delay[:, 1]) of signals_data array
    :type delay: np.ndarray
    :param curves_list: zero-based indices of curves for which
                        you want to find the zero level offset
    :type curves_list: list
    :param start_stop_tuples: list of (bg_start_x, bg_stop_x) tuples for each
                              curves in curves list.
                              You can specify one tuple or list, and
                              it will be applied to all the curves.
    :type start_stop_tuples: list of tuple
    :return: new delays structure (ndarray(ndim=2))
    :rtype: np.ndarray
    """

    assert len(curves_list) == len(start_stop_tuples), \
        "Error! The number of (start_x, stop_x) tuples ({}) " \
        "does not match the number of the specified curves " \
        "({}).".format(len(start_stop_tuples), len(curves_list))

    delays_shape = (signals_data.cnt_curves, 2)
    delays = np.zeros(shape=delays_shape, dtype=np.float64)
    for tuple_idx, curve_idx in enumerate(curves_list):
        # get curve copy
        curve = signals_data.get_single_curve(curve_idx)
        # get SignalsData instance with only selected curve
        signals_data_single = SignalsData(curve.data.copy(), [curve.label], [curve.units], curve.t_units)
        signals_data_single.data = multiplier_and_delay(signals_data_single.data,
                                                        # need multiplier and delay ndarray(ndim=2)
                                                        # for single curve
                                                        multiplier[curve_idx:curve_idx + 1],
                                                        delay[curve_idx:curve_idx + 1])
        delays[curve_idx, 1] = get_y_zero_offset(signals_data_single.get_single_curve(0),
                                                 *start_stop_tuples[tuple_idx])
        # print(f"New delay for curve #{curve_idx} = {delays[curve_idx, 1]}")

    return delays


def update_delays_by_zero_level_offset(data, args):
    """Calculates zero level correction for specified curves.
    Returns modified delays structure for all columns in SignalsData.
    Delays for Y-columns of specified curves will be filled with
    the Y zero level offset values.
    For all other Y columns and for all X columns delay will not be modified.

    Use it for zero level correction before PeakProcess.

    :param data: SignalsData structure with all curve data
    :type data: SignalsData
    :param args: namespace with args
    :type args: argparse.Namespace
    :return: modified delays structure
    :rtype: np.ndarray
    """
    curves_list = [val[0] for val in args.y_auto_zero]
    start_stop_tuples = [(val[1], val[2]) for val in args.y_auto_zero]

    # data has ndim=3 with this structure: SignalsData.data[curve, axis, point]
    # ndarray_copy = data.data[curves_list, :, :].copy()
    # ndarray_copy = multiplier_and_delay(ndarray_copy, args.multiplier[curves_list, :], args.delay[curves_list, :])
    # data_copy = SignalsData(ndarray_copy, data.labels[curves_list, :], data.units[curves_list, :], data.time_units)

    zero_level_offset_delays = get_delays_with_y_zero_offsets(data,
                                                              args.multiplier,
                                                              args.delay,
                                                              curves_list,
                                                              start_stop_tuples)
    args.delay = args.delay + zero_level_offset_delays
    return args.delay


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


def get_front_point(signals_data, args, multiplier, delay,
                    front_plot_name, search_bounds=None, plot_bounds=None, interactive=False):
    """Finds the most left front point where the curve
    amplitude is greater (lower - for negative curve) than
    the level (args[1]) value.
    To improve accuracy, the signal is smoothed by
    the savgol filter.

    Saves a curve plot with the point on the front.

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
    :param search_bounds: the left and right time limits to search for the curve front
    :param plot_bounds: save a curve plot with the point on the front with specified bounds
    :param interactive: turns on interactive mode of the smooth filter
                        parameters selection

    :type signals_data: SignalsData
    :type args: tuple or list
    :type multiplier: tuple or list
    :type delay: tuple or list
    :type front_plot_name: str
    :type search_bounds: list or None
    :type plot_bounds: tuple ot list
    :type interactive: bool

    :return: front point
    :rtype: SinglePeak
    """
    # TODO: fix for new SignalsData class
    if not os.path.isdir(os.path.dirname(front_plot_name)):
        try:
            os.makedirs(os.path.dirname(front_plot_name))
        except FileExistsError as e:
            assert os.path.isdir(os.path.dirname(front_plot_name)), e.strerror

    curve_idx = args[0]
    level = args[1]
    window = args[2]
    poly_order = args[3]

    # curve = data_types.SingleCurve(signals_data.get_single_curve(curve_idx))
    # curve = signals_data.get_single_curve(curve_idx).copy()

    # make local copy of target curve
    curve = SignalsData(np.copy(signals_data.get_curve_2d_arr(curve_idx)),
                        labels=[signals_data.get_curve_label(curve_idx)],
                        units=[signals_data.get_curve_units(curve_idx)],
                        time_units=signals_data.time_units)

    # polarity = check_polarity(curve.get_single_curve(0))
    front = "fall" if level < 0 else "rise"

    # TODO: user enter front type

    # if is_pos(polarity):
    #     level = abs(level)
    # else:
    #     level = -abs(level)

    cur_mult = None
    if multiplier is not None:
        cur_mult = multiplier[curve_idx: curve_idx + 1]

    cur_del = None
    if delay is not None:
        cur_del = delay[curve_idx: curve_idx + 1]

    # apply multiplier and delay to local copy
    curve.data = multiplier_and_delay(curve.data, cur_mult, cur_del)
    # get x and y columns
    data_x = curve.get_x(0)
    data_y = curve.get_y(0)

    # handling empty array and nan only array
    if data_y.size - np.isnan(data_y).sum() == 0:
        return None

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

    if search_bounds is not None:
        if search_bounds[0] is not None and search_bounds[1] is not None:
            left = search_bounds[0]
            right = search_bounds[1]
            assert len(search_bounds) == 2, \
                (f"search_bounds list must consist of exactly 2 elements, "
                 f"got {len(search_bounds)} instead.")
            assert left < right, \
                (f"The left edge of the search front ({left}) should be "
                 f"smaller than the right ({right})")
        else:
            search_bounds = None

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
                                            level,
                                            front,
                                            bounds=search_bounds,
                                            interpolate=True)

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

    peaks = [None, None]
    peaks[0] = [front_point]
    """peaks is the list of single curve peaks (list)
    peaks[curve_idx][peak_idx] == SinglePeak instance
    
    if curve has no peaks then peaks[curve] == None
    """
    # sent to plot only two curves
    plotter.plot_multiple_curve(curve,
                                curve_list=[0, 1],
                                peaks=peaks,
                                xlim=plot_bounds,
                                title=plot_title)

    plt.savefig(front_plot_name, dpi=400)
    # show plot to avoid tkinter warning "can't invoke "event" command..."
    # plt.show(block=False)
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
                                  search_bounds=cl_args.off_front_bounds,
                                  plot_bounds=cl_args.t_bounds,
                                  interactive=cl_args.it_offset)
    # update delays
    new_delay = None
    if cl_args.delay is None:
        new_delay = np.zeros(shape=(signals_data.cnt_curves, 2))
    else:
        new_delay = np.copy(cl_args.delay)

    if front_point is not None:
        for idx in range(0, new_delay.shape[0]):
            new_delay[idx, 0] += front_point.time
        return new_delay
    return None


def do_smooth_curves_and_add(signals, param_list, shot_name):
    """

    The param_list should be in the form of a list of dictionaries:

    [
      {
      'idx': <idx_value>,
      'window': <window_value>,
      'order': <order_value>,
      'label': <label_value>
      },
      etc.
    ]

    :param signals: signals data
    :type signals: SignalsData
    :param param_list: list of dict with smooth parameters
    :type param_list: list of dict
    :param shot_name: the name (usually number) of shot
    :type shot_name: str
    :return:
    :rtype:
    """
    if param_list is None:
        return

    assert all(val["label"] not in signals.labels for val in param_list), \
        f"The label for the smoothed curve already exists."

    for param_dict in param_list:
        cur_idx = param_dict['idx']
        smoothed = smooth_curve_2d_arr(signals.get_curve_2d_arr(cur_idx), param_dict)
        signals.add_from_array(smoothed,
                               labels=[param_dict["label"]],
                               units=[signals.get_curve_label(cur_idx)])



def smooth_curve_2d_arr(data, params):
    """Smooth values along y-axis of 2-dim ndarray.
    Expected [axis][points] data structure.

    X: axis == 0,
    Y: axis == 1.

    The 'params' dictionary should have the structure given below:

    {
    'idx': <int_value>,
    'window': <int_value>,
    'order': <int_value>,
    'label': <str_value>
    }


    :param data: curve data
    :type data: np.ndarray
    :param params: smooth parameters: window size, order of the polynomial, etc.
    :type params: dict
    :return: data curve with smoothed Y values
    :rtype: np.ndarray
    """
    axis_x = 0
    axis_y = 1

    assert isinstance(data, np.ndarray), f"Wrong type. Expected 'numpy.ndarray', got {type(data)}."
    assert data.ndim == 2, f"Incorrect number of input data dimensions. Expected {2}, got {data.ndim}."

    data_y_smooth = smooth_voltage(data[axis_y], params["window"], params["order"])
    smooth_data = np.stack((data[axis_x], data_y_smooth))
    return smooth_data



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
    return signals


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

    # other args check
    options = arg_checker.check_utility_args(options)

    # analysis args check
    options = arg_checker.check_analysis_args(options)

    return options


def convert_only(args, shot_idx, num_mask):
    """ The main script of the file conversion process.

    :param args: command line arguments, entered by the user
    :type args: argparse.Namespace
    :param shot_idx: zero-based index of the shot (set of data files)
    :type shot_idx: int
    :param num_mask: first and last index of the real shot number in file[0] name
    :type num_mask: tuple
    :return: None
    :rtype: None
    """
    file_list = args.gr_files[shot_idx]
    # shot_name = file_list[0]
    shot_name = file_handler.get_shot_number_str(file_list[0], num_mask,
                                                 args.ext_list)

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


def full_process(args, shot_idx, num_mask):
    """ Signal Processing Main Script.

    :param args: command line arguments, entered by the user
    :type args: argparse.Namespace
    :param shot_idx: zero-based index of the shot (set of data files)
    :type shot_idx: int
    :param num_mask: first and last index of the real shot number in file[0] name
    :type num_mask: tuple
    :return: None
    :rtype: None
    """
    file_list = args.gr_files[shot_idx]
    shot_name = file_handler.get_shot_number_str(file_list[0], num_mask,
                                                 args.ext_list)

    # get SignalsData
    data = file_handler.read_signals(file_list, start=args.partial[0],
                                     step=args.partial[1], points=args.partial[2],
                                     labels=args.labels, units=args.units,
                                     time_units=args.time_units)

    # checks the number of columns with data,
    # as well as the number of multipliers, delays, labels
    args.multiplier = arg_checker.check_multiplier(args.multiplier,
                                                   curves_count=data.cnt_curves)
    args.delay = arg_checker.check_delay(args.delay,
                                         curves_count=data.cnt_curves)

    # multiplier and delay are 2D ndarrays [curve][axis]
    arg_checker.check_coeffs_number(data.cnt_curves * 2, ["multiplier", "delay"],
                                    args.multiplier, args.delay)
    arg_checker.check_coeffs_number(data.cnt_curves, ["label", "unit"],
                                    args.labels, args.units)

    # reset to zero
    if args.zero is not None:
        data = zero_curves(data, args.zero, verbose)

    if args.y_auto_zero:
        args.delay = update_delays_by_zero_level_offset(data, args)

    # check offset_by_voltage parameters (if idx is out of range)
    new_delay = None
    if args.offset_by_front is not None:
        new_delay = do_offset_by_front(data, args, shot_name)

    # multiplier and delay
    if new_delay is None:
        data = multiplier_and_delay(data, args.multiplier, args.delay)
    else:
        data = multiplier_and_delay(data, args.multiplier, new_delay)

    if args.smooth is not None:
        do_smooth_curves_and_add(data, args.smooth, shot_name)

    correlate_data = list()
    if args.correlate is not None:
        correlate_data.extend(do_correlate(data, args))

    correlate_part_data = list()
    if args.correlate_part is not None:
        correlate_part_data.extend(do_correlate_part(data, args))

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
    # # front_points = get_two_fronts_delay(data.get_single_curve(1), 1.0, "rise",
    # #                                     data.get_single_curve(2), 1.0, "rise",
    # #                                     save=True, prefix="fronts_"+shot_name)
    # # peaks[6] = front_points[0]
    # # peaks[4] = front_points[1]
    # #
    # # front_points = get_two_fronts_delay(data.get_single_curve(1), -6.0, "fall",
    # #                                     data.get_single_curve(4), -6.0, "fall",
    # #                                     save=True, prefix="fronts_" + shot_name)
    # # peaks[1] = front_points[0]
    #
    # # front_points = [None] * 3
    # pulse_front = print_pulse_duration(data.get_single_curve(1), 1.0, "rise",
    #                                    save=True, prefix="pulse_")
    # peaks[1] = pulse_front[0]
    # pulse_front = print_pulse_duration(data.get_single_curve(2), 0.4, "rise",
    #                                    save=True, prefix="pulse_")
    # peaks[2] = pulse_front[0]
    #
    # # if args.multiplot is not None:
    # #     plotter.do_multiplots(data, args, shot_name,
    # #                           peaks=peaks,
    # #                           verbose=verbose)
    #
    # print("DELAY between CamExp left and laser right = {:.3f}".format(peaks[1][0].time - peaks[2][1].time))
    # print("DELAY between CamExp right and laser left = {:.3f}".format(peaks[2][0].time - peaks[1][1].time))
    #
    # if args.multicurve is not None:
    #     plotter.do_multicurve_plots(data, args, shot_name,
    #                                 peaks=peaks,
    #                                 verbose=verbose)

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

    if args.correlate_dir is not None:
        if args.correlate is not None:
            for idx in range(len(args.correlate)):
                name = "{:04d}_correlate_{}to{}.csv" \
                       "".format(shot_idx, args.correlate[idx][0], args.correlate[idx][1])
                save_as = os.path.join(args.correlate_dir, name)
                saved_as = file_handler.do_save(correlate_data[idx], args, name,
                                                save_as=save_as,
                                                verbose=verbose)
        if args.correlate_part is not None:
            for idx in range(len(args.correlate_part)):
                name = plotter.get_correlate_part_plot_name(shot_idx, args.correlate_part[idx])
                name += ".csv"
                save_as = os.path.join(args.correlate_dir, name)
                saved_as = file_handler.do_save(correlate_part_data[idx], args, name,
                                                save_as=save_as,
                                                verbose=verbose)

    if args.correlate_plot_dir is not None:
        plotter.do_plot_correlate_all(data, args, shot_idx, correlate_data, correlate_part_data)


# ============================================================================
# --------------   MAIN    ----------------------------------------
# ======================================================


def main():
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

    num_mask = file_handler.numbering_parser([files[0] for
                                              files in args.gr_files])
    # MAIN LOOP
    if args.convert_only:
        with Pool(args.threads) as p:
            p.starmap(convert_only, [(args, shot_idx, num_mask) for shot_idx in range(len(args.gr_files))])

    elif (args.save or
          args.plot or
          args.multiplot or
          args.multicurve or
          args.offset_by_front or
          args.zero or
          args.correlate or
          args.correlate_part):

        with Pool(args.threads) as p:
            p.starmap(full_process, [(args, shot_idx, num_mask) for shot_idx in range(len(args.gr_files))])

    check_files_for_duplicates(args)

    # if verbose:
    #     arg_checker.print_duplicates(args.gr_files, 30)

    # except Exception as e:
    #     print()
    #     sys.exit(e)
    # ========================================================================


if __name__ == "__main__":
    main()
    # TODO: description
    # TODO: file names validity check
    # TODO: test plot and multiplot save
    # TODO: test refactored SignalProcess
    # TODO: add support for ISF files
    # TODO: --as-log-sequence file loading
    # TODO: refactor docstrings
