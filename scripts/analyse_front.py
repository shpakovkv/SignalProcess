import os
import sys
import time

import bisect
import scipy.integrate as integrate

from matplotlib import pyplot as plt
import numpy as np
from numba import njit
from scipy.signal import correlate as scipy_correlate
from scipy import interpolate
from scipy.signal import savgol_filter

from data_types import SignalsData
from data_types import SinglePeak
from data_types import SingleCurve

from file_handler import get_real_num_bounds_1d
from file_handler import get_front_plot_name

from arg_checker import check_idx_list

from plotter import plot_multiple_curve

from data_manipulation import multiplier_and_delay

from analyse_peak import find_curve_front


pos_polarity_labels = {'pos', 'positive', '+'}
neg_polarity_labels = {'neg', 'negative', '-'}


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
    :type multiplier: np.ndarray
    :type delay: np.ndarray
    :type front_plot_name: str
    :type search_bounds: list or None
    :type plot_bounds: tuple ot list
    :type interactive: bool

    :return: front point
    :rtype: SinglePeak
    """
    # TODO: fix for new SignalsData class
    if front_plot_name is not None:
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

    first_num, last_num = get_real_num_bounds_1d(data_y)

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

        smoothed_curve = SingleCurve(smooth_data,
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
            front_point = SinglePeak(front_x, front_y, 0)
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
            plot_multiple_curve(curve,
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
    plot_multiple_curve(curve,
                        curve_list=[0, 1],
                        peaks=peaks,
                        xlim=plot_bounds,
                        title=plot_title)

    if front_plot_name is not None:
        plt.savefig(front_plot_name, dpi=400)
    # show plot to avoid tkinter warning "can't invoke "event" command..."
    # plt.show(block=False)
    plt.close('all')

    return front_point


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


def do_offset_by_front(signals_data, cl_args, shot_name, plot=True):
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
    :param plot: save front point graph flag

    :type signals_data: SignalsData
    :type cl_args: argparse.Namespace
    :type shot_name: str
    :type plot: bool

    :return: new delay structure
    :rtype: np.ndarray
    """

    check_idx_list(cl_args.offset_by_front[0], signals_data.cnt_curves - 1,
                   "--offset-by-curve-front")

    front_plot_name = None
    if plot:
        front_plot_name = get_front_plot_name(cl_args.offset_by_front,
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
