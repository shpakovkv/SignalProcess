from __future__ import print_function

import numpy as np
import os
from matplotlib import pyplot as plt

import SignalProcess as sp
import PeakProcess as pp


# def align_by_voltage_front_old(data,                # an instance of SignalsData class
#                            voltage_col,         # index or label of voltage curve in SignalsData
#                            level=-0.2,          # level of voltage front to be set as time zero
#                            is_positive=False):  # positive polarity of curve or not
#     # INPUT CHECK
#     if not isinstance(data, sp.SignalsData):
#         raise TypeError("Data must be an instance of SignalsData class")
#     if isinstance(voltage_col, str):
#         curve_idx = data.labels[voltage_col]
#     else:
#         curve_idx = voltage_col
#
#     # VOLTAGE FRONT TIME POS
#     smoothed_voltage = sp.smooth_voltage(data.curves[curve_idx].get_x(), data.curves[curve_idx].get_y())
#     front_x, front_y = pp.find_voltage_front(data.curves[curve_idx].get_x(), smoothed_voltage, level, is_positive)
#
#     print "Voltage front time = " + str(front_x) + " ns"
#
#     # TIMELINE CORRECTION
#     if front_x:
#         delay = [front_x if idx % 2 == 0 else 0 for idx in range(data.count * 2)]
#         data = sp.multiplier_and_delay(data, None, delay)
#     return data

ext_of_type = {"CSV": ".csv", "DPO7054": ".wfm", "TDS2024C": ".csv",
               "HMO3004": ".csv", "LECROY": ".txt"}


def add_to_log(m="", end='\n'):
    global log
    log += m + end
    print(m, end=end)


def read_signals(file_list, file_type='csv', delimiter=","):
    '''
    Function returns SignalsData object filled with
    data from files in file_list.
    Do not forget to sort the list of files
    for the correct order of curves.

    file_list -- list of full paths or 1 path (str)
    type -- type of files in file_list:
            CSV, DPO7054, HMO3004, TDS2024C, LeCroy

    returns -- SignalsData object
    '''

    # check inputs
    if isinstance(file_list, str):
        file_list = [file_list]
    elif not isinstance(file_list, list):
        raise TypeError("file_list must be an instance of str or list of str")
    for filename in file_list:
        if not isinstance(filename, str):
            raise TypeError("A non-str element is found in the file_list")
        if not os.path.isfile(filename):
            raise Exception("File \"{}\" not found.".format(filename))

    # read
    if file_type.upper() == "CSV":
        data = sp.SignalsData()
        for filename in file_list:
            data.append(np.genfromtxt(filename, delimiter=delimiter))
        return data
    else:
        raise TypeError("Unknown type of file \"{}\"".format(type))


def zero_single_curve(curve, lenght=30):
    curve.data = curve.data[0:lenght + 1, :]
    for row in range(lenght):
        curve.data[row, 1] = 0
    return curve


def zero_and_save(filename, file_type="CSV", curves=None, length=30):
    # read
    file_type = file_type.upper()
    # print("Reading " + filename)
    add_to_log("Reading " + filename)
    data = read_signals(filename, file_type)

    # zero
    if curves is None:
        curves = range(data.count)
    # print("Set to zero curves: ", end="")
    add_to_log("Set to zero curves: ", end="")
    for idx in curves:
        # print("{}  ".format(idx), end="")
        add_to_log("{}  ".format(idx), end="")
        data.curves[idx] = zero_single_curve(data.curves[idx], length)
    # print()
    add_to_log("")

    # save
    save_as = filename
    # if isinstance(filename, list):
    #     save_as = filename[0]
    # save_as = save_as[0:-4] + "_zero.csv"
    # print("Saveing " + save_as)
    add_to_log("Saveing " + save_as)
    sp.save_ndarray_csv(save_as, data.get_array())
    # print("Done!\n")
    add_to_log("Done!\n")


def get_voltage_front(voltage, level,
                      time_multiplier=1, polarity='auto',
                      save_plot=False, plot_name="voltage_plot.png"):
    '''
    This function finds the time point of voltage front on level
    'level', recalculates input delays of all time columns (odd 
    elements of the input delay list) to make that time point be zero
    and correspondingly offset other curves
    
    NOTE: the voltage curve is not multiplied yet, so the 'level' 
    value must be in raw voltage data units
    
    voltage -- an instance of SingleCurve
    level -- level of voltage front
    delay -- list of floats (delays) for each column in SignalsData
    time_multiplier -- multiplier of time column of voltage curve
        default 1 (for time points in seconds) (1e9 for nanoseconds)
    is_positive -- set to True if voltage have positive polarity
        else False (default)
        
    return -- list of recalculated delays (floats)
    '''

    if not isinstance(voltage, sp.SingleCurve):
        raise TypeError("voltage_curve must be an instance "
                        "of SingleCurve class")
    # smooth voltage curve to improve accuracy of front search
    smoothed_voltage = sp.smooth_voltage(voltage.get_x(),
                                         voltage.get_y(),
                                         time_multiplier)
    smoothed_voltage_curve = sp.SingleCurve(voltage.time, smoothed_voltage)
    front_x, front_y = pp.find_voltage_front(smoothed_voltage_curve,
                                             level, polarity,
                                             save_plot=save_plot,
                                             plot_name=plot_name)
    # plt.plot(voltage.get_x(), voltage.get_y(), '-b')
    # plt.plot(voltage.get_x(), smoothed_voltage, '-r')
    # plt.plot([front_x], [front_y], '*g')
    # plt.show()
    return front_x


def get_offset_by_front(curve, level, polarity, time_multiplier=1, time_delay=0,
                        save=False, save_name='voltage_plot.png'):
    level_raw = ((voltage_front_level +
                  delays[volt_cr_idx * 2 + 1]) /
                 multipliers[volt_cr_idx * 2 + 1])
    if not isinstance(curve, sp.SingleCurve):
        raise TypeError("voltage_curve must be an instance "
                        "of SingleCurve class")
    # smooth voltage curve to improve accuracy of front search
    smoothed_voltage = sp.smooth_voltage(curve.get_x(),
                                         curve.get_y(),
                                         time_multiplier)
    smoothed_voltage_curve = sp.SingleCurve(curve.time, smoothed_voltage)
    front_x, front_y = pp.find_voltage_front(smoothed_voltage_curve,
                                             level, polarity,
                                             save_plot=save,
                                             plot_name=save_name)
    time_offset = (front_raw * time_multiplier - time_delay)
    return time_offset


def make_final():
    global log
    # GET FOLDERs PATHs
    path_dict = dict()
    # path_list = list()
    osc_list = ["DPO7054", "HMO3004", "TDS2024C", "LeCroy"]

    # path_dict["DPO7054"] = ("/media/shpakovkv/6ADA8899DA886365/"
    #                         "WORK/2017/2017 05 12-19 ERG/"
    #                         "2017 05 13 ERG Input united/"
    #                         "2017 05 13 DPO7054")
    # path_dict["HMO3004"] = ("/media/shpakovkv/6ADA8899DA886365/"
    #                         "WORK/2017/2017 05 12-19 ERG/"
    #                         "2017 05 13 ERG Input united/"
    #                         "2017 05 13 HMO3004")
    #
    # path_dict["TDS2024C"] = ("/media/shpakovkv/6ADA8899DA886365/"
    #                          "WORK/2017/2017 05 12-19 ERG/"
    #                          "2017 05 13 ERG Input united/"
    #                          "2017 05 13 TDS2024C")
    #
    # path_dict["LeCroy"] = ("/media/shpakovkv/6ADA8899DA886365/"
    #                        "WORK/2017/2017 05 12-19 ERG/"
    #                        "2017 05 13 ERG Input united/"
    #                        "2017 05 13 LeCroy")
    #
    # save_to_folder = ("/media/shpakovkv/6ADA8899DA886365/WORK/2017/"
    #                   "2017 05 12-19 ERG/2017 05 13 ERG Output FINAL")

    path_dict["DPO7054"] = ("H:\\WORK\\ERG\\2017\\2017 05 12-19 ERG\\"
                            "2017 05 13-19 ERG Output final\\"
                            "2017 05 18 ERG Input final\\"
                            "2017 05 18 DPO7054")
    path_dict["HMO3004"] = ("H:\\WORK\\ERG\\2017\\2017 05 12-19 ERG\\"
                            "2017 05 13-19 ERG Output final\\"
                            "2017 05 18 ERG Input final\\"
                            "2017 05 18 HMO3004")

    path_dict["TDS2024C"] = ("H:\\WORK\\ERG\\2017\\2017 05 12-19 ERG\\"
                             "2017 05 13-19 ERG Output final\\"
                             "2017 05 18 ERG Input final\\"
                             "2017 05 18 TDS2024C")

    path_dict["LeCroy"] = ("H:\\WORK\\ERG\\2017\\2017 05 12-19 ERG\\"
                           "2017 05 13-19 ERG Output final\\"
                           "2017 05 18 ERG Input final\\"
                           "2017 05 18 LeCroy")

    save_to_folder = ("H:\\WORK\\ERG\\2017\\2017 05 12-19 ERG\\"
                      "2017 05 18 ERG Output FINAL")


    save_log_to = os.path.join(save_to_folder, "SignalProcess.log")

    file_dict = dict()
    multiplier_dict = dict()
    delay_dict = dict()
    cols_label = dict()

    # GET MULTIPLIERs & DELAYs
    multiplier_dict["DPO7054"] = [1e9, 1.216, 1e9, 0.991,
                                  1e9, 1, 1e9, 0.994]
    delay_dict["DPO7054"] = [134.7, 0, 117.7, 0,
                             121.7, 0, 131.7, 0]
    cols_label["DPO7054"] = ["Time1", "PMTD1", "Time2", "PMTD2",
                             "Time3", "PMTD3", "Time4", "PMTD4"]

    multiplier_dict["HMO3004"] = [1e9, 0.272, 1e9, 0.951,
                                  1e9, 1.138, 1e9, 1.592]
    # delay_dict["HMO3004"] = [87, 0, 124, 0,   # 14.05 shoot #9 and below
    #                          142, 0, 124, 0]
    delay_dict["HMO3004"] = [87, 0, 144, 0,
                             142, 0, 124, 0]

    multiplier_dict["TDS2024C"] = [1e9, 1.261, 1e9, 1.094,
                                   1e9, 1.222, 1e9, 1.59]
    delay_dict["TDS2024C"] = [126.5, 0, 111.5, 0,
                              104.5, 0, 129.5, 0]

    multiplier_dict["LeCroy"] = [1e9, 0.697, 1e9, 4.554,
                                 1e9, 0.25]
    delay_dict["LeCroy"] = [80.3, 0, 367.1, 0, 0, 0]

    all_multipliers = [multiplier_dict[osc][i]
                      for osc in osc_list
                      for i in range(len(multiplier_dict[osc]))]
    all_delays = [delay_dict[osc][i]
                 for osc in osc_list
                 for i in range(len(delay_dict[osc]))]

    # no_voltage = {1, 64, 76, 79}    # 2017 05 13
    # no_voltage = {75, 87, 88, 89, 90, 91}  # 2017 05 14
    # no_voltage = {8, 30, 70, 77, 80, 92}  # 2017 05 15
    no_voltage = {19, 23}  # 2017 05 18
    # no_voltage = {0}  # 2017 05 19
    voltage_idx = 12  # zero-based index of voltage curve
    voltage_front_level = -0.2
    duplicate_check = True

    # GET FILE GROUPS
    for key in path_dict:
        file_dict[key] = sp.get_file_list_by_ext(path_dict[key],
                                                 ".CSV", sort=True)

    shoots_count = len(file_dict["DPO7054"])
    for key in path_dict:
        if shoots_count != len(file_dict[key]):
            raise IndexError("The number of .csv files in "
                             "osc-directories must be the same.")
    for number in range(shoots_count):  # zero-based index of shoot
        # number = 80
        data = sp.SignalsData()

        file_name = "ERG_" + sp.get_name_from_group(
            [file_dict["DPO7054"][number]], 4, 3)
        # READ CSV
        for osc in osc_list:  # sorted order
            add_to_log("Reading " + file_dict[osc][number])
            temp = np.genfromtxt(file_dict[osc][number], delimiter=",")
            filename_current = os.path.basename(file_dict[osc][number - 1])
            filename_before = os.path.basename(file_dict[osc][number])
            if number > 0 and duplicate_check:
                if sp.compare_2_files(file_dict[osc][number - 1],
                                      file_dict[osc][number]):
                    add_to_log("DUPLICATE files: "
                               "\"{}\" and \"{}\"".format(
                                    filename_before, filename_current))
            data.append(temp)

        # if is_positive:
        #     add_to_log("Polarity: positive")
        # else:
        #     add_to_log("Polarity: negative")

        # FIND VOLTAGE FRONT
        local_delay = None
        if number not in no_voltage:
            add_to_log("Polarity:  ", end="")
            polarity = pp.check_polarity(data.curves[voltage_idx])
            add_to_log(str(polarity) + "  (autodetect)")
            volt_plot_name = file_name + "_voltage.png"
            volt_plot_name = os.path.join(save_to_folder,
                                          "voltage_front", volt_plot_name)
            if pp.is_pos(polarity):
                voltage_front_level = abs(voltage_front_level)
            else:
                voltage_front_level = -abs(voltage_front_level)

            add_to_log("Searching voltage front at level = " +
                       str(voltage_front_level))

            level_raw = ((voltage_front_level +
                          all_delays[voltage_idx * 2 + 1]) /
                         all_multipliers[voltage_idx * 2 + 1])
            front_raw = get_voltage_front(data.curves[voltage_idx], level_raw,
                                          save_plot=True, polarity=polarity,
                                          plot_name=volt_plot_name)

            # CORR DELAYs
            if front_raw:
                add_to_log("Raw_voltage_front_level = " + str(level_raw))
                # apply multiplier and compensate voltage delay
                time_offset = (front_raw * all_multipliers[voltage_idx * 2] -
                               all_delays[voltage_idx * 2])
                add_to_log("Time_offset_by_voltage = " +
                           str(time_offset))
                # recalculate delays for time columns
                local_delay = [all_delays[idx] + time_offset
                               if idx % 2 == 0 else all_delays[idx]
                               for idx in range(len(all_delays))]
        else:
            front_raw = None
        if not front_raw:
            add_to_log("Time offset by voltage = 0"
                       "    (No voltage front detected)")
            local_delay = all_delays

        # CORR DATA
        data = sp.multiplier_and_delay(data, all_multipliers, local_delay)
        table = data.get_array()
        add_to_log("Curves_count = " + str(data.count) +
                   "     Columns_count = " + str(table.shape[1]) +
                   "     Rows_count = " + str(table.shape[0]))

        # print table[len(table) - 1,:]

        # for idx in range(0, data.count, 1):
        #     plt.plot(data.curves[idx].get_x(), data.curves[idx].get_y())
        #     print("Curve #0" + str(idx + 1))
        #     plt.show()

        # for idx in range(0, table.shape[1], 2):
        #     if int(idx // 2 + 1) == 12:
        #         plt.plot(table[:, idx], table[:, idx + 1])
        #         print("Curve #0" + str(idx // 2 + 1))
        #         plt.show()

        # SAVE FILE

        save_to = os.path.join(save_to_folder, file_name)
        file_ver = 1
        version = ""
        while os.path.isfile(save_to + version + ".csv"):
            file_ver += 1
            version = "_v" + str(file_ver)
        save_to += version
        add_to_log("Saving " + save_to + ".csv")
        sp.save_ndarray_csv(save_to, table)  # auto adds ".csv"
        add_to_log("Done!\n")

        # break

    print("Saving log file...")
    with open(save_log_to, 'a') as f:
        f.write(log)
    print("Done!")


def plot_single_curve(curve, peaks=None, xlim=None,
                      save=False, show=False, save_as=""):
    plt.close('all')
    if xlim is not None:
        plt.xlim(xlim)
    plt.plot(curve.time, curve.val, '-', color='#999999', linewidth=0.5)

    if peaks is not None:
        peak_x = [peak.time for peak in peaks if peak is not None]
        peak_y = [peak.val for peak in peaks if peak is not None]
        # plt.plot(peak_x, peak_y, 'or')
        plt.scatter(peak_x, peak_y, s=50, edgecolors='#ff7f0e', facecolors='none', linewidths=2)
        plt.scatter(peak_x, peak_y, s=80, edgecolors='#dd3328', facecolors='none', linewidths=2)
        # plt.plot(peak_x, peak_y, '*g')
    if save:
        # print("Saveing " + save_as)
        plt.savefig(save_as)
    if show:
        plt.show()
    plt.close('all')

def calc_ylim(time, y, time_bounds=None, reserve=0.1):
    if time_bounds is None:
        time_bounds = (None, None)
    if time_bounds[0] is None:
        time_bounds = (time[0], time_bounds[1])
    if time_bounds[1] is None:
        time_bounds = (time_bounds[0], time[-1])
    start = pp.find_nearest_idx(time, time_bounds[0], side='right')
    stop = pp.find_nearest_idx(time, time_bounds[1], side='left')
    y_max = np.amax(y[start:stop])
    y_min = np.amin(y[start:stop])
    y_range = y_max - y_min
    reserve *= y_range
    if y_max == 0 and y_min == 0:
        y_max = 1.4
        y_min = -1.4
    return y_min - reserve, y_max + reserve


def plot_peaks_all(data, peak_data, curves_list, xlim=None,
                   show=False, save=False, save_as=""):
    from cycler import cycler
    plt.close('all')
    fig, axes = plt.subplots(len(curves_list), 1, sharex='all')
    colors = ['#1f22dd', '#ff7f0e', '#9467bd', '#d62728', '#2ca02c',
              '#8c564b', '#17becf', '#bcbd22', '#e377c2']
    for wf in range(len(curves_list)):
        axes[wf].plot(data.time(curves_list[wf]),
                      data.value(curves_list[wf]), '-', color='#999999', linewidth=0.5)
        if xlim is not None:
            axes[wf].set_xlim(xlim)
            axes[wf].set_ylim(calc_ylim(data.time(curves_list[wf]),
                                        data.value(curves_list[wf]),
                                        xlim, reserve=0.1))
        color_idx = 0
        if peak_data is not None:
            for pk in peak_data[wf]:
                color = colors[color_idx]
                color_idx += 1
                if color_idx == len(colors):
                    color_idx = 0
                if pk is not None:
                    # axes[wf].plot([pk.time], [pk.val], '*',
                    #               color=color, markersize=5)
                    axes[wf].scatter([pk.time], [pk.val], s=20,
                                     edgecolors=color, facecolors='none',
                                     linewidths=1.5)
    # figManager = plt.get_current_fig_manager()
    # figManager.window.showMaximized()
    if save:
        # print("Saving plot " + save_as)
        plt.savefig(save_as, dpi=400)
        # print("Done!")
    if show:
        plt.show()
    plt.close('all')


def go_peak_process(data, curves_list, params, group_diff,
                    filename=None):
    # data_folder = ("/media/shpakovkv/6ADA8899DA886365/WORK/2017/"
    #           "2017 05 12-19 ERG/2017 05 15 ERG Output FINAL")
    # save_peaks_to = ("/media/shpakovkv/6ADA8899DA886365/WORK/2017/"
    #                  "2017 05 12-19 ERG/2017 05 13 ERG Peaks")

    # folder = ("H:\\WORK\\ERG\\2017\\2017 05 12-19 ERG\\"
    #           "2017 05 14 ERG Output FINAL")

    # folder = "H:\\WORK\\ERG\\2017\\2017 05 12-19 ERG\\TEMP\\"

    # GET PEAKS
    add_to_log("Curves count = " + str(data.count) + "\n")
    peaks = []
    for idx in curves_list:
        # sp.save_ndarray_csv("neg_peaks.csv", data.curves[idx].data)
        add_to_log("Curve #" + str(idx), end="    ")
        new_peaks, peak_log = pp.peak_finder(
            data.time(idx), data.value(idx), **params)
        peaks.append(new_peaks)
        add_to_log(peak_log, end="")

    # max_peaks = max([len(x) for x in peaks])
    # for pk in range(max_peaks):
    #     for wf in range(len(peaks)):
    #         try:
    #             s = str(peaks[wf][pk].time)
    #         except IndexError:
    #             s = "---------"
    #         print(s, end="\t")
    #     print()
    # print()

    # GROUP PEAKS
    peak_data, peak_map = pp.group_peaks(peaks, group_diff)
    # for gr in range(len(peak_map[0])):
    #     for wf in range(len(peak_map)):
    #         print(peak_map[wf][gr], end="\t")
    #     print()

    # GRAPH ALL PEAKS
    add_to_log("Saving peaks and plots...")
    peaks_filename = os.path.join(data_folder, "Peaks_all",
                                  os.path.basename(filename))
    save_peaks_csv(peaks_filename, peak_data)
    # plt.show()
    plot_filename = os.path.join(data_folder, "Peaks_all",
                                 os.path.basename(filename))
    if plot_filename.upper().endswith('.CSV'):
        plot_filename = plot_filename[:-4]
    plot_filename += ".plot.png"
    plot_peaks_all(data, peak_data, curves_list,
                   params.get("time_bounds", None),
                   save=True, save_as=plot_filename, show=True)
    add_to_log("Saving all peaks as " + plot_filename)
    for idx in range(len(curves_list)):
        curve_filename = os.path.join(data_folder, "Peaks_single")
        if not os.path.isdir(curve_filename):
            os.makedirs(curve_filename)
        curve_filename = os.path.join(curve_filename,
                                      os.path.basename(filename))
        if curve_filename.upper().endswith('.CSV'):
            curve_filename = curve_filename[:-4]
        curve_filename += "_curve" + str(curves_list[idx]) + ".png"
        plot_single_curve(data.curves[curves_list[idx]], peak_data[idx],
                          params.get("time_bounds", None),
                          save=True, save_as=curve_filename)

    # print("Saving log...")
    # log_filename = os.path.join(data_folder, "PeakProcess.log")
    # with open(log_filename, 'w') as f:
    #     f.write(log)
    # print("Done!")
    add_to_log("Done!\n")


def save_peaks_csv(filename, peaks):
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
                pk = pp.SinglePeak(0, 0, 0)
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


def test_peak_process(filename, curves_list, params, save=False):
    add_to_log("Reading " + filename)
    data = sp.SignalsData(np.genfromtxt(filename, delimiter=','))
    # data = np.genfromtxt(filename, delimiter=',')
    # data[:,1] = -data[:,1]
    # sp.save_ndarray_csv("pos_peaks.csv", data)

    # curves_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    peaks = []
    add_to_log("Finding peaks...")
    for idx in curves_list:
        add_to_log("Curve #" + str(idx), end="    ")
        new_peaks, peak_log = pp.peak_finder(
            data.time(idx), data.value(idx), **params)
        peaks.append(new_peaks)
        add_to_log(peak_log, end="")
    add_to_log("Grouping peaks...")
    peak_data, peak_map = pp.group_peaks(peaks, 25)

    plot_peaks_all(data, peak_data, curves_list,
                   params.get("time_bounds", None),
                   show=True)


def plot_one_curve_peaks(filename, idx, params):
    add_to_log("Reading " + filename)
    data = sp.SignalsData(np.genfromtxt(filename, delimiter=','))
    peaks = []
    add_to_log("Finding peaks...")
    add_to_log("Curve #" + str(idx), end="    ")
    new_peaks, peak_log = pp.peak_finder(
        data.time(idx), data.value(idx), **params)
    peaks.append(new_peaks)
    add_to_log(peak_log, end="")
    # add_to_log("Grouping peaks...")
    # peak_data, peak_map = pp.group_peaks(peaks, 25)
    plot_single_curve(data.curves[idx], peaks[0],
                      xlim=params.get("time_bounds", None), show=True)
    # plot_peaks_all(data, peak_data, curves_list, [-200, 750],
    #                show=True)


def save_curve_for_test(file_idx, curve_idx):
    folder = ("/media/shpakovkv/6ADA8899DA886365/WORK/2017/"
              "2017 05 12-19 ERG/2017 05 13 ERG Output final")
    file_list = sp.get_file_list_by_ext(folder, ".CSV", sort=True)
    filename = file_list[file_idx]
    print("Reading " + filename)
    data = sp.SignalsData(np.genfromtxt(filename, delimiter=','))
    save_as = "file{}_curve{}.csv".format(file_idx, curve_idx)
    print("Saving " + save_as)
    sp.save_ndarray_csv(save_as, data.curves[curve_idx].data)
    print("Done!\n")


def read_single_peak(filename):
    data = np.genfromtxt(filename, delimiter=',')
    sh = data.shape
    peaks = []
    for idx in range(data.shape[0]):
        peaks.append([])    # new group
        new_peak = pp.SinglePeak(time=data[idx, 1], value=data[idx, 2])
        if new_peak.time != 0 and new_peak.val != 0:
            peaks[idx].append(new_peak)
        else:
            peaks[idx].append(None)
    return peaks


# def peak_gr_file_list(data_file, folder):
#     data_file_name = os.path.basename(data_file)
#     data_file_name = data_file_name[0:-4]
#     peak_file_list = []
#     for name in sp.get_file_list_by_ext(folder, '.csv', sort=True):
#         if os.path.basename(name).startswith(data_file_name):
#             peak_file_list.append(name)
#     return peak_file_list


def get_peak_files(data_file, peak_folder=None, peak_dir_name='Peaks_all'):
    '''
    Returns the list of the peak files for specified data file.
    If peak files are not found or the folder containing 
    peak data is not found, returns [].
    
    data_file       -- the path to the file with SignalsData
                       The extension of the file must be '.csv'
                       (case insensitive).
    peaks_folder    -- the path to the files with peaks data
                       The default is None, which means: 
                       <data_file_path>/<peak_dir_name>
    peak_dir_name   -- the name of the folder containing peak data
                       Default == 'Peaks_all'
    '''
    assert os.path.isfile(data_file), \
        "Error! Can not find file '{}'.".format(data_file)
    path = os.path.dirname(data_file)
    if peak_folder is None:
        peak_folder = os.path.join(path, peak_dir_name)
    if os.path.isdir(peak_folder):
        data_file_name = os.path.basename(data_file)
        data_file_name = data_file_name[0:-4]
        peak_file_list = []
        for name in sp.get_file_list_by_ext(peak_folder, '.csv', sort=True):
            if os.path.basename(name).startswith(data_file_name):
                peak_file_list.append(name)
        return peak_file_list
    return []


def read_peaks(file_list):
    if file_list is None or len(file_list) == 0:
        return None
    else:
        groups = read_single_peak(file_list[0])
        curves_number = len(groups)
        for file_idx in range(1, len(file_list)):
            new_group = read_single_peak(file_list[file_idx])
            for wf in range(curves_number):     # wavefrorm number
                groups[wf].append(new_group[wf][0])
        return groups


def plot_peaks_from_file(data_file, peaks_folder, curves_list, params):
    peak_file_list = get_peak_files(data_file, peaks_folder)
    peaks = read_peaks(peak_file_list)
    if peaks is None:
        print("No peaks.")
    elif isinstance(peaks, list):
        print("Number of peaks: {}".format(len(peaks[0])))

    data = sp.SignalsData(np.genfromtxt(data_file, delimiter=','))
    # curves_list = [0, 1, 2, 3, 4, 5, 6, 7, 12, 13]
    plot_name = os.path.basename(data_file)
    plot_name = plot_name[0:-4] + ".plot.png"
    plot_name = os.path.join(peaks_folder, plot_name)
    print("Saving as " + plot_name)
    plot_peaks_all(data, peaks, curves_list,
                   xlim=params.get("time_bounds", None),
                   show=True, save=True, save_as=plot_name)


def replot_peaks(path, curves_list, params, filename=None):
    data_file_list = sp.get_file_list_by_ext(path, ".csv", sort=True)
    current_peaks_folder = os.path.join(path, "Peaks_all")
    assert os.path.isdir(current_peaks_folder), \
        "Can not find folder with peaks data '" + current_peaks_folder + "'."
    if filename is not None:
        print("Reading " + filename)
        plot_peaks_from_file(filename, current_peaks_folder, curves_list, params)
        print("Done!\n")
    else:
        for idx, name in enumerate(data_file_list):
            print("Reading " + name)
            plot_peaks_from_file(name, current_peaks_folder, curves_list, params)
            print("Done!\n")


def get_y_zero_offset(signal, start_x, stop_x):
    '''
    Returns the Y zero level offset value.
    Use it for zero level correction before PeakProcess.
    
    signal -- SingleCurve instance
    start_x and stop_x -- define the limits of the 
            X interval where Y is filled with noise only.
    '''
    if start_x < signal.time[0]:
        start_x = signal.time[0]
    if stop_x > signal.time[-1]:
        stop_x = signal.time[-1]
    assert stop_x > start_x, \
        "Error! start_x value must be lower than stop_x value."
    if start_x > signal.time[-1] or stop_x < signal.time[0]:
        return 0

    start_idx = pp.find_nearest_idx(signal.time, start_x, side='right')
    stop_idx = pp.find_nearest_idx(signal.time, stop_x, side='left')
    sum = 0.0
    for val in signal.val[start_idx:stop_idx]:
        sum += val
    return sum / (stop_idx - start_idx + 1)


def get_all_y_zero_offset(signals_data, curves_list, start_stop_tuples):
    '''
    Return delays list for all columns in SignalsData stored 
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
    '''

    assert len(curves_list) == len(start_stop_tuples), \
        "Error! The number of (start_x, stop_x) tuples ({}) " \
        "does not match the number of specified curves " \
        "({}).".format(len(start_stop_tuples), len(curves_list))

    delays = [0 for _ in range(2 * signals_data.count)]
    for curve_idx in curves_list:
        y_data_idx = curve_idx * 2 + 1
        delays[y_data_idx] = get_y_zero_offset(signals_data.curves[curve_idx],
                                               *start_stop_tuples[curve_idx])
    # print("Delays = ")
    # for i in range(0, len(delays), 2):
    #     print("{}, {},".format(delays[i], delays[i + 1]))
    return delays


def pretty_print_nums(nums, prefix='', postfix='',
                      s=u'{pref}{val:.2f}{postf}', show=True):
    '''
    Prints template 's' filled with values from 'nums',
    'prefix' and 'postfix' arrays for all numbers in 'nums'.
    
    nums    -- array of float or int
    prefix  -- array of prefixes for all values in 'nums'
               or single prefix string for all elements. 
    postfix -- array of postfixes for all values in 'nums'
               or single postfix string for all elements. 
    s       -- template string.
    '''
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


def save_all_single_plots(signals, peaks, time_bounds, curves_list,
                          fileprefix, dir_name="./Peaks_single"):
    for idx in range(len(curves_list)):
        fname = fileprefix + "_curve" + str(curves_list[idx]) + ".png"
        plot_single_curve(signals.curves[curves_list[idx]],
                          peaks[idx], time_bounds,
                          save=True, save_as=fname)


if __name__ == '__main__':
    # =================================================================
    # -----     MAIN     -----------------------------
    # =================================================================
    # import sys
    # filename = sys.argv[1]

    log = ""
    # make_final()
    # save_curve(0, 6)

    # filename = ("H:\\WORK\\ERG\\2017\\2017 05 12-19 ERG\\"
    #             "2017 05 13 ERG Output FINAL\\ERG_106.csv")

    # filename = ("/media/shpakovkv/6ADA8899DA886365/WORK/2017/"
    #             "2017 05 12-19 ERG/2017 05 19 ERG Output FINAL/ERG_020.csv")

    # filename = ("H:\\WORK\\ERG\\2015\\2015 06 25 ERG\\"
    #             "2015 06 25 UnitedData\\ERG_002.csv")

    # step 1 - set parameters
    start_stop_tuples = [(-350, -100),  # 0 grad
                         (-350, -100),  # 10 grad
                         (-350, -100),  # 20 grad
                         (-350, -100),  # 30 grad
                         (-350, -100),  # 40 grad
                         (-350, -100),  # 50 grad
                         (-350, -100),  # 60 grad
                         (-350, -100),  # 70 grad
                         (-350, -100),  # 80 grad
                         (-350, -100), ]  # 90 grad
    multipliers = None
    delays = [0, 0,  # 0 grad
              0, 0,  # 10 grad
              0, 0,  # 20 grad
              0, 0,  # 30 grad
              -21, 0,  # 40 grad
              -21, 0,  # 50 grad
              -21, 0,  # 60 grad
              -21, 0,  # 70 grad
              -10, 0,  # 80 grad
              -10, 0,  # 90 grad
              -10, 0,
              -10, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    need_corr_by_voltage = False
    save_changed_data = False
    Y_zero_offset = True
    save_single_plots = True

    # filename = ("/media/shpakovkv/6ADA8899DA886365/WORK/2015/"
    #             "2015 05 15 ERG VNIIA/2015 05 15 UnitedData/ERG_055.csv")
    filename = ("H:\\WORK\\ERG\\2015\\2015 05 15 ERG VNIIA\\"
                "2015 05 15 UnitedData\\ERG_096.csv")

    data_folder = os.path.dirname(filename)
    save_dir = data_folder

    params = {"level": -0.34, "diff_time": 5, "tnoise": 50, "graph": False,
              "time_bounds": [-100, 500], "noise_attenuation": 0.75}
    curves_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    curves_labels = [u'0\xb0: ', u'10\xb0: ', u'20\xb0: ', u'30\xb0: ', u'40\xb0: ',
                     u'50\xb0: ', u'60\xb0: ', u'70\xb0: ', u'80\xb0: ', u'90\xb0: ']

    pk_group_diff = 20
    curve_idx = 8

    multiplot_name = os.path.basename(filename)
    multiplot_name = multiplot_name[0:-4] + ".plot.png"
    peaks_folder = os.path.join(data_folder, "Peaks_all")
    multiplot_name = os.path.join(peaks_folder, multiplot_name)

    # step 2 - read data
    add_to_log("Reading " + filename)
    signals_data = read_signals([filename], file_type='csv', delimiter=",")
    add_to_log("Curves count = " + str(signals_data.count) + "\n")
    shot_name = os.path.basename(filename)[0:-4]

    # step 3 - update delays with accordance to Y zero offset
    if Y_zero_offset:
        for idx, new_val in enumerate(get_all_y_zero_offset(signals_data,
                                                            curves_list,
                                                            start_stop_tuples)):
            delays[idx] += new_val

    add_to_log(pretty_print_nums([delays[idx] for idx in
                                  range(0, len(delays), 2)],
                                 curves_labels, " ns", show=False))
    add_to_log()
    add_to_log(pretty_print_nums([delays[idx] for idx in
                                  range(1, len(delays), 2)],
                                 curves_labels, " V", show=False))
    add_to_log()

    # step 4 - update delays whith accordance to voltage front
    if need_corr_by_voltage:
        voltage_front_level = 0.2
        volt_cr_idx = 12
        polarity = pp.check_polarity(signals_data.curves[volt_cr_idx])
        add_to_log(str(polarity) + "  (autodetect)")

        volt_plot_name = shot_name + "_voltage.png"
        volt_plot_name = os.path.join(save_dir, "voltage_front", volt_plot_name)
        if pp.is_pos(polarity):
            voltage_front_level = abs(voltage_front_level)
        else:
            voltage_front_level = -abs(voltage_front_level)

        add_to_log("Searching voltage front at level = " +
                   str(voltage_front_level))

        level_raw = ((voltage_front_level +
                      delays[volt_cr_idx * 2 + 1]) /
                      multipliers[volt_cr_idx * 2 + 1])
        front_raw = get_voltage_front(signals_data.curves[volt_cr_idx], level_raw,
                                      save_plot=True, polarity=polarity,
                                      plot_name=volt_plot_name)
        if front_raw:
            add_to_log("Raw_voltage_front_level = " + str(level_raw))
            # apply multiplier and compensate voltage delay
            time_offset = (front_raw * multipliers[volt_cr_idx * 2] -
                           delays[volt_cr_idx * 2])
            add_to_log("Time_offset_by_voltage = " +
                       str(time_offset))
            for idx in range(0, len(delays), 2):
                delays[idx] += time_offset

    # step 5 - apply multipliers and delays
    add_to_log("Applying multipliers and delays...", )
    signals_data = sp.multiplier_and_delay(signals_data, multipliers, delays)

    # step 6 - find peaks [and plot single graphs]
    if True:
        unsorted_peaks = []
        for idx in curves_list:
            add_to_log("Curve #" + str(idx), end="    ")
            new_peaks, peak_log = pp.peak_finder(
                signals_data.time(idx), signals_data.value(idx), **params)
            unsorted_peaks.append(new_peaks)
            add_to_log(peak_log, end="")

        # step 7 - group peaks [and plot all curves with peaks]
        peak_data, peak_map = pp.group_peaks(unsorted_peaks, pk_group_diff)

        # step 8 - save peaks data
        add_to_log("Saving peak data...")
        peaks_filename = os.path.join(data_folder, "Peaks_all",
                                      os.path.basename(filename))
        save_peaks_csv(peaks_filename, peak_data)

        # step 9 - save multicurve plot
        add_to_log("Saving all peaks as " + multiplot_name)
        plot_peaks_all(signals_data, peak_data, curves_list,
                       xlim=params.get("time_bounds", None),
                       show=True, save=True, save_as=multiplot_name)

    # step 10 - replot all curves with peaks from files
    if input("Re-plot subplots from files? >> "):
        peak_file_list = get_peak_files(filename)
        peak_data = read_peaks(peak_file_list)

        # # swap the peaks of curves #0 and #2 (0 grad and 20 grad)
        # # and save them
        # tmp = peaks[0]
        # peaks[0] = peaks[2]
        # peaks[2] = tmp
        # peak_file_name = os.path.basename(filename)[0:-4]
        # peak_file_name = os.path.join(data_folder, "Peaks_all", peak_file_name)
        # print(peak_file_name)
        # input("Continue? >> ")
        # save_peaks_csv(peak_file_name, peaks)

        add_to_log("Saving all peaks as " + multiplot_name)
        plot_peaks_all(signals_data, peak_data, curves_list,
                       xlim=params.get("time_bounds", None),
                       show=True, save=True, save_as=multiplot_name)

    # step 11 - save single plots with peaks
    single_plot_name = os.path.join(data_folder, "Peaks_single")
    if not os.path.isdir(single_plot_name):
        os.makedirs(single_plot_name)
    single_plt_file_prefix = os.path.join(single_plot_name, shot_name)
    if save_single_plots:
        add_to_log("Saving single plots with peaks...")
        save_all_single_plots(signals_data, peak_data,
                              params.get("time_bounds", None),
                              curves_list,
                              single_plt_file_prefix,
                              dir_name="./Peaks_single")

    # step 12 - save changed signals data
    if save_changed_data:
        add_to_log("Saving...")
        if False:
            # save with rewrite
            sp.save_ndarray_csv(filename, signals_data.get_array())
        elif False:
            # save as
            new_filename = ""
            sp.save_ndarray_csv(new_filename, signals_data.get_array())

    add_to_log("Done.")

    # OLD---------------------------------------------------------------------
    # data_file_list = sp.get_file_list_by_ext(data_folder, ".csv", sort=True)
    # for name in data_file_list:
    #     voltage_front_level = -0.2
    #     voltage_idx = 0
    #     all_delays = [125, 0, 78, 0, 135, 0, 135, 0]  # for 2014.11.10-11
    #     # all_delays = [117, 0, 70, 0, 127, 0, 127, 0]  # for 2014.11.13
    #     # all_delays = [132, 0, 85, 0, 142, 0, 142, 0]  # for 2014.11.14
    #     all_multipliers = [1e9, 0.17771, 1e9, 0.46, 1e9, 1, 1e9, 1]
    #     data = sp.SignalsData(np.genfromtxt(name, delimiter=","))
    #     level_raw = ((voltage_front_level +
    #                   all_delays[voltage_idx * 2 + 1]) /
    #                  all_multipliers[voltage_idx * 2 + 1])
    #     front_raw = offset_by_voltage(data.curves[voltage_idx], level_raw,
    #                                   save_plot=False, polarity="neg")
    #     if front_raw is not None:
    #         time_offset = (front_raw * all_multipliers[voltage_idx * 2] -
    #                        all_delays[voltage_idx * 2])
    #         print(time_offset)
    #     else:
    #         print("None")

    # data_folder_list = []
    # data_folder_list.append("/media/shpakovkv/6ADA8899DA886365/WORK/2017/"
    #                         "2017 05 12-19 ERG/2017 05 13 ERG Output FINAL")
    # data_folder_list.append("/media/shpakovkv/6ADA8899DA886365/WORK/2017/"
    #                         "2017 05 12-19 ERG/2017 05 14 ERG Output FINAL")
    # data_folder_list.append("/media/shpakovkv/6ADA8899DA886365/WORK/2017/"
    #                         "2017 05 12-19 ERG/2017 05 15 ERG Output FINAL")
    # data_folder_list.append("/media/shpakovkv/6ADA8899DA886365/WORK/2017/"
    #                         "2017 05 12-19 ERG/2017 05 16 ERG Output FINAL")
    # data_folder_list.append("/media/shpakovkv/6ADA8899DA886365/WORK/2017/"
    #                         "2017 05 12-19 ERG/2017 05 18 ERG Output FINAL")
    # data_folder_list.append("/media/shpakovkv/6ADA8899DA886365/WORK/2017/"
    #                         "2017 05 12-19 ERG/2017 05 19 ERG Output FINAL")
    # for item in data_folder_list:
    #     data_file_list = sp.get_file_list_by_ext(item, ".csv", sort=True)
    #     for name in data_file_list:
    #         print("Reading " + name)
    #         current_peaks_folder = os.path.join(item, "Peaks_all")
    #         plot_peaks_from_file(name, current_peaks_folder, curves_list, params)
    #         print("Done!\n")

    # plot_peaks_from_file(filename, peaks_folder, curves_list, params)

    # sp.compare_files_in_folder("H:\\WORK\\ERG\\2017\\2017 05 12-19 ERG\\"
    #                            "2017 05 13-19 ERG Output final\\"
    #                            "2017 05 19 ERG Input final\\"
    #                            "2017 05 19 DPO7054")

    # folder = ("/media/shpakovkv/6ADA8899DA886365/WORK/2017/"
    #           "2017 05 12-19 ERG/2017 05 13 ERG Output FINAL")
    # file_list = sp.get_file_list_by_ext(folder, ".CSV", sort=True)
    # DPO7054_fail_list = []
    # HMO3004_fail_list = []
    # TDS2024C_fail_list = [1, 64, 79, ]
    # LeCroy_fail_list = [1, 64, 76, 79, ]
    # for idx in range(len(file_list)):
    #     curves = []
    #     if idx in DPO7054_fail_lis
    #         curves += [0, 1, 2, 3]
    #     if idx in HMO3004_fail_list:
    #         curves += [4, 5, 6, 7]
    #     if idx in TDS2024C_fail_list:
    #         curves += [8, 9, 10, 11]
    #     if idx in LeCroy_fail_list:
    #         curves += [12, 13, 14]
    #     if len(curves) > 0:
    #         zero_and_save(file_list[idx], file_type="csv", curves=curves)
    # zero_log_filename = os.path.join(folder, "zero_curves.log")
    # with open(zero_log_filename, 'w') as f:
    #     f.write(log)

    '''
    # UNION AND SAVE
    group_size = 4
    data_folder = "H:\\WORK\\ERG\\2015\\2015 06 25 ERG\\Data_CSV"
    data_folder = ("/media/shpakovkv/6ADA8899DA886365/WORK/2015/"
                   "2015 05 15 ERG VNIIA/2015 05 15 DataSheets")
    save_to = "H:\\WORK\\ERG\\2015\\2015 06 25 ERG\\2015 06 25 UnitedData"
    save_to = ("/media/shpakovkv/6ADA8899DA886365/WORK/2015/"
               "2015 05 15 ERG VNIIA/2015 05 15 UnitedData")
    if not os.path.isdir((save_to)):
        os.makedirs(save_to)
    prefix = "ERG_"
    postfix = ".csv"
    grouped_list = []
    save_as = []
    file_list = sp.get_file_list_by_ext(data_folder, ".csv", sort=True)

    # # REPLACE DELIMITERS
    # # for direct EXPORTED from OriginPro csv files
    # import re
    # for name in file_list:
    #     print("Replacing delimiters in " + name, end="    ")
    #     with open(name, 'r') as fid:
    #         lines = fid.readlines()
    #         for idx, line in enumerate(lines):
    #             lines[idx] = re.sub(r',', '.', line)
    #             lines[idx] = re.sub(r';', ',', lines[idx])
    #     with open(name, 'w') as fid:
    #         fid.writelines(lines)
    #     print("Done.")
    # # raise Exception("STOP HERE!")

    shots_count = len(file_list) // group_size
    for shot in range(shots_count):
        grouped_list.append([file_list[idx] for idx in
                             range(shot, len(file_list), shots_count)])
    num_start, num_end = \
        sp.numbering_parser(names[0] for names in grouped_list)

    for filename in (os.path.basename(name[0]) for name in grouped_list):
        shot_number = filename[num_start: num_end]
        if shot_number.lower().endswith(".csv"):
            shot_number = shot_number[:-4]
        save_as.append(os.path.join(save_to, prefix + shot_number + postfix))
    for idx, group in enumerate(grouped_list):
        print('Reading files: ')
        print('\n'.join(group))
        data = read_signals(group, delimiter=",")
        np.savetxt(save_as[idx], data.get_array(), delimiter=",")
        print('Saved as: {}'.format(save_as[idx]))
    '''
