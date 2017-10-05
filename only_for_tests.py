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
    if file_type.upper() == 'CSV':
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


def offset_by_voltage(voltage, level,
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
            front_raw = offset_by_voltage(data.curves[voltage_idx], level_raw,
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
                    axes[wf].plot([pk.time], [pk.val], '*', color=color, markersize=5)
    # figManager = plt.get_current_fig_manager()
    # figManager.window.showMaximized()
    if save:
        # print("Saving plot " + save_as)
        plt.savefig(save_as, dpi=400)
        # print("Done!")
    if show:
        plt.show()
    plt.close('all')


def go_peak_process(data_folder, curves_list, params, group_diff, single_file_name=None):
    # data_folder = ("/media/shpakovkv/6ADA8899DA886365/WORK/2017/"
    #           "2017 05 12-19 ERG/2017 05 15 ERG Output FINAL")
    # save_peaks_to = ("/media/shpakovkv/6ADA8899DA886365/WORK/2017/"
    #                  "2017 05 12-19 ERG/2017 05 13 ERG Peaks")

    # folder = ("H:\\WORK\\ERG\\2017\\2017 05 12-19 ERG\\"
    #           "2017 05 14 ERG Output FINAL")

    # folder = "H:\\WORK\\ERG\\2017\\2017 05 12-19 ERG\\TEMP\\"

    # curves_list = [0, 1, 2, 3, 4, 5, 6, 7, 12, 13]
    file_list = sp.get_file_list_by_ext(data_folder, ".CSV", sort=True)
    # GET PEAKS
    for filename in file_list:
        if single_file_name is not None:
            filename = single_file_name
        add_to_log("Reading " + filename)
        data = sp.SignalsData(np.genfromtxt(filename, delimiter=','))
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
                       save=True, save_as=plot_filename)
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
        add_to_log("Done!\n")
        if single_file_name is not None:
            break

    print("Saving log...")
    log_filename = os.path.join(data_folder, "PeakProcess.log")
    with open(log_filename, 'w') as f:
        f.write(log)
    print("Done!")
    # input("Press Enter...")
    # break


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


def read_peaks_group(filename):
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


def read_all_groups(file_list):
    if file_list is None or len(file_list) == 0:
        return None
    else:
        groups = read_peaks_group(file_list[0])
        curves_number = len(groups)
        for file_idx in range(1, len(file_list)):
            new_group = read_peaks_group(file_list[file_idx])
            for wf in range(curves_number):     # wavefrorm number
                groups[wf].append(new_group[wf][0])
        return groups


def peak_gr_file_list(data_file, folder):
    data_file_name = os.path.basename(data_file)
    data_file_name = data_file_name[0:-4]
    peak_file_list = []
    for name in sp.get_file_list_by_ext(folder, '.csv', sort=True):
        if os.path.basename(name).startswith(data_file_name):
            peak_file_list.append(name)
    return peak_file_list

def plot_peaks_from_file(data_file, peaks_folder, curves_list, params):
    peak_file_list = peak_gr_file_list(data_file, peaks_folder)
    peaks = read_all_groups(peak_file_list)
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
                   save=True, save_as=plot_name)


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

    filename = ("H:\\WORK\\ERG\\2015\\2015 06 25 ERG\\"
                "2015 06 25 UnitedData\\ERG_002.csv")

    data_folder = os.path.dirname(filename)
    peaks_folder = os.path.join(data_folder, "Peaks_all")

    params = {"level": -0.26, "diff_time": 30, "tnoise": 100, "graph": True,
              "time_bounds": [-100, 600], "noise_attenuation": 0.75}
    curves_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    group_params = 15
    curve_idx = 8

    # plot_one_curve_peaks(filename, curve_idx, params)

    # test_peak_process(filename, curves_list, params)


    # go_peak_process(data_folder, curves_list, params, group_params,)

    go_peak_process(data_folder, curves_list, params, group_params, filename)
    input("Press enter")
    replot_peaks(data_folder, curves_list, params, filename)

    # replot_peaks(data_folder, curves_list, params)

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
    save_to = "H:\\WORK\\ERG\\2015\\2015 06 25 ERG\\2015 06 25 UnitedData"
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
