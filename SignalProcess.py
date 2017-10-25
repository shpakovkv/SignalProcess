# Python 2.7
from __future__ import print_function, with_statement

import re
import os

import numpy as np
from scipy.signal import savgol_filter

import wfm_reader_lite as wfm


class SingleCurve:
    def __init__(self, in_x=None, in_y=None):
        self.data = np.empty([0, 2], dtype=float, order='F')
        if in_x is not None and in_y is not None:
            self.append(in_x, in_y)

    def append(self, x_in, y_in):
        # INPUT DATA CHECK
        x_data = x_in
        y_data = y_in
        if not isinstance(x_data, np.ndarray) or not isinstance(y_data, np.ndarray):
            raise TypeError("Input time and value arrays must be "
                            "instances of numpy.ndarray class.")
        if len(x_data) != len(y_data):
            raise IndexError("Input time and value arrays must "
                             "have same length.")
        # self.points = len(x_data)

        if np.ndim(x_data) == 1:  # check if X array has 2 dimensions
            x_data = np.expand_dims(x_in, axis=1)  # add dimension to the array
        if np.ndim(y_data) == 1:  # check if Y array has 2 dimensions
            y_data = np.expand_dims(y_in, axis=1)  # add dimension to the array

        if x_data.shape[1] != 1 or y_data.shape[1] != 1:  # check if X and Y arrays have 1 column
            raise ValueError("Input time and value arrays must "
                             "have 1 column and any number of rows.")

        start_idx = None
        stop_idx = None
        for i in range(len(x_data) - 1, 0, -1):
            tmp_x = x_data[i]
            tmp_y= y_data[i]
            if not np.isnan(x_data[i]) and not np.isnan(y_data[i]):
                stop_idx = i
                break
        for i in range(0, stop_idx + 1):
            if not np.isnan(x_data[i]) and not np.isnan(y_data[i]):
                start_idx = i
                break
        if start_idx == None or stop_idx == None:
            raise ValueError("Can not append array of empty "
                             "values to SingleCurve's data.")
        # CONVERT TO NDARRAY
        temp = np.append(x_data[start_idx:stop_idx + 1],
                         y_data[start_idx:stop_idx + 1],
                         axis=1)
        self.data = np.append(self.data, temp, axis=0)

    def get_x(self):
        return self.data[:, 0]

    def get_y(self):
        return self.data[:, 1]

    def get_points(self):
        return self.data.shape[0]

    time = property(get_x, doc="Get curve's 1D array of time points")
    val = property(get_y, doc="Get curve's 1D array of value points")
    points = property(get_points, doc="Get the number of points in curve")


class SignalsData:
    def __init__(self, input_data=None, curve_labels=None):
        # EMPTY INSTANCE
        self.count = 0          # number of curves
        self.curves = []        # list of curves data (SingleCurve instances)
        self.labels = dict()    # dict with curve labels as keys and curve indexes as values

        # FILL WITH VALUES
        if input_data is not None:
            self.append(input_data, curve_labels)

    def append(self, input_data, curve_labels=None):
        # appends new SingleCurves to the self.curves list
        data = np.array(input_data, dtype=float, order='F')  # convert input data to numpy.ndarray
        self.check_input(data)                               # check inputs
        for curve_idx in range(0, data.shape[1], 2):
            self.curves.append(SingleCurve(data[:, curve_idx], data[:, curve_idx + 1]))  # adds new SingleCurve
            self.count += 1                                                              # updates the number of curves
            if curve_labels:
                self.labels[curve_labels[curve_idx]] = self.count - 1                 # adds label-index pair to dict
            else:
                self.labels[str(self.count - 1)] = self.count - 1                     # adds 'index'-index pair to dict

    def check_input(self, data, curve_labels=None):
        # CHECK INPUT DATA
        if np.ndim(data) != 2:                              # number of dimension of the input array check
            raise ValueError("Input array must have 2 dimensions.")
        if data.shape[1] % 2 != 0:                          # number of columns of the input array check
            raise IndexError("Input array must have even number of columns.")
        if curve_labels:
            if not isinstance(curve_labels, list):          # labels array type checck
                raise TypeError("Variable curve_labels must be an instance of the list class.")
            if data.shape[1] // 2 != len(curve_labels):     # number of labels check
                raise IndexError("Number of curves (pair of time-value columns) in data "
                                 "and number of labels must be the same.")
            for label in curve_labels:                      # label duplicate check
                if label in self.labels:
                    raise ValueError("Label \"" + label + "\" is already exist.")

    def get_array(self):
        # return all curves data as united 2D array
        # short curve arrays are supplemented with required amount of rows (filled with 'nan')
        return align_and_append_ndarray(*[curve.data for curve in self.curves])

    def by_label(self, label):   # returns SingleCurve by name
        return self.curves[self.labels[label]]

    def get_label(self, idx):    # return label of the SingelCurve by index
        for key, value in self.labels.items():
            if value == idx:
                return key

    def get_idx(self, label):    # returns index of the SingelCurve by label
        if label in self.labels:
            return self.labels[label]

    def time(self, curve):
        return self.curves[curve].get_x()

    def value(self, curve):
        return self.curves[curve].get_y()


def get_subdir_list(path):
    # return list of subdirectories
    return [os.path.join(path, x) for x in os.listdir(path) if os.path.isdir(os.path.join(path, x))]


def get_file_list_by_ext(path, ext, sort=True):
    # return a list of files (with the specified extension) contained in the folder (path)
    # each element of the returned list is a full path to the file
    file_list = [os.path.join(path, x) for x in os.listdir(path)
                 if os.path.isfile(os.path.join(path, x)) and x.upper().endswith(ext.upper())]
    if sort:
        file_list.sort()
    return file_list


def group_files_by_count(file_list, files_in_group=4):
    # group files for DPO7054 / TDS2024
    # (first 4(default) files form first group, second 4 files form second group, etc.)
    # files in group mus have equal prefix, equal number and different postfix
    # example: "erg_001_Ch1.wfm", "erg_001_Ch2.wfm", "erg_001_Ch3.wfm", "erg_001_Ch4.wfm"
    groups_list = []
    if len(file_list) % files_in_group != 0:
        raise IndexError("Not enough files in file_list. Or wrong value of variable files_in_group.")
    for i in range(0, len(file_list), files_in_group):
        new_group = []
        for j in range(files_in_group):  # makes new group of files
            new_group.append(file_list[i + j])
        # print("Group[" + str(int(i / files_in_group + 1)) + "] : " + str(new_group))
        groups_list.append(new_group)   # adds new group to the list
    return groups_list


def group_files_lecroy(file_list, files_in_group=4):
    # group files for LeCroy
    # Every N files in the file_list contains single channel data,
    # where N is shoots count (N = len(file_list) / files_in_group)
    #
    # files in group mus have equal postfix, equal number and different prefix ("C1_", "C2_", "C3_" or "C4_")
    # example: "C1_LeCroy_0001.txt", "C2_LeCroy_0001.txt", "C3_LeCroy_0001.txt", "C4_LeCroy_0001.txt"
    if len(file_list) % files_in_group != 0:
        raise IndexError("Not enough files in file_list. Or wrong value of variable files_in_group.")

    groups_list = []
    shoots_count = int(len(file_list) / files_in_group)
    for i in range(0, shoots_count):
        new_group = []
        for j in range(files_in_group):  # makes new group of files
            new_group.append(file_list[i + j * shoots_count])
        # print("Group[" + str(i) + "] : " + str(new_group))    # shows group's content
        groups_list.append(new_group)  # adds new group to the list
    return groups_list


def get_name_from_group(group, ch_postfix_len=8, ch_prefix_len=0):
    # INPUT
    # group:            list of files corresponding to one shoot
    #                   each file is a single channel record from the oscilloscope
    # DEFAULT:
    # ch_postfix_len:   DPO7054: 8  |  TDS2024C: 7  | HMO3004: 4 |   LeCroy: 4
    # ch_prefix_len:    DPO7054: 0  |  TDS2024C: 0  | HMO3004: 0 |   LeCroy: 3
    #
    # OUTPUT:   name of the shoot

    path = os.path.abspath(group[0])
    name = os.path.basename(path)
    if ch_postfix_len > 0:
        return name[ch_prefix_len:-ch_postfix_len]
    return name[ch_prefix_len:]


def combine_wfm_to_csv(dir_path,
                       save_to=None,
                       files_in_group=4,
                       ch_postfix_len=8,
                       delimiter=", ",
                       target_ext=".wfm",
                       save_with_ext=".CSV",
                       silent_mode=False):
    # READS a group of wfm files (from DPO7054) with single shoot data (each file contains single channel data)
    # COMBINES all data to one table (each 2 columns [time, value] represent 1 file )
    # SAVES data to one csv file
    # REPEATS first 3 action
    #
    # dir_path          - folder containing target files
    # save_to           - folder to save files to
    # target_ext        - target files extension
    # files_in_group    = number of channels(records/files) of the oscilloscope corresponding to one shoot
    # ch_postfix_len    - DPO7054 wfm filename postfix length (default = 8) Example of postfix: "_Ch1.wfm"
    # delimiter         - delimiter for csv file
    # save_with_ext     - The data will be saved to a file with specified extension

    # CHECK PATH
    if save_to and not os.path.isdir(save_to):
        os.makedirs(save_to)

    if not silent_mode:
        print("Working directory: \n" + dir_path)

    # GET LIST OF FILES---------------------------------------------
    file_list = get_file_list_by_ext(dir_path, target_ext)
    file_list.sort()

    # GROUP FILES for DPO7054 --------------------------------------
    groups_list = group_files_by_count(file_list, files_in_group)

    # READ & SAVE
    for group in groups_list:
        # READ WFM
        data = wfm.read_wfm_group(group)

        if save_to:
            # SAVE CSV
            file_name = get_name_from_group(group, ch_postfix_len)
            file_name += save_with_ext
            # file_full_name = save_to + file_name
            np.savetxt(os.path.join(save_to, file_name), data, delimiter=delimiter)
            if not silent_mode:
                print("File \"" + file_name + "\" saved to " + os.path.join(save_to, file_name))

    if not silent_mode:
        print()


def combine_tds2024c_csv(dir_path,
                         save_to=None,
                         files_in_group=4,
                         ch_postfix_len=7,
                         skip_header=18,
                         usecols=(3, 4),
                         delimiter=", ",
                         target_ext=".CSV",
                         save_with_ext=".CSV",
                         silent_mode=False):
    # READS a group of csv files (from TDS2024C) with single shoot data (each file contains single channel data)
    # COMBINES all data to one table (each 2 columns [time, value] represent 1 file )
    # SAVES data to one csv file
    # REPEATS first 3 action
    #
    # dir_path          - folder containing target files
    # save_to           - folder to save files to
    # target_ext        - target files extension
    # files_in_group    = number of channels(records/files) of the oscilloscope corresponding to one shoot
    # ch_postfix_len    - TDS2024C filename postfix length (default = 7) Example of postfix: "CH1.CSV"
    # delimiter         - delimiter for csv file
    # save_with_ext     - The data will be saved to a file with specified extension

    # CHECK PATH
    if save_to and not os.path.isdir(save_to):
        os.makedirs(save_to)

    if not silent_mode:
        print("Working directory: " + dir_path)

    # GET LIST OF FILES---------------------------------------------
    file_list = get_file_list_by_ext(dir_path, target_ext)
    file_list.sort()
    data = None

    # GROUP FILES for DPO7054 --------------------------------------
    groups_list = group_files_by_count(file_list, files_in_group)

    # READ & SAVE
    for group in groups_list:
        # READ SINGLE CSVs
        data = read_csv_group(group, skip_header=skip_header, usecols=usecols)

        if save_to:
            # SAVE 'ALL-IN' CSV
            file_name = get_name_from_group(group, ch_postfix_len)
            file_name += save_with_ext
            # file_full_name = save_to + file_name
            np.savetxt(os.path.join(save_to, file_name), data, delimiter=delimiter)
            if not silent_mode:
                print("File \"" + file_name + "\" saved to " + os.path.join(save_to, file_name))

    if not silent_mode:
        print()
    return data


def combine_hmo3004_csv(dir_path,
                        save_to=None,           # save to filename (don't save if None)
                        files_in_group=1,
                        ch_postfix_len=4,
                        skip_header=1,
                        usecols=tuple(),
                        delimiter=", ",
                        target_ext=".CSV",
                        save_with_ext=".CSV",
                        silent_mode=False):
    # READS a csv file (from HMO3004) with single shoot data (one time column and several value columns)
    # COMBINES all data to default-shape table (each 2 columns [time, value] represent 1 channel )
    # SAVES data to one csv file
    # REPEATS first 3 action for all files in dir
    #
    # dir_path          - folder containing target files
    # save_to           - folder to save files to
    # target_ext        - target files extension
    # files_in_group    = number of channels(records/files) of the oscilloscope corresponding to one shoot
    # ch_postfix_len    - HMO3004 filename postfix length (default = 4) Example of postfix: ".CSV"
    # delimiter         - delimiter for csv file
    # save_with_ext     - The data will be saved to a file with specified extension

    # CHECK PATH
    if save_to and not os.path.isdir(save_to):
        os.makedirs(save_to)

    if not silent_mode:
        print("Working directory: " + dir_path)

    # GET LIST OF FILES---------------------------------------------
    file_list = get_file_list_by_ext(dir_path, target_ext)
    file_list.sort()
    data = None

    # GROUP FILES for DPO7054 --------------------------------------
    groups_list = group_files_by_count(file_list, files_in_group)

    # READ & SAVE
    for group in groups_list:
        # READ SINGLE CSVs
        old_format_data = read_csv_group(group, skip_header=skip_header, usecols=usecols)

        # ADD "time" column to each "value" column
        data = np.c_[old_format_data[:, 0:2],      # 0:2 means [0:2) and includes only columns 0 and 1
                     old_format_data[:, 0], old_format_data[:, 2],
                     old_format_data[:, 0], old_format_data[:, 3],
                     old_format_data[:, 0], old_format_data[:, 4]]

        if save_to:
            # SAVE 'ALL-IN' CSV
            file_name = get_name_from_group(group, ch_postfix_len)
            file_name = add_zeros_to_filename(file_name, 4)
            file_name += save_with_ext
            # file_full_name = save_to + file_name
            np.savetxt(os.path.join(save_to, file_name), data, delimiter=delimiter)
            if not silent_mode:
                print("File \"" + file_name + "\" saved to " + os.path.join(save_to, file_name))
    if not silent_mode:
        print()
    return data


def combine_lecroy_csv(dir_path,
                       save_to=None,           # save to filename (don't save if None)
                       files_in_group=4,
                       ch_postfix_len=4,
                       ch_prefix_len=3,
                       skip_header=5,
                       usecols=tuple(),
                       delimiter=", ",
                       target_ext=".txt",
                       save_with_ext=".CSV",
                       silent_mode=False):
    # READS a group of wfm files (from LeCroy) with single shoot data (each file contains single channel data)
    # COMBINES all data to one table (each 2 columns [time, value] represent 1 file )
    # SAVES data to one csv file
    # REPEATS first 3 action
    #
    # dir_path          - folder containing target files
    # save_to           - folder to save files to
    # target_ext        - target files extension
    # files_in_group    = number of channels(records/files) of the oscilloscope corresponding to one shoot
    # ch_postfix_len    - LeCroy csv filename postfix length (default = 4) Example of postfix: ".txt"
    # ch_prefix_len     - LeCroy csv filename prefix length (default = 3) Example of prefix: "C4_"
    # delimiter         - delimiter for csv file
    # save_with_ext     - The data will be saved to a file with specified extension

    # CHECK PATH
    if save_to and not os.path.isdir(save_to):
        os.makedirs(save_to)

    if not silent_mode:
        print("Working directory: " + dir_path)

    # GET LIST OF FILES---------------------------------------------
    file_list = get_file_list_by_ext(dir_path, target_ext)
    file_list.sort()
    data = None

    # GROUP FILES for LeCroy --------------------------------------
    groups_list = group_files_lecroy(file_list, files_in_group)

    # READ & SAVE
    for group in groups_list:
        # READ SINGLE CSVs
        data = read_csv_group(group, skip_header=skip_header, usecols=usecols)

        if save_to:
            # SAVE CSV
            file_name = get_name_from_group(group, ch_postfix_len, ch_prefix_len=ch_prefix_len)
            file_name += save_with_ext
            # file_full_name = save_to + file_name
            np.savetxt(os.path.join(save_to, file_name), data, delimiter=delimiter)
            if not silent_mode:
                print("File \"" + file_name + "\" saved to " + os.path.join(save_to, file_name))
    if not silent_mode:
        print()
    return data


def add_zeros_to_filename(full_path, count):
    # adds zeros to number in filename
    # Example: f("shoot22.csv", 4) => "shoot0022.csv"

    name = os.path.basename(full_path)
    folder_path = os.path.dirname(full_path)

    num = re.search(r'd+', name).group(0)
    match = re.match(r'^D+', name)
    if match:
        prefix = match.group(0)
    else:
        prefix = ""

    match = re.search(r'D+$', name)
    if match:
        postfix = match.group(0)
    else:
        postfix = ""

    while len(num) < count:
        num = "0" + num
    name = prefix + num + postfix

    return os.path.join(folder_path, name)


def read_csv_group(group_of_files,
                   delimiter=",",       # Default = ","
                   skip_header=18,      # Defaults: TDS2024C = 18  |  HMO3004 = 1  |  LeCroy = 5  |  EasyScope = 0
                   usecols=tuple()):    # Default:  tuple()   |   TDS2024C = (3.4)

    # READS a number of wfm files, UNITES columns to 1 table and RETURNS it as 2-dimensional ndarray

    data = np.genfromtxt(group_of_files[0], delimiter=delimiter, skip_header=skip_header, usecols=usecols)
    for i in range(1, len(group_of_files), 1):
        new_data = np.genfromtxt(group_of_files[i], delimiter=delimiter, skip_header=skip_header, usecols=usecols)
        data = np.c_[data, new_data]  # adds data to the array
    return data


def add_new_dir_to_path(old_leaf_dir, new_parent_dir):
    # adds last dir name from old_leaf_dir to new_parent_dir path
    # dir_name = re.search(r'[^/]+/$', old_leaf_dir).group(0)
    if old_leaf_dir.endswith("/") or old_leaf_dir.endswith("\\"):
        old_leaf_dir = old_leaf_dir[0:-1]
    dir_name = os.path.basename(old_leaf_dir)
    return os.path.join(new_parent_dir, dir_name)    # new leaf dir


def compare_2_files(first_file_name, second_file_name, lines=30):
    # compare a number of first lines of two files
    # return True if lines matches exactly
    with open(first_file_name, 'r') as file:
        with open(second_file_name, 'r') as file2:
            for idx in range(lines):
                if file.readline() != file2.readline():
                    return False
            return True


def compare_files_in_folder(path, ext=".CSV"):
    file_list = get_file_list_by_ext(path, ext=ext, sort=True)
    print("Current PATH = " + path)
    for idx in range(len(file_list) - 1):
        if compare_2_files(file_list[idx], file_list[idx + 1]):
            print(os.path.basename(file_list[idx]) + " == " + os.path.basename(file_list[idx + 1]))
    print()


def compare_files_in_subfolders(path, ext=".CSV"):
    if os.path.isdir(path):
        path = os.path.abspath(path)
        for subpath in get_subdir_list(path):
            compare_files_in_folder(subpath, ext=ext)
    else:
        print("Path " + path + "\n does not exist!")


def add_to_log(s, print_to_console=True):
    if print_to_console:
        print(s, end="")
    return s


def get_max_min_from_file_cols(file_list,       # list of fullpaths of target files
                               col_list,        # list of zero-based indexes of column to be processed
                               col_corr=list(),     # list of correction multiplayer for each col in the col_list
                               delimiter=',',   # file column separator
                               skip_header=0,   # number of headerlines in files
                               verbose_mode=True):   # print log to console
    # For each specified column, finds the maximum and minimum values for all files in the list
    # returns the file processing log
    log = ""
    min_data = {}
    max_data = {}
    for item in col_list:
        min_data[item] = 0
        max_data[item] = 0
    if not col_corr:
        for _ in col_list:
            col_corr.append(1)
    for file_i in range(len(file_list)):
        log += add_to_log("FILE \"" + file_list[file_i] + "\"\n", verbose_mode)
        data = np.genfromtxt(file_list[file_i], delimiter=delimiter, skip_header=skip_header, usecols=col_list)
        for col_i in range(len(col_list)):
            col = col_list[col_i]

            current_max = max(data[:, col_i])
            if max_data[col] < current_max:
                max_data[col] = current_max

            current_min = min(data[:, col_i])
            if min_data[col] > current_min:
                min_data[col] = current_min
            log += add_to_log("Col [" + str(col) + "] \t MAX = " + str(current_max)
                              + "\t MAX_corr = " + str(current_max * col_corr[col_i])
                              + "\t MIN = " + str(current_min)
                              + "\t MIN_corr = " + str(current_min * col_corr[col_i]) + "\n", verbose_mode)
        log += add_to_log("\n", verbose_mode)
    log += add_to_log("OVERALL STATS (from " + str(len(file_list)) + " files)\n", verbose_mode)
    for col_i in range(len(col_list)):
        col = col_list[col_i]
        log += add_to_log("Column [" + str(col) + "] (x" + str(col_corr[col_i]) + ")\t MAX = " + str(max_data[col])
                          + "\t MAX_corr = " + str(max_data[col] * col_corr[col_i])
                          + "\t MIN = " + str(min_data[col])
                          + "\t MIN_corr = " + str(min_data[col] * col_corr[col_i]) + "\n", verbose_mode)
    log += add_to_log("\n", verbose_mode)
    return log


def get_max_min_from_dir(dir_path,        # target folder
                         col_list,        # list of zero-based indexes of column to be processed
                         col_corr=list(),     # list of correction multiplayer for each col in the col_list
                         ext='.CSV',      # target files extension
                         delimiter=',',   # file column separator
                         skip_header=0,   # number of headerlines in files
                         log_file_name='log.txt',   # log file name
                         verbose_mode=True):        # print log to console
    file_list = get_file_list_by_ext(dir_path, ext, sort=True)

    log = get_max_min_from_file_cols(file_list, col_list,
                                     col_corr=col_corr,
                                     delimiter=delimiter,
                                     skip_header=skip_header,
                                     verbose_mode=verbose_mode)
    dir_name = os.path.split(dir_path)[1]
    if dir_name == '':
        dir_path = os.path.split(dir_path)[0]
        dir_name = os.path.split(dir_path)[1]

    log_file_path = os.path.join(dir_path, dir_name + ' ' + log_file_name)
    with open(log_file_path, "w") as fid:
        fid.write(log)
    return log


def col_and_param_number_check(data, *arg):
    for param_list in arg:
        if isinstance(data, np.ndarray):
            cols = data.shape[1]
        elif isinstance(data, SignalsData):
            cols = data.count
        else:
            raise TypeError("Can not check columns count. Data must be an instance of numpy.ndarray or SignalsData")

        if cols > len(param_list):
            raise IndexError("The number of columns exceeds the number of parameters.")
        elif cols < len(param_list):
            raise IndexError("The number of parameters exceeds the number of columns.")
    return True


def multiplier_and_delay(data,          # an instance of SignalsData class OR 2D numpy.ndarray
                         multiplier,    # list of multipliers for each columns in data.curves
                         delay):        # list of delays (subtrahend) for each columns in data.curves
    if isinstance(data, np.ndarray):
        row_number = data.shape[0]
        col_number = data.shape[1]

        if not multiplier:
            multiplier = [1 for _ in range(col_number)]
        if not delay:
            delay = [0 for _ in range(col_number)]

        if col_and_param_number_check(data, multiplier, delay):
            for col_idx in range(col_number):
                for row_idx in range(row_number):
                    data[row_idx][col_idx] = data[row_idx][col_idx] * multiplier[col_idx] - delay[col_idx]
            return data
    elif isinstance(data, SignalsData):
        if not multiplier:
            multiplier = [1 for _ in range(data.count * 2)]
        if not delay:
            delay = [0 for _ in range(data.count * 2)]

        if len(multiplier) != data.count * 2:
            raise IndexError("List of multipliers must contain values for each time and value columns in data.curves.")
        for curve_idx in range(data.count):
            col_idx = curve_idx * 2             # index of time-column of current curve
            data.curves[curve_idx].data = multiplier_and_delay(data.curves[curve_idx].data,
                                                               multiplier[col_idx:col_idx + 2],
                                                               delay[col_idx:col_idx + 2])
        return data


def smooth_voltage(x, y, x_multiplier=1):
    '''This function returns smoothed copy of 'y'.
    Optimized for voltage pulse of ERG installation.
    
    x -- 1D numpy.ndarray of time points (in seconds by default)
    y -- 1D numpy.ndarray value points
    x_multiplier -- multiplier applied to time data 
        (default 1 for x in seconds)
    
    return -- smoothed curve (1D numpy.ndarray)
    '''
    poly_order = 3       # 3 is optimal polyorder value for speed and accuracy
    window_len = 101    # value 101 is optimal for 1 ns (1e9 multiplier) resolution of voltage waveform
    #                     for 25 kV charging voltage of ERG installation

    # window_len correction
    time_step = (x[1] - x[0]) * 1e9 / x_multiplier      # calc time_step and converts to nanoseconds
    window_len = int(window_len / time_step)            # calc savgol filter window
    if len(y) < window_len:
        window_len = len(y) - 1
    if window_len % 2 == 0:
        window_len += 1     # window must be even number
    if window_len < 5:
        window_len = 5

    # smooth
    if len(y) >= 5:
        y_smoothed = savgol_filter(y, window_len, poly_order)
        return y_smoothed
    return y


def save_ndarray_csv(filename, data, delimiter=",", precision=18):
    # Saves 2-dimensional numpy.ndarray as .csv file
    # Replaces 'nan' values with ''
    # precision - a number of units after comma

    # check precision value
    if not isinstance(precision, int):
        raise ValueError("Precision must be integer")
    if precision > 18:
        precision = 18
    value_format = '%0.' + str(precision) + 'e'

    # check filename value
    if len(filename) < 4 or filename[-4:].upper() != ".CSV":
        filename += ".csv"
    folder_path = os.path.dirname(filename)
    if folder_path and not os.path.isdir(folder_path):
        os.makedirs(folder_path)

    with open(filename, 'w') as fid:
        lines = []
        for row in range(data.shape[0]):
            s = delimiter.join([value_format % data[row, col] for col in range(data.shape[1])]) + "\n"
            s = re.sub(r'nan', '', s)
            lines.append(s)
        fid.writelines(lines)

def align_and_append_ndarray(*args):
    # returns 2D numpy.ndarray containing all input 2D numpy.ndarrays
    # if input arrays have different number of rows, fills missing values with 'nan'

    # CHECK TYPE & LENGTH
    for arr in args:
        if not isinstance(arr, np.ndarray):
            raise TypeError("Input arrays must be instances of numpy.ndarray class.")
        if np.ndim(arr) != 2:
            raise ValueError("Input arrays must have 2 dimensions.")

    # ALIGN & APPEND
    max_rows = max([arr.shape[0] for arr in args])                   # output array's rows number == max rows number
    data = np.empty(shape=(max_rows, 0), dtype=float, order='F')     # empty 2-dim array
    for arr in args:
        miss_rows = max_rows - arr.shape[0]                                         # number of missing rows
        nan_arr = np.empty(shape=(miss_rows, arr.shape[1]), dtype=float, order='F') # array with missing rows...
        nan_arr = nan_arr * np.nan                                                  # ...filled with 'nan'

        aligned_arr = np.append(arr, nan_arr, axis=0)                               # combine existing & missing rows
        data = np.append(data, aligned_arr, axis=1)                                 # append arr to the data
    return data


def numbering_parser(group):
    '''
    Finds serial number (substring) in the file name string.

    group -- list of files (each file must corresponds 
             to different shot)

    return -- (start, digits), where: 
        start -- index of the first digit of the serial number
                 in a file name
        end -- index of the last digit of the serial number
    '''
    names = []
    for raw in group:
        names.append(os.path.basename(raw))
    assert all(len(name) for name in names), \
        ("Error! All file names must have the same length")
    numbers = []
    # for idx in range(2):
    #     numbers.append(parse_filename(names[idx]))
    for name in names:
        numbers.append(parse_filename(name))

    # numbers[filename_idx][number_idx]{}
    num_count = len(numbers[0])
    if len(numbers) == 1:
        return (numbers[0][0]['start'], numbers[0][0]['end'])

    for match_idx in range(num_count):
        unique = True
        for num in range(len(names)):
            for name_idx in range(len(names)):
                start = numbers[num][match_idx]['start']
                end = numbers[num][match_idx]['end']
                if num != name_idx and numbers[num][match_idx]['num'] == names[name_idx][start:end]:
                    unique = False
                    break
            if not unique:
                break
        if unique:
            return (numbers[0][match_idx]['start'],
                    numbers[0][match_idx]['end'])
    return (0, len(names[0]))


def parse_filename(name):
    '''
    Finds substrings with numbers in the input string.

    name -- string with numbers.

    return -- list of dicts. [{...}, {...}, ...]
        where each dict contains info about one founded number:
        'start' -- index of the first digit of the found number in the string
        'end' -- index of the last digit of the found number in the string
        'num' -- string representation of the number
    '''
    import re
    matches = re.finditer(r'\d+', name)
    match_list = []
    for match in matches:
        # num = int(match)
        match_list.append({'start': match.start(),
                           'end': match.end(),
                           'num': match.group()})
    return match_list


# ================================================================================================
# --------------   MAIN    ----------------------------------------
# ======================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(prog='Process Signals',
                                     description='', epilog='', fromfile_prefix_chars='@')

    # input files ------------------------------------------------------------
    parser.add_argument('-d', '--scr', '--source-dir',
                        action='store',
                        metavar='SOURCE_DIR',
                        dest='src_dir',
                        default='',
                        help='sets the directory containing data files '
                             '(default=current folder).'
                        )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-f', '--input-files',
                        action='append',
                        nargs='+',
                        metavar='INPUT_FILES',
                        dest='files',
                        help='specify one or more (space separated) input '
                             'file names after the flag. '
                             'It is assumed that the files belong '
                             'to the same shot. '
                             'In order to process multiple shots enter '
                             'multiple \'-i\' parameters. '
                        )
    group.add_argument('-g', '--grouped-by',
                        action='store',
                        type=int,
                        metavar='GROUPED_BY',
                        dest='group_size',
                        help='sets the size of the groups (default=1). '
                             'A group is a set of files, corresponding '
                             'to one shot. \nNOTE: if \'-g\' parameter'
                             'is specified, then all the files in the '
                             'specified directory will be processed.'
                        )
    parser.add_argument('-c', '--ch',  '--sorted-by-channel',
                        action='store_true',
                        dest='sorted_by_ch',
                        help='this options tells the program that the files '
                             'are sorted by the oscilloscope/channel '
                             '(firstly) and by the shot number (secondly). '
                             'By default, the program considers that the '
                             'files are sorted by the shot number (firstly) '
                             'and by the oscilloscope/channel (secondly).\n'
                             'ATTENTION: files from all oscilloscopes must '
                             'be in the same folder and be sorted '
                             'in one style.'
                        )

    # process parameters and options -----------------------------------------
    parser.add_argument('--multiplier',
                        action='store',
                        type=float,
                        metavar='MULTIPLIER',
                        nargs='+',
                        dest='multiplier',
                        default=None,
                        help='the list of multipliers data file(s). '
                             'NOTE: you must enter values for all the '
                             'columns in data file(s). Two columns (X and Y) '
                             'expected for each curve.'
                        )
    parser.add_argument('--delay',
                        action='store',
                        type=float,
                        metavar='DELAY',
                        nargs='+',
                        dest='delay',
                        default=None,
                        help=''
                        )
    parser.add_argument('--offset-by-voltage',
                        action='store',
                        metavar=('CURVE_IDX', 'LEVEL'),
                        nargs=2,
                        dest='voltage_idx',
                        default=None,
                        help=''
                        )
    parser.add_argument('--y-offset',
                        action='append',
                        metavar=('CURVE_IDX', 'BG_START', 'BG_STOP'),
                        nargs=3,
                        dest='delay',
                        help=''
                        )
    parser.add_argument('-s', '--save',
                        action='store_true',
                        dest='save_data',
                        help='saves data files.\n'
                             'NOTE: If in the input data one '
                             'shot corresponds to one file and output '
                             'directory is not specified, input files '
                             'will be overwritten!'
                        )

    # output settings --------------------------------------------------------
    parser.add_argument('-t', '--save-to', '--target-dir',
                        action='store',
                        metavar='SAVE_TO',
                        dest='save_to',
                        help='specify the output directory after the flag.'
                        )
    parser.add_argument('--prefix',
                        action='store',
                        metavar='FILE_PREFIX',
                        dest='prefix',
                        help='specify the prefix after the flag. This prefix '
                             'will be added to the output file names during '
                             'the automatic generation of file names. '
                             'Default=\'\'.'
                        )
    parser.add_argument('--postfix',
                        action='store',
                        metavar='FILE_POSTFIX',
                        dest='postfix',
                        help='specify the postfix after the flag. This '
                             'postfix will be added to the output file '
                             'names during the automatic generation '
                             'of file names. '
                             'Default=\'\'.'
                        )
    parser.add_argument('-o', '--output-files',
                        action='store',
                        nargs='*',
                        metavar='OUTPUT_FILES',
                        dest='out_names',
                        help='specify the list of file names after the flag. '
                             'The output files with data will be save with '
                             'the names from this list. '
                             'This will override the automatic generation '
                             'of file names.'
                        )
    parser.add_argument('-p', '--save-plots-to',
                        action='store',
                        dest='save_plots_to',
                        metavar='SAVE_PLOTS_TO',
                        help='specify the directory after the flag. Each '
                             'curve from data will be plotted ans saved '
                             'as a single plot.png.'
                        )
    parser.add_argument('-m', '--multiplot',
                        action='append',
                        dest='multiplot',
                        nargs='+',
                        help='specify the indexes of curves to be added to '
                             'plot after \'-m\' flag. You may use as many '
                             '\'-m\' flags (with different lists of curves)'
                             ' as you want.'
                        )
    parser.add_argument('--save-multiplot-as',
                        action='append',
                        dest='save_mp_as',
                        metavar='SAVE_MULTIPLOT_AS',
                        nargs=1,
                        help='the postfix to the multiplot file names. '
                             'The prefix will be equal to the output data '
                             'file.\n'
                             'If this flag is omitted the multiplots '
                             'will not be saved.\n'
                             'NOTE: if you use \'-s\' flags, then you must '
                             'use as many \'-s\' flags as '
                             'you used \'-m\' flags.'
                        )
    parser.add_argument('--hide', '--hide-multiplot',
                        action='store_true',
                        dest='hide_mplt',
                        help='if the flag is specified the multiplots '
                             'will be saved (if the \'-s\' flag was '
                             'specified as well) but not shown. This '
                             'option can reduce the runtime of the program.'
                        )

    args = parser.parse_args()

    # input files configuration check
    if args.src_dir:
        assert os.path.isdir(args.src_dir), \
            "Can not find directory {}".format(args.src_dir)
    if args.files:
        args.group_size = len(args.files[0])
        for idx, shot_files in enumerate(args.files):
            assert len(shot_files) == args.group_size, \
                ("The number of files in each shot must be the same.\n"
                 "Shot[1] = {} files ({})\nShot[{}] = {} files ({})"
                 "".format(args.group_size, ", ".join(args.files[0]),
                           idx + 1, len(shot_files), ", ".join(shot_files))
                )
            for filename in shot_files:
                assert os.path.isfile(os.path.join(args.src_dir, filename)), \
                    "Can not find file {}".format(filename)
    else:
        # user input: directory, group_size and sorted_by
        assert args.src_dir, ("Specify the directory (-d) containing the "
                              "data files. See help for more details.")
        file_list = get_file_list_by_ext(args.src_dir, ".csv", sort=True)
        assert len(file_list) % args.group_size == 0, \
            ("The number of .csv files ({}) in the specified folder "
             "is not a multiple of group size ({})."
             "".format(len(file_list), args.group_size))
        args.files = []
        if args.sorted_by_ch:
            shots_count = len(file_list) // args.group_size
            for shot in range(shots_count):
                args.files.append([file_list[idx] for idx in
                                     range(shot, len(file_list), shots_count)])
        else:
            for idx in range(0, len(file_list), args.group_size):
                args.files.append(file_list[idx: idx + args.group_size])

    # raw check offset_by_voltage parameters (types)
    
    # raw check y_zero_offset parameters (types)

    # MAIN LOOP starts with file read
    for shot_idx, group in enumerate(args.files):
        pass

    # check multiplier and delay

    # check offset_by_voltage parameters (if idx out of range)

    # check y_zero_offset parameters (if idx out of range)

    # updates delay values with accordance to voltage front

    # updates delays with accordance to Y zero offset

    # multiplier and delay

    # TODO: check len(multiplier, delay) == len(columns)
    # TODO: process fake files (not recorded)
    # TODO: file duplicates check
    # TODO: check for different number of columns in data files
    # TODO: user interactive input checker
    # TODO: partial import


    print(args)
