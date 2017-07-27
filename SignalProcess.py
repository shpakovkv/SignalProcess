# Python 2.7
from __future__ import print_function, with_statement
import re
import os
import numpy as np
import wfm_reader_lite as wfm


class SingleCurve:
    def __init__(self, in_x, in_y):
        # INPUT DATA CHECK
        x = in_x
        y = in_y
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError("Input time and value arrays must be instances of numpy.ndarray class.")
        if len(x) != len(y):
            raise IndexError("Input time and value arrays must have same length.")
        self.points = len(x)

        if np.ndim(x) == 1:                     # check if X array has 2 dimensions
            x = np.expand_dims(in_x, axis=1)    # add dimension to the array
        if np.ndim(y) == 1:                     # check if Y array has 2 dimensions
            y = np.expand_dims(in_y, axis=1)    # add dimension to the array

        if x.shape[1] != 1 or y.shape[1] != 1:  # check if X and Y arrays have 1 column
            raise ValueError("Input time and value arrays must have 1 column and any number of rows.")

        # CONVERT TO NDARRAY
        self.data = np.append(x, y, axis=1)

    def get_x(self):
        return self.data[:, 0]

    def get_y(self):
        return self.data[:, 1]


class SignalsData:
    def __init__(self, input_data=None):
        # EMPTY INSTANCE
        self.count = 0      # number of curves
        self.curves = []    # list of curves data (SingleCurve instances)

        # FILL WITH VALUES
        if input_data:
            self.append(input_data)

    def check_input(self, data):
        # CHECK INPUT DATA
        if np.ndim(data) != 2:
            raise ValueError("Input array must have 2 dimensions.")
        if data.shape[1] % 2 != 0:
            raise IndexError("Input array must have even number of columns.")

    def append(self, input_data):
        # appends new SingleCurves to the self.curves list

        data = np.array(input_data, dtype=float, order='F') # convert input data to numpy.ndarray
        self.check_input(data)                              # check inputs
        self.count += data.shape[1] // 2                    # update the number of curves
        for curve_idx in range(0, data.shape[1], 2):
            self.curves.append(SingleCurve(data[:,curve_idx], data[:, curve_idx + 1]))  # add new SingleCurve

    def get_array(self):
        # return all curves data as united 2D array
        # short curve arrays are supplemented with required amount of rows (filled with 'nan')
        return align_and_append_ndarray(*[curve.data for curve in self.curves])


def get_subdir_list(path):
    # return list of subdirectories
    return [os.path.join(path, x) for x in os.listdir(path) if os.path.isdir(os.path.join(path, x))]


def get_file_list_by_ext(path, ext, sort=False):
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


def get_name_from_group_of_files(group, ch_postfix_len=8, ch_prefix_len=0):
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
            file_name = get_name_from_group_of_files(group, ch_postfix_len)
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
            file_name = get_name_from_group_of_files(group, ch_postfix_len)
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
            file_name = get_name_from_group_of_files(group, ch_postfix_len)
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
            file_name = get_name_from_group_of_files(group, ch_postfix_len, ch_prefix_len=ch_prefix_len)
            file_name += save_with_ext
            # file_full_name = save_to + file_name
            np.savetxt(os.path.join(save_to, file_name), data, delimiter=delimiter)
            if not silent_mode:
                print("File \"" + file_name + "\" saved to " + os.path.join(save_to, file_name))
    if not silent_mode:
        print()
    return data


def add_new_dir_to_path(old_leaf_dir, new_parent_dir):
    # adds last dir name from old_leaf_dir to new_parent_dir path
    # dir_name = re.search(r'[^/]+/$', old_leaf_dir).group(0)
    if old_leaf_dir.endswith("/") or old_leaf_dir.endswith("\\"):
        old_leaf_dir = old_leaf_dir[0:-1]
    dir_name = os.path.basename(old_leaf_dir)
    return os.path.join(new_parent_dir, dir_name)    # new leaf dir


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


def compare_2_files(first_file_name, second_file_name, lines=10):
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
    with open(log_file_path, "w") as file:
        file.write(log)
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


def multiplier_and_delay_arr(data,          # 2D numpy.ndarray
                             multiplier,
                             delay):
    if col_and_param_number_check(data, multiplier, delay):
        row_number = data.shape[0]
        col_number = data.shape[1]
        for col_idx in range(col_number):
            for row_idx in range(row_number):
                data[row_idx][col_idx] = multiplier[col_idx] * data[row_idx][col_idx] - delay[col_idx]
        return data


def multiplier_and_delay_obj(data,          # SignalsData obj
                             multiplier,
                             delay):
    if col_and_param_number_check(data, multiplier, delay):
        for curve_idx in range(data.count):
            multiplier_and_delay_arr(data.curves[curve_idx], multiplier, delay)
        return data


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

    with open(filename, 'w') as file:
        lines = []
        for row in range(data.shape[0]):
            s = delimiter.join([value_format % data[row, col] for col in range(data.shape[1])]) + "\n"
            s = re.sub(r'nan', '', s)
            lines.append(s)
        file.writelines(lines)

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
    max_len = max([arr.shape[0] for arr in args])  # output array's rows number == max rows number
    data = np.empty(shape=(max_len, 0), dtype=float, order='F')  # empty 2-dim array
    for arr in args:
        miss_rows = max_len - arr.shape[0]  # number of missing rows
        cols = arr.shape[1]
        nan_arr = np.empty(shape=(miss_rows, arr.shape[1]), dtype=float, order='F')  # array with missing rows...
        nan_arr *= np.nan  # ...filled with 'nan'
        aligned_arr = np.append(arr, nan_arr, axis=0)  # fill missing row with nans
        data = np.append(data, aligned_arr, axis=1)  # append arr to data
    return data

# ================================================================================================
# --------------   MAIN    ----------------------------------------
# ======================================================
if __name__ == "__main__":
    # ----------  INPUT PARAMETERS   ---------------------------------------------
    # ==================================================================================================
    # ----------     DPO7054     -----------------------------------------------------------------------
    # =======================================
    dir_dpo7054 = "/media/shpakovkv/6ADA8899DA886365/WORK/2017/2017 05 12-19 ERG/DPO7054/2017 05 13"
    save_dpo7054_to_path = "/media/shpakovkv/6ADA8899DA886365/WORK/2017/2017 05 12-19 ERG/DPO7054_CSV/"

    files_in_group_dpo7054 = 4                          # number of files of the DPO7054 (default 4)
    read_dpo7054 = False                                # read files and convert to csv
    max_min_log_dpo7054 = False                         # read converted csv and find max and min for all files
    log_cols_dpo7054 = [1, 3, 5, 7]                     # search max and min only for this columns (zero-based indexes)
    corr_cols_dpo7054 = [1.216, 0.991, 1.000, 0.994]    # correction multiplyer for each column (for max-min log)
    log_filename_dpo7054 = 'max_min_dpo7054.log'    # max-min filename postfix (containing dir name will be the prefix)

    # ==================================================================================================
    # ----------     TDS2024C     ----------------------------------------------------------------------
    # =======================================
    dir_tds2024 = "/media/shpakovkv/6ADA8899DA886365/WORK/2017/2017 05 12-19 ERG/TDS2024C"
    save_tds2024_to_path = "/media/shpakovkv/6ADA8899DA886365/WORK/2017/2017 05 12-19 ERG/TDS2024C_CSV"

    files_in_group_tds2024 = 4                          # number of files of the TDS2024C (default 4)
    read_tds2024 = False                                # read files and convert to csv
    max_min_log_tds2024 = False                         # read converted csv and find max and min for all files
    log_cols_tds2024 = [1, 3, 5, 7]                     # search max and min only for this columns (zero-based indexes)
    corr_cols_tds2024 = [1.261, 1.094, 1.222, 1.590]    # correction multiplyer for each column (for max-min log)
    log_filename_tds2024 = 'max_min_tds2024c.log'   # max-min filename postfix (containing dir name will be the prefix)

    # ==================================================================================================
    # ----------     HMO3004     -----------------------------------------------------------------------
    # =======================================
    dir_hmo3004 = "/media/shpakovkv/6ADA8899DA886365/WORK/2017/2017 05 12-19 ERG/HMO 3004"
    save_hmo3004_to_path = "/media/shpakovkv/6ADA8899DA886365/WORK/2017/2017 05 12-19 ERG/HMO3004_CSV"

    files_in_group_hmo3004 = 1                          # number of files of the HMO3004 (default 1)
    read_hmo3004 = False                                # read files and convert to csv
    max_min_log_hmo3004 = False                         # read converted csv and find max and min for all files
    log_cols_hmo3004 = [1, 3, 5, 7]                     # search max and min only for this columns (zero-based indexes)
    corr_cols_hmo3004 = [0.272, 0.951, 1.138, 1.592]    # correction multiplyer for each column (for max-min log)
    log_filename_hmo3004 = 'max_min_hmo3004.log'    # max-min filename postfix (containing dir name will be the prefix)

    # ==================================================================================================
    # ----------     LeCroy     ------------------------------------------------------------------------
    # =======================================
    dir_lecroy = "/media/shpakovkv/6ADA8899DA886365/WORK/2017/2017 05 12-19 ERG/LeCroy"
    save_lecroy_to_path = "/media/shpakovkv/6ADA8899DA886365/WORK/2017/2017 05 12-19 ERG/LeCroy_CSV"

    files_in_group_lecroy = 3                           # number of files of the LeCroy (default 4)
    read_LeCroy = False                                 # read files and convert to csv
    max_min_log_lecroy = False                          # read converted csv and find max and min for all files
    log_cols_lecroy = [1, 3, 5]                         # search max and min only for this columns (zero-based indexes)
    corr_cols_lecroy = [0.62791, 4.554, 4.0]            # correction multiplyer for each column (for max-min log)
    log_filename_lecroy = 'max_min_lecroy.log'      # max-min filename postfix (containing dir name will be the prefix)

    # ==================================================================================================
    # ----------     PROCESS     -----------------------------------------------------------------------
    # =======================================
    # READ WFM from DPO7054
    if read_dpo7054:
        if os.path.isdir(dir_dpo7054):
            dir_dpo7054 = os.path.abspath(dir_dpo7054)
            save_lecroy_to_path = os.path.abspath(save_dpo7054_to_path)
            for old_path in get_subdir_list(dir_dpo7054):
                new_path = add_new_dir_to_path(old_path, save_dpo7054_to_path)
                combine_wfm_to_csv(old_path, new_path, files_in_group=files_in_group_dpo7054)
        else:
            print("Path " + dir_dpo7054 + "\n does not exist!")
    # FIND MAX & MIN from DPO7054
    if max_min_log_dpo7054:
        if os.path.isdir(save_dpo7054_to_path):
            save_dpo7054_to_path = os.path.abspath(save_dpo7054_to_path)
            for path in get_subdir_list(save_dpo7054_to_path):
                get_max_min_from_dir(path, log_cols_dpo7054, corr_cols_dpo7054,
                                     ext='.CSV',
                                     log_file_name=log_filename_dpo7054)
        else:
            print("Path " + save_dpo7054_to_path + "\n does not exist!")

    # ====================================================================================
    # READ CSV from Tektronix TDS2024C
    if read_tds2024:
        if os.path.isdir(dir_tds2024):
            dir_tds2024 = os.path.abspath(dir_tds2024)
            save_tds2024_to_path = os.path.abspath(save_tds2024_to_path)
            for old_path in get_subdir_list(dir_tds2024):
                new_path = add_new_dir_to_path(old_path, save_tds2024_to_path)
                combine_tds2024c_csv(old_path, new_path, files_in_group=files_in_group_tds2024)
        else:
            print("Path " + save_tds2024_to_path + "\n does not exist!")
    # FIND MAX & MIN from TDS2024C
    if max_min_log_tds2024:
        if os.path.isdir(save_tds2024_to_path):
            save_tds2024_to_path = os.path.abspath(save_tds2024_to_path)
            for path in get_subdir_list(save_tds2024_to_path):
                get_max_min_from_dir(path, log_cols_tds2024, corr_cols_tds2024,
                                     ext='.CSV',
                                     log_file_name=log_filename_tds2024)
        else:
            print("Path " + save_tds2024_to_path + "\n does not exist!")

    # =====================================================================================
    # READ CSV from Rohde&Schwarz HMO 3004
    if read_hmo3004:
        if os.path.isdir(dir_hmo3004):
            dir_hmo3004 = os.path.abspath(dir_hmo3004)
            save_hmo3004_to_path = os.path.abspath(save_hmo3004_to_path)
            for old_path in get_subdir_list(dir_hmo3004):
                new_path = add_new_dir_to_path(old_path, save_hmo3004_to_path)
                combine_hmo3004_csv(old_path, new_path)
        else:
            print("Path " + dir_hmo3004 + "\n does not exist!")
    # FIND MAX & MIN from HMO3004
    if max_min_log_hmo3004:
        if os.path.isdir(save_hmo3004_to_path):
            save_hmo3004_to_path = os.path.abspath(save_hmo3004_to_path)
            for path in get_subdir_list(save_hmo3004_to_path):
                get_max_min_from_dir(path, log_cols_hmo3004, corr_cols_hmo3004,
                                     ext='.CSV',
                                     log_file_name=log_filename_hmo3004)
        else:
            print("Path " + save_hmo3004_to_path + "\n does not exist!")

    # =====================================================================================
    # READ TXT from LeCroy
    if read_LeCroy:
        if os.path.isdir(dir_lecroy):
            dir_lecroy = os.path.abspath(dir_lecroy)
            save_hmo3004_to_path = os.path.abspath(save_hmo3004_to_path)
            for old_path in get_subdir_list(dir_lecroy):
                new_path = add_new_dir_to_path(old_path, save_lecroy_to_path)
                combine_lecroy_csv(old_path, new_path, files_in_group=files_in_group_lecroy)
        else:
            print("Path " + dir_lecroy + "\n does not exist!")

    # FIND MAX & MIN from LeCroy
    if max_min_log_lecroy:
        if os.path.isdir(save_lecroy_to_path):
            save_lecroy_to_path = os.path.abspath(save_lecroy_to_path)
            for path in get_subdir_list(save_lecroy_to_path):
                get_max_min_from_dir(path, log_cols_lecroy, corr_cols_lecroy,
                                     ext='.CSV',
                                     log_file_name=log_filename_lecroy)
        else:
            print("Path " + save_lecroy_to_path + "\n does not exist!")

    # =======================================================================================
    # FIND DUPLACATES
    compare_files_in_subfolders(save_dpo7054_to_path)
    compare_files_in_subfolders(save_tds2024_to_path)
    compare_files_in_subfolders(save_hmo3004_to_path)
    compare_files_in_subfolders(save_lecroy_to_path)
    # import matplotlib.pyplot as mplt
    # data = np.genfromtxt('F0009CH1.CSV', delimiter=",", skip_header=18, usecols=tuple())
    # mplt.plot(data[:,0], data[:,1])
    # mplt.show()
