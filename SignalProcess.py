# Python 2.7
from __future__ import print_function, with_statement

import re
import os

import colorsys
import numpy as np
from scipy.signal import savgol_filter
from matplotlib import pyplot as plt
from matplotlib import axes

import wfm_reader_lite as wfm
import PeakProcess as pp


verbose = True


class ColorRange():
    '''Color code iterator. Generates contrast colors.
    Returns the hexadecimal RGB color code (for example '#ffaa00')
    '''
    def __init__(self, start_hue=0, hue_step=140, min_hue_diff=20,
                  saturation=(90, 90, 60),
                  luminosity=(55, 30, 50)):
        self.start = start_hue
        self.step = hue_step
        self.window = min_hue_diff
        self.hue_range = 360
        self.s_list = saturation
        self.l_list = luminosity

    def too_close(self, val_list, val, window=10):
        for item in val_list:
            if val < item + window and val > item - window:
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

    def hsl_to_rgb_code(self, hue, saturation, luminosity):
        hue = hue / 360.0
        saturation = saturation / 100.0
        luminosity = luminosity / 100.0
        # print([hue, saturation, luminosity])
        rgb_float = colorsys.hls_to_rgb(hue, luminosity, saturation)
        # print(rgb_float)
        rgb = [val * 255 for val in rgb_float]
        rgb_int = [int(round(val * 255)) for val in rgb_float]
        # print("{:.2f}, {:.2f}, {:.2f}".format(*rgb), end=" == ")
        rgb_code = "#{:02x}{:02x}{:02x}".format(*rgb_int)
        return rgb_code

    def __iter__(self):
        while True:
            offset = 0
            for sat, lum in zip(self.s_list, self.l_list):
                last = self.start + offset
                offset += 10
                yield self.hsl_to_rgb_code(*[0, sat, lum])
                for i in range(0, self.calc_count()):
                    new_hue = last + self.step
                    if new_hue > 360:
                        new_hue -= 360
                    last = new_hue
                    yield self.hsl_to_rgb_code(*[new_hue, sat, lum])


class SingleCurve:
    def __init__(self, in_x=None, in_y=None,
                 label=None, unit=None, time_unit=None):
        self.data = np.empty([0, 2], dtype=float, order='F')
        self.label = label
        self.unit = unit
        self.time_unit = time_unit
        if in_x is not None and in_y is not None:
            self.append(in_x, in_y)

    def append(self, x_in, y_in):
        # INPUT DATA CHECK
        x_data = x_in
        y_data = y_in
        if (not isinstance(x_data, np.ndarray) or
                not isinstance(y_data, np.ndarray)):
            raise TypeError("Input time and value arrays must be "
                            "instances of numpy.ndarray class.")
        if len(x_data) != len(y_data):
            raise IndexError("Input time and value arrays must "
                             "have same length.")
        # self.points = len(x_data)

        # dimension check
        if np.ndim(x_data) == 1:
            x_data = np.expand_dims(x_in, axis=1)
        if np.ndim(y_data) == 1:
            y_data = np.expand_dims(y_in, axis=1)

        if x_data.shape[1] != 1 or y_data.shape[1] != 1:
            raise ValueError("Input time and value arrays must "
                             "have 1 column and any number of rows.")

        start_idx = None
        stop_idx = None
        for i in range(len(x_data) - 1, 0, -1):
            if not np.isnan(x_data[i]) and not np.isnan(y_data[i]):
                stop_idx = i
                break
        for i in range(0, stop_idx + 1):
            if not np.isnan(x_data[i]) and not np.isnan(y_data[i]):
                start_idx = i
                break
        if start_idx is None or stop_idx is None:
            raise ValueError("Can not append array of empty "
                             "values to SingleCurve's data.")
        # CONVERT TO NDARRAY
        temp = np.append(x_data[start_idx: stop_idx + 1],
                         y_data[start_idx: stop_idx + 1],
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
    def __init__(self, input_data=None, labels=None,
                 units=None, time_units=None):
        # EMPTY INSTANCE
        # number of curves:
        self.count = 0
        # dict of curves data (SingleCurve instances):
        self.curves = {}
        # dict with curve labels as keys and curve indexes as values:
        self.label_to_idx = dict()
        self.idx_to_label = dict()

        # FILL WITH VALUES
        if input_data is not None:
            self.append(input_data, labels, units, time_units)

    def append(self, input_data, labels=None, units=None, time_unit=None):
        # appends one or more new SingleCurves to the self.curves list
        # and updates the corresponding self parameters
        data = np.array(input_data, dtype=float, order='F')
        self.check_input(data)
        if data.shape[1] % 2 != 0:
            multiple_X_columns = False
            add_count = data.shape[1] - 1
        else:
            multiple_X_columns = True
            add_count = data.shape[1] // 2

        for curve_idx in range(0, add_count):
            if labels:
                current_label = labels[curve_idx]
            else:
                current_label = "Curve[{}]".format(curve_idx)
            if units:
                current_unit = units[curve_idx]
            else:
                current_unit = "a.u."
            if time_unit:
                current_time_unit = time_unit
            else:
                current_time_unit = "a.u."

            if multiple_X_columns:
                self.curves[self.count] = \
                    SingleCurve(data[:, curve_idx * 2],
                                data[:, curve_idx * 2 + 1],
                                current_label, current_unit,
                                current_time_unit)
            else:
                self.curves[self.count] = \
                    SingleCurve(data[:, 0], data[:, curve_idx + 1],
                                current_label, current_unit,
                                current_time_unit)
            self.curves[current_label] = self.curves[self.count]
            self.count += 1

            self.label_to_idx[current_label] = curve_idx
            self.idx_to_label[curve_idx] = current_label


    def check_input(self, data, new_labels=None, new_units=None):
        '''
        Checks the correctness of input data.
        Raises exception if data check fails.
        
        data        -- ndarray with data to add to a SignlsData instance
        new_labels  -- the list of labels for the new curves
        new_units   -- the list of units for the new curves
        '''
        # CHECK INPUT DATA
        if np.ndim(data) != 2:
            raise ValueError("Input array must have 2 dimensions.")
        if data.shape[1] % 2 != 0:
            if verbose:
                print("The input array has an odd number of columns! "
                      "The first column is considered as X column, "
                      "and the rest as Y columns.")

        if data.shape[1] % 2 != 0:
            # multiple_X_columns = False
            curves_count = data.shape[1] - 1
        else:
            # multiple_X_columns = True
            curves_count = data.shape[1] // 2
        if new_labels:
            if not isinstance(new_labels, list):
                raise TypeError("The variable 'labels' must be "
                                "an instance of the list class.")
            if len(new_labels) != curves_count:
                raise IndexError("The number of curves ({}) (the pair "
                                 "of time-value columns) in the appending "
                                 "data and tne number of new labels ({}) "
                                 "must be the same."
                                 "".format(curves_count, len(new_labels)))
            for label in new_labels:
                if label in self.label_to_idx.keys():
                    raise ValueError("Label \"{}\" is already exist."
                                     "".format(label))
        if new_units:
            if not isinstance(new_labels, list):
                raise TypeError("The variable 'units' must be "
                                "an instance of the list class.")
            if len(new_labels) != curves_count:
                raise IndexError("The number of curves ({}) (the pair "
                                 "of time-value columns) in the appending "
                                 "data and tne number of new units ({}) "
                                 "must be the same."
                                 "".format(curves_count, len(new_labels)))

    def get_array(self):
        '''Returns all curves data as united 2D array
        short curve arrays are supplemented with 
        required amount of rows (filled with 'nan')
        
        return -- 2d ndarray 
        '''
        return align_and_append_ndarray(*[curve.data for
                                          curve in self.curves])

    def by_label(self, label):
        # returns SingleCurve by name
        return self.curves[self.label_to_idx[label]]

    def get_label(self, idx):
        # return label of the SingelCurve by index
        for key, value in self.label_to_idx.items():
            if value == idx:
                return key

    def get_idx(self, label):
        # returns index of the SingelCurve by label
        if label in self.label_to_idx:
            return self.label_to_idx[label]

    def time(self, curve):
        return self.curves[curve].get_x()

    def value(self, curve):
        return self.curves[curve].get_y()
    def label(self, curve):
        return self.curves[curve].label
    def unit(self, curve):
        return self.curves[curve].unit
    def time_unit(self, curve):
        return self.curves[curve].time_unit


def get_subdir_list(path):
    '''Returns list of subdirectories for the given directory.
    
    path -- path to the target directory 
    '''
    return [os.path.join(path, x) for x in os.listdir(path)
            if os.path.isdir(os.path.join(path, x))]


def get_file_list_by_ext(path, ext_list, sort=True):
    '''Returns a list of files (with the specified extensions) 
    contained in the folder (path).
    Each element of the returned list is a full path to the file
    
    path -- target directory.
    ext -- the list of file extensions or one extension (str).
    sort -- by default, the list of results is sorted
            in lexicographical order.
    '''
    if isinstance(ext_list, str):
        ext_list = [ext_list]
    file_list = [os.path.join(path, x) for x in os.listdir(path)
                 if os.path.isfile(os.path.join(path, x)) and
                 any(x.upper().endswith(ext.upper()) for
                 ext in ext_list)]
    if sort:
        file_list.sort()
    return file_list


def add_zeros_to_filename(full_path, count):
    '''Adds zeros to number in filename.
    Example: f("shot22.csv", 4) => "shot0022.csv"
    
    full_path -- filename or full path to file
    count -- number of digits
    '''
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


def read_signals(file_list, start=0, step=1, points=-1,
                 labels=None, units=None, time_unit=None):
    '''
    Function returns one SignalsData object filled with
    data from files in file_list.
    Do not forget to sort the list of files
    for the correct order of curves.
    
    Can handle different type of files.
    Supported formats: CSV, WFM
    For more information about the supported formats, 
    see the help of the function load_from_file().

    file_list -- list of full paths or 1 path (str)
    start -- start read data points from this index
    step -- read data points with this step
    points -- data points to read (-1 == all)
    '''

    # check inputs
    if isinstance(file_list, str):
        file_list = [file_list]
    elif not isinstance(file_list, list):
        raise TypeError("file_list must be an instance of str or list of str")

    # read
    data = SignalsData()
    current_count = 0
    for filename in file_list:
        if verbose:
            print("Loading \"{}\"".format(filename))
        new_data = load_from_file(filename, start, step, points)
        if new_data.shape[1] % 2 != 0:
            # multiple_X_columns = False
            add_count = new_data.shape[1] - 1
        else:
            # multiple_X_columns = True
            add_count = new_data.shape[1] // 2
        current_labels = None
        current_units = None
        if labels:
            current_labels = labels[current_count: current_count + add_count]
        if units:
            current_units = units[current_count: current_count + add_count]

        data.append(new_data, current_labels, current_units, time_unit)

        if verbose:
            print()
        current_count += add_count
    return data


def origin_to_csv(readed_lines):
    '''
    Replaces ',' with '.' and then ';' with ',' in the
    given list of lines of a .csv file.

    Use it for OriginPro ascii export format with ';' as delimiter 
    and ',' as decimal separator.

    read_lines -- array of a .csv file lines (array of string)
    '''
    import re
    converted = [re.sub(r',', '.', line) for line in readed_lines]
    converted = [re.sub(r';', ',', line) for line in converted]
    # if verbose:
    #     print("OriginPro ascii format detected.")
    return converted


def valid_cols(read_lines, skip_header, delimiter=','):
    '''
    Returns the list of valid (non empty) columns
    for the given list of lines of a .csv file.

    read_lines -- array of a .csv file lines (array of string)
    skip_header -- the number of header lines
    delimiter -- .csv data delimiter
    :return: 
    '''
    cols = read_lines[skip_header].strip().split(delimiter)
    valid_col_list = []
    for idx, val in enumerate(cols):
        if val != '':
            valid_col_list.append(idx)
    # if verbose and len(cols) != len(valid_col_list):
    #     print("\nThe columns ", end = '')
    #     print(", ".join([str(idx) for idx in range(0, len(cols))
    #                      if idx not in valid_col_list]), end=' ')
    #     print(" is empty!", end='')
    return tuple(valid_col_list)


def get_csv_headers(read_lines, delimiter=',', except_list=('', 'nan')):
    '''
    Returns the number of header lines 
    for the given list of lines of a .csv file.

    read_lines -- array of a .csv file lines (array of string)
    delimiter -- .csv data delimiter
    except_list -- the list of special strings, that can not be converted
                   to numeric with the float() function, but the csv reader
                   may considered it as a valid value.
    '''
    cols_count = len(read_lines[-1].strip().split(delimiter))
    headers = 0
    idx = 0
    lines = len(read_lines)

    # column number difference
    while (len(read_lines[-1].strip().split(delimiter)) !=
               cols_count and idx < lines and idx < lines):
        idx += 1
        headers += 1

    # non numeric value check
    for idx in range(headers, lines):
        try:
            [float(val) for val in read_lines[idx].strip().split(delimiter) if
             val not in except_list]
        except ValueError:
            headers += 1
        else:
            break
    if verbose:
        print("Header lines = {}   |   ".format(headers), end="")
        # print("Columns count = {}  |  ".format(cols_count), end="")
    return headers


def load_from_file(filename, start=0, step=1, points=-1):
    '''
    Return ndarray instance filled with data from csv or wfm file.
    
    filename -- file name (full/relative path)
    start -- start read data points from this index
    step -- read data points with this step
    points -- data points to read (-1 == all)
     '''

    valid_delimiters = [',', ';', ' ', '\t']

    import csv

    if filename[-3:].upper() != 'WFM':
        with open(filename, "r") as datafile:
            dialect = csv.Sniffer().sniff(datafile.read(2048))
            datafile.seek(0)
            if dialect.delimiter not in valid_delimiters:
                dialect.delimiter = ','
            if verbose:
                print("Delimiter = \"{}\"   |   ".format(dialect.delimiter),
                      end="")
            text_data = datafile.readlines()
            if dialect.delimiter == ";":
                text_data = origin_to_csv(text_data)
                dialect.delimiter = ','
            skip_header = get_csv_headers(text_data)
            usecols = valid_cols(text_data, skip_header,
                                 delimiter=dialect.delimiter)
            if verbose:
                print("Valid columns = {}   |   ".format(usecols), end="")
            data = np.genfromtxt(text_data,
                                 delimiter=dialect.delimiter,
                                 skip_header=skip_header,
                                 usecols=usecols)

    else:
        data = wfm.read_wfm_group([filename], start_index=start,
                                  number_of_points=points,
                                  read_step=step)
    if verbose:
        if data.shape[1] % 2 ==0:
            curves_count = data.shape[1] // 2
        else:
            curves_count = data.shape[1] - 1
        print("Curves count = {}".format(curves_count))
    return data


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
    file_list = get_file_list_by_ext(path, ext, sort=True)
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


def multiplier_and_delay(data, multiplier, delay):
    '''Returns the modified data.
    Each column of the data is first multiplied by 
    the corresponding multiplier value and
    then the corresponding delay value is subtracted from it.
    
    data -- an instance of the SignalsData class OR 2D numpy.ndarray
    multiplier -- the list of multipliers for each columns in the input data.
    delay -- the list of delays (subtrahend) for each columns in the input data.
    '''
    if multiplier is None and delay is None:
        return data

    if isinstance(data, np.ndarray):
        row_number = data.shape[0]
        col_number = data.shape[1]

        if not multiplier:
            multiplier = [1 for _ in range(col_number)]
        if not delay:
            delay = [0 for _ in range(col_number)]
        # check_coeffs_number(col_number, ["multiplier", "delay"],
        #                     multiplier, delay)
        for col_idx in range(col_number):
            for row_idx in range(row_number):
                data[row_idx][col_idx] = (data[row_idx][col_idx] *
                                          multiplier[col_idx] -
                                          delay[col_idx])
        return data
    elif isinstance(data, SignalsData):
        if not multiplier:
            multiplier = [1 for _ in range(data.count * 2)]
        if not delay:
            delay = [0 for _ in range(data.count * 2)]

        # check_coeffs_number(data.count * 2, ["multiplier", "delay"],
        #                     multiplier, delay)
        for curve_idx in range(data.count):
            col_idx = curve_idx * 2             # index of time-column of current curve
            data.curves[curve_idx].data = multiplier_and_delay(data.curves[curve_idx].data,
                                                               multiplier[col_idx:col_idx + 2],
                                                               delay[col_idx:col_idx + 2])
        return data
    else:
        raise TypeError("Data must be an instance of numpy.ndarray or SignalsData.")


def multiplier_and_delay_peak(peaks, multiplier, delay, curve_idx):
    '''Returns the modified peaks data.
    Each time and amplitude of each peak is first multiplied by 
    the corresponding multiplier value and
    then the corresponding delay value is subtracted from it.
    
    peaks       -- the list of the SinglePeak instance
    multiplier  -- the list of multipliers for each columns 
                   in the SignalsData.
    delay       -- the list of delays (subtrahend) 
                   for each columns in the SignalsData.
    curve_idx   -- the index of the curve to which the peaks belong
    '''
    for peak in peaks:
        if not isinstance(peak, pp.SinglePeak):
            raise ValueError("'peaks' must be a list of "
                             "the SinglePeak instance.")
    if not multiplier and not delay:
        raise ValueError("Specify multipliers or delays.")
    if not multiplier:
        multiplier = [1 for _ in range(len(delay))]
    if not delay:
        delay = [0 for _ in range(multiplier)]
    if curve_idx >= len(multiplier):
        raise IndexError("The curve index({}) is greater than the length "
                         "({}) of multiplier/delay."
                         "".format(curve_idx, len(multiplier)))
    corr_peaks = []
    time_mult = multiplier[curve_idx * 2]
    time_del = delay[curve_idx * 2]
    amp_mult = multiplier[curve_idx * 2 + 1]
    amp_del = delay[curve_idx * 2 + 1]
    for peak in peaks:
        corr_peaks.append(pp.SinglePeak(peak.time * time_mult - time_del,
                                        peak.val * amp_mult - amp_del,
                                        peak.idx, peak.sqr_l, peak.sqr_r))
    return corr_peaks


def OLD_smooth_voltage(x, y, x_multiplier=1):
    '''This function returns smoothed copy of 'y'.
    Optimized for voltage pulse of ERG installation.
    
    x -- 1D numpy.ndarray of time points (in seconds by default)
    y -- 1D numpy.ndarray value points
    x_multiplier -- multiplier applied to time data 
        (default 1 for x in seconds)
    
    return -- smoothed curve (1D numpy.ndarray)
    '''
    # 3 is optimal polyorder value for speed and accuracy
    poly_order = 3

    # value 101 is optimal for 1 ns (1e9 multiplier) resolution
    # of voltage waveform for 25 kV charging voltage of ERG installation
    window_len = 101

    # calc time_step and converts to nanoseconds
    time_step = (x[1] - x[0]) * 1e9 / x_multiplier
    window_len = int(window_len / time_step)
    if len(y) < window_len:
        window_len = len(y) - 1
    if window_len % 2 == 0:
        # window must be even number
        window_len += 1
    if window_len < 5:
        # lowest possible value
        window_len = 5

    if len(y) >= 5:
        print("WINDOW LEN = {}  |  POLY ORDER = {}".format(window_len, poly_order))
        y_smoothed = savgol_filter(y, window_len, poly_order)
        return y_smoothed
    # too short array to be processed
    return y


def smooth_voltage(y_data, window=101, poly_order=3):
    '''This function returns smoothed copy of 'y_data'.

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
    '''

    # calc time_step and converts to nanoseconds
    if len(y_data) < window:
        window = len(y_data) - 1
    if window % 2 == 0:
        # window must be even number
        window += 1
    if window < 5:
        # lowest possible value
        window = 5

    if len(y_data) >= 5:
        # print("WINDOW LEN = {}  |  POLY ORDER = {}".format(window, poly_order))
        y_smoothed = savgol_filter(y_data, window, poly_order)
        return y_smoothed

    # too short array to be processed
    return y_data


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
            s = delimiter.join([value_format % data[row, col] for
                                col in range(data.shape[1])]) + "\n"
            s = re.sub(r'nan', '', s)
            lines.append(s)
        fid.writelines(lines)


def align_and_append_ndarray(*args):
    '''Returns 2D numpy.ndarray containing all input 2D numpy.ndarrays.
    If input arrays have different number of rows, 
    fills missing values with 'nan'.
    
    args -- 2d ndarrays
    '''
    # CHECK TYPE & LENGTH
    for arr in args:
        if not isinstance(arr, np.ndarray):
            raise TypeError("Input arrays must be instances "
                            "of numpy.ndarray class.")
        if np.ndim(arr) != 2:
            raise ValueError("Input arrays must have 2 dimensions.")

    # ALIGN & APPEND
    max_rows = max([arr.shape[0] for arr in args])
    data = np.empty(shape=(max_rows, 0), dtype=float, order='F')
    for arr in args:
        miss_rows = max_rows - arr.shape[0]
        nan_arr = np.empty(shape=(miss_rows, arr.shape[1]),
                           dtype=float, order='F')
        nan_arr = nan_arr * np.nan

        aligned_arr = np.append(arr, nan_arr, axis=0)
        data = np.append(data, aligned_arr, axis=1)
    return data


def numbering_parser(group):
    '''
    Finds serial number (substring) in the file name string.

    group -- list of files (each file must corresponds 
             to different shot)

    return -- (start, end), where: 
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
                if (num != name_idx and
                            numbers[num][match_idx]['num'] ==
                            names[name_idx][start:end]):
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
        'end'   -- index of the last digit of the found number in the string
        'num'   -- string representation of the number
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


def get_grouped_file_list(dir, ext_list, group_size, sorted_by_ch=False):
    '''Return the list of files grouped by shots.
    
    dir             -- the directory containing the target files
    group_size      -- the size of the groups (number of 
                       files for each shot)
    sorted_by_ch    -- this options tells the program that the files
                       are sorted by the oscilloscope/channel
                       (firstly) and by the shot number (secondly).
                       By default, the program considers that the
                       files are sorted by the shot number (firstly)
                       and by the oscilloscope/channel (secondly).
    '''
    assert dir, ("Specify the directory (-d) containing the "
                          "data files. See help for more details.")
    file_list = get_file_list_by_ext(dir, ext_list, sort=True)
    assert len(file_list) % group_size == 0, \
        ("The number of .csv files ({}) in the specified folder "
         "is not a multiple of group size ({})."
         "".format(len(file_list), group_size))
    grouped_files = []
    if sorted_by_ch:
        shots_count = len(file_list) // group_size
        for shot in range(shots_count):
            grouped_files.append([file_list[idx] for idx in
                               range(shot, len(file_list), shots_count)])
    else:
        for idx in range(0, len(file_list), group_size):
            grouped_files.append(file_list[idx: idx + group_size])
    return grouped_files


def global_check_front_params(params):
    '''Parses user input of offset_by_curve_level parameter.
    Returns (idx, level), where:
        idx -- [int] the index of the curve, inputted by user
        level -- [float] the curve front level (amplitude),
                 which will be found and set as time zero

    params -- user input (list of two strings)
    '''
    try:
        idx = int(params[0])
        if idx < 0:
            raise ValueError
    except ValueError:
        raise ValueError("Unsupported value for curve index ({}) at "
                         "--offset_by_curve_level parameters\n"
                         "Only positive integer values are allowed."
                         "".format(params[0]))
    try:
        level = float(params[1])
    except ValueError:
        raise ValueError("Unsupported value for curve front "
                         "level ({}) at "
                         "--offset_by_curve_level parameters\n"
                         "Only float values are allowed."
                         "".format(params[0]))
    try:
        window = int(params[2])
        if window < 5:
            raise ValueError
    except ValueError:
        raise ValueError("Unsupported value for filter window "
                         "length ({}) at "
                         "--offset_by_curve_level parameters\n"
                         "Only integer values >=5 are allowed."
                         "".format(params[0]))
    try:
        poly_order = int(params[3])
        if poly_order < 1:
            raise ValueError
    except ValueError:
        raise ValueError("Unsupported value for filter polynomial "
                         "order ({}) at "
                         "--offset_by_curve_level parameters\n"
                         "Only integer values >=1 are allowed."
                         "".format(params[0]))
    return idx, level, window, poly_order


def check_file_list(dir, grouped_files):
    '''Checks the list of file names, inputted by user.
    The file names must be grouped by shots.
    Raises an exception if any test fails.
    
    Returns the list of full paths, grouped by shots. 
    
    grouped_files -- the list of the files, grouped by shots 
    '''
    group_size = len(grouped_files[0])
    for shot_idx, shot_files in enumerate(grouped_files):
        assert len(shot_files) == group_size, \
            ("The number of files in each shot must be the same.\n"
             "Shot[1] = {} files ({})\nShot[{}] = {} files ({})"
             "".format(group_size, ", ".join(grouped_files[0]),
                       shot_idx + 1, len(shot_files), ", ".join(shot_files))
             )
        for file_idx, filename in enumerate(shot_files):
            full_path = os.path.join(dir, filename.strip())
            grouped_files[shot_idx][file_idx] = full_path
            assert os.path.isfile(full_path), \
                "Can not find file \"{}\"".format(full_path)
    return grouped_files

def global_check_y_auto_zero_params(y_auto_zero_params):
    '''The function checks y zero offset parameters inputted by user,
    converts string values to numeric and returns it.
    
    The structure of input/output parameters list:
        [
            [curve_idx, bg_start, bg_stop]
            [curve_idx, bg_start, bg_stop]
            ...etc.
        ]
    
    y_auto_zero_params -- the list of --y-offset parameters inputted by user:
    '''
    # 'CURVE_IDX', 'BG_START', 'BG_STOP'
    output = []
    for params in y_auto_zero_params:
        try:
            idx = int(params[0])
            if idx < 0:
                raise ValueError
        except ValueError:
            raise ValueError("Unsupported value for curve index ({}) in "
                             "--y-offset parameter\n"
                             "Only positive integer values are allowed."
                             "".format(params[0]))
        try:
            start = float(params[1])
            stop = float(params[2])
        except ValueError:
            raise ValueError("Unsupported value for background start or stop "
                             "time ({}, {}) in --y-offset parameter\n"
                             "Only float values are allowed."
                             "".format(params[1], params[2]))
        output.append([idx, start, stop])
    return output


def check_y_auto_zero_params(data, y_auto_zero_params):
    '''Checks if the curve indexes of y auto zero parameters list
     is out of range. Raises an exception if true. 

    data                -- SignalsData instance
    y_auto_zero_params  -- y auto zero parameters (see --y-auto-zero
                           argument's hep for more detail)
    '''
    for params in y_auto_zero_params:
        assert params[0] < data.count, ("Index ({}) is out of range in "
                                        "--y-auto-zero parameters."
                                        "".format(params[0]))


def check_coeffs_number(need_count, coeff_names, *coeffs):
    '''Checks the needed and the actual number of the coefficients.
    Raises an exception if they are not equal.
    
    need_count  -- the number of needed coefficients
    coeff_names -- the list of the coefficient names
    coeffs      -- the list of coefficients to check 
                   (multiplier, delay, etc.)
    '''
    for idx, coeff_list in enumerate(coeffs):
        if coeff_list is not None:
            coeffs_count = len(coeff_list)
            if coeffs_count < need_count:
                raise IndexError("Not enough {} values.\n"
                                 "Expected ({}), got ({})."
                                 "".format(coeff_names[idx],
                                           need_count, coeffs_count))
            elif coeffs_count > need_count:
                raise IndexError("Too many {} values.\n"
                                 "Expected ({}), got ({})."
                                 "".format(coeff_names[idx],
                                           need_count, coeffs_count))

def y_zero_offset(curve, start_x, stop_x):
    '''
    Returns the Y zero level offset value.
    Use it for zero level correction before PeakProcess.

    curve               -- SingleCurve instance
    start_x and stop_x  -- define the limits of the 
                           X interval where Y is filled with noise only.
    '''
    if start_x < curve.time[0]:
        start_x = curve.time[0]
    if stop_x > curve.time[-1]:
        stop_x = curve.time[-1]
    assert stop_x > start_x, \
        ("Error! start_x value ({}) must be lower than stop_x({}) value."
        "".format(start_x, stop_x))
    if start_x > curve.time[-1] or stop_x < curve.time[0]:
        return 0
    start_idx = pp.find_nearest_idx(curve.time, start_x, side='right')
    stop_idx = pp.find_nearest_idx(curve.time, stop_x, side='left')
    amp_sum = 0.0
    DEBUG_VAL = {}
    for val in curve.val[start_idx:stop_idx + 1]:
        amp_sum += val
        DEBUG_VAL[int(val * 100)] = (DEBUG_VAL.setdefault(int(val * 100), 0) +
                                     1)
    # DEBUG
    # print("\n".join("{}: {} : {}".format(float(key) / 100, count, count * float(key) / 100) for
    #                  key, count in DEBUG_VAL.items()))
    # print("AMP_SUM = {}  |  stop_idx={}  |  start_idx={}  | result = {}"
    #       "".format(amp_sum, stop_idx, start_idx, amp_sum / (stop_idx - start_idx + 1)))
    return amp_sum / (stop_idx - start_idx + 1)


def y_zero_offset_all(signals_data, curves_list, start_stop_tuples):
    '''Return delays list for all columns in SignalsData stored 
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
    '''Prints template 's' filled with values from 'nums',
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

def check_labels(labels):
    '''Checks if all the labels contains only 
    Latin letters, numbers and underscore.
    Raises an exception if the test fails.
    
    labels -- the list of labels
    '''
    import re
    for label in labels:
        if re.search(r'[^\w]+', label):
            raise ValueError("{}\nLatin letters, numbers and underscore"
                             " are allowed only.".format(label))


def global_check_plot(args, name, allow_all=False):
    '''Checks the values of a parameter,
    converts it to integers and returns it.
    Returns [-1] if only 'all' is entered (if allow_all==True)
    
    args -- the list of str parameters to be converted to integers
    name -- the name of the parameter (needed for error message)
    allow_all -- turns on/off support of the value 'all'
    '''
    error_text = ("Unsupported curve index ({val}) at {name} parameter."
                  "\nOnly positive integer values ")
    if allow_all:
        error_text += "and string 'all' "
    error_text += "are allowed."

    if len(args) == 1 and args[0].upper() == 'ALL' and allow_all:
        args = [-1]
    else:
        for idx, _ in enumerate(args):
            try:
                args[idx] = int(args[idx])
                assert args[idx] >= 0, ""
            except (ValueError, AssertionError):
                raise ValueError(error_text.format(val=args[idx], name=name))
    return args


def check_plot_param(args, curves_count, param_name):
    '''Checks if any index from args list is greater than the curves count.
    
    args            -- the list of curve indexes
    curves_count    -- the count of curves
    param_name      -- the name of the parameter through which 
                       the value was entered
    '''
    error_text = ("The curve index ({idx}) from ({name}) parameters "
                  "is greater than the number of curves ({count}).")
    for idx in args:
        assert idx < curves_count, \
            (error_text.format(idx=idx, name=param_name, count=curves_count))


def raw_y_auto_zero(params, multiplier, delay):
    '''Returns raw values for y_auto_zero parameters.
    'Raw values' are values before applying the multiplier.
    
    params      -- y auto zero parameters (see --y-auto-zero
                   argument's hep for more detail)
    multiplier  -- the list of multipliers for all the curves
                   in the SignalsData  
    delay       -- the list of delays for all the curves
                   in the SignalsData 
    '''
    raw_params=[]
    for item_idx, item in enumerate(params):
        curve_idx = item[0]
        start_x = (item[1] + delay[curve_idx * 2]) / multiplier[curve_idx * 2]
        stop_x = (item[2] + delay[curve_idx * 2]) / multiplier[curve_idx * 2]
        raw_params.append([curve_idx, start_x, stop_x])
    return raw_params


def apdate_delays_by_y_auto_zero(data, y_auto_zero_params,
                                 multiplier, delay, verbose=True):
    '''The function analyzes the mean amplitude in 
    the selected X range of the selected curves. 
    The Y zero offset values obtained are added to 
    the corresponding items of the original list of delays.
    The new list of delays is returned.
    
    NOTE: adjusting the delay values before applying 
    the multipliers and the delays to the input data, 
    decreases the execution time of the code.
    
    data                -- SignalsData instance
    y_auto_zero_params  -- y auto zero parameters (see --y-auto-zero
                           argument's hep for more detail)
    multiplier          -- the list of multipliers for all the curves
                           in the SignalsData
    delay               -- the list of delays for all the curves
                           in the SignalsData
    verbose             -- show/hide information during function execution
    '''
    new_delay = delay[:]
    curves_list = [item[0] for item in y_auto_zero_params]
    start_stop_tuples = [(item[1], item[2]) for item in
                         raw_y_auto_zero(y_auto_zero_params,
                                         args.multiplier,
                                         args.delay)]
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


def calc_ylim(time, y, time_bounds=None, reserve=0.1):
    '''Returns (min_y, max_y) tuple with y axis bounds.
    The axis boundaries are calculated in such a way 
    as to show all points of the curve with a indent 
    (default = 10% of the span of the curve) from 
    top and bottom.
    
    time        -- the array of time points
    y           -- the array of amplitude points
    time_bounds -- the tuple/list with the left and 
                   the right X bounds in X units.
    reserve     -- the indent size (the fraction of the curve's range)
    '''
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


def plot_multiplot(data, peak_data, curves_list,
                   xlim=None, amp_unit=None,
                   time_unit=None, title=None):
    '''Plots subplots for all curves with index in curve_list.
    Optional: plots peaks.
    Subplots are located one under the other. 
    
    data -- the SignalsData instance
    peak_data -- the list of list of peaks (SinglePeak instance) 
                 of curves with index in curve_list
                 peak_data[0] == list of peaks for data.curves[curves_list[0]]
                 peak_data[1] == list of peaks for data.curves[curves_list[1]]
                 etc.
    curves_list -- the list of curve indexes in data to be plotted
    xlim        -- the tuple/list with the left and 
                   the right X bounds in X units.
    show        -- (bool) show/hide the graph window
    save_as     -- filename (full path) to save the plot as .png
                   Does not save by default.
    amp_unit    -- the unit of Y scale for all subplots.
                   If not specified, the curve.unit parameter will be used
    time_unit   -- the unit of time scale for all subplots.
                   If not specified, the time_unit parameter of 
                   the first curve in curves_list will be used
    title       -- the main title of the figure.
    '''
    plt.close('all')
    fig, axes = plt.subplots(len(curves_list), 1, sharex='all')
    # # an old color scheme
    # colors = ['#1f22dd', '#ff7f0e', '#9467bd', '#d62728', '#2ca02c',
    #           '#8c564b', '#17becf', '#bcbd22', '#e377c2']
    if title is not None:
        fig.suptitle(title)

    for wf in range(len(curves_list)):
        # plot curve
        axes[wf].plot(data.time(curves_list[wf]),
                      data.value(curves_list[wf]),
                      '-', color='#9999aa', linewidth=0.5)
        axes[wf].tick_params(direction='in', top=True, right=True)

        # set bounds
        if xlim is not None:
            axes[wf].set_xlim(xlim)
            axes[wf].set_ylim(calc_ylim(data.time(curves_list[wf]),
                                        data.value(curves_list[wf]),
                                        xlim, reserve=0.1))
        # y label (units only)
        if amp_unit is None:
            amp_unit = data.curves[curves_list[wf]].unit
        axes[wf].set_ylabel(amp_unit, size=10, rotation='horizontal')

        # subplot title
        amp_label = data.curves[curves_list[wf]].label
        if data.curves[curves_list[wf]].unit:
            amp_label += ", " + data.curves[curves_list[wf]].unit
        axes[wf].text(0.99, 0.01, amp_label, verticalalignment='bottom',
                      horizontalalignment='right',
                      transform=axes[wf].transAxes, size=8)

        # Time axis label
        if wf == len(curves_list) - 1:
            time_label = "Time"
            if time_unit:
                time_label += ", " + time_unit
            else:
                time_label += ", " + data.curves[curves_list[wf]].time_unit
            axes[wf].set_xlabel(time_label, size=10)
        axes[wf].tick_params(labelsize=8)

        # plot peaks scatter
        if peak_data is not None:
            for pk in peak_data[wf]:
                color_iter = iter(ColorRange())
                color = next(color_iter)
                if pk is not None:
                    axes[wf].scatter([pk.time], [pk.val], s=20,
                                     edgecolors=color, facecolors='none',
                                     linewidths=1.5)
                    axes[wf].scatter([pk.time], [pk.val], s=50,
                                     edgecolors='none', facecolors=color,
                                     linewidths=1.5, marker='x')
    fig.subplots_adjust(hspace=0)


def plot_multiple_curve(curve_list, peaks=None,
                        xlim=None, amp_unit=None,
                        time_unit=None, title=None):
    '''Draws one or more curves on one graph.
    Additionally draws peaks on the underlying layer 
    of the same graph, if the peaks exists.
    The color of the curves iterates though the ColorRange iterator.
    
    NOTE: after the function execution you need to show() or save() pyplot
          otherwise the figure will not be saved or shown
    
    curve_list  -- the list of SingleCurve instances
                   or SingleCurve instance
    peaks       -- the list or SinglePeak instances
    title       -- the title for plot
    amp_unit    -- the units for curves Y scale
    time_unit   -- the unit for time scale
    xlim        -- the tuple with the left and the right X bounds
    '''
    plt.close('all')
    if xlim is not None:
        plt.xlim(xlim)
    color_iter = iter(ColorRange())
    if isinstance(curve_list, SingleCurve):
        curve_list = [curve_list]
    for curve in curve_list:
        if len(curve_list) > 1:
            color = next(color_iter)
        else:
            color = '#9999aa'
        # print("||  COLOR == {} ===================".format(color))
        plt.plot(curve.time, curve.val, '-',
                 color=color, linewidth=1)
        axes_obj = plt.gca()
        axes_obj.tick_params(direction='in', top=True, right=True)

    time_label = "Time"
    amp_label = "Amplitude"
    if time_unit is not None:
        time_label += ", " + time_unit
    elif curve_list[0].time_unit:
        time_label += ", " + curve_list[0].time_unit

    if amp_unit is not None:
        amp_label += ", " + amp_unit
    elif all(curve_list[0].unit == curve.unit for curve in curve_list):
        amp_label += ", " + curve_list[0].unit

    if title is not None:
        plt.title(title)
    elif len(curve_list) > 1:
        # LEGEND
        pass
    else:
        plt.title(curve_list[0].label)
    plt.xlabel(time_label)
    plt.ylabel(amp_label)

    if peaks is not None:
        peak_x = [peak.time for peak in peaks if peak is not None]
        peak_y = [peak.val for peak in peaks if peak is not None]
        plt.scatter(peak_x, peak_y, s=50, edgecolors='#ff7f0e',
                    facecolors='none', linewidths=2)
        plt.scatter(peak_x, peak_y, s=90, edgecolors='#dd3328',
                    facecolors='none', linewidths=2)
        plt.scatter(peak_x, peak_y, s=150, edgecolors='none',
                    facecolors='#133cac', linewidths=1.5, marker='x')
        # plt.scatter(peak_x, peak_y, s=40, edgecolors='#ff5511',
        #           facecolors='none', linewidths=2)
        # plt.scatter(peak_x, peak_y, s=100, edgecolors='#133cac',
        #           facecolors='none', linewidths=2)
        # plt.scatter(peak_x, peak_y, s=160, edgecolors='#62e200',
        #           facecolors='none', linewidths=1.5)
    # if save_as is not None:
    #     if not os.path.isdir(os.path.dirname(save_as)):
    #         os.makedirs(os.path.dirname(save_as))
    #     # print("Saveing " + save_as)
    #     plt.savefig(save_as, dpi=400)
    # if show:
    #     plt.show()
    # plt.close('all')


def update_delays_by_curve_front(data, offset_by_curve_params,
                                 multiplier, delay, smooth=True,
                                 plot=False):
    '''This function finds the time point of the selected curve front on level
    'level', recalculates input delays of all time columns (odd 
    elements of the input delay list) to make that time point be zero.

    NOTE: the curve is not multiplied yet, so the 'level' 
    value must be in raw data units

    data                    -- an instance of SingleCurve
    offset_by_curve_params  -- level of voltage front
    multiplier              -- the list of multipliers for all the curves
                               in the SignalsData
    delay                   -- the list of delays for all the curves
                               in the SignalsData
    smooth                  -- smooth curve before front detection
                               NOTE: the smooth process is optimized for 
                               Voltage pulse of ERG installation 
                               (~500 ns width)

    return -- list of recalculated delays (floats)
    '''

    curve_idx = offset_by_curve_params[0]
    level = offset_by_curve_params[1]
    window = offset_by_curve_params[2]
    poly_order = offset_by_curve_params[3]

    polarity = pp.check_polarity(data.curves[curve_idx])

    if pp.is_pos(polarity):
        level = abs(level)
    else:
        level = -abs(level)

    print("Time offset by curve front process.")
    print("Searching curve[{}] front at level = {}".format(curve_idx, level))

    level_raw = ((level + delay[curve_idx * 2 + 1]) /
                 multiplier[curve_idx * 2 + 1])
    # smooth curve to improve accuracy of front search
    if smooth:
        print("Smoothing is turned ON.")
        smoothed_y = smooth_voltage(data.curves[curve_idx].get_y(),
                                    window, poly_order)
        smoothed_curve = SingleCurve(data.curves[curve_idx].get_x(),
                                     smoothed_y)
    else:
        print("Smoothing is turned OFF.")
        smoothed_curve = data.curves[curve_idx]
    front_x, front_y = pp.find_curve_front(smoothed_curve,
                                           level, polarity)
    if plot:
        if front_x:
            front_point = [pp.SinglePeak(front_x, front_y, 0)]
        else:
            front_point = None
        plot_title = "Curve[{}]\nRaw data before applying multipliers and " \
                     "delays".format(data.curves[curve_idx].label)
        plot_multiple_curve([data.curves[curve_idx], smoothed_curve],
                            front_point, amp_unit="a.u.",
                            time_unit="a.u.", title=plot_title)

    new_delay = delay[:]
    if front_x:
        # add_to_log("Raw_voltage_front_level = " + str(level_raw))

        # apply multiplier and compensate voltage delay
        time_offset = (front_x * multiplier[curve_idx * 2] -
                       delay[curve_idx * 2])
        print("Time offset by curve front = " + str(time_offset))
        print()
        for idx in range(0, len(delay), 2):
            new_delay[idx] += time_offset
    return new_delay, pp.SinglePeak(front_x, front_y, 0)


def check_param_path(path, param_name):
    '''Path checker. 
    Verifies the syntactic correctness of the entered path.
    Does not check the existence of the path.
    Returns abs version of the path. 
    Recursive creates all directories from the path 
    if they are not exists. 
    
    path        -- the path to check
    param_name  -- the key on which the path was entered.
                   Needed for correct error message.
    '''
    try:
        path = os.path.abspath(path)
        if not os.path.isdir(path):
            os.makedirs(path)
    except:
        raise ValueError("Unsupported path ({path}) was entered via key "
                         "{param}.".format(path=path, param=param_name))
    return path


def global_check_labels(labels):
    '''Checks if any label contains non alphanumeric character.
     Underscore are allowed.
     Returns true if all the labels passed the test.
    
    labels -- the list of labels for graph process
    '''
    for label in labels:
        if re.search(r'[\W]', label):
            return False
    return True


# ============================================================================
# --------------   MAIN    ----------------------------------------
# ======================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(prog='Process Signals',
                                     description='', epilog='',
                                     fromfile_prefix_chars='@')

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
    group.add_argument('-e', '--ext', '--extension',
                        action='store',
                        nargs='+',
                        metavar='EXT',
                        dest='ext_list',
                        help='specify one or more (space separated) '
                             'extensions of the files with data.'
                        )
    parser.add_argument('--labels',
                        action='store',
                        metavar='LABEL',
                        nargs='+',
                        dest='labels',
                        help='specify the labels for all the curves '
                             'in the data files. Latin letters, numbers and '
                             'underscore are allowed only.\n'
                             'Using of non latin letters and special '
                             'characters are allowed but may cause '
                             'an error.'
                        )
    parser.add_argument('--units',
                        action='store',
                        metavar='UNIT',
                        nargs='+',
                        dest='units',
                        help='specify the units for all the curves '
                             'in the data files.'
                        )
    parser.add_argument('--time-unit',
                        action='store',
                        metavar='UNIT',
                        dest='time_unit',
                        help='specify the unit for time scale.'
                        )
    parser.add_argument('-g', '--grouped-by',
                        action='store',
                        type=int,
                        metavar='GROUPED_BY',
                        dest='group_size',
                        default=1,
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
    parser.add_argument('--silent',
                        action='store',
                        dest='silent',
                        help='enables the silent mode, in which only '
                             'error messages and input prompts '
                             'are displayed.')
    parser.add_argument('--multiplier',
                        action='store',
                        type=float,
                        metavar='MULT',
                        nargs='+',
                        dest='multiplier',
                        default=None,
                        help='the list of multipliers for each data columns.'
                             '\nNOTE: you must enter values for all the '
                             'columns in data file(s). Two columns (X and Y) '
                             'are expected for each curve.'
                        )
    parser.add_argument('--delay',
                        action='store',
                        type=float,
                        metavar='DELAY',
                        nargs='+',
                        dest='delay',
                        default=None,
                        help='the list of delays (subtrahend) for each data '
                             'columns.\n'
                             'NOTE: the data is first multiplied by '
                             'a multiplier and then the delay '
                             'is subtracted from them.'
                        )
    parser.add_argument('--offset-by-curve_level',
                        action='store',
                        metavar=('IDX', 'LEVEL', 'WIN', 'ORDER'),
                        nargs=4,
                        dest='offset_by_front',
                        default=None,
                        help=''
                        )
    parser.add_argument('--y-auto-zero',
                        action='append',
                        metavar=('CURVE_IDX', 'BG_START', 'BG_STOP'),
                        nargs=3,
                        dest='y_auto_zero',
                        help='auto zero level correction for curve. '
                             'Specify curve_idx, bg_start, bg_stop '
                             'for the curve whose zero level you want '
                             'to offset.\n'
                             'Where curve_idx is the zero-based index '
                             'of the curve; and the time interval '
                             '[bg_start: bg_stop] does not contain signals '
                             'or random bursts (background interval).\n'
                             'You can specify as many --y-auto-zero '
                             'parameters as you want.'
                        )

    # output settings --------------------------------------------------------
    parser.add_argument('-s', '--save',
                        action='store_true',
                        dest='save_data',
                        help='saves data files.\n'
                             'NOTE: If in the input data one '
                             'shot corresponds to one file and output '
                             'directory is not specified, input files '
                             'will be overwritten!'
                        )
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
    parser.add_argument('-p', '--plot',
                        action='store',
                        nargs='+',
                        metavar='CURVE',
                        dest='plot',
                        help='specify the indexes of curves to be plotted '
                             '(or \'all\' to plot all the curves). Each '
                             'curve from this list will be plotted separately.\n'
                             'You can specify --p-save for saveing them '
                             'as .png files.'
                        )
    parser.add_argument('--p-save', '--save-plots-to',
                        action='store',
                        dest='plot_dir',
                        metavar='PLOT_DIR',
                        help='specify the directory after the flag. Each '
                             'curve from the --plot list will be plotted '
                             'and saved separately as .png file.'
                        )
    parser.add_argument('-m', '--multiplot',
                        action='append',
                        dest='multiplot',
                        nargs='+',
                        help='specify the indexes of curves to be plotted '
                             'as subplots one under the other. You may use as many '
                             '\'-m\' flags (with different lists of curves)'
                             ' as you want.'
                        )
    parser.add_argument('--mp-save', '--save-multiplot-as',
                        action='store',
                        dest='multiplot_dir',
                        metavar='MULTIPLOT_DIR',
                        help='specify the directory after the flag. The '
                             'subplots will be saved as .png file for all '
                             'the --multiplot lists.'
                        )
    parser.add_argument('--mp-hide', '--multiplot-hide',
                        action='store_true',
                        dest='mp_hide',
                        help='if the flag is specified the multiplots '
                             'will be saved (if the \'--mp-save\' flag was '
                             'specified as well) but not shown. This '
                             'option can reduce the runtime of the program.'
                        )
    parser.add_argument('--p-hide', '--plot-hide',
                        action='store_true',
                        dest='p_hide',
                        help='if the flag is specified the plots '
                             'will be saved (if the \'--p-save\' flag was '
                             'specified as well) but not shown. This '
                             'option can reduce the runtime of the program.'
                        )


    args = parser.parse_args()
    verbose = not args.silent

    # input directory and files check
    if args.src_dir:
        args.src_dir = args.src_dir.strip()
        assert os.path.isdir(args.src_dir), \
            "Can not find directory {}".format(args.src_dir)
    if args.files:
        grouped_files = check_file_list(args.src_dir, args.files)
        if not args.src_dir:
            args.src_dir = os.path.dirname(grouped_files[0][0])
    else:
        grouped_files = get_grouped_file_list(args.src_dir, args.ext_list,
                                           args.group_size, args.sorted_by_ch)

    # Now we have the list of files, grouped by shots:
    # grouped_files == [
    #                    ['shot001_osc01.wfm', 'shot001_osc02.csv', ...],
    #                    ['shot002_osc01.wfm', 'shot002_osc02.csv', ...],
    #                    ...etc.
    #                  ]

    # raw check offset_by_voltage parameters (types)
    if args.offset_by_front:
        args.offset_by_front = global_check_front_params(args.offset_by_front)

    # raw check labels
    # if args.labels:
    #     assert global_check_labels(args.labels), \
    #         "Label value error! Only latin letters, " \
    #         "numbers and underscore are allowed."

    args.plot_dir = check_param_path(args.plot_dir, '--p_save')
    args.multiplot_dir = check_param_path(args.multiplot_dir, '--mp-save')

    # raw check plot and multiplot
    if args.plot:
        args.plot = global_check_plot(args.plot, '--plot',
                                            allow_all=True)
    if args.multiplot:
        for idx, m_param in enumerate(args.multiplot):
            args.multiplot[idx] = global_check_plot(m_param, '--multiplot')

    # raw check y_auto_zero parameters (types)
    if args.y_auto_zero:
        args.y_auto_zero = global_check_y_auto_zero_params(args.y_auto_zero)

    # raw check multiplier and delay
    if args.multiplier is not None and args.delay is not None:
        assert len(args.multiplier) == len(args.delay), \
            "The number of multipliers ({}) is not equal to the number of " \
            "delays ({}).".format(len(args.multiplier), len(args.delay))

    number_start, number_end = numbering_parser(files[0] for
                                                files in grouped_files)
    labels_dict = {'labels': args.labels, 'units': args.units,
                   'time': args.time_unit}

    # MAIN LOOP
    for shot_idx, file_list in enumerate(grouped_files):
        shot_name = os.path.basename(file_list[0])[number_start:number_end]
        data = read_signals(file_list, start=0, step=1, points=-1,
                            labels=args.labels, units=args.units,
                            time_unit=args.time_unit)
        if verbose:
            print("The number of curves = {}".format(data.count))

        # checks multipliers, delays and labels numbers
        check_coeffs_number(data.count * 2, ["multiplier", "delay"],
                            args.multiplier, args.delay)
        check_coeffs_number(data.count, ["label", "unit"],
                            args.labels, args.units)

        # check y_zero_offset parameters (if idx is out of range)
        check_y_auto_zero_params(data, args.y_auto_zero)

        # updates delays with accordance to Y zero offset
        args.delay = apdate_delays_by_y_auto_zero(data, args.y_auto_zero,
                                                  args.multiplier, args.delay,
                                                  verbose=True)

        # check offset_by_voltage parameters (if idx is out of range)
        assert args.offset_by_front[0] < data.count, \
            "Index ({}) is out of range in --y-offset-by-curve-level " \
            "parameters.\nCurves count = {}" \
            "".format(args.offset_by_front[0], data.count)

        # updates delay values with accordance to voltage front
        raw_front_point = None
        if args.offset_by_front:
            front_plot_name = shot_name
            front_plot_name += ("_curve{:03d}_front_level_{:.3f}.png"
                               "".format(args.offset_by_front[0],
                                         args.offset_by_front[1]))
            front_plot_name = os.path.join(args.src_dir,
                                           'FrontBeforeTimeOffset',
                                           front_plot_name)
            print("FRONT PLOT NAME = {}".format(front_plot_name))
            args.delay, raw_front_point = \
                update_delays_by_curve_front(data,
                                             args.offset_by_front,
                                             args.multiplier,
                                             args.delay,
                                             smooth=True,
                                             plot=True)
            if not os.path.isdir(os.path.dirname(front_plot_name)):
                os.makedirs(os.path.dirname(front_plot_name))
            plt.savefig(front_plot_name, dpi=400)
            plt.show()
            plt.close('all')


        # multiplier and delay
        data = multiplier_and_delay(data, args.multiplier, args.delay)
        # if raw_front_point:
        #     front_point_list = \
        #         multiplier_and_delay_peak([raw_front_point],
        #                                   args.multiplier,
        #                                   args.delay,
        #                                   args.offset_by_front[0])
        #     plot_multiple_curve(data.curves)

        # plot_multiplot(data, None, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], show=True)

        # plot preview and save
        if args.plot:
            # args.plot = check_plot(args.plot)
            if args.plot[0] == -1:       # 'all'
                args.plot = list(range(0, data.count))
            else:
                check_plot_param(args.plot, data.count, '--plot')
            for idx in args.plot:
                plot_multiple_curve(data.curves[idx])

                if args.plot_dir is not None:
                    plot_name = ("{shot}_curve_{idx}_{label}.plot.png"
                                 "".format(shot=shot_name, idx=idx,
                                           label=data.curves[idx].label))
                    plot_path = os.path.join(args.plot_dir, plot_name)
                    plt.savefig(plot_path, dpi=400)
                    print("saved as {}".format(plot_path))
                if not args.p_hide:
                    plt.show()
                plt.close('all')

        # save plot

        # plot multi-plots
        if args.multiplot:
            for curve_list in args.multiplot:
                check_plot_param(curve_list, data.count, '--multiplot')
            for curve_list in args.multiplot:
                plot_multiplot(data, None, curve_list)
                if args.multiplot_dir is not None:
                    idx_list = "_".join(str(i) for i in sorted(curve_list))
                    mplot_name = ("{shot}_curves_{idx_list}.multiplot.png"
                                 "".format(shot=shot_name,
                                           idx_list=idx_list))
                    mplot_path = os.path.join(args.multiplot_dir, mplot_name)
                    plt.savefig(mplot_path, dpi=400)
                    print("saved as {}".format(mplot_path))
                if not args.mp_hide:
                    plt.show()
                plt.close('all')

        # save multiplot
        # multiplot file name constructor

        # save data

        # DEBUG
        # from only_for_tests import plot_multiplot
        # plot_multiplot(data, None, [0, 4, 8, 11], show=True)
        #
        # plot_multiple_curve(data.curves[12], show=True)
        # plot_multiple_curve(data.curves[0], show=True)

    # TODO: interactive offset_by_curve smooth process
    # TODO: process fake files (not recorded)
    # TODO: file duplicates check
    # TODO: user interactive input commands?
    # TODO: partial import (start, step, points)
    # TODO: add labels to CSV files
    # TODO: add log file with multipliers and delays applied to data saved to CSV

    print(args.y_auto_zero)
    print()
    print(args)
