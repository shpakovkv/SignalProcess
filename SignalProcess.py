# Python 2.7
from __future__ import print_function, with_statement

import re
import os

import numpy as np
from scipy.signal import savgol_filter
import colorsys

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
    def __init__(self, in_x=None, in_y=None):
        self.data = np.empty([0, 2], dtype=float, order='F')
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
    def __init__(self, input_data=None, curve_labels=None):
        # EMPTY INSTANCE
        # number of curves:
        self.count = 0
        # list of curves data (SingleCurve instances):
        self.curves = []
        # dict with curve labels as keys and curve indexes as values:
        self.labels = dict()

        # FILL WITH VALUES
        if input_data is not None:
            self.append(input_data, curve_labels)

    def append(self, input_data, curve_labels=None):
        # appends new SingleCurves to the self.curves list
        data = np.array(input_data, dtype=float, order='F')
        self.check_input(data)
        if data.shape[1] % 2 != 0:
            new_curves = data.shape[1] - 1
            for curve_idx in range(0, new_curves):
                self.curves.append(SingleCurve(data[:, 0],
                                               data[:, curve_idx + 1]))
                self.count += 1
                if curve_labels:
                    self.labels[curve_labels[curve_idx]] = self.count - 1
                else:
                    self.labels[str(self.count - 1)] = self.count - 1
        else:
            for curve_idx in range(0, data.shape[1], 2):
                self.curves.append(SingleCurve(data[:, curve_idx],
                                               data[:, curve_idx + 1]))
                self.count += 1
                if curve_labels:
                    self.labels[curve_labels[curve_idx]] = self.count - 1
                else:
                    self.labels[str(self.count - 1)] = self.count - 1

    def check_input(self, data, curve_labels=None):
        '''
        Checks the correctness of input data.
        Raises exception if data check fails.
        
        data -- ndarray with data to add to a SignlsData instance
        curve_labels -- the list of labels for new curves
        '''
        # CHECK INPUT DATA
        if np.ndim(data) != 2:
            raise ValueError("Input array must have 2 dimensions.")
        if data.shape[1] % 2 != 0:
            if verbose:
                print("The input array has an odd number of columns! "
                      "The first column is considered as X column, "
                      "and the rest as Y columns.")
        if curve_labels:
            if not isinstance(curve_labels, list):
                raise TypeError("Variable curve_labels must be "
                                "an instance of the list class.")
            if data.shape[1] // 2 != len(curve_labels):
                raise IndexError("Number of curves (pair of "
                                 "time-value columns) in data "
                                 "and number of labels must be the same.")
            for label in curve_labels:
                if label in self.labels:
                    raise ValueError("Label \"{}\" is already exist."
                                     "".format(label))

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
        return self.curves[self.labels[label]]

    def get_label(self, idx):
        # return label of the SingelCurve by index
        for key, value in self.labels.items():
            if value == idx:
                return key

    def get_idx(self, label):
        # returns index of the SingelCurve by label
        if label in self.labels:
            return self.labels[label]

    def time(self, curve):
        return self.curves[curve].get_x()

    def value(self, curve):
        return self.curves[curve].get_y()


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


def read_signals(file_list, start=0, step=1, points=-1):
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
    for filename in file_list:
        if verbose:
            print("Loading \"{}\"".format(filename))
        data.append(load_from_file(filename, start, step, points))
        if verbose:
            print()
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


def smooth_voltage(x, y, x_multiplier=1):
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
        raise ValueError("Unsupported value for curve front level ({}) at "
                         "--offset_by_curve_level parameters\n"
                         "Only float values are allowed.".format(params[0]))
    try:
        time_multiplier = float(params[2])
    except ValueError:
        raise ValueError("Unsupported value for time multiplier ({}) at "
                         "--offset_by_curve_level parameters\n"
                         "Only float values are allowed.".format(params[0]))
    return idx, level, time_multiplier


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
                                           coeffs_count, need_count))
            elif coeffs_count > need_count:
                raise IndexError("Too many {} values.\n"
                                 "Expected ({}), got ({})."
                                 "".format(coeff_names[idx],
                                           coeffs_count, need_count))

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
        #                              curves_labels_DEBUG, " ns", show=False))
        print(pretty_print_nums([new_delay[idx] for idx in
                                 range(1, len(new_delay), 2)],
                                curves_labels, show=False))
    return new_delay


def plot_multiple_curve(curve_list, peaks=None, xlim=None,
                        show=False, save_as=None):
    '''Draws one or more curves on one graph.
    Additionally draws peaks on the underlying layer 
    of the same graph, if the peaks exists.
    The color of the curves iterates though the list 'corols'.
    
    curve_list  -- the list of SingleCurve instances
                   or SingleCurve instance
    peaks       -- the list or SinglePeak instances
    xlim        -- the tuple with the left and the right X bounds
    show        -- (bool) show/hide the graph window
    save_as     -- filename (full path) to save the plot as .png
                   Does not save by default.
    '''
    from matplotlib import pyplot as plt
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
            color = '#9595aa'
        # print("||  COLOR == {} ===================".format(color))
        plt.plot(curve.time, curve.val, '-',
                 color=color, linewidth=1)

    if peaks is not None:
        peak_x = [peak.time for peak in peaks if peak is not None]
        peak_y = [peak.val for peak in peaks if peak is not None]
        plt.scatter(peak_x, peak_y, s=50, edgecolors='#ff7f0e', facecolors='none', linewidths=2)
        plt.scatter(peak_x, peak_y, s=90, edgecolors='#dd3328', facecolors='none', linewidths=2)
        # plt.scatter(peak_x, peak_y, s=40, edgecolors='#ff5511', facecolors='none', linewidths=2)
        # plt.scatter(peak_x, peak_y, s=100, edgecolors='#133cac', facecolors='none', linewidths=2)
        # plt.scatter(peak_x, peak_y, s=160, edgecolors='#62e200', facecolors='none', linewidths=1.5)
        plt.scatter(peak_x, peak_y, s=150, edgecolors='none', facecolors='#133cac', linewidths=1.5, marker='x')
    if save_as is not None:
        # print("Saveing " + save_as)
        plt.savefig(save_as)
    if show:
        plt.show()
    plt.close('all')


def apdate_delays_by_curve_front(data, offset_by_curve_params,
                                 multiplier, delay, smooth=True,
                                 show_plot=False):
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
    time_multiplier = offset_by_curve_params[2]

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
        smoothed_y = smooth_voltage(data.curves[curve_idx].get_x(),
                                    data.curves[curve_idx].get_y(),
                                    time_multiplier)
        smoothed_curve = SingleCurve(data.curves[curve_idx].get_x(),
                                     smoothed_y)
    else:
        print("Smoothing is turned OFF.")
        smoothed_curve = data.curves[curve_idx]
    front_x, front_y = pp.find_curve_front(smoothed_curve,
                                           level, polarity)
    if show_plot:
        if front_x:
            front_point = [pp.SinglePeak(front_x, front_y, 0)]
        else:
            front_point = None
        plot_multiple_curve([data.curves[curve_idx], smoothed_curve],
                            front_point, show=True)

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
    return new_delay


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
                             'in data files. Latin letters, numbers and '
                             'underscore are allowed only.'
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
                        metavar=('CURVE_IDX', 'LEVEL', 'MULTIPLIER'),
                        nargs=3,
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
    verbose = not args.silent

    # input directory and files check
    if args.src_dir:
        args.src_dir = args.src_dir.strip()
        assert os.path.isdir(args.src_dir), \
            "Can not find directory {}".format(args.src_dir)
    if args.files:
        grouped_files = check_file_list(args.src_dir, args.files)
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

    # raw check y_auto_zero parameters (types)
    if args.y_auto_zero:
        args.y_auto_zero = global_check_y_auto_zero_params(args.y_auto_zero)

    # raw check multiplier and delay
    if args.multiplier is not None and args.delay is not None:
        assert len(args.multiplier) == len(args.delay), \
            "The number of multipliers ({}) is not equal to the number of " \
            "delays ({}).".format(len(args.multiplier), len(args.delay))

    # MAIN LOOP
    for shot_idx, file_list in enumerate(grouped_files):
        data = read_signals(file_list, start=0, step=1, points=-1)
        if verbose:
            print("The number of curves = {}".format(data.count))

        # checks multipliers, delays and labels numbers
        check_coeffs_number(data.count * 2, ["multiplier", "delay"],
                            args.multiplier, args.delay)
        check_coeffs_number(data.count, ["label"],  args.labels)

        # check y_zero_offset parameters (if idx out of range)
        check_y_auto_zero_params(data, args.y_auto_zero)

        # updates delays with accordance to Y zero offset
        args.delay = apdate_delays_by_y_auto_zero(data, args.y_auto_zero,
                                                  args.multiplier, args.delay,
                                                  verbose=True)

        # check offset_by_voltage parameters (if idx out of range)
        assert args.offset_by_front[0] < data.count, \
            "Index ({}) is out of range in --y-offset-by-curve-level " \
            "parameters.\nCurves count = {}" \
            "".format(args.offset_by_front[0], data.count)

        # updates delay values with accordance to voltage front
        args.delay = apdate_delays_by_curve_front(data, args.offset_by_front,
                                                  args.multiplier, args.delay,
                                                  smooth=True, show_plot=True)

        # multiplier and delay
        data = multiplier_and_delay(data, args.multiplier, args.delay)

        # DEBUG
        from only_for_tests import plot_peaks_all
        # plot_peaks_all(data, None, [0, 4, 8, 11], show=True)

        from only_for_tests import plot_single_curve
        # plot_single_curve(data.curves[12], show=True)
        # plot_single_curve(data.curves[11], show=True)

    # TODO: process fake files (not recorded)
    # TODO: file duplicates check
    # TODO: user interactive input commands?
    # TODO: partial import (start, step, points)
    # TODO: add labels to plots
    # TODO: add labels to CSV files
    # TODO: add log file with multipliers and delays applied to data saved to CSV

    print(args.y_auto_zero)
    print()
    print(args)
