# Python 2.7
from __future__ import print_function, with_statement

import re
import os
import colorsys
import sys
import math
import datetime

import numpy as np
from scipy.signal import savgol_filter
from matplotlib import pyplot as plt

import wfm_reader_lite as wfm
import PeakProcess as pp

verbose = True
global_log = ""


# ========================================
# -----     CLASSES     ------------------
# ========================================
class ColorRange:
    """Color code iterator. Generates contrast colors.
    Returns the hexadecimal RGB color code (for example '#ffaa00')
    """

    def __init__(self, start_hue=0, hue_step=140, min_hue_diff=20,
                 saturation=(90, 90, 60),
                 luminosity=(55, 30, 50)):
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

    def hsl_to_rgb_code(self, hue, saturation, luminosity):
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
            for sat, lum in zip(self.s_list, self.l_list):
                last = self.start + offset
                offset += 10
                yield self.hsl_to_rgb_code(0, sat, lum)
                for i in range(0, self.calc_count()):
                    new_hue = last + self.step
                    if new_hue > 360:
                        new_hue -= 360
                    last = new_hue
                    yield self.hsl_to_rgb_code(new_hue, sat, lum)


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
        # dict with indexes as keys and SingleCurve instances as values:
        self.curves = {}
        # dict with curve labels as keys and curve indexes as values:
        self.label_to_idx = dict()
        # dict with curve indexes as keys and curve labels as values:
        self.idx_to_label = dict()

        # FILL WITH VALUES
        if input_data is not None:
            self.append(input_data, labels, units, time_units)

    def append(self, input_data, labels=None, units=None, time_unit=None):
        """Separates the input ndarray data into SingleCurves
        and adds it to the self.curves dict.

        input_data -- ndarray with curves data. if the array has
                      an even number of columns, then the first
                      two curves will form the first curve,
                      the next two columns will result in the second
                      curve, etc.
                      If the array has an odd number of columns,
                      then the first column will be considered as
                      an X-column for all the curves added. And the
                      rest of the columns will be treated as Y-columns.
        labels     -- the list of labels for the added curves
        units      -- the list of labels for the added curves
        time_unit  -- the unit of the time scale
        """
        # appends one or more new SingleCurves to the self.curves list
        # and updates the corresponding self parameters
        new_data = np.array(input_data, dtype=float, order='F')
        self.check_input(new_data)
        if new_data.shape[1] % 2 != 0:
            multiple_x_columns = False
            add_count = new_data.shape[1] - 1
        else:
            multiple_x_columns = True
            add_count = new_data.shape[1] // 2

        for old_idx in range(0, add_count):
            # old_idx -- the curve index inside the adding ndarray
            # new_idx -- the curve index inside the output SignalsData
            new_idx = self.count
            if labels:
                current_label = labels[old_idx]
            else:
                current_label = "Curve[{}]".format(old_idx)
            if units:
                current_unit = units[old_idx]
            else:
                current_unit = "a.u."
            if time_unit:
                current_time_unit = time_unit
            else:
                current_time_unit = "a.u."

            if multiple_x_columns:
                self.curves[self.count] = \
                    SingleCurve(new_data[:, old_idx * 2],
                                new_data[:, old_idx * 2 + 1],
                                current_label, current_unit,
                                current_time_unit)
            else:
                self.curves[self.count] = \
                    SingleCurve(new_data[:, 0], new_data[:, old_idx + 1],
                                current_label, current_unit,
                                current_time_unit)
            self.curves[current_label] = self.curves[self.count]
            self.count += 1

            self.label_to_idx[current_label] = new_idx
            self.idx_to_label[new_idx] = current_label

    def check_input(self, data_ndarray, new_labels=None, new_units=None):
        """
        Checks the correctness of input data_ndarray.
        Raises exception if data_ndarray check fails.

        data_ndarray  -- ndarray with data to add to a SignlsData instance
        new_labels    -- the list of labels for the new curves
        new_units     -- the list of units for the new curves
        """
        # CHECK INPUT DATA
        if np.ndim(data_ndarray) != 2:
            raise ValueError("Input array must have 2 dimensions.")
        if data_ndarray.shape[1] % 2 != 0:
            if verbose:
                print("The input array has an odd number of columns! "
                      "The first column is considered as X column, "
                      "and the rest as Y columns.")

        if data_ndarray.shape[1] % 2 != 0:
            # multiple_X_columns = False
            curves_count = data_ndarray.shape[1] - 1
        else:
            # multiple_X_columns = True
            curves_count = data_ndarray.shape[1] // 2
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
        """Returns all curves data as united 2D array
        short curve arrays are supplemented with
        required amount of rows (filled with 'nan')

        return -- 2d ndarray
        """
        list_of_2d_arr = [self.curves[idx].data for
                          idx in sorted(self.idx_to_label.keys())]
        print("len(lest of 2D arr) = {}".format(len(list_of_2d_arr)))
        return align_and_append_ndarray(*list_of_2d_arr)

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


# ========================================
# -----     CHECKERS     -----------------
# ========================================
def check_partial_args(args):
    """The validator of partial import args.

    Returns converted args.

    args -- partial import args
    """
    if args is not None:
        start, step, count = args
        assert start >= 0, "The start point must be non-negative integer."
        assert step > 0, "The step of the import must be natural number."
        assert count > 0 or count == -1, \
            ("The count of the data points to import " 
             "must be natural number or -1 (means all).")
        return args
    return 0, 1, -1


def global_check_labels(labels):
    """Checks if any label contains non alphanumeric character.
     Underscore are allowed.
     Returns true if all the labels passed the test.

    labels -- the list of labels for graph process
    """
    for label in labels:
        if re.search(r'[\W]', label):
            return False
    return True


def check_header_label(s, count):
    """Checks if header line (from csv file) contains
    contains the required number of labels/units/etc.

    Returns the list of labels or None.

    s -- the header line (string) to check
    count -- the requered number of labels/units/etc.
    """
    labels = [word.strip() for word in s.split(",")]
    if len(labels) == count:
        return labels
    return None


def check_labels(labels):
    """Checks if all the labels contains only
    Latin letters, numbers and underscore.
    Raises an exception if the test fails.

    labels -- the list of labels
    """
    import re
    for label in labels:
        if re.search(r'[^\w]+', label):
            raise ValueError("{}\nLatin letters, numbers and underscore"
                             " are allowed only.".format(label))


def global_check_idx_list(args, name, allow_all=False):
    """Checks the values of a parameter,
    converts it to integers and returns it.
    Returns [-1] if only 'all' is entered (if allow_all==True)

    args -- the list of str parameters to be converted to integers
    name -- the name of the parameter (needed for error message)
    allow_all -- turns on/off support of the value 'all'
    """
    error_text = ("Unsupported curve index ({val}) at {name} parameter."
                  "\nOnly positive integer values ")
    if allow_all:
        error_text += "and string 'all' (without the qoutes) "
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


def check_coeffs_number(need_count, coeff_names, *coeffs):
    """Checks the needed and the actual number of the coefficients.
    Raises an exception if they are not equal.

    need_count  -- the number of needed coefficients
    coeff_names -- the list of the coefficient names
    coeffs      -- the list of coefficient lists to check
                   (multiplier, delay, etc.)
    """
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


def global_check_front_params(params, window=101, polyorder=3):
    """ Parses user input of offset_by_curve_level parameter.

    Returns (idx, level, window, order), where:
        idx -- [int] the index of the curve, inputted by user
        level -- [float] the curve front level (amplitude),
                 which will be found and set as time zero
        window -- The length of the smooth filter window
                  (i.e. the number of coefficients)
        order -- The order of the polynomial used to fit the samples
                 (for smooth filter)

    params      -- user input (list of 2, 3 or 4 strings)
    window      -- the default value of filter window (if not
                   specified by user)
    polyorder   -- the default value of polynomial order of the
                   smooth filter (if not specified by user)

    """
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

    if len(params) > 2:
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
    if len(params) == 4:
        try:
            polyorder = int(params[3])
            if polyorder < 1:
                raise ValueError
        except ValueError:
            raise ValueError("Unsupported value for filter polynomial "
                             "order ({}) at "
                             "--offset_by_curve_level parameters\n"
                             "Only integer values >=1 are allowed."
                             "".format(params[0]))
    return idx, level, window, polyorder


def check_file_list(dir, grouped_files):
    """Checks the list of file names, inputted by user.
    The file names must be grouped by shots.
    Raises an exception if any test fails.

    Returns the list of full paths, grouped by shots.

    grouped_files -- the list of the files, grouped by shots
    """
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
    """The function checks y zero offset parameters inputted by user,
    converts string values to numeric and returns it.

    The structure of input/output parameters list:
        [
            [curve_idx, bg_start, bg_stop]
            [curve_idx, bg_start, bg_stop]
            ...etc.
        ]

    y_auto_zero_params -- the list of --y-offset parameters inputted by user:
    """
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
    """Checks if the curve indexes of y auto zero parameters list
     is out of range. Raises an exception if true.

    data                -- SignalsData instance
    y_auto_zero_params  -- y auto zero parameters (see --y-auto-zero
                           argument's hep for more detail)
    """
    for params in y_auto_zero_params:
        assert params[0] < data.count, ("Index ({}) is out of range in "
                                        "--y-auto-zero parameters."
                                        "".format(params[0]))


def check_plot_param(args, curves_count, param_name):
    """Checks if any index from args list is greater than the curves count.

    args            -- the list of curve indexes
    curves_count    -- the count of curves
    param_name      -- the name of the parameter through which
                       the value was entered
    """
    error_text = ("The curve index ({idx}) from ({name}) parameters "
                  "is greater than the number of curves ({count}).")
    for idx in args:
        assert idx < curves_count, \
            (error_text.format(idx=idx, name=param_name, count=curves_count))


def check_param_path(path, param_name):
    """Path checker.
    Verifies the syntactic correctness of the entered path.
    Does not check the existence of the path.
    Returns abs version of the path.
    Recursive creates all directories from the path
    if they are not exists.

    path        -- the path to check
    param_name  -- the key on which the path was entered.
                   Needed for correct error message.
    """
    if path:
        try:
            path = os.path.abspath(path)
            if not os.path.isdir(path):
                os.makedirs(path)
        except:
            raise ValueError("Unsupported path ({path}) was entered via key "
                             "{param}.".format(path=path, param=param_name))
    return path


def check_idx_list(idx_list, max_idx, arg_name):
    """Checks if the curve index in idx_list is greater
    than max_idx. Raises an error if true.

    max_idx  -- the limit
    idx_list -- the list of indexes to check
    arg_name -- the name of the argument through which
                indexes were entered by the user.
                Needed for correct error message.
    """
    if isinstance(idx_list, (int, bool)):
        idx_list = [idx_list]
    for idx in idx_list:
        assert idx < max_idx, \
            "Index ({idx}) is out of range in {name} " \
            "parameters.\nMaximum index = {max}" \
            "".format(idx=idx, name=arg_name, max=max_idx)


def compare_2_files(first_file_name, second_file_name, lines=30):
    """Compares a number of first lines of two files
    return True if lines matches exactly.

    first_file_name  --  the full path to the first file
    second_file_name --  the full path to the second file
    lines            --  the number of lines to compare
    """
    with open(first_file_name, 'r') as file:
        with open(second_file_name, 'r') as file2:
            for idx in range(lines):
                if file.readline() != file2.readline():
                    return False
            return True


def compare_grouped_files(group_list, lines=30):
    """Compares files with corresponding indexes
    in neighboring groups if the number of files in
    these groups is the same.

    Returns a list of pairs of matching files.
    The files in each pair are sorted by
    modification time in ascending order.

    group_list -- the list of groups of files
    lines      -- the number of files lines to compare
    """
    match_list = []
    for group_1, group_2 in zip(group_list[0:-2], group_list[1:]):
        if len(group_1) == len(group_2):
            for file_1, file_2 in zip(group_1, group_2):
                if compare_2_files(file_1, file_2, lines=lines):
                    if os.path.getmtime(file_1) > os.path.getmtime(file_2):
                        match_list.append([file_2, file_1])
                    else:
                        match_list.append([file_1, file_2])
    return match_list


def print_duplicates(group_list, lines=30):
    """Compares files with corresponding indexes
    in neighboring groups if the number of files in
    these groups is the same.

    Prints the information about file duplicates.

    group_list -- the list of groups of files
    lines      -- the number of files lines to compare
    """
    duplicates = compare_grouped_files(group_list, lines=lines)
    if duplicates:
        print()
        print("Warning! Files duplicates are detected.")
        for pair in duplicates:
            if os.path.getmtime(pair[0]) != os.path.getmtime(pair[1]):
                print("File '{}' is a copy of '{}'"
                      "".format(os.path.basename(pair[1]),
                                os.path.basename(pair[0])))
            else:
                print("Files '{}' and '{}' is the same."
                      "".format(os.path.basename(pair[0]),
                                os.path.basename(pair[1])))


# ========================================
# -----    FILES HANDLING     ------------
# ========================================
def get_subdir_list(path):
    """Returns list of subdirectories for the given directory.

    path -- path to the target directory
    """
    return [os.path.join(path, x) for x in os.listdir(path)
            if os.path.isdir(os.path.join(path, x))]


def get_file_list_by_ext(path, ext_list, sort=True):
    """Returns a list of files (with the specified extensions)
    contained in the folder (path).
    Each element of the returned list is a full path to the file

    path -- target directory.
    ext  -- the list of file extensions or one extension (str).
    sort -- by default, the list of results is sorted
            in lexicographical order.
    """
    if isinstance(ext_list, str):
        ext_list = [ext_list]
    target_files = [os.path.join(path, x) for x in os.listdir(path)
                    if os.path.isfile(os.path.join(path, x)) and
                    any(x.upper().endswith(ext.upper()) for
                    ext in ext_list)]
    if sort:
        target_files.sort()
    return target_files


def add_zeros_to_filename(full_path, count):
    """Adds zeros to number in filename.
    Example: f("shot22.csv", 4) => "shot0022.csv"

    full_path -- filename or full path to file
    count -- number of digits
    """
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


def numbering_parser(group):
    """
    Finds serial number (substring) in the file name string.

    group -- list of files (each file must corresponds
             to different shot)

    return -- (start, end), where:
        start -- index of the first digit of the serial number
                 in a file name
        end -- index of the last digit of the serial number
    """
    names = []
    for raw in group:
        names.append(os.path.basename(raw))
    assert all(len(name) for name in names), \
        "Error! All file names must have the same length"
    if len(group) == 1:
        return 0, len(names[0])

    numbers = []
    # for idx in range(2):
    #     numbers.append(parse_filename(names[idx]))
    for name in names:
        numbers.append(parse_filename(name))

    # numbers[filename_idx][number_idx]{}
    num_count = len(numbers[0])
    if len(numbers) == 1:
        return numbers[0][0]['start'], numbers[0][0]['end']

    for match_idx in range(num_count):
        unique = True
        for num in range(len(names)):
            for name_idx in range(len(names)):
                start = numbers[num][match_idx]['start']
                end = numbers[num][match_idx]['end']
                if num != name_idx and (numbers[num][match_idx]['num'] ==
                                            names[name_idx][start:end]):
                    unique = False
                    break
            if not unique:
                break
        if unique:
            return (numbers[0][match_idx]['start'],
                    numbers[0][match_idx]['end'])
    return 0, len(names[0])


def parse_filename(name):
    """
    Finds substrings with numbers in the input string.

    name -- string with numbers.

    return -- list of dicts. [{...}, {...}, ...]
        where each dict contains info about one founded number:
        'start' -- index of the first digit of the found number in the string
        'end'   -- index of the last digit of the found number in the string
        'num'   -- string representation of the number
    """
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
    """Return the list of files grouped by shots.

    dir             -- the directory containing the target files
    group_size      -- the size of the groups (number of
                       files for each shot)
    sorted_by_ch    -- this options tells the program that the files
                       are sorted by the oscilloscope/channel
                       (firstly) and by the shot number (secondly).
                       By default, the program considers that the
                       files are sorted by the shot number (firstly)
                       and by the oscilloscope/channel (secondly).
    """
    assert dir, ("Specify the directory (-d) containing the "
                 "data files. See help for more details.")
    file_list = get_file_list_by_ext(dir, ext_list, sort=True)
    assert len(file_list) % group_size == 0, \
        ("The number of data files ({}) in the specified folder "
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


def get_front_plot_name(args, save_to, shot_name):
    """Generates plot name for offset_by_curve_front
    function's plot.
    Returns full path for the plot.

    args        -- offset_by_curve_front args
    save_to     -- the directory to save data files
    shot_name   -- the name of current shot (number of shot)
    """
    front_plot_name = str(shot_name)
    front_plot_name += ("_curve{:03d}_front_level_{:.3f}.png"
                        "".format(args[0],
                                  args[1]))
    front_plot_name = os.path.join(save_to,
                                   'Offset_By_Front',
                                   front_plot_name)
    return front_plot_name


def trim_ext(filename, ext_list):
    """Return the filename without extension,
    if the extension is in the ext_list.

    filename -- the file name
    ext_list -- the list of the extensions to check
    """
    filename = os.path.basename(filename)
    filename.strip()
    if isinstance(ext_list, str):
        ext_list = [ext_list]
    if not ext_list:
        ext_list = ['.CSV', '.WFM', '.DAT', '.TXT']
        # have bug: len(ext_list[idx]) == 3
    for ext in ext_list:
        if filename.upper().endswith(ext.upper()):
            ext_len = len(ext)
            if not ext.startswith("."):
                ext_len += 1
            return filename[0: - ext_len]
    return filename


# ========================================
# -----    READ/SAVE DATA     ------------
# ========================================
def read_signals(file_list, start=0, step=1, points=-1,
                 labels=None, units=None, time_unit=None):
    """Function returns one SignalsData object filled with
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
    """

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
        new_data, new_header = load_from_file(filename, start,
                                              step, points, h_lines=3)
        if new_data.shape[1] % 2 != 0:
            # multiple_X_columns = False
            add_count = new_data.shape[1] - 1
        else:
            # multiple_X_columns = True
            add_count = new_data.shape[1] // 2

        current_labels = None
        current_units = None
        current_time_unit = None

        if labels:
            # user input
            current_labels = labels[current_count: current_count + add_count]
        elif new_header:
            # from file
            current_labels = check_header_label(new_header[0], add_count)

        if units:
            # user input
            current_units = units[current_count: current_count + add_count]
        elif new_header:
            # from file
            current_units = check_header_label(new_header[1], add_count)

        if time_unit:
            current_time_unit = time_unit
        elif new_header:
            current_time_unit = check_header_label(new_header[2], 1)[0]

        data.append(new_data, current_labels, current_units, current_time_unit)

        if verbose:
            print()
        current_count += add_count
    return data


def load_from_file(filename, start=0, step=1, points=-1, h_lines=3):
    """
    Return ndarray instance filled with data
    read from csv or wfm file.

    filename -- file name (full/relative path)
    start -- start read data points from this index
    step -- read data points with this step
    points -- data points to read (-1 == all)
    """

    valid_delimiters = [',', ';', ' ', ':', '\t']

    import csv
    from csv import Error as CSV_Error

    if filename[-3:].upper() != 'WFM':
        with open(filename, "r") as datafile:
            # analyze structure
            try:
                dialect = csv.Sniffer().sniff(datafile.read(2096))
            except CSV_Error:
                datafile.seek(0)
                dialect = csv.Sniffer().sniff(datafile.readline())
            datafile.seek(0)
            if dialect.delimiter not in valid_delimiters:
                dialect.delimiter = ','
            if verbose:
                print("Delimiter = \"{}\"   |   ".format(dialect.delimiter),
                      end="")
            # read
            text_data = datafile.readlines()
            header = text_data[0: h_lines]
            if dialect.delimiter == ";":
                text_data = origin_to_csv(text_data)
                dialect.delimiter = ','
            skip_header = get_csv_headers(text_data)
            usecols = valid_cols(text_data, skip_header,
                                 delimiter=dialect.delimiter)

            # remove excess
            last_row = len(text_data) - 1
            if points > 0:
                available = int(math.ceil((len(text_data) - skip_header -
                                           start) / float(step)))
                if points < available:
                    last_row = points * step + skip_header + start - 1
            text_data = text_data[h_lines + start : last_row + 1 : step]
            assert len(text_data) >= 2, \
                "\nError! Not enough data lines in the file."

            if verbose:
                print("Valid columns = {}".format(usecols))
            data = np.genfromtxt(text_data,
                                 delimiter=dialect.delimiter,
                                 usecols=usecols)

    else:
        data = wfm.read_wfm_group([filename], start_index=start,
                                  number_of_points=points,
                                  read_step=step)
        header = None
    if verbose:
        if data.shape[1] % 2 == 0:
            curves_count = data.shape[1] // 2
        else:
            curves_count = data.shape[1] - 1
        print("Curves count = {}".format(curves_count))
        print("Points count = {}".format(data.shape[0]))
    return data, header


def origin_to_csv(readed_lines):
    """
    Replaces ',' with '.' and then ';' with ',' in the
    given list of lines of a .csv file.

    Use it for OriginPro ascii export format with ';' as delimiter
    and ',' as decimal separator.

    read_lines -- array of a .csv file lines (array of string)
    """
    import re
    converted = [re.sub(r',', '.', line) for line in readed_lines]
    converted = [re.sub(r';', ',', line) for line in converted]
    # if verbose:
    #     print("OriginPro ascii format detected.")
    return converted


def valid_cols(read_lines, skip_header, delimiter=','):
    """
    Returns the list of valid (non empty) columns
    for the given list of lines of a .csv file.

    read_lines -- array of a .csv file lines (array of string)
    skip_header -- the number of header lines
    delimiter -- .csv data delimiter
    :return:
    """
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
    """
    Returns the number of header lines
    for the given list of lines of a .csv file.

    read_lines -- array of a .csv file lines (array of string)
    delimiter -- .csv data delimiter
    except_list -- the list of special strings, that can not be converted
                   to numeric with the float() function, but the csv reader
                   may considered it as a valid value.
    """
    cols_count = len(read_lines[-1].strip().split(delimiter))
    headers = 0
    idx = 0
    lines = len(read_lines)

    # column number difference
    while (len(read_lines[-1].strip().split(delimiter)) !=
           cols_count and idx < lines):
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


def save_signals_csv(filename, signals, delimiter=",", precision=18):
    """

    :param filename:
    :param signals:
    :param delimiter:
    :param precision:
    :return:
    """
    # check precision value
    table = signals.get_array()
    print("Save columns count = {}".format(table.shape[1]))
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
        # add headers
        labels = [signals.curves[idx].label for
                  idx in signals.idx_to_label.keys()]
        labels = [re.sub(r'[^-.\w_]', '_', label) for label in labels]
        labels = delimiter.join(labels) + "\n"
        units = [signals.curves[idx].unit for
                 idx in signals.idx_to_label.keys()]
        units = [re.sub(r'[^-.\w_]', '_', unit) for unit in units]
        units = delimiter.join(units) + "\n"
        time_unit = "{}\n".format(signals.curves[0].time_unit)
        lines.append(labels)
        lines.append(units)
        lines.append(time_unit)
        # add data
        for row in range(table.shape[0]):
            s = delimiter.join([value_format % table[row, col] for
                                col in range(table.shape[1])]) + "\n"
            s = re.sub(r'nan', '', s)
            lines.append(s)
        fid.writelines(lines)


def save_m_log(src, save_as, labels, multiplier=None, delays=None,
               offset_by_front=None, y_auto_offset=None, partial_params=None):
    """Save modifications log. 
    Saves log file describing the changes made to the data.

    If the log file exists and any new changes were made to the data and
    saved at the same data file, appends new lines to the log file.

    src             -- the list of source files the data was read from
    save_as         -- the full path to the file the data was saved to
    labels          -- the list of curves labels
    multiplier      -- the list of multipliers that were applied to the data
    delays          -- the list of delays that were applied to the data
    offset_by_front -- the --offset-by-curve-front args
    y_auto_offset   -- the list of --y-auto-zero args (list of lists)
    """
    now = datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S")
    lines = list()
    lines.append("Modified on {}\n".format(now))
    src_line = "Source files = {}\n".format(str(src))
    src_line = re.sub(r"\\\\", "\*", src_line)
    src_line = re.sub(r"\*", "", src_line)

    lines.append(src_line)
    if not partial_params:
        partial_str = "start = 0, step = 1, points = all"
    else:
        partial_str = ("start = {}, stop = {}, points = "
                       "".format(partial_params[0], partial_params[1]))
        if partial_params[2] == -1:
            partial_str += "all"
        else:
            partial_str += str(partial_params[2])
    lines.append("Partial import: " + partial_str + "\n")
    lines.append("\n")

    if multiplier:
        lines.append("Applied multipliers:\n")
        lines.append("    Time         Amplitude\n")
        for pair in zip(multiplier[0:-1:2], multiplier[1:-1:2]):
            lines.append("{: <15.5e} {:.5e}\n".format(*pair))
        lines.append("\n")
    else:
        lines.append("No multipliers were applied.\n")
    if delays:
        lines.append("Applied delays:\n")
        lines.append("Time            Amplitude\n")
        for pair in zip(delays[0:-1:2], delays[1:-1:2]):
            lines.append("{: <15.5e} {:.5e}\n".format(*pair))
        lines.append("\n")
    else:
        lines.append("No delays were applied.\n")
    if offset_by_front:
        lines.append("Delays was modified by --offset-by-curve-front "
                     "option with values:\n")
        lines.append("Curve idx = {}\n".format(offset_by_front[0]))
        lines.append("Level     = {}\n".format(offset_by_front[1]))
        lines.append("Smooth filter window = {}\n"
                     "".format(offset_by_front[2]))
        lines.append("Smooth filter polyorder = {}\n"
                     "".format(offset_by_front[3]))
        lines.append("\n")
    if y_auto_offset:
        lines.append("Delays was modified by --y-auto-zero "
                     "option with values:\n")
        for args in y_auto_offset:
            lines.append("Curve idx = {}     ({})\n"
                         "".format(args[0], labels[args[0]]))
            lines.append("Background start time = {}\n".format(args[1]))
            lines.append("Background stop time  = {}\n".format(args[2]))
            lines.append("\n")
    lines.append("----------------------------------------------------\n")
    if src == save_as and os.path.isfile(save_as + ".log"):
        mode = "a"
    else:
        mode = "w"
    # save log file if changes were made or data was saved as a new file
    if len(lines) > 7 or src != save_as:
        with open(save_as + ".log", mode) as f:
            f.writelines(lines)


# ========================================
# -----    WORKFLOW     ------------------
# ========================================
def add_to_log(s, print_to_console=True):
    global global_log
    global_log += s
    if print_to_console:
        print(s, end="")


def multiplier_and_delay(data, multiplier, delay):
    """Returns the modified data.
    Each column of the data is first multiplied by
    the corresponding multiplier value and
    then the corresponding delay value is subtracted from it.

    data       -- an instance of the SignalsData class
                  OR 2D numpy.ndarray
    multiplier -- the list of multipliers for each columns
                  in the input data.
    delay      -- the list of delays (subtrahend) for each
                  columns in the input data.
    """
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
            col_idx = curve_idx * 2  # index of time-column of current curve
            data.curves[curve_idx].data = \
                multiplier_and_delay(data.curves[curve_idx].data,
                                     multiplier[col_idx:col_idx + 2],
                                     delay[col_idx:col_idx + 2]
                                     )
        return data
    else:
        raise TypeError("Data must be an instance of "
                        "numpy.ndarray or SignalsData.")


def multiplier_and_delay_peak(peaks, multiplier, delay, curve_idx):
    """Returns the modified peaks data.
    Each time and amplitude of each peak is first multiplied by
    the corresponding multiplier value and
    then the corresponding delay value is subtracted from it.

    peaks       -- the list of the SinglePeak instance
    multiplier  -- the list of multipliers for each columns
                   in the SignalsData.
    delay       -- the list of delays (subtrahend)
                   for each columns in the SignalsData.
    curve_idx   -- the index of the curve to which the peaks belong
    """
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
        y_smoothed = savgol_filter(y_data, window, poly_order)
        return y_smoothed

    # too short array to be processed
    return y_data


def align_and_append_ndarray(*args):
    """Returns 2D numpy.ndarray containing all input 2D numpy.ndarrays.
    If input arrays have different number of rows,
    fills missing values with 'nan'.

    args -- 2d ndarrays
    """
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
        nan_arr[:] = np.nan
        # nan_arr[:] = np.NAN

        aligned_arr = np.append(arr, nan_arr, axis=0)
        data = np.append(data, aligned_arr, axis=1)
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
    start_idx = pp.find_nearest_idx(curve.time, start_x, side='right')
    stop_idx = pp.find_nearest_idx(curve.time, stop_x, side='left')
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
    """
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


def update_by_front(signals_data, args, multiplier, delay,
                    front_plot_name, interactive=False):
    """

    :param signals_data:
    :param args:
    :param multiplier:
    :param delay:
    :param front_plot_name:
    :param interactive:
    :return:
    """
    cancel = False

    if not os.path.isdir(os.path.dirname(front_plot_name)):
        os.makedirs(os.path.dirname(front_plot_name))

    curve_idx = args[0]
    level = args[1]
    window = args[2]
    poly_order = args[3]

    polarity = pp.check_polarity(signals_data.curves[curve_idx])
    if pp.is_pos(polarity):
        level = abs(level)
    else:
        level = -abs(level)

    x_mult = multiplier[curve_idx * 2]
    y_mult = multiplier[curve_idx * 2 + 1]
    x_del = delay[curve_idx * 2]
    y_del = delay[curve_idx * 2 + 1]

    # make local copy of target curve
    curve = SingleCurve(signals_data.time(curve_idx),
                        signals_data.value(curve_idx),
                        signals_data.label(curve_idx),
                        signals_data.unit(curve_idx),
                        signals_data.time_unit(curve_idx))
    # apply multiplier and delay to local copy
    curve.data = multiplier_and_delay(curve.data,
                                      [x_mult, y_mult],
                                      [x_del, y_del])
    # get x and y columns
    data_x = curve.get_x()
    data_y = curve.get_y()

    print("Time offset by curve front process.")
    print("Searching curve[{idx}] \"{label}\" front at level = {level}"
          "".format(idx=curve_idx, label=curve.label, level=level))

    plot_title = None
    smoothed_curve = None
    front_point = None
    if interactive:
        print("\n----------- Interactive offset_by_curve_front process "
              "--------------\n"
              "Enter two space separated positive integer value\n"
              "for WINDOW and POLYORDER parameters.\n"
              "\n"
              "WINDOW must be an odd value greater or equal to 5,\n"
              "POLYORDER must be >= 0, <= 5 and less than WINDOW.\n"
              "\n"
              "The larger the WINDOW value, the greater the smooth effect.\n"
              "\n"
              "The larger the POLYORDER value, the more accurate the result\n"
              "of smoothing.\n"
              "\n"
              "Close graph window to continue.")
    # interactive cycle
    while not cancel:
        # smooth curve
        data_y_smooth = smooth_voltage(data_y, window, poly_order)
        smoothed_curve = SingleCurve(data_x, data_y_smooth,
                                     curve.label, curve.unit,
                                     curve.time_unit)
        # find front
        front_x, front_y = pp.find_curve_front(smoothed_curve,
                                               level, polarity)
        plot_title = ("Curve[{idx}] \"{label}\"\n"
                      "".format(idx=curve_idx, label=curve.label))
        if front_x:
            front_point = pp.SinglePeak(front_x, front_y, 0)
            plot_title += "Found front at [{},  {}]".format(front_x, front_y)
        else:
            plot_title += "No front found."
            front_point = None

        # get input
        if interactive:
            # plot
            plot_multiple_curve([curve, smoothed_curve],
                                [front_point], title=plot_title)

            print("\nPrevious values:  {win}  {ord}\n"
                  "".format(win=window, ord=poly_order))
            plt.show()
            print("Press enter without entering values to save t"
                  "he last values and quit.")
            while True:
                try:
                    print("Enter WINDOW POLYORDER >>>", end="")
                    user_input = sys.stdin.readline().strip()
                    if user_input == "":
                        # breaks both cycles
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
        else:
            break

    # update delays
    new_delay = delay[:]
    if front_point is not None:
        for idx in range(0, len(delay), 2):
            new_delay[idx] += front_point.time
    # save final version of the plot
    plot_multiple_curve([curve, smoothed_curve],
                        [front_point], title=plot_title)
    plt.savefig(front_plot_name, dpi=400)
    # show plot to avoid tkinter warning "can't invoke "event" command..."
    plt.show(block=False)
    plt.close('all')

    return new_delay


def zero_one_curve(curve, max_rows=30):
    """Reesets the y values to 0, leaves first 'max_rows' rows
    and deletes the others.

    Returns reset curve.

    curve    -- the SingleCurve instance
    max_rows -- the number of rows to leave
    """
    curve.data = curve.data[0:max_rows + 1, :]
    for row in range(max_rows + 1):
        curve.data[row, 1] = 0
    return curve


def zero_curves(signals, curve_indexes, max_rows=30):
    """Reesets the y values to 0, leaves first 'max_rows' rows
    and deletes the others for all curves with index in
    curve_indexes.

    Returns modified SingnalsData

    data          -- the SignalsData instance
    curve_indexes -- the list of curves indexes to reset
    max_rows      -- the number of rows to leaved
    """
    if curve_indexes == -1:
        curve_indexes = list(range(0, signals.count))
    for idx in curve_indexes:
        signals.curves[idx] = zero_one_curve(signals.curves[idx], max_rows)
    return data


# ========================================
# -----    PLOT     ----------------------
# ========================================
def calc_ylim(time, y, time_bounds=None, reserve=0.1):
    """Returns (min_y, max_y) tuple with y axis bounds.
    The axis boundaries are calculated in such a way
    as to show all points of the curve with a indent
    (default = 10% of the span of the curve) from
    top and bottom.

    time        -- the array of time points
    y           -- the array of amplitude points
    time_bounds -- the tuple/list with the left and
                   the right X bounds in X units.
    reserve     -- the indent size (the fraction of the curve's range)
    """
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
    """Plots subplots for all curves with index in curve_list.
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
    """
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
    """Draws one or more curves on one graph.
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
    """
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


# ============================================================================
# --------------   MAIN    ----------------------------------------
# ======================================================
if __name__ == "__main__":
    import argparse
    prog_discr = ("")

    prog_epilog = ("")

    parser = argparse.ArgumentParser(prog='python SignalProcess.py',
                                     description=prog_discr, epilog=prog_epilog,
                                     fromfile_prefix_chars='@',
                                     formatter_class=argparse.RawTextHelpFormatter)

    # input files ------------------------------------------------------------
    parser.add_argument(
        '-d', '--scr', '--source-dir',
        action='store',
        metavar='DIR',
        dest='src_dir',
        default='',
        help='specify the directory containing data files.\n'
             'Default= the folder containing this code.\n\n')

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        '-f', '--input-files',
        action='append',
        nargs='+',
        metavar='FILE',
        dest='files',
        help='specify one or more (space separated) input file names \n'
             'after the flag. It is assumed that the files belong to \n'
             'the same shot. In order to process multiple shots enter\n'
             'multiple \'-f\' parameters.\n\n')

    group.add_argument(
        '-e', '--ext', '--extension',
        action='store',
        nargs='+',
        metavar='EXT',
        dest='ext_list',
        help='specify one or more (space separated) extensions of \n'
             'the files with data.\n\n')

    parser.add_argument(
        '--labels',
        action='store',
        metavar='LABEL',
        nargs='+',
        dest='labels',
        help='specify the labels for all the curves in the data \n'
             'files. It is recommended to use only Latin letters, \n'
             'numbers, dashes and underscores, any other symbols\n'
             'will be replaced with underscores.\n'
             'Needed for correct graph labels.\n\n')

    parser.add_argument(
        '--units',
        action='store',
        metavar='UNIT',
        nargs='+',
        dest='units',
        help='specify the units for each of the curves in the \n'
             'data files. Do not forget to take into account the \n'
             'influence of the corresponding multiplier value.\n'
             'Needed for correct graph labels.\n\n')

    parser.add_argument(
        '--time-unit',
        action='store',
        metavar='UNIT',
        dest='time_unit',
        help='specify the unit of time scale (uniform for all \n'
             'curves). Needed for correct graph labels.\n\n')

    parser.add_argument(
        '-g', '--grouped-by',
        action='store',
        type=int,
        metavar='NUMBER',
        dest='group_size',
        default=1,
        help='specify the size of the files groups. Default=1. \n'
             'A group is a set of files, corresponding to one shot. \n'
             'For correct import all shots in folder must consist of\n'
             'the same number of files.\n'
             'NOTE: if \'-g\' parameter is specified, then all the \n'
             '      files (with specified extensions via -e flag) in\n'
             '      the specified directory will be processed.\n\n')

    parser.add_argument(
        '-c', '--ch', '--sorted-by-channel',
        action='store_true',
        dest='sorted_by_ch',
        help='this options tells the program that the files are \n'
             'sorted by the oscilloscope/channel firstly and secondly\n'
             'by the shot number. By default, the program considers \n'
             'that the files are sorted by the shot number firstly\n'
             'and secondly by the oscilloscope/channel.\n\n'
             'ATTENTION: files from all oscilloscopes must be in the\n'
             'same folder and be sorted in one style.\n\n')

    parser.add_argument(
        '--partial-import',
        action='store',
        type=int,
        metavar=('START', 'STEP', 'COUNT'),
        nargs=3,
        dest='partial',
        help='Specify START STEP COUNT after the flag. \n'
             'START: the index of the data point with which you want\n'
             'to start the import of data points,\n'
             'STEP: the reading step, \n'
             'COUNT: the number of points that you want to import \n'
             '(-1 means till the end of the file).\n\n')

    # process parameters and options -----------------------------------------
    parser.add_argument(
        '--silent',
        action='store_true',
        dest='silent',
        help='enables the silent mode, in which only most important\n'
             'messages are displayed.\n\n')

    parser.add_argument(
        '--multiplier',
        action='store',
        type=float,
        metavar='MULT',
        nargs='+',
        dest='multiplier',
        default=None,
        help='the list of multipliers for each data columns.\n'
             'NOTE: you must enter values for all the columns in\n'
             '      data file(s). Each curve have two columns: \n'
             '      filled with X and Y values correspondingly.\n\n')

    parser.add_argument(
        '--delay',
        action='store',
        type=float,
        metavar='DELAY',
        nargs='+',
        dest='delay',
        default=None,
        help='the list of delays (subtrahend) for each data columns.\n'
             'NOTE: the data is first multiplied by a corresponding \n'
             '      multiplier and then the delay is subtracted \n'
             '      from them.\n\n')

    parser.add_argument(
        '--offset-by-curve-front',
        action='store',
        metavar='VAL',
        nargs='+',
        dest='offset_by_front',
        default=None,
        help='Enter: IDX LEVEL WINDOW ORDER, where:\n'
             'IDX    - the index of the curve\n'
             'LEVEL  - the amplitude level\n'
             'WINDOW - length of the filter window (must be an odd \n'
             '         integer greater than 4)\n'
             'ORDER  - the order of the polinomial used to fit the\n'
             '         samples (must be less than window length)\n\n'
             'Description:\n'
             '1. Finds the most left front point where the curve\n'
             'amplitude is greater (lower - for negative peak) than\n'
             'level value.\n'
             '2. Makes this point the origin of the time axis for\n'
             'all signals (changes the list of the delays).\n\n'
             'To improve accuracy, the signal is smoothed by \n'
             'the savgol filter.\n\n'
             'NOTE: you can enter only two parameters (IDX LEVEL),\n'
             '      then the interactive mode of the smooth filter\n'
             '      parameters selection will start.\n\n')

    parser.add_argument(
        '--y-auto-zero',
        action='append',
        metavar=('CURVE_IDX', 'BG_START', 'BG_STOP'),
        nargs=3,
        dest='y_auto_zero',
        help='auto zero level correction of the specified curve.\n'
             'CURVE_IDX is the zero-based index of the curve; \n'
             'BG_START and BG_STOP are the left and the right bound\n'
             'of the time interval at which the curve does not\n'
             'contain signals (background interval).\n'
             'You can use as many --y-auto-zero flags\n'
             'as you want (one flag for one curve).\n\n')

    parser.add_argument(
        '--set-to-zero',
        action='store',
        metavar='CURVE_IDX',
        nargs='+',
        dest='zero',
        default=None,
        help='specify the indexes of the fake curves, whose values\n'
             'you want to set to zero.\n'
             'Enter \'all\' (without the quotes) to set all curves\n'
             'values to zero.\n\n')

    # output settings --------------------------------------------------------
    parser.add_argument(
        '-s', '--save',
        action='store_true',
        dest='save',
        help='saves the shot data to a CSV file after all the changes\n'
             'have been applied.\n'
             'NOTE: if one shot corresponds to one CSV file, and\n'
             '      the output directory is not specified, the input\n'
             '      files will be overwritten.\n\n')

    parser.add_argument(
        '-t', '--save-to', '--target-dir',
        action='store',
        metavar='DIR',
        dest='save_to',
        default='',
        help='specify the output directory.\n\n')

    parser.add_argument(
        '--prefix',
        action='store',
        metavar='PREFIX',
        dest='prefix',
        default='',
        help='specify the file name prefix. This prefix will be added\n'
             'to the output file names during the automatic\n'
             'generation of file names.\n'
             'Default=\'\'.\n\n')

    parser.add_argument(
        '--postfix',
        action='store',
        metavar='POSTFIX',
        dest='postfix',
        default='',
        help='specify the file name postfix. This postfix will be\n'
             'added to the output file names during the automatic\n'
             'generation of file names.\n'
             'Default=\'\'.\n\n')

    parser.add_argument(
        '-o', '--output',
        action='store',
        nargs='+',
        metavar='FILE',
        dest='out_names',
        help='specify the list of file names after the flag.\n'
             'The output files with data will be save with the names\n'
             'from this list. This will override the automatic\n'
             'generation of file names.\n'
             'NOTE: you must enter file names for \n'
             '      all the input shots.\n\n')

    parser.add_argument(
        '-p', '--plot',
        action='store',
        nargs='+',
        metavar='CURVE_IDX',
        dest='plot',
        help='specify the indexes of the curves you want to plot\n'
             'or enter \'all\' (without quotes) to plot all the\n'
             'curves).\n'
             'Each curve from this list will be plotted separately\n'
             'as a single graph.\n'
             'You can specify --p-save flag in order to save them\n'
             'as .png files.\n\n')

    parser.add_argument(
        '--p-hide', '--plot-hide',
        action='store_true',
        dest='p_hide',
        help='if the --plot, --p-save and this flag is specified\n'
             'the single plots will be saved but not shown.\n'
             'This option can reduce the running time of the program.\n\n')

    parser.add_argument(
        '--p-save', '--save-plots-to',
        action='store',
        dest='plot_dir',
        metavar='PLOT_DIR',
        help='specify the directory.\n'
             'Each curve from the list, entered via --plot flag\n'
             'will be plotted and saved separately as .png file\n'
             'to this directory.\n\n')

    parser.add_argument(
        '-m', '--multiplot',
        action='append',
        dest='multiplot',
        metavar='CURVE_IDX',
        nargs='+',
        help='specify the indexes of the curves you want to plot\n'
             'at one graph (one curve under the other with uniform\n'
             'time scale).\n'
             'You may use as many \'-m\' flags (with different lists\n'
             'of curves) as you want. One flag for one graph.\n\n')

    parser.add_argument(
        '--mp-hide', '--multiplot-hide',
        action='store_true',
        dest='mp_hide',
        help='if the --multiplot, --mp-save and this flag is specified\n'
             'the multiplots will be saved but not shown.\n'
             'This option can reduce the running time of the program.\n\n')

    parser.add_argument(
        '--mp-save', '--save-multiplot-as',
        action='store',
        dest='multiplot_dir',
        metavar='MULTIPLOT_DIR',
        help='specify the directory.\n'
             'Each multiplot, entered via --multiplot flag(s)\n'
             'will be plotted and saved separately as .png file\n'
             'to this directory.\n\n')

    args = parser.parse_args()
    verbose = not args.silent

    try:
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
                                                  args.group_size,
                                                  args.sorted_by_ch)

        # Now we have the list of files, grouped by shots:
        # grouped_files == [
        #                    ['shot001_osc01.wfm', 'shot001_osc02.csv', ...],
        #                    ['shot002_osc01.wfm', 'shot002_osc02.csv', ...],
        #                    ...etc.
        #                  ]

        # check partial import args
        args.partial = check_partial_args(args.partial)

        # raw check offset_by_voltage parameters (types)
        it_offset = False  # interactive offset process
        if args.offset_by_front:
            assert len(args.offset_by_front) in [2, 4], \
                ("error: argument {arg_name}: expected 2 or 4 arguments.\n"
                 "[IDX LEVEL] or [IDX LEVEL WINDOW POLYORDER]."
                 "".format(arg_name="--offset-by-curve_level"))
            if len(args.offset_by_front) < 4:
                it_offset = True
            args.offset_by_front = global_check_front_params(args.offset_by_front)

        # # raw check labels
        # if args.labels:
        #     assert global_check_labels(args.labels), \
        #         "Label value error! Only latin letters, " \
        #         "numbers and underscore are allowed."

        args.plot_dir = check_param_path(args.plot_dir, '--p_save')
        args.multiplot_dir = check_param_path(args.multiplot_dir, '--mp-save')
        args.save_to = check_param_path(args.save_to, '--save-to')
        if not args.save_to:
            args.save_to = os.path.dirname(grouped_files[0][0])

        # checks if postfix and prefix can be used in filename
        if args.prefix:
            args.prefix = re.sub(r'[^-.\w]', '_', args.prefix)
        if args.postfix:
            args.postfix = re.sub(r'[^-.\w]', '_', args.postfix)

        # check and convert plot and multiplot args
        if args.plot:
            args.plot = global_check_idx_list(args.plot, '--plot',
                                          allow_all=True)
        if args.multiplot:
            for idx, m_param in enumerate(args.multiplot):
                args.multiplot[idx] = global_check_idx_list(m_param,
                                                            '--multiplot')

        # raw check y_auto_zero parameters (types)
        if args.y_auto_zero:
            args.y_auto_zero = global_check_y_auto_zero_params(args.y_auto_zero)

        # check and convert set-to-zero args
        if args.zero:
            args.zero = global_check_idx_list(args.zero, '--set-to-zero',
                                              allow_all=True)

        # raw check multiplier and delay
        if args.multiplier is not None and args.delay is not None:
            assert len(args.multiplier) == len(args.delay), \
                ("The number of multipliers ({}) is not equal"
                 " to the number of delays ({})."
                 "".format(len(args.multiplier), len(args.delay)))

        number_start, number_end = numbering_parser([files[0] for
                                                    files in grouped_files])
        labels_dict = {'labels': args.labels, 'units': args.units,
                       'time': args.time_unit}

        # MAIN LOOP
        if (args.save or
                args.plot or
                args.multiplot or
                args.offset_by_front):

            for shot_idx, file_list in enumerate(grouped_files):
                # get current shot name (number)
                shot_name = os.path.basename(file_list[0])[number_start:number_end]
                shot_name = trim_ext(shot_name, args.ext_list)
                # get SignalsData
                data = read_signals(file_list, start=args.partial[0],
                                    step=args.partial[1], points=args.partial[2],
                                    labels=args.labels, units=args.units,
                                    time_unit=args.time_unit)
                labels = [data.label(cr) for cr in data.idx_to_label.keys()]
                if verbose:
                    print("The number of curves = {}".format(data.count))

                # checks multipliers, delays and labels numbers
                check_coeffs_number(data.count * 2, ["multiplier", "delay"],
                                    args.multiplier, args.delay)
                check_coeffs_number(data.count, ["label", "unit"],
                                    args.labels, args.units)

                # check y_zero_offset parameters (if idx is out of range)
                if args.y_auto_zero:
                    check_y_auto_zero_params(data, args.y_auto_zero)

                    # updates delays with accordance to Y zero offset
                    args.delay = update_by_y_auto_zero(data, args.y_auto_zero,
                                                       args.multiplier,
                                                       args.delay,
                                                       verbose=True)

                # check offset_by_voltage parameters (if idx is out of range)
                if args.offset_by_front:
                    # check_offset_by_front(data, args.offset_by_front)
                    check_idx_list(args.offset_by_front[0], data.count - 1,
                                   "--offset-by-curve-front")

                    # updates delay values with accordance to voltage front
                    front_plot_name = get_front_plot_name(args.offset_by_front,
                                                          args.save_to, shot_name)
                    args.delay = update_by_front(data, args.offset_by_front,
                                                 args.multiplier, args.delay,
                                                 front_plot_name,
                                                 interactive=it_offset)

                # reset to zero
                if args.zero:
                    check_idx_list(args.zero, data.count - 1, "--set-to-zero")
                    if verbose:
                        print("Resetting to zero the values of the curves "
                              "with index in {}".format(args.zero))
                    data = zero_curves(data, args.zero)

                # multiplier and delay
                data = multiplier_and_delay(data, args.multiplier, args.delay)

                # plot preview and save
                if args.plot:
                    # args.plot = check_plot(args.plot)
                    if args.plot[0] == -1:  # 'all'
                        args.plot = list(range(0, data.count))
                    else:
                        check_plot_param(args.plot, data.count, '--plot')
                    for curve_idx in args.plot:
                        plot_multiple_curve(data.curves[curve_idx])
                        if args.plot_dir is not None:
                            plot_name = ("{shot}_curve_{idx}_{label}.plot.png"
                                         "".format(shot=shot_name, idx=curve_idx,
                                                   label=data.curves[curve_idx].label))
                            plot_path = os.path.join(args.plot_dir, plot_name)
                            plt.savefig(plot_path, dpi=400)
                            if verbose:
                                print("Plot is saved as {}".format(plot_path))
                        if not args.p_hide:
                            plt.show()
                        else:
                            plt.show(block=False)
                        plt.close('all')

                # plot and save multi-plots
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
                            if verbose:
                                print("Multiplot is saved {}".format(mplot_path))
                        if not args.mp_hide:
                            plt.show()
                        else:
                            plt.show(block=False)
                        plt.close('all')

                # save data
                if args.save:
                    save_name = ("{pref}{number}{postf}.csv"
                                 "".format(pref=args.prefix, number=shot_name,
                                           postf=args.postfix))
                    save_path = os.path.join(args.save_to, save_name)
                    save_signals_csv(save_path, data)
                    if verbose:
                        max_rows = max(curve.data.shape[0] for curve in
                                       data.curves.values())
                        print("Curves count = {}\n"
                              "Rows count = {} ".format(data.count, max_rows))
                        print("Saved as {}".format(save_path))
                    save_m_log(file_list, save_path, labels, args.multiplier,
                               args.delay, args.offset_by_front,
                               args.y_auto_zero, args.partial)

        print_duplicates(grouped_files, 30)
    except Exception as e:
        print()
        sys.exit(e)
    # =========================================================================
    # TODO: comments
    # TODO: description

    # print(args.y_auto_zero)
    print()
    print(args)
