# Python 3.6
"""
File handling functions.

Maintainer: Shpakov Konstantin
Link: https://github.com/shpakovkv/ProcessSignals_Python
"""

import os
import re
import math
import datetime
import numpy as np
import argparse
import WFMReader

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../isf-converter-py/'))

from isfconverter import isfreader as isf

from data_types import SignalsData
from data_types import SinglePeak


LOGDIRECTORY = "LOG_SignalProcess"
VERBOSE = 1
DEBUG = 0


# =======================================================================
# -----    CHECKERS     -------------------------------------------------
# =======================================================================
def check_header_label(s, count):
    """Checks if header line (from csv file) contains
    contains the required number of labels/units/etc.

    Returns the list of labels or None.

    s -- the header line (string) to check
    count -- the required number of labels/units/etc.
    """
    labels = [word.strip() for word in s.split(",")]
    if len(labels) == count:
        return labels
    return None


# =======================================================================
# -----    FILES HANDLING     -------------------------------------------
# =======================================================================
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


def get_shot_number_str(filename, num_mask, ext_list):
    """Returns the string contains the shot number.
    That string is the substring of the file name.

    NOTE: may return the full name of the file without the extension.
          Read numbering_parser docstring for more info.

    :param filename: the name of the file with experiments data.
                     That filename contains the number of the shot.
    :param num_mask: contains the first and last index of substring
                     of filename. See numbering_parser docstring for
                     more details.
    :param ext_list: the list of extension of data files

    :type filename: str
    :type num_mask: tuple/list
    :type ext_list: tuple/list

    :return: the string with the shot name
    :rtype: str
    """
    # get current shot name (number)
    shot_name = os.path.basename(filename)
    shot_name = shot_name[num_mask[0]:num_mask[1]]

    # in case of full file name
    shot_name = trim_ext(shot_name, ext_list)
    return shot_name


def numbering_parser(group_of_files):
    """Finds serial number (shot number) in the file name string.

    :param group_of_files: a list of files. Each file must corresponds
                           to different shot. All file names must have
                           the same length.

    :return: a tuple that contains the first and last index of
             the substring in the file name that contains the shot number.
             The last idx is excluded: [first, last)
             Example of the data file of experiment number 137
             Experiment_0137_Ch02_specific_environment.wfm
                       /    \
                    first   last
             'first' numeral is '0' with idx = 11, 'last' is '7' with idx = 14
             The interval (num_mask) = [11, 15)
             The 'Ch02' is the oscilloscope channel number
             whose data is written to that file.
             In most cases the channel number have 2 digits,
             while the shot number have 4 or more digits.
    :rtype: tuple
    """
    names = []
    for raw in group_of_files:
        names.append(os.path.basename(raw))
    assert all(len(name) == len(names[0]) for name in names), \
        "Error! All input file names must have the same length"
    if len(group_of_files) == 1:
        return 0, len(names[0])

    numbers = []  # list of lists of dictionaries
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


def get_grouped_file_list(folder, ext_list, group_size, sorted_by_ch=False):
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
    assert folder, ("Specify the directory (-d) containing the "
                    "data files. See help for more details.")
    file_list = get_file_list_by_ext(folder, ext_list, sort=True)
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


# =======================================================================
# -----    READ/SAVE DATA     -------------------------------------------
# =======================================================================
def read_log(file_list, start=0, step=1, points=-1,
             labels=None, units=None, time_unit=None):
    """Function returns one SignalsData object filled with
    data from files in file_list.

    The curves will be formed from the data
    of the first file. Then, points from the remaining
    files will be added to these curves (one after the other).

    Do not forget to sort the list of files
    for the correct order of points.

    Can handle different type of files.
    Supported formats: CSV, WFM
    For more information about the supported formats,
    see the help of the function load_from_file().

    :param file_list: list of full paths or 1 path (str)
    :param start: start read data points from this index
    :param step: read data points with this step
    :param points: data points to read (-1 == all)
    :param labels: list of labels for curves. The length of
                   the list and the number of curves in
                   each file should be the same.
    :param units: list of units for curves. The length of
                   the list and the number of curves in
                   each file should be the same.
    :param time_unit: time unit for all the curves.

    :type file_list: list
    :type start: int
    :type step: int
    :type points: int
    :type labels: list of str
    :type units: list of str
    :type time_unit: str

    :return: SignalData with loaded curves
    :rtype: SignalsData
    """
    data = read_signals(file_list[0], start=start, step=step, points=points,
                        labels=labels, units=units, time_unit=time_unit)
    for logfile in file_list[1:]:
        if VERBOSE:
            print("Loading {}".format(logfile))
        new_data, head = load_from_file(logfile, start=start,
                                        step=step, points=points)
        data.append(new_data)
    return data


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
        if VERBOSE:
            print("Loading {}".format(filename))
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
            head = check_header_label(new_header[2], 1)
            if head:
                current_time_unit = head[0]
            else:
                current_time_unit = None

        data.add_curves(new_data, current_labels, current_units, current_time_unit)

        if VERBOSE:
            print()
        current_count += add_count
    return data


def load_from_file(filename, start=0, step=1, points=-1, h_lines=0):
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

    # data type initialization
    data = np.ndarray(shape=(2, 2), dtype=np.float64, order='F')
    header = None

    ext_upper = filename[-3:].upper()

    if ext_upper == 'WFM':
        # TODO: add header output to the WFMReader
        data = WFMReader.read_wfm_group([filename], start_index=start,
                                        number_of_points=points,
                                        read_step=step)
    elif ext_upper == 'ISF':
        x_data, y_data, head = isf.read_isf(filename)
        x_data = np.expand_dims(x_data, axis=1)
        y_data = np.expand_dims(y_data, axis=1)
        data = np.append(x_data, y_data, axis=1)
        header = ["{}: {}".format(key, val) for key, val in head.items()]
    else:
        with open(filename, "r") as datafile:
            text_data = datafile.readlines()
            try:
                dialect = csv.Sniffer().sniff(text_data[-1])
            except csv.Error:
                datafile.seek(0)
                dialect = csv.Sniffer().sniff(datafile.read(4096))

            # # old version
            # try:
            #     dialect = csv.Sniffer().sniff(datafile.read(4096))
            # except csv.Error:
            #     datafile.seek(0)
            #     dialect = csv.Sniffer().sniff(datafile.readline())
            # datafile.seek(0)

            if dialect.delimiter not in valid_delimiters:
                dialect.delimiter = ','
            if ';' in text_data[-1]:
                dialect.delimiter = ';'

            if VERBOSE:
                print("Delimiter = \"{}\"   |   ".format(dialect.delimiter),
                      end="")
            header = text_data[0: h_lines]
            if dialect.delimiter == ";":
                text_data = origin_to_csv(text_data)
                dialect.delimiter = ','
            skip_header = get_csv_headers(text_data, delimiter=dialect.delimiter)
            usecols = valid_cols(text_data, skip_header,
                                 delimiter=str(dialect.delimiter))
            header = text_data[0: skip_header]

            # remove excess
            last_row = len(text_data) - 1
            if points > 0:
                available = int(math.ceil((len(text_data) - skip_header -
                                           start) / float(step)))
                if points < available:
                    last_row = points * step + skip_header + start - 1
            text_data = text_data[skip_header + start: last_row + 1: step]
            assert len(text_data) >= 2, \
                "\nError! Not enough data lines in the file."

            if VERBOSE:
                print("Valid columns = {}".format(usecols))
            data = np.genfromtxt(text_data,
                                 delimiter=str(dialect.delimiter),
                                 usecols=usecols)

    if VERBOSE:
        columns = 1
        if data.ndim > 1:
            columns = data.shape[1]

        if columns % 2 == 0:
            curves_count = columns // 2
        else:
            curves_count = columns - 1
        print("Curves loaded: {}".format(curves_count))
        print("Maximum points: {}".format(data.shape[0]))
    return data, header


def origin_to_csv(read_lines):
    """
    Replaces ',' with '.' and then ';' with ',' in the
    given list of lines of a .csv file.

    Use it for OriginPro ascii export format with ';' as delimiter
    and ',' as decimal separator.

    read_lines -- array of a .csv file lines (array of string)
    """
    import re
    converted = [re.sub(r',', '.', line) for line in read_lines]
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

    # get last line with data
    last_idx = len(read_lines) - 1
    while last_idx > 0:
        try:
            vals = [float(val) for val in
                    read_lines[last_idx].strip().split(delimiter)
                    if val not in except_list]
            # find at least 1 numeric value
            if len(vals) > 0:
                break
        except ValueError:
            # pass non numeric values at the end of a file
            pass
        last_idx -= 1

    cols_count = len(read_lines[last_idx].strip().split(delimiter))

    headers = 0
    idx = 0
    lines = len(read_lines)

    for idx in range(0, lines):
        # column number difference check
        if len(read_lines[idx].strip().split(delimiter)) == cols_count:

            # non numeric values check
            try:
                [float(val) for val in read_lines[idx].strip().split(delimiter) if
                 val not in except_list]
            except ValueError:
                headers += 1
            else:
                break
        else:
            headers += 1

    if VERBOSE:
        print("Header lines = {}   |   ".format(headers), end="")
        # print("Columns count = {}  |  ".format(cols_count), end="")
    return headers


def do_save(signals_data, cl_args, shot_name, save_as=None, verbose=False, separate_files=False):
    """Makes filename, saves changed data,
    prints info about saving process.

    :param signals_data: SignalsData instance
    :param cl_args: user-entered arguments (namespace from parser)
    :param shot_name: shot number, needed for saving
    :param save_as: fulllpath to save as
    :param verbose: show additional information or no
    :param separate_files: save each curve in separate file

    :type signals_data: SignalsData
    :type cl_args: argparse.Namespace
    :type shot_name: str
    :type save_as: str
    :type verbose: bool
    :type separate_files: bool

    :return: filename (full path)
    :rtype: str
    """
    if verbose:
        print('Saving data...')
    if save_as is None:
        save_name = ("{pref}{number}{postf}.csv"
                     "".format(pref=cl_args.prefix, number=shot_name,
                               postf=cl_args.postfix))
        save_as = os.path.join(cl_args.save_to, save_name)

    if separate_files:
        # delete extension
        if len(save_as) > 4 and save_as[-4] == ".":
            save_as = save_as[:-4]
        # save single curve
        if signals_data.count == 1:
            save_curve_as = "{}.csv".format(save_as)
            save_signals_csv(save_curve_as, signals_data, curves_list=[0])
        else:
            for curve in sorted(signals_data.idx_to_label.keys()):
                save_curve_as = "{}.curve{}.csv".format(save_as, curve)
                save_signals_csv(save_curve_as, signals_data, curves_list=[curve])
        # restore extension
        save_as = "{}.csv".format(save_as)

    else:
        save_signals_csv(save_as, signals_data)

    #TODO: logging with separate_files==True
    if verbose:
        max_rows = max(curve.data.shape[0] for curve in
                       signals_data.curves.values())
        if verbose:
            print("Curves count = {};    "
                  "Rows count = {} ".format(signals_data.count, max_rows))
        print("Saved as {}".format(save_as))
    return save_as


def save_signals_csv(filename, signals, delimiter=",", precision=18, curves_list=None):
    """Saves SignalsData to a CSV file.
    First three lines will be filled with header:
        1) the labels
        2) the curves units
        3) the time unit (only 1 column at this row)

    filename  -- the full path
    signals   -- SignalsData instance
    delimiter -- the CSV file delimiter
    precision -- the precision of storing numbers
    """
    # check precision value
    table = signals.get_array(curves_list)
    if DEBUG:
        print("Save columns count = {}".format(table.shape[1]))
    if not isinstance(precision, int):
        raise ValueError("Precision must be integer")
    if precision > 18:
        precision = 18
    value_format = '%0.' + str(precision) + 'e'

    # check filename
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

        # replaces forbidden characters
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


def save_peaks_csv(filename, peaks, labels=None):
    """Saves peaks data.
    Writes one line for each peak.
    Each line contains (comma separated):
        zero-based curve index,
        time value,
        amplitude value,
        sqr_l,
        sqr_r,
        the sum of sqr_l and sqr_r.

    :param filename: the peak data file name prefix
    :param peaks: the list of lists of peaks [curve_idx][peak_idx]
    :return: None
    """
    folder_path = os.path.dirname(filename)
    if folder_path and not os.path.isdir(folder_path):
        os.makedirs(folder_path)

    if len(filename) > 4 and filename[-4:].upper() == ".CSV":
        filename = filename[0:-4]

    if labels is None:
        labels = ['' for _ in range(len(peaks))]

    for gr in range(len(peaks[0])):
        content = ""
        for wf in range(len(peaks)):
            pk = peaks[wf][gr]
            if pk is None:
                pk = SinglePeak(0, 0, 0)
            content = (content +
                       "{idx:3d},{time:0.18e},{amp:0.18e},"
                       "{sqr_l:0.3f},{sqr_r:0.3f},{label}\n".format(
                           idx=wf, time=pk.time, amp=pk.val,
                           sqr_l=pk.sqr_l, sqr_r=pk.sqr_r,
                           label=labels[wf]
                       )
                       )
        postfix = "_peak{:03d}.csv".format(gr + 1)
        with open(filename + postfix, 'w') as fid:
            fid.writelines(content)
            # print("Done!")


def create_log(src, saved_as, labels, multiplier=None, delays=None,
               offset_by_front=None, y_auto_offset=None, partial_params=None):
    """Creates data changes log.
    Returns a list of log file lines.

    :param src:             the list of source files the data was read from
    :param saved_as:        the full path to the file the signals data was saved to
    :param labels:          the list of curves labels
    :param multiplier:      the list of multipliers that were applied to the data
    :param delays:          the list of delays that were applied to the data
    :param offset_by_front: the --offset-by-curve-front args
    :param y_auto_offset:   the list of --y-auto-zero args (list of lists)
    :param partial_params:  the list of input parameters (start, stop, points)
    :return:                the list of log file lines
    """
    now = datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S")
    lines = list()
    lines.append("Modified on {}\n".format(now))
    lines.append("Input files = {}\n".format(str(src)))
    lines.append("Output file = {}\n".format(saved_as))

    # # removes escape backslashes
    # for idx, src_line in enumerate(lines):
    #     lines[idx] = re.sub(r"\\\\", "\*", src_line)
    #     lines[idx] = re.sub(r"\*", "", src_line)

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
        lines.append("Time            Amplitude     Labels\n")
        for idx, pair in enumerate(zip(multiplier[0:len(multiplier):2],
                                       multiplier[1:len(multiplier):2])):
            lines.append("{: <15.5e} {:.5e}   {}\n".format(pair[0], pair[1],
                                                           labels[idx]))
        lines.append("\n")
    else:
        lines.append("No multipliers were applied.\n")

    if delays:
        lines.append("Applied delays:\n")
        lines.append("Time            Amplitude     Labels\n")
        for idx, pair in enumerate(zip(delays[0:len(delays):2],
                                       delays[1:len(delays):2])):
            lines.append("{: <15.5e} {:.5e}   {}\n".format(pair[0], pair[1],
                                                           labels[idx]))
        lines.append("\n")
    else:
        lines.append("No delays were applied.\n")

    if offset_by_front:
        lines.append("Delays was modified by --offset-by-curve-front "
                     "option with values:\n")
        lines.append("Curve idx = {}   ({})\n"
                     "".format(offset_by_front[0],
                               labels[offset_by_front[0]]))
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
    return lines


def save_m_log(src, saved_as, labels, multiplier=None, delays=None,
               offset_by_front=None, y_auto_offset=None, partial_params=None):
    """Saves the log of data modifications.
    Saves log file describing the changes made to the data.

    If the log file exists and any new changes were made to the data and
    saved at the same data file, appends new lines to the log file.

    :param src:             the list of source files the data was read from
    :param saved_as:        the full path to the file the signals data was saved to
    :param labels:          the list of curves labels
    :param multiplier:      the list of multipliers that were applied to the data
    :param delays:          the list of delays that were applied to the data
    :param offset_by_front: the --offset-by-curve-front args
    :param y_auto_offset:   the list of --y-auto-zero args (list of lists)
    :param partial_params:  the list of input parameters (start, stop, points)
    :return:                None
    """

    # log folder path
    save_log_as = os.path.join(os.path.dirname(saved_as),
                               LOGDIRECTORY)
    if not os.path.isdir(save_log_as):
        os.makedirs(save_log_as)

    # log file path
    save_log_as = os.path.join(save_log_as,
                               os.path.basename(saved_as) + ".log")

    log_lines = create_log(src, saved_as, labels, multiplier, delays,
                           offset_by_front, y_auto_offset, partial_params)

    if src == saved_as and os.path.isfile(save_log_as):
        mode = "a"
    else:
        mode = "w"
    # save log file if changes were made or data was saved as a new file
    if len(log_lines) > 8 or src != saved_as:
        with open(save_log_as, mode) as f:
            f.writelines(log_lines)
