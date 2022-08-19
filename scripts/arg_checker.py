# Python 3.6
"""
Maintainer: Shpakov Konstantin
Link: https://github.com/shpakovkv/SignalProcess
"""

import re
import os
import hashlib
import numpy as np
from file_handler import get_grouped_file_list, get_csv_headers, get_dialect

ENCODING = 'latin-1'


# ========================================
# -----     CHECKERS     -----------------
# ========================================
def check_partial_args(options):
    """The validator of partial import args.

    Returns converted args.

    args -- partial import args
    """
    if options.partial is not None:
        start, step, count = options.partial
        assert start >= 0, "The start point must be non-negative integer."
        assert step > 0, "The step of the import must be natural number."
        assert count > 0 or count == -1, \
            ("The count of the data points to import "
             "must be natural number or -1 (means all).")
    else:
        options.partial = (0, 1, -1)
    return options


# def global_check_labels(labels):
#     """Checks if any label contains non alphanumeric character.
#      Underscore are allowed.
#      Returns true if all the labels passed the test.
#
#     labels -- the list of labels for graph process
#     """
#     for label in labels:
#         if re.search(r'[\W]', label):
#             return False
#     return True


def label_check(labels):
    """Checks if all the labels contains only
    Latin letters, numbers and underscore.

    Raises an exception if the test fails.

    WARNING: use it only after file_arg_check()

    :param labels: the list of labels or None
    :return: None
    """
    import re
    if labels is not None:
        for label in labels:
            if re.search(r'[^\w-]+', label):
                raise ValueError("{}\nWrong labels values!\n"
                                 "Latin letters, numbers, underscore and dash"
                                 " are allowed only.".format(label))


def filename_is_valid(name):
    """Checks the validity of a file name.

    :param name: file name (can be full name)
    :return: True if the file name is valid, else False.
    """
    import re
    if re.search(r'[^\w\s-]+', os.path.basename(name)):
        return False
    return True


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
        error_text += "and string 'all' (without the quotes) "
    error_text += "are allowed."

    if len(args) == 1 and args[0].upper() == 'ALL' and allow_all:
        args = [-1]
    elif len(args) == 1 and args[0] == -1:
        pass
    else:
        for idx, _ in enumerate(args):
            try:
                args[idx] = int(args[idx])
                assert args[idx] >= 0, ""
            except (ValueError, AssertionError):
                raise ValueError(error_text.format(val=args[idx], name=name))
    return args


def check_coeffs_number(need_count, coeff_names, *coeffs):
    """
    Checks the needed and the actual number of the coefficients.
    Raises an exception if they are not equal.

    Dimensions:
    multiplier[ ShotNumber ][ AxisNumber ]
    delay[ ShotNumber ][ AxisNumber ]

    :param need_count: the number of needed coefficients
    :param coeff_names: the list of the coefficient names
    :param coeffs: the list of coefficient lists to check (multiplier, delay, etc.)
    :return: None
    """

    for idx, coeff_list in enumerate(coeffs):
        if coeff_list is not None:
            coeffs_count = len(coeff_list)
            if isinstance(coeff_list, np.ndarray) and coeff_list.ndim == 2:
                # for multiplier and delay
                coeffs_count = coeff_list.shape[0] * coeff_list.shape[1]
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


def global_check_offset_by_front_params(params, window=101, polyorder=3):
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


def check_file_list(folder, grouped_files):
    """Checks if all files in the list of files
    exists.
    The file names must be grouped by shots.
    Each group must have the same number of files.
    Raises an exception if any test fails.

    :param folder: the path to the files
    :param grouped_files: the list of the filenames or fullpaths,
                          grouped by shots
    :type grouped_files: list
    :return: the list of full paths, grouped by shots.
    :rtype: list
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
            full_path = os.path.join(folder, filename.strip())
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
        assert params[0] < data.count, ("Index ({}) is out of range ({} curves) in "
                                        "--y-auto-zero parameters."
                                        "".format(params[0], data.count))


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
        except Exception:
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
        assert idx <= max_idx, \
            "Index ({idx}) is out of range in {name} " \
            "parameters.\nMaximum index = {max}" \
            "".format(idx=idx, name=arg_name, max=max_idx)


# def files_are_equal(first_file_name, second_file_name):
#     """Compares md5 sum of 2 files.
#     Returns True or False.
#
#     :param first_file_name: the full path to the first file
#     :param second_file_name: the full path to the second file
#
#     :type first_file_name: str
#     :type second_file_name: str
#
#     :return: True if md5 is equal, else False
#     :rtype: bool
#     """
#
#     chunk_size = 1024
#
#     file1_md5 = hashlib.md5()
#     file2_md5 = hashlib.md5()
#
#     with open(first_file_name, 'r', encoding=ENCODING) as file1:
#         for chunk in iter(file1.read(chunk_size)):
#             file1_md5.update(chunk.encode(ENCODING))
#
#     with open(second_file_name, 'r', encoding=ENCODING) as file2:
#         for chunk in iter(file2.read(chunk_size)):
#             file2_md5.update(chunk.encode(ENCODING))
#
#     if file1_md5 != file2_md5:
#         return False
#     return True


def files_are_equal(first_file_name, second_file_name):
    """Compares md5 sum of 2 files.
    Returns True or False.

    :param first_file_name: the full path to the first file
    :param second_file_name: the full path to the second file

    :type first_file_name: str
    :type second_file_name: str

    :return: True if md5 is equal, else False
    :rtype: bool
    """
    dialect1, text1 = get_dialect(first_file_name)
    dialect2, text2 = get_dialect(second_file_name)

    head1 = get_csv_headers(text1, delimiter=dialect1.delimiter, except_list=('', 'nan'))
    head2 = get_csv_headers(text2, delimiter=dialect2.delimiter, except_list=('', 'nan'))
    for file_1_line, file_2_line in zip(text1[head1:], text2[head2:]):
        if file_1_line != file_2_line:
            return False
    return True


def compare_grouped_files(group_list, lines=30):
    """Compares files with neighboring shot (group) indexes.
    Compares only

    Returns a list of pairs of matching files.

    :param group_list: the list of groups of files
    :param lines: the number of files lines to compare
    :return: a list of pairs (sub-list) of matching files
    """

    match_list = []
    for group_1, group_2 in zip(group_list[0:-1], group_list[1:]):
        if len(group_1) == len(group_2):
            for file_1, file_2 in zip(group_1, group_2):
                if files_are_equal(file_1, file_2):
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


def check_and_prepare_multiplier_and_delay(options, data_axes=2, dtype=np.float64):
    mult = options.multiplier
    delay = options.delay
    if mult is not None:
        assert len(mult) % data_axes == 0, \
            "Еhe number of elements in multiplier ({}) must be a multiple of " \
            "the number of data axes ({}).".format(len(mult), data_axes)

        assert isinstance(mult, list), \
            "The multiplier must be of type list. " \
            "Got {} instead.".format(type(mult))

        mult = np.array(mult, dtype=dtype)

        # input structure: cur0_x_mult, cur0_y_mult, cur1_x_mult, cur1_y_mult, etc.
        # first convert list to 'table' with x_column and y_column:
        # cur0_x_mult, cur0_y_mult
        # cur1_x_mult, cur1_y_mult
        # cur2_x_mult, cur2_y_mult
        # -- etc.

        mult = mult.reshape(len(mult) // data_axes, data_axes)
        # options.multiplier = mult.reshape(data_axes, len(mult) // data_axes)
        options.multiplier = mult

    if delay is not None:
        assert len(delay) % data_axes == 0, \
            "Еhe number of elements in delay ({}) must be a multiple of " \
            "the number of data axes ({}).".format(len(delay), data_axes)

        assert isinstance(delay, list), \
            "The delay argument must be of type list. " \
            "Got {} instead.".format(type(delay))

        shape = (len(delay) // data_axes, data_axes)
        delay = np.array(delay, dtype=dtype)

        # input structure: cur0_x_delay, cur0_y_delay, cur1_x_delay, cur1_y_delay, etc.
        # first convert list to 'table' with x_column and y_column:
        # cur0_x_delay, cur0_y_delay
        # cur1_x_delay, cur1_y_delay
        # cur2_x_delay, cur2_y_delay
        # -- etc.

        delay = delay.reshape(shape)
        # options.delay = delay.reshape(data_axes, len(delay) // data_axes)
        options.delay = delay

    if delay is None and mult is not None:
        options.delay = np.zeros(shape=mult.shape, dtype=dtype)

    if mult is None and delay is not None:
        options.multiplier = np.ones(shape=delay.shape, dtype=dtype)


def check_multiplier(m, curves_count=1):
    """Checks 'multiplier' argument, return it's copy
     or generates a list of ones.

    :param m:     the list of multipliers
    :param curves_count: the number of data curves
    :return:      the list of multipliers (copy)
    """
    if m is None:
        # need two multipliers for each curve (for X and Y columns)
        m_shape = (curves_count, 2)
        return np.ones(shape=m_shape, dtype=np.float64)
    check_coeffs_number(curves_count * 2, ["multiplier"], m)
    return m


def check_delay(d, curves_count=1):
    """Checks 'delay' argument, return it's copy
     or generates a list of zeros.

    :param d:     the list of delays
    :param curves_count: the number of data curves
    :return:      the list of delays (copy)
    """
    if d is None:
        # need two delays for each curve (for X and Y columns)
        m_shape = (curves_count, 2)
        return np.zeros(shape=m_shape, dtype=np.float64)
    check_coeffs_number(curves_count * 2, ["delay"], d)
    return d


def file_arg_check(options):
    """Needed for data file import process.

    Checks import files parameters.
    Groups files by shots and makes
    and makes shots list with
    files sublists.
    gr_files == [
                  ['shot001_osc01.wfm', 'shot001_osc02.csv', ...],
                  ['shot002_osc01.wfm', 'shot002_osc02.csv', ...],
                   ...etc.
                ]

    Checks:
    --input-files,
    --source-dir,
    --ext,
    --grouped-by,
    --sorted-by-channel,
    --partial-import
    --as-log-sequence

    :param options: namespace with args
    :type options: argparse.Namespace

    :return: changed options
    :rtype: argparse.Namespace
    """
    # input directory and files check
    if options.src_dir:
        dir_list = list()
        for value in options.src_dir:
            dir_list.append(value.strip())
            assert os.path.isdir(value), \
                "Can not find directory {}".format(value)

    if options.files:
        gr_files = check_file_list(options.src_dir, options.files)
        if not options.src_dir:
            options.src_dir = os.path.dirname(gr_files[0][0])
    else:
        gr_files = get_grouped_file_list(options.src_dir,
                                         options.ext_list,
                                         options.group_size,
                                         options.sorted_by_ch)
    options.gr_files = gr_files

    if options.sequence:
        assert (options.group_size == 1), ("When importing logger data "
                                           "files, all specified files "
                                           "must belong to one group "
                                           "(--groped-by 1).")

    assert len(options.gr_files) > 0, "No data files found!"

    # Now we have the list of files, grouped by shots
    return options


def plot_arg_check(options):
    """Needed for plotting process.

    Check plot parameters args.
    --plot,
    --multiplot,
    --save-plot-to,
    --save-multiplot-to

    :param options: namespace with args
    :type options: argparse.Namespace

    :return: changed options
    :rtype: argparse.Namespace
    """
    options.plot_dir = check_param_path(options.plot_dir, '--p_save')
    options.multiplot_dir = check_param_path(options.multiplot_dir,
                                             '--mp-save')
    options.multicurve_dir = check_param_path(options.multicurve_dir,
                                              '--multicurve-plot-save')
    # check and convert plot and multiplot options
    if options.plot:
        options.plot = global_check_idx_list(options.plot, '--plot',
                                             allow_all=True)
    if options.multiplot:
        for idx, m_param in enumerate(options.multiplot):
            options.multiplot[idx] = global_check_idx_list(m_param,
                                                           '--multiplot')

    if options.multicurve:
        for idx, mc_param in enumerate(options.multicurve):
            options.multicurve[idx] = global_check_idx_list(mc_param,
                                                           '--multicurve-plot')

    return options


def save_arg_check(options):
    """Needed for data saving process.

    Check file save arguments.
    --save-to,
    --prefix,
    --postfix,
    --output-files

    WARNING: use it only after file_arg_check()

    :param options: namespace with args
    :type options: argparse.Namespace

    :return: changed options
    :rtype: argparse.Namespace
    """
    if options.separate_save:
        options.save = True

    options.save_to = check_param_path(options.save_to, '--save-to')
    if not options.save_to:
        options.save_to = os.path.dirname(options.gr_files[0][0])

    # checks if postfix and prefix can be used in filename
    if options.prefix:
        options.prefix = re.sub(r'[^-.\w]', '_', options.prefix)
    if options.postfix:
        options.postfix = re.sub(r'[^-.\w]', '_', options.postfix)

    if options.out_names is None and not options.convert_only:
        options.out_names = [None for _ in range(len(options.gr_files))]
    return options


def data_corr_arg_check(options):
    """Needed for data manipulation/correction process.

    Check data manipulation arguments:
    --offset-by-curve-front,
    --y-auto-zero,
    --set-to-zero,
    --multiplier,
    --delay,

    :param options: namespace with args
    :type options: argparse.Namespace

    :return: changed options
    :rtype: argparse.Namespace
    """

    # raw check offset_by_voltage parameters (types)
    options.it_offset = False  # interactive offset process
    if options.offset_by_front:
        assert len(options.offset_by_front) in [2, 4], \
            ("error: argument {arg_name}: expected 2 or 4 arguments.\n"
             "[IDX LEVEL] or [IDX LEVEL WINDOW POLYORDER]."
             "".format(arg_name="--offset-by-curve_level"))
        if len(options.offset_by_front) < 4:
            options.it_offset = True
        options.offset_by_front = global_check_offset_by_front_params(options.offset_by_front)

    # # raw check labels (not used)
    # # instead: the forbidden symbols are replaced during CSV saving
    # if options.labels:
    #     assert global_check_labels(options.labels), \
    #         "Label value error! Only latin letters, " \
    #         "numbers and underscore are allowed."

    # raw check y_auto_zero parameters (types)
    if options.y_auto_zero:
        options.y_auto_zero = global_check_y_auto_zero_params(options.y_auto_zero)

    # check and convert set-to-zero options
    if options.zero:
        options.zero = global_check_idx_list(options.zero, '--set-to-zero',
                                             allow_all=True)

    # raw check multiplier and delay
    if options.multiplier is not None and options.delay is not None:
        assert len(options.multiplier) == len(options.delay), \
            ("The number of multipliers ({}) is not equal"
             " to the number of delays ({})."
             "".format(len(options.multiplier), len(options.delay)))

    return options


def convert_only_arg_check(options):
    """If convert-only was specified
     checks prohibited options, creates output filename list.

     Use it only after file_arg_check() and save_arg_check().

    :param options: namespace with args
    :type options: argparse.Namespace

    :return: changed options
    :rtype: argparse.Namespace
    """
    if options.convert_only:
        # turn on save options
        options.save = True

        # prohibited args check
        assert options.out_names is None, "Argument --convert-only: not allowed with argument -o/--output-files"
        assert options.multiplier is None, "Argument --convert-only: not allowed with argument --multiplier"
        assert options.delay is None, "Argument --convert-only: not allowed with argument --delay"
        assert options.offset_by_front is None, "Argument --convert-only: not allowed with argument --offset-by-curve-front"
        assert options.y_auto_zero is None, "Argument --convert-only: not allowed with argument --y-auto-zero"
        assert options.zero is None, "Argument --convert-only: not allowed with argument --set-to-zero"
        assert options.plot is None, "Argument --convert-only: not allowed with argument -p/--plot"
        assert options.multiplot is None, "Argument --convert-only: not allowed with argument -m/--multiplot"
        assert options.group_size == 1, "Argument --convert-only: not allowed with group size > 1 (-g/--grouped-by)"
        # assert options. is None, "Argument --convert-only: not allowed with argument "

        # fill output files name
        options.out_names = [os.path.join(options.save_to, os.path.basename(group[0])[:-4])
                             for group in options.gr_files]
    return options


def peak_param_check(options):
    # original PeakProcess args
    if any([options.level, options.pk_diff, options.gr_width, options.curves]):
        assert all([options.level, options.pk_diff, options.gr_width, options.curves]), \
            "To start the process of finding peaks, '--level', " \
            "'--diff-time', '--group-diff', '--curves' arguments are needed."
        assert options.pk_diff >= 0, \
            "'--diff-time' value must be non negative real number."
        assert options.gr_width >= 0, \
            "'--group-diff' must be non negative real number."
        assert all(idx >= 0 for idx in options.curves), \
            "Curve index must be non negative integer"

    if options.t_noise:
        assert options.t_noise >= 0, \
            "'--noise-half-period' must be non negative real number."
    assert options.noise_att > 0, \
        "'--noise-attenuation' must be real number > 0."

    if all(bound is not None for bound in options.t_bounds):
        assert options.t_bounds[0] < options.t_bounds[1], \
            "The left time bound must be less then the right one."

    return options


def check_utility_args(options):
    assert options.threads > 0, "The number of threads must be > 0."

    if options.hide_all:
        options.p_hide = True
        options.mp_hide = True
        options.mcp_hide = True
        options.peak_hide = True

        # select non-interactive offset by curve front process
        options.it_offset = False
    return options


def check_analysis_args(options):
    if options.correlate_dir is not None:
        assert options.correlate is not None or \
               options.correlate_part is not None, \
               "Correlate data save option is added, " \
               "but no correlate parameters was specified!"
        if not os.path.isdir(options.correlate_dir):
            os.makedirs(options.correlate_dir)

    if options.correlate_plot_dir is not None:
        assert options.correlate is not None or \
               options.correlate_part is not None, \
               "Correlate plot save option is added, " \
               "but no correlate parameters was specified!"
        if not os.path.isdir(options.correlate_plot_dir):
            os.makedirs(options.correlate_plot_dir)

    if options.correlate_part is not None:
        for idx in range(len(options.correlate_part)):
            options.correlate_part[idx][0] = int(options.correlate_part[idx][0])  # curve_1 index
            options.correlate_part[idx][3] = int(options.correlate_part[idx][3])  # curve_2 index
            options.correlate_part[idx][6] = int(options.correlate_part[idx][6])  # add to signals_data or no
    return options


def front_delay_check(options):
    """Needed for front delay calculation process.

    Check data manipulation arguments:
    --front-delay
    --front-delay-save-plot-to

    :param options: namespace with args
    :type options: argparse.Namespace

    :return: changed options
    :rtype: argparse.Namespace
    """

    front_delay_count = 0
    save_to_count = 0
    bounds_count = 0
    if options.front_delay is not None:
        front_delay_count = len(options.front_delay)

        if options.front_delay_save_to is not None:
            save_to_count = len(options.front_delay_save_to)
            assert save_to_count == front_delay_count, \
                "You entered {} --front-delay flag(s). " \
                "Expected the same number (or none) of " \
                "--front-delay-save-plot-to flags, but found {}" \
                "".format(front_delay_count, save_to_count)

        if options.front_bounds is not None:
            bounds_count = len(options.front_bounds)
            assert bounds_count == front_delay_count, \
                "You entered {} --front-delay flag(s). " \
                "Expected the same number (or none) of " \
                "--front-bounds flags, but found {}" \
                "".format(front_delay_count, bounds_count)

        for idx in range(front_delay_count):
            front_dict = dict()
            front_dict["cur1"] = int(options.front_delay[idx][0])
            front_dict["level1"] = options.front_delay[idx][1]
            front_dict["slope1"] = "fall" if options.front_delay[idx][2] < 0 else "rise"
            front_dict["cur2"] = int(options.front_delay[idx][3])
            front_dict["level2"] = options.front_delay[idx][4]
            front_dict["slope2"] = "fall" if options.front_delay[idx][5] < 0 else "rise"

            save_to = None
            if options.front_delay_save_to is not None:
                if options.front_delay_save_to[idx] and not options.front_delay_save_to[idx].lower() == "none":
                    save_to = options.front_delay_save_to[idx]
            front_dict["save_to"] = save_to

            bounds1 = None
            bounds2 = None
            if options.front_bounds is not None:
                # if all --front-bounds were entered
                if options.front_bounds[idx]:
                    bounds1, bounds2 = check_and_prepare_front_bounds(options.front_bounds[idx])
            front_dict["bounds1"] = bounds1
            front_dict["bounds2"] = bounds2

            options.front_delay[idx] = front_dict

    else:
        if options.front_delay_save_to is not None:
            raise ValueError("--front-delay-save-plot-to flag was specified "
                             "but no --front-delay flag were entered!")
        if options.front_bounds is not None:
            raise ValueError("--front-bounds flag was specified "
                             "but no --front-delay flag were entered!")
    return options


def check_and_prepare_front_bounds(user_entered_bounds):
    """ Checks one --front-bounds flag values:
        - Checks for skipping word "none" and set value to None
        - Checks the number of values
        - Checks that left border < right border

        Returns prepared bounds.

        Possible outputs:
        [float, float]
        [None, float]
        [float, None]
        None
    """
    # if all --front-bounds were entered
    bounds1 = [None, None]
    bounds2 = [None, None]
    if user_entered_bounds:
        tmp_bounds = user_entered_bounds
        assert len(tmp_bounds) == 4, \
            "You must enter four bounds (left1, right1, left2, right2) " \
            "after the --front-bounds flag."

        # check for skipping word
        for j in range(4):
            if isinstance(tmp_bounds[j], str):
                if tmp_bounds[j].lower() == "none":
                    tmp_bounds[j] = None
                else:
                    try:
                        tmp_bounds[j] = float(tmp_bounds[j])
                    except ValueError as e:
                        raise ValueError("Wrong value entered with --front-bounds flag. "
                                         "Expected float type or 'none' keyword, found {}"
                                         "".format(tmp_bounds[j]))
        bounds1[0], bounds1[1] = tmp_bounds[0], tmp_bounds[1]
        bounds2[0], bounds2[1] = tmp_bounds[2], tmp_bounds[3]

        for bounds in (bounds1, bounds2):
            if bounds[0] is not None and bounds[1] is not None:
                assert bounds[1] > bounds[0], \
                    "Wrong value entered with --front-bounds flag. " \
                    "Еhe left border is greater than or equal to the right. " \
                    "({}, {})".format(bounds[0], bounds[1])

        if all(val is None for val in bounds1):
            bounds1 = None

        if all(val is None for val in bounds2):
            bounds2 = None

    return bounds1, bounds2
