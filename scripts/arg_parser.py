# Python 3.6
"""
Universal and specific command line interface argument parsers.


Maintainer: Shpakov Konstantin
Link: https://github.com/shpakovkv/ProcessSignals_Python
"""


# =======================================================================
# -----     IMPORT     -------------------------------------------------
# =======================================================================
import os
import argparse

# =======================================================================
# -----     CONST     ---------------------------------------------------
# =======================================================================
NOISEATTENUATION = 0.75
SAVETODIR = 'Peaks'
PEAKDATADIR = 'PeakData'


# =======================================================================
# -----     LOAD/SAVE PARSERS     ---------------------------------------
# =======================================================================
def get_output_args_parser():
    """Returns the parser of parameters of save data.

    :return: file save arguments parser
    :rtype: argparse.ArgumentParser
    """
    output_params_parser = argparse.ArgumentParser(add_help=False)
    output_params_parser.add_argument(
        '-s', '--save',
        action='store_true',
        dest='save',
        help='saves the shot data to a CSV file after all the changes\n'
             'have been applied.\n'
             'NOTE: if one shot corresponds to one CSV file, and\n'
             '      the output directory is not specified, the input\n'
             '      files will be overwritten.\n\n')

    output_params_parser.add_argument(
        '-t', '--save-to', '--target-dir',
        action='store',
        metavar='DIR',
        dest='save_to',
        default='',
        help='specify the output directory.\n\n')

    output_params_parser.add_argument(
        '--prefix',
        action='store',
        metavar='PREFIX',
        dest='prefix',
        default='',
        help='specify the file name prefix. This prefix will be added\n'
             'to the output file names during the automatic\n'
             'generation of file names.\n'
             'Default=\'\'.\n\n')

    output_params_parser.add_argument(
        '--postfix',
        action='store',
        metavar='POSTFIX',
        dest='postfix',
        default='',
        help='specify the file name postfix. This postfix will be\n'
             'added to the output file names during the automatic\n'
             'generation of file names.\n'
             'Default=\'\'.\n\n')

    output_params_parser.add_argument(
        '-o', '--output-files',
        action='store',
        nargs='+',
        metavar='FILE',
        dest='out_names',
        default=None,
        help='specify the list of file names after the flag.\n'
             'The output files with data will be save with the names\n'
             'from this list. This will override the automatic\n'
             'generation of file names.\n'
             'NOTE: you must enter file names for \n'
             '      all the input shots.\n\n')
    return output_params_parser


def get_input_files_args_parser():
    """Returns the parser of parameters of read data.

    :return: input file arguments parser
    :rtype: argparse.ArgumentParser
    """
    input_params_parser = argparse.ArgumentParser(add_help=False)

    input_params_parser.add_argument(
        '-d', '--source-dir',
        action='store',
        metavar='DIR',
        dest='src_dir',
        default='',
        help='specify the directory containing data files.\n'
             'Default= the folder containing this code.\n\n')

    group = input_params_parser.add_mutually_exclusive_group(required=True)

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

    input_params_parser.add_argument(
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

    input_params_parser.add_argument(
        '--units',
        action='store',
        metavar='UNIT',
        nargs='+',
        dest='units',
        help='specify the units for each of the curves in the \n'
             'data files. Do not forget to take into account the \n'
             'influence of the corresponding multiplier value.\n'
             'Needed for correct graph labels.\n\n')

    input_params_parser.add_argument(
        '--time-unit',
        action='store',
        metavar='UNIT',
        dest='time_unit',
        help='specify the unit of time scale (uniform for all \n'
             'curves). Needed for correct graph labels.\n\n')

    input_params_parser.add_argument(
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

    input_params_parser.add_argument(
        '--sorted-by-channel',
        action='store_true',
        dest='sorted_by_ch',
        help='this options tells the program that the files are \n'
             'sorted by the oscilloscope/channel firstly and secondly\n'
             'by the shot number. By default, the program considers \n'
             'that the files are sorted by the shot number firstly\n'
             'and secondly by the oscilloscope/channel.\n\n'
             'ATTENTION: files from all oscilloscopes must be in the\n'
             'same folder and be sorted in one style.\n\n')

    input_params_parser.add_argument(
        '--partial-import',
        action='store',
        type=int,
        metavar=('START', 'STEP', 'COUNT'),
        nargs=3,
        dest='partial',
        default=None,
        help='Specify START STEP COUNT after the flag. \n'
             'START: the index of the data point with which you want\n'
             'to start the import of data points,\n'
             'STEP: the reading step, \n'
             'COUNT: the number of points that you want to import \n'
             '(-1 means till the end of the file).\n\n')

    input_params_parser.add_argument(
        '--silent',
        action='store_true',
        dest='silent',
        help='enables the silent mode, in which only most important\n'
             'messages are displayed.\n\n')

    input_params_parser.add_argument(
        '--as-log-sequence',
        action='store_true',
        dest='sequence',
        help='All files will be loaded as pieces of one file.\n'
             'This means that curves will be formed from the data\n'
             'of the first file. Then, points from the remaining \n'
             'files will be added to these curves (one after the other \n'
             'in the order in which the file names were entered, \n'
             'or in alphabetical order if a folder with files \n'
             'was specified).\n\n'
             'Useful when processing logger data files.\n\n')

    return input_params_parser


# =======================================================================
# -----     DATA MANIPULATION PARSERS     -------------------------------
# =======================================================================
def get_mult_del_args_parser():
    """Returns multiplier and delay options parser.

    :return: multiplier and delay arguments parser
    :rtype: argparse.ArgumentParser
    """
    coeffs_parser = argparse.ArgumentParser(add_help=False)

    coeffs_parser.add_argument(
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

    coeffs_parser.add_argument(
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
    return coeffs_parser


def get_data_corr_args_parser():
    """Returns data manipulation options parser.
    Special for SignalProcess.py.

    :return: data manipulation arguments parser
    :rtype: argparse.ArgumentParser
    """
    data_corr_args_parser = argparse.ArgumentParser(add_help=False)

    data_corr_args_parser.add_argument(
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

    data_corr_args_parser.add_argument(
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

    data_corr_args_parser.add_argument(
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
    return data_corr_args_parser


# =======================================================================
# -----     PLOT PARSERS     --------------------------------------------
# =======================================================================
def get_plot_args_parser():
    """Returns plot options parser.

    :return: plot arguments parser
    :rtype: argparse.ArgumentParser
    """
    plot_args_parser = argparse.ArgumentParser(add_help=False)

    plot_args_parser.add_argument(
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

    plot_args_parser.add_argument(
        '--p-hide', '--plot-hide',
        action='store_true',
        dest='p_hide',
        help='if the --plot, --p-save and this flag is specified\n'
             'the single plots will be saved but not shown.\n'
             'This option can reduce the running time of the program.\n\n')

    plot_args_parser.add_argument(
        '--p-save', '--save-plot-to',
        action='store',
        dest='plot_dir',
        metavar='PLOT_DIR',
        help='specify the directory.\n'
             'Each curve from the list, entered via --plot flag\n'
             'will be plotted and saved separately as .png file\n'
             'to this directory.\n\n')

    plot_args_parser.add_argument(
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

    plot_args_parser.add_argument(
        '--mp-hide', '--multiplot-hide',
        action='store_true',
        dest='mp_hide',
        help='if the --multiplot, --mp-save and this flag is specified\n'
             'the multiplots will be saved but not shown.\n'
             'This option can reduce the running time of the program.\n\n')

    plot_args_parser.add_argument(
        '--mp-save', '--save-multiplot-to',
        action='store',
        dest='multiplot_dir',
        metavar='MULTIPLOT_DIR',
        help='specify the directory.\n'
             'Each multiplot, entered via --multiplot flag(s)\n'
             'will be plotted and saved separately as .png file\n'
             'to this directory.\n\n')

    plot_args_parser.add_argument(
        '--plot-bounds',
        action='store',
        dest='t_bounds',
        metavar=('LEFT', 'RIGHT'),
        nargs=2,
        type=float,
        default=(None, None),
        help='Specify the left and the right time boundaries.\n'
             'The plot and multiplot will only display data within \n'
             'these boundaries.\n\n')

    plot_args_parser.add_argument(
        '--unixtime',
        action='store_true',
        dest='unixtime',
        help='Handle all time values as unix timestamp. \n'
             'The values will be converted to matplotlib datetime.\n'
             'Note:\n'
             'The unix time stamp is merely the number of seconds \n'
             'between a particular date and the Unix Epoch\n'
             '(January 1st, 1970 at UTC).\n\n')

    return plot_args_parser


# =======================================================================
# -----     PEAK SEARCH PARSERS     -------------------------------------
# =======================================================================
def get_peak_args_parser():
    """Returns peak search options parser.
    Special for PeakProcess.py.

    :return: peak process arguments parser
    :rtype: argparse.ArgumentParser
    """
    peak_args_parser = argparse.ArgumentParser(add_help=False)

    peak_args_parser.add_argument(
        '--level',
        action='store',
        dest='level',
        metavar='LEVEL',
        type=float,
        help='The threshold of peak (all amplitude values \n'
             'below this level will be ignored).\n\n')

    peak_args_parser.add_argument(
        '--diff', '--diff-time',
        action='store',
        dest='pk_diff',
        metavar='DIFF_TIME',
        type=float,
        help='The minimum difference between two neighboring peaks. \n'
             'If two peaks are spaced apart from each other by less than \n'
             'diff_time, then the lower one will be ignored. \n'
             'Or if the next peak is at the edge (fall or rise) \n' 
             'of the previous peak, and the "distance" (time) from \n' 
             'its maximum to that edge (at the same level) is less \n' 
             'than the diff_time, this second peak will be ignored.\n\n')

    peak_args_parser.add_argument(
        '--curves',
        action='store',
        dest='curves',
        metavar='CURVE',
        nargs='+',
        type=int,
        help='The list of zero-based indexes of curves for which \n'
             'it is necessary to find peaks.\n'
             'The order of the curves corresponds to the order of \n'
             'the columns with data in the files \n'
             'and the order of reading the files\n\n')

    peak_args_parser.add_argument(
        '--peak-bounds',
        action='store',
        dest='t_bounds',
        metavar=('LEFT', 'RIGHT'),
        nargs=2,
        type=float,
        default=(None, None),
        help='Specify the left and the right search boundaries.\n'
             'The program will search for peaks only within this \n'
             'interval.\n\n')

    peak_args_parser.add_argument(
        '--noise-half-period', '--t-noise',
        action='store',
        dest='t_noise',
        metavar='T',
        type=float,
        help='Maximum half-period of noise fluctuation.\n'
             'Decreasing the value of this parameter will increase \n'
             'the number of false positive errors. \n'
             'Too high value will result in missing some real peaks.\n\n')

    peak_args_parser.add_argument(
        '--noise-attenuation',
        action='store',
        dest='noise_att',
        type=float,
        default=NOISEATTENUATION,
        help='Attenuation of the second half-wave of noise with a polarity\n'
             'reversal. If too many parasitic (noise) peaks are defined \n'
             'as real peaks, reduce this value.\n\n')

    peak_args_parser.add_argument(
        '--group-width',
        action='store',
        dest='gr_width',
        metavar='GR_DIFF',
        type=float,
        help='The maximum time difference between peaks that '
             'will be grouped.\n\n')

    peak_args_parser.add_argument(
        '--hide-found-peaks',
        action='store_true',
        dest='peak_hide',
        help='Hides the multiplot with overall found peaks.\n'
             'The multiplot will be saved at the default folder \n'
             'but not shown.\n'
             'Needed for automation.\n\n')

    peak_args_parser.add_argument(
        '--hide-all',
        action='store_true',
        dest='hide_all',
        help='Hides all plots. The plots will be saved at their \n'
             'default folders but not shown.\n'
             'Needed for automation.\n\n')

    peak_args_parser.add_argument(
        '--read',
        action='store_true',
        dest='read',
        help='Read the peak from files at the default folder\n'
             '({folder}).\n'
             'If the arguments needed for the searching of peaks \n'
             'were specified, the program will find peaks, save them\n'
             'and show the multiplot with all found peaks.\n'
             'You may delete some peak files or edit them.\n'
             'When the multiplot window is closed, the program \n'
             'will read the edited peak data and the new plots\n'
             'will be plotted.\n\n'
             ''.format(folder=os.path.join(SAVETODIR, PEAKDATADIR)))

    return peak_args_parser
