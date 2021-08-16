# Python 3.6
"""

Maintainer: Shpakov Konstantin
Link: https://github.com/shpakovkv/SignalProcess
"""

import argparse
import arg_parser
import arg_checker
import file_handler as fh
import plotter


def get_parser():
    usage = ('python %(prog)s [options]\n'
             '       python %(prog)s @file_with_options\n\n'
             'Supports: comma delimited (CSV, TXT, DAT)\n'
             '          WFM (Tektronics standart only)\n\n')
    desc = ('Plots graph with single curve (plot), '
            'multiple curves in one/different panels (multiplot).\n'
            '\n'
            '===================================================\n'
            '-----   MINIMUM IMPORT OPTIONS   ------------------\n'
            '===================================================\n'
            '\n'
            'Single file:\n\n'
            '       --input-files data001_ch1.csv\n'
            '\n'
            'Multiple files, single shot:\n\n'
            '       --input-files data001_ch1.csv data001_ch2.wfm\n'
            '\n'
            'Multiple files, multiple shots:\n\n'
            '       --input-files data001_ch1.csv data001_ch2.wfm '
            '--input-files data002_ch1.csv data002_ch2.wfm\n'
            '\n'
            'Directory with files:\n\n'
            '       --source-dir ./data --ext csv wfm --grouped-by 2\n'
            '\n'
            '===================================================\n'
            '-----   MINIMUM PLOT OPTIONS   --------------------\n'
            '===================================================\n'
            '\n'
            'Plot 4 single plots for first 4 curves and save:\n\n'
            '       --plot 0, 1, 2, 3 --save-plots-to ./plotdir\n'
            '\n'
            'Plot 1 multiplot with first 4 curves and save:\n\n'
            '       --plot 0, 1, 2, 3 --save-multiplots-to ./multiplotdir\n'
            '\n'
            '===================================================\n'
            )

    final_parser = argparse.ArgumentParser(
        prog='PlotFile.py',
        usage=usage,
        description=desc,
        epilog='Maintainer: Konstantin Shpakov',
        parents=[arg_parser.get_input_files_args_parser(),
                 arg_parser.get_plot_args_parser()],
        formatter_class=argparse.RawTextHelpFormatter,
        fromfile_prefix_chars='@')

    return final_parser


def global_check(options):
    """Checks user args and corrects it.

    :param options: CLI args
    :type options: argparse.Namespace

    :return: changed CLI args
    :rtype: argparse.Namespace
    """
    # file import args check
    options = arg_checker.file_arg_check(options)

    # partial import args check
    options = arg_checker.check_partial_args(options)

    # plot args check
    options = arg_checker.plot_arg_check(options)

    # curve labels check
    arg_checker.label_check(options.labels)

    return options


def load_data(file_list, args):
    """Returns SIgnalsData instance with curves loaded from files.

    :param file_list: list of full paths or 1 path (str)
    :param args: namespace with args

    :type file_list: list of str or str
    :type args: argparse.Namespace

    :return:
    """
    if args.sequence:
        # load curves whose data are recorded in several files
        # (first part, second part, etc. in different files)
        return fh.read_log(file_list, start=args.partial[0],
                           step=args.partial[1], points=args.partial[2],
                           labels=args.labels, units=args.units,
                           time_unit=args.time_unit)

    else:
        # load curves whose data are recorded in single file
        # (new curves are loaded from each file and added to
        # the array of already loaded curves)
        return fh.read_signals(file_list, start=args.partial[0],
                               step=args.partial[1], points=args.partial[2],
                               labels=args.labels, units=args.units,
                               time_unit=args.time_unit)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    verbose = not args.silent

    args = global_check(args)

    '''
    num_mask (tuple) - contains the first and last index
    of substring of filename
    That substring contains the shot number.
    The last idx is excluded: [first, last).
    Read numbering_parser docstring for more info.
    '''
    num_mask = fh.numbering_parser([files[0] for
                                   files in args.gr_files])

    # MAIN LOOP
    if args.plot or args.multiplot:
        # Load files group by group from args.gr_files list
        for shot_idx, file_list in enumerate(args.gr_files):
            shot_name = fh.get_shot_number_str(file_list[0], num_mask,
                                               args.ext_list)

            # get SignalsData
            data = load_data(file_list, args)

            # checks the number of data curves,
            # and the number of labels and units
            arg_checker.check_coeffs_number(data.curves_count, ["label", "unit"],
                                            args.labels, args.units)

            # preview and save single plots
            if args.plot:
                plotter.do_plots(data, args, shot_name, verbose=verbose)

            # preview and save multi-plots
            if args.multiplot:
                plotter.do_multiplots(data, args, shot_name, verbose=verbose)
