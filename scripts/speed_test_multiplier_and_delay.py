# Python 3.6
"""
Speed tests for various variation of function.
This set of tests aimed to find the algorithms
with best performance for several method
used in SignalProcess program.

Maintainer: Shpakov Konstantin
Link: https://github.com/shpakovkv/SignalProcess
"""

import numpy as np

import random
import time

from numba import jit, njit, vectorize, guvectorize, float64, float32, int32, int64, cuda
from numba.typed import List

import logging
import os
import psutil
process = psutil.Process(os.getpid())
# print(process.memory_info().rss)  # in bytes

numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.DEBUG)

VERBOSE = True
DEBUG = True


# ===================================================================================
# --------    DECORATORS    ---------------------------------------------------------
# ...................................................................................


def print_func_time(func):

    def wrapper(*args, **kwargs):
        start = time.time_ns()
        func(*args, **kwargs)
        end = time.time_ns()
        print('[*] Runtime of the \'{}\': {:.15f} seconds.'.format(func.__name__, (end - start) / 1.0E+9))
        # print('[*] Runtime of the \'{}\': {:.15f} seconds.'.format("func.__name__", (end - start) / 1.0E+9))

    return wrapper


# ===================================================================================
# --------    DATA CLASS    ---------------------------------------------------------
# ...................................................................................

class SingleCurve:
    def __init__(self, in_x=None, in_y=None,
                 label=None, unit=None, time_unit=None):
        self.data = np.empty([0, 2], dtype=np.float64, order='F')
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


# -----------------------------------------------------------------------------------


class SignalsDataEnhanced:

    def __init__(self, from_array=None, labels=None,
                 units=None, time_units=None, dtype=np.float64):
        """

        :param from_array:
        :param labels:
        :param units:
        :param time_units:
        :param dtype:

        :type from_array: np.ndarray
        :type labels: list
        :type units: list
        :type time_units: list
        :type dtype: np.dtype
        """
        # CONST  --------------------------------------------------------------

        # [0]: time axis;  [1]: amplitude axis
        self._axes_number = 2

        # [curve-idx][axis-idx][point-idx]
        self._data_ndim = 3

        # VARIABLES   ---------------------------------------------------------

        # dict with curve labels as keys and curve indexes as values:
        self.label_to_idx = dict()
        # dict with curve indexes as keys and curve labels as values:
        self.idx_to_label = dict()

        self._time_units = ""
        self.units = list()
        self.labels = list()

        # EMPTY INSTANCE
        # number of curves:
        self.cnt_curves = 0
        self.max_points = 0
        self.min_points = 0
        self.dtype = dtype

        self.data = np.zeros(shape=(0, self._axes_number, 0), dtype=self.dtype, order="C")

        # fill with input data
        if isinstance(from_array, np.ndarray):
            self.add_from_array(from_array, labels=labels, units=units, time_units=time_units)
        # wrong input type
        else:
            raise TypeError("Wrong data inpyt option type ({})"
                            "".format(type(from_array)))

    def append_2d_array(self, input_data, labels=None, units=None,
                        force_single_time_row=False,
                        column_oriented=False):
        """
        Appends new points from ndarray to existing curves in self.
        Rise exception if the number of curves in the self
        and the input data do not match.

        :param input_data: ndarray with curves data. if the array has
                           an even number of columns, then the first
                           two curves will form the first curve,
                           the next two columns will result in the second
                           curve, etc.
                           If the array has an odd number of columns,
                           then the first column will be considered as
                           an X-column for all the curves added. And the
                           rest of the columns will be treated as Y-columns.

        :type input_data: np.ndarray

        :param labels: list of labels for new curves (one y_label for all new curves)
        :type labels: list

        :param units: list of units for new curves (one y_unit for all new curves)
        :type units: list

        :param force_single_time_row:
        :type force_single_time_row:

        :param column_oriented:
        :type column_oriented:

        :return: number of added curves
        :rtype: int
        """

        # check data
        assert isinstance(input_data, np.ndarray), ("Expected 2-dimensional numpy.ndarray as input data, " 
                                                    "got {} instead.".format(type(input_data)))
        assert input_data.ndim == 2, ("Expected 2-dimensional numpy.ndarray as input data, " 
                                      "got {} dimensions instead.".format(input_data.ndim))
        if column_oriented:
            input_data = np.transpose(input_data)

        # expected structure: time_row1, val_row1, time_row2, val_row2, etc.
        new_curves = input_data.shape[0] // self._axes_number

        # check if rows number is even/odd
        if not force_single_time_row:
            if input_data.shape[0] % 2 != 0:
                force_single_time_row = True

        if force_single_time_row:
            new_curves = input_data.shape[0] - 1
            # if rows number is odd add time rows for all input curves except first
            input_data = multiply_time_column_2d(input_data, self._axes_number)

        self.data = SignalsDataEnhanced.align_and_append_2d_arrays(self.data,
                                                                   input_data,
                                                                   dtype=self.dtype,
                                                                   axes_number=self._axes_number)

        # check & append new labels and units
        # TODO: append labels and units
        return new_curves

    def add_from_array(self, input_data, labels=None, units=None, time_units=None, force_single_time_row=False):
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
        # TODO: append multiple ndarrays (mixed 2-dimension and 3-dimension)
        assert isinstance(input_data, np.ndarray), \
            "Expected ndarray, got {} instead.".format(type(input_data))
        new_curves = 0
        new_min_points = 0
        if input_data.ndim == 2:
            new_min_points = input_data.shape[1]
            new_curves = self.append_2d_array(input_data, labels=labels, units=units,
                                              force_single_time_row=force_single_time_row)
        elif input_data.ndim == 3:
            new_curves = input_data.shape[0]
            new_min_points = input_data.shape[2]
            self.data = SignalsDataEnhanced.align_and_append_3d_arrays(self.data, input_data, dtype=self.dtype)
        else:
            raise AssertionError("Input array mast be 2-dimension or 3-dimension.")

        # update curves count
        self.cnt_curves += new_curves

        # update minimum points (some curve may have less points than others)
        if self.cnt_curves == new_curves:
            self.min_points = new_min_points
        elif new_min_points < self.min_points:
            self.min_points = new_min_points

        # update maximum points (some curves may have more points than others)
        self.max_points = self.data.shape[2]

        # check and append labels and units
        self.append_labels(new_curves, labels, units)

    def check_new_labels(self, new_curves, new_labels=None, new_units=None):
        """
        Checks the correctness of input new labels and units.
        Raises exception if check fails.

        :param new_curves: number of added curves
        :param new_labels: the list of labels for the new curves
        :param new_units: the list of units for the new curves

        :return: None
        :rtype: None
        """

        if new_labels is not None:
            assert isinstance(new_labels, list), \
                "New labels list must be list type. Got {} instead." \
                "".format(type(new_labels))

            assert len(new_labels) == new_curves, \
                "The number of new labels ({}) differs from the number of new curves ({})" \
                "".format(len(new_labels), new_curves)
            for idx, item in enumerate(new_labels):
                assert isinstance(item, str), \
                    "The type of each labels in list must be str. Got {} instead at index {}." \
                    "".format(type(item), idx)
                assert item not in new_labels[: idx], \
                    "All new labels must be unique. Label[{}] '{}' was used as label[{}]" \
                    "".format(idx, item, new_labels[: idx].index(item))
                assert item not in self.labels, \
                    "Label[ '{}' is already used as label[{}]" \
                    "".format(item, self.labels.index(item))

        if new_units is not None:
            assert isinstance(new_units, list), \
                "New units list must be list type. Got {} instead." \
                "".format(type(new_units))
            assert len(new_units) == new_curves, \
                "The number of new units ({}) differs from the number of new curves ({})" \
                "".format(len(new_units), new_curves)
            for idx, item in enumerate(new_units):
                assert isinstance(item, str), \
                    "The type of each units in new units list must be str. Got {} instead at index {}." \
                    "".format(type(item), idx)

    def append_labels(self, new_curves, new_labels=None, new_units=None):
        """
        Checks the correctness of input new labels and units.
        Raises exception if check fails.

        Appends fills labels and units for new curves

        :param new_curves: number of added curves
        :param new_labels: the list of labels for the new curves
        :param new_units: the list of units for the new curves

        :return: None
        :rtype: None
        """
        if new_curves > 0:
            if new_labels is not None:
                new_labels = list(new_labels)
            else:
                start = self.cnt_curves - new_curves
                new_labels = ["curve{}".format(idx) for idx in range(start, self.cnt_curves)]
            if new_units is not None:
                new_units = list(new_units)
            else:
                new_units = ["a.u." for _ in range(new_curves)]

            self.check_new_labels(new_curves, new_labels, new_units)
            self.labels.extend(new_labels)
            self.units.extend(new_units)

    def get_array_to_print_new(self, curves_list=None):
        # return align_and_append_ndarray(*list_of_2d_arr)
        raise NotImplementedError("get_array_to_print is not inplemented in SignalsDataEnhanced")

    def get_x(self, curve_idx):
        return self.data[curve_idx][0]

    def get_y(self, curve_idx):
        return self.data[curve_idx][1]

    def get_curve(self, curve_idx):
        return self.data[curve_idx]

    def get_labels(self):
        return list(self.labels)

    def get_units(self):
        return list(self.units)

    def get_label(self, curve_idx):
        return self.labels[curve_idx]

    def get_idx(self, label):
        for idx, name in enumerate(self.labels):
            if label == name:
                return idx
        return None

    def get_curve_units(self, curve_idx):
        return self.units[curve_idx]

    def set_units(self, units_list):
        units_list = list(units_list)
        assert len(units_list) == self.cnt_curves, \
            "The number of new units ({}) differs from the number of self curves ({})" \
            "".format(len(units_list), self.cnt_curves)
        for idx, item in enumerate(units_list):
            assert isinstance(item, str), \
                "The type of each units in list must be str. Got {} instead at index {}." \
                "".format(type(item), idx)
        self.units = units_list

    def set_labels(self, labels_list):
        labels_list = list(labels_list)
        assert len(labels_list) == self.cnt_curves, \
            "The number of new labels ({}) differs from the number of self curves ({})" \
            "".format(len(labels_list), self.cnt_curves)
        for idx, item in enumerate(labels_list):
            assert isinstance(item, str), \
                "The type of each labels in list must be str. Got {} instead at index {}." \
                "".format(type(item), idx)
            assert item not in labels_list[: idx], \
                "Label[{}] '{}' is already used as label[{}]" \
                "".format(idx, item, labels_list[: idx].index(item))
        self.labels = labels_list

    @property
    def time_units(self):
        return self._time_units

    @time_units.setter
    def time_units(self, s):
        """Sets new time unit for all curves
        (need for graph captions).

        :param s: new time units for all curves
        :type s: str

        :return: None
        :rtype: NoneType
        """
        assert isinstance(s, str), \
            "Wrong type. Expected str, got {} instead.".format(type(s))
        self.time_units = s

    def is_empty(self):
        """Check if there are no curves (no data) in this structure.

        :return: True if there are no curves, else False
        :rtype: bool
        """
        return True if self.cnt_curves == 0 else False

    @staticmethod
    def align_and_append_2d_arrays(base_3d_array, *arrays, dtype=np.float64, axes_number=2):
        """Returns 3-dimension numpy.ndarray containing all input curves.

        First array must be 3-dimension ndarray [curve_idx][axis_idx][point_idx]
        Curves from other arrays will be added to the copy of this array.

        All other ndarray must be 2D: [row_idx][point_idx] where rows 0, 2, 4, etc. are time-rows
        and rows 1, 3, 5, etc. are amplitude-rows

        Output data:
        Each curve consist of 2 rows: time-row and amplitude-row
        and have the shape == (1, 2, N),
        where N is the number of maximum points among all curves
        if the curve have n points and N > n:
        then the last (N - n) values of time-row and amp-row
        is filled with NaNs.

        :param base_3d_array: 3-dimension ndarray to append data to
        :type base_3d_array: np.ndarray

        :param arrays: a number of 2-dimension ndarray to append data from
        :type arrays: np.ndarray

        :param dtype: values type output array
        :type dtype: np.dtype

        :param axes_number: number of axes for output array
        :type axes_number: int

        :return: united ndarray
        :rtype: np.ndarray
        """

        # @jit(nopython=True, nogil=True)
        # def fill_data(data, args):
        #     """
        #
        #     :param data: 3-dimension ndarray to copy values to
        #     :param args: 2-dimension ndarrays to copy data from
        #     :return: data,- 3-dimension ndarray filled with data from other arrays
        #     """
        #     axes_number = data.shape[1]
        #     max_points = data.shape[2]
        #     curve_idx = 0
        #     for arr in args:
        #         # print("Adding array with shape {}".format(arr.shape))
        #         for sub_cur in range(arr.shape[0] // axes_number):
        #             # print("process subcurve {} from [0..{}]".format(sub_cur, arr.shape[0] // axes_number - 1))
        #             for axis in range(axes_number):
        #                 # print("Current axis: {} from [0..{}]".format(axis, axes_number - 1))
        #                 for idx in range(arr.shape[1]):
        #                     data[curve_idx][axis][idx] = arr[sub_cur * axes_number + axis][idx]
        #                 for idx in range(arr.shape[1], max_points):
        #                     data[curve_idx][axis][idx] = np.nan
        #             curve_idx += 1
        #
        # @jit(nopython=True, nogil=True)
        # def fill_data_once(data, source, curve_idx):
        #     """
        #
        #     :param data: 3-dimension ndarray to copy values to
        #     :param args: 2-dimension ndarrays to copy data from
        #     :return: data,- 3-dimension ndarray filled with data from other arrays
        #     """
        #     axes_number = data.shape[1]
        #     max_points = data.shape[2]
        #     for axis in range(axes_number):
        #         # print("Current axis: {} from [0..{}]".format(axis, axes_number - 1))
        #         for idx in range(source.shape[1]):
        #             data[curve_idx][axis][idx] = source[axis][idx]
        #         for idx in range(source.shape[1], max_points):
        #             data[curve_idx][axis][idx] = np.nan

        # this method works for 2-d arrays only (except first 3-d array)
        input_ndim = 2
        base_ndim = 3

        for arr in (base_3d_array, *arrays):
            assert isinstance(arr, np.ndarray), \
                "Expected ndarray, got {} instead.".format(type(arr))

        # use fix_axes_number instead
        # cnt_axes = arrays[0].shape[1]

        assert base_3d_array.ndim == base_ndim, \
            "Base ndarray must have 3 dimension. " \
            "Got {} instead.".format(base_3d_array.ndim)

        assert base_3d_array.shape[1] == axes_number, \
            "Wrong axes number in base ndarray. " \
            "Expected {}, got {}.".format(axes_number, base_3d_array.shape[0])

        for idx in range(len(arrays)):
            assert arrays[idx].ndim == input_ndim, \
                "Number of dimension must be {} for all ndarrays\n" \
                "Got {} in arrays[{}]".format(input_ndim, arrays[idx].ndim, idx)

            assert arrays[idx].shape[0] % axes_number == 0, \
                "Unsuitable number of rows ({}) for self data axes number ({})" \
                "".format(arrays[idx].shape[0], axes_number)

        # TODO: handle arrays with odd row number (or not...)

        max_points = max(arr.shape[2] if (arr.ndim == 3) else arr.shape[1]
                         for arr in (base_3d_array, *arrays))

        curves_count = sum(arr.shape[0] if (arr.ndim == 3) else (arr.shape[0] // axes_number)
                           for arr in (base_3d_array, *arrays))

        # tmp buffer
        data = np.full(shape=(curves_count, axes_number, max_points), fill_value=np.nan, dtype=dtype, order='C')

        # prepare 2-dimension slices [axis][point] to represent each curve
        # each slice have 2xN size,
        # where N - number of points in current sub array
        # first row is time row
        # and 2nd row is amplitude row
        slices_2d = list()

        for arr in (*(arr_2d for arr_2d in base_3d_array), *arrays):
            new_curves = arr.shape[0] // axes_number
            for idx in range(0, arr.shape[0], axes_number):
                slices_2d.append(arr[idx: idx + axes_number, :])

        for idx in range(curves_count):
            points = slices_2d[idx].shape[1]
            fill_2d_array(slices_2d[idx], data[idx, :, 0: points])

        return data

    @staticmethod
    def align_and_append_3d_arrays(*args, dtype=np.float64, axes_number=2):
        """Returns 3-dimension numpy.ndarray containing all input numpy.ndarrays.
        Input ndarray must be 3D: [curve_idx][axis][point_idx] where axes are time-axis and amplitude-axis

        Output data:
        Each curve consist of 2 rows: time-row and amplitude-row
        and have the shape == (1, 2, N),
        where N is the number of maximum points among all curves
        if the curve have n points and N > n:
        then the last (N - n) values of time-row and amp-row
        is filled with NaNs.

        :param args: a number of 3-dimension ndarray
        :type args: np.ndarray

        :param dtype: values type output array
        :type dtype: np.dtype

        :param axes_number: number of axes for output array
        :type axes_number: int

        :return: united ndarray
        :rtype: np.ndarray
        """

        # slower version
        #
        # @jit(nopython=True, nogil=True)
        # def fill_data(data, *args):
        #     curve_idx = 0
        #     for arr in args:
        #         for sub_cur in range(arr.shape[0]):
        #             for axis in range(2):
        #                 for idx in range(arr.shape[2]):
        #                     data[curve_idx][axis][idx] = arr[sub_cur][axis][idx]
        #                 for idx in range(arr.shape[2], max_points):
        #                     data[curve_idx][axis][idx] = np.nan
        #             curve_idx += 1

        fix_ndim = 3
        axes_number = 2

        for arr in args:
            assert isinstance(arr, np.ndarray), \
                "Expected ndarray, got {} instead.".format(type(arr))

        # use fix_axes_number instead
        # cnt_axes = args[0].shape[1]

        for idx in range(len(args)):
            assert args[idx].ndim == fix_ndim, \
                "Number of dimension must be {} for all ndarrays\n" \
                "Got {} in args[{}]".format(fix_ndim, args[idx].ndim, idx)
            assert args[idx].shape[1] == axes_number, \
                "Number of axes (shape[1] value) must be the same for all ndarrays\n" \
                "Expected {}, got {} in args[{}]".format(axes_number, args[idx].shape[0], idx)

        max_points = max(arr.shape[2] if (arr.ndim == 3) else arr.shape[1] for arr in args)
        curves_count = sum(arr.shape[0] if (arr.ndim == 3) else (arr.shape[0] // 2) for arr in args)

        # tmp buffer
        # data = np.ndarray(shape=(curves_count, 2, max_points), dtype=dtype, order='C')
        data = np.full(shape=(curves_count, axes_number, max_points), fill_value=np.nan, dtype=dtype, order='C')

        slices_2d = list()

        for arr in args:
            for idx in range(arr.shape[0]):
                slices_2d.append(arr[idx])

        for idx in range(curves_count):
            points = slices_2d[idx].shape[1]
            fill_2d_array(slices_2d[idx], data[idx, :, 0: points])

        return data


@guvectorize([(float64[:, :], float64[:, :])], '(a,n)->(a,n)', nopython=True, target="cpu")
def fill_2d_array(source, res):
    """

    :param source: 2-dimension ndarrays to copy data from
    :type source: np.ndarray

    :param res: 2-dimension ndarray to copy values to
    :type res: np.ndarray

    :return: None
    """
    for axis in range(source.shape[0]):
        for idx in range(source.shape[1]):
            res[axis][idx] = source[axis][idx]


# @guvectorize([(float64[:], float64[:, :], int64, float64[:, :])], '(n),(a,n),()->(b,n)', nopython=True, target="cpu")
# def fill_2d_array_and_multiply_time_guvector(time, source, axes_num, res):
#     """
#
#     :param source: 2-dimension ndarrays to copy data from
#     :type source: np.ndarray
#
#     :param res: 2-dimension ndarray to copy values to
#     :type res: np.ndarray
#
#     :return: None
#     """
#     time_rows = 0
#     for row in range(res.shape[0]):
#         if row % axes_num == 0:
#             time_rows += 1
#             for idx in range(source.shape[1]):
#                 res[row][idx] = time[idx]
#         else:
#             for idx in range(source.shape[1]):
#                 res[row][idx] = source[row - time_rows][idx]


@jit(nopython=True)
def fill_2d_array_and_multiply_time_jit(time, source, axes_num, res):
    """

    :param source: 2-dimension ndarrays to copy data from
    :type source: np.ndarray

    :param res: 2-dimension ndarray to copy values to
    :type res: np.ndarray

    :return: None
    """
    time_rows = int64(0)
    for row in range(res.shape[0]):
        if row % axes_num == 0:
            time_rows += 1
            for idx in range(source.shape[1]):
                res[row][idx] = time[idx]
        else:
            for idx in range(source.shape[1]):
                res[row][idx] = source[row - time_rows][idx]
    return res


def multiply_time_column_2d(arr_2d, axes_number):
    """

    :param arr_2d: 2-dimension ndarray
    :param axes_number:
    :return:
    """

    assert arr_2d.ndim == 2, \
        "Expected 2-dimensional numpy.ndarray as input data, " \
        "got {} dimensions instead.".format(arr_2d.ndim)

    curves = (arr_2d.shape[0] - 1) // (axes_number - 1)

    # for axes_number > 2
    assert (arr_2d.shape[0] - 1) % (axes_number - 1) == 0, \
        "Unsuitable number of non-time rows ({}) for self data axes number ({})." \
        "".format(arr_2d.shape[0] - 1, axes_number)

    points = arr_2d.shape[1]
    data = np.ndarray(shape=(curves * axes_number, points), dtype=arr_2d.dtype, order='C')

    # print()
    # print("EMPTY DATA")
    # print(data)
    for cur in range(curves):
        cur_row = cur * axes_number
        fill_2d_array(arr_2d[0: 1, :], data[cur_row: cur_row + 1, :])

    # print()
    # print("DATA WITH TIME")
    # print(data)

    time_rows = 0
    for row in range(1, arr_2d.shape[0]):
        fill_2d_array(arr_2d[row: row + 1, :], data[row + time_rows: row + time_rows + 1, :])
        # print("Copy from [{} : {}]  to  [{} : {}]".format(row, row + 1, row + time_rows, row + time_rows + 1))
        if row % (axes_number - 1) == 0:
            time_rows += 1

    # print()
    # print("DATA FILLED")
    # print(data)

    # fill_2d_array_and_multiply_time_jit(time, arr_2d[1:], axes_number, data)
    return data


# ===================================================================================
# --------    FUNCTIONS    ----------------------------------------------------------
# ...................................................................................

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
    col_len_list = [arr.shape[0] for arr in args]
    max_rows = max(col_len_list)
    data = np.empty(shape=(max_rows, 0), dtype=float, order='C')
    for arr in args:
        miss_rows = max_rows - arr.shape[0]
        nan_arr = np.empty(shape=(miss_rows, arr.shape[1]),
                           dtype=float, order='C')
        nan_arr[:] = np.nan
        # nan_arr[:] = np.NAN

        aligned_arr = np.append(arr, nan_arr, axis=0)
        data = np.append(data, aligned_arr, axis=1)
    if DEBUG:
        print("aligned array shape = {}".format(data.shape))
    return data


def align_and_append_row_ordered_ndarray(*args):
    """Returns 2D numpy.ndarray containing all input 2D numpy.ndarrays.
    (Each 2 rows is one data curve, and each column is data point)
    If input arrays have different number of points,
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

    data_type = args[0].dtype

    # ALIGN & APPEND
    # each curve should consist of 2 rows: x-row (time values) and y-row (amplitude values)
    max_points = max(arr.shape[1] for arr in args)
    data = np.empty(shape=(0, max_points), dtype=data_type, order='C')
    for arr in args:
        miss_points = max_points - arr.shape[1]
        nan_arr = np.full(shape=(arr.shape[0], miss_points),
                          fill_value=np.NAN,
                          dtype=data_type,
                          order='C')
        aligned_arr = np.append(arr, nan_arr, axis=1)
        data = np.append(data, aligned_arr, axis=0)
    # if DEBUG:
    #     print("aligned array shape = {}".format(data.shape))
    return data


def align_and_append_row_ordered_3d_array(*args):
    """Returns 2D numpy.ndarray containing all input 2D numpy.ndarrays.
    (Each 2 rows is one data curve, and each column is data point)
    If input arrays have different number of points,
    fills missing values with 'nan'.

    args -- 2d ndarrays
    """
    # CONST
    # each curve consist of 2 rows: time-row and amplitude-row
    # and have the shape == (1, 2, N),
    # where N is the number of maximum points among all curves
    # if the curve have n points and N > n:
    # then the last (N - n) values of time-row and amp-row
    # is filled with NaN
    rows_in_curve = 2

    # CHECK TYPE & LENGTH
    for arr in args:
        if not isinstance(arr, np.ndarray):
            raise TypeError("Input arrays must be instances "
                            "of numpy.ndarray class.")
        if np.ndim(arr) != 3:
            raise ValueError("Input arrays must have 3 dimensions.")

    data_type = args[0].dtype

    # TODO: handle arrays with odd row number (one time row for several value rows)

    # ALIGN & APPEND
    # each curve should consist of 2 rows: x-row (time values) and y-row (amplitude values)
    max_points = max(arr.shape[2] for arr in args)
    # print("max_points = {}".format(max_points))
    data = np.empty(shape=(0, 2, max_points), dtype=data_type, order='C')
    for arr in args:
        for curve_idx in range(arr.shape[0]):
            # take two rows (time-row and amplitude-row)
            # last index not included
            # print("from args: arr.shape = {}".format(arr.shape))
            single_curve = arr[curve_idx: curve_idx + 1, :, :]
            # print("single_curve.shape = {}".format(single_curve.shape))

            # single_curve have 3 dimension and consist of two rows:
            # time row [0, 0, :] and amplitude row [0, 1, :]
            miss_points = max_points - single_curve.shape[2]
            # print("miss_points = {}".format(miss_points))
            nan_arr = np.full(shape=(1, rows_in_curve, miss_points),
                              fill_value=np.NAN,
                              dtype=data_type,
                              order='C')
            # print("nan_arr.shape = {}".format(nan_arr.shape))
            aligned_arr = np.append(single_curve, nan_arr, axis=2)
            data = np.append(data, aligned_arr, axis=0)
    # if DEBUG:
    #     print("aligned array shape = {}".format(data.shape))
    return data


def align_and_append_row_ordered_3d_array_copy(*args):
    """Returns 3D numpy.ndarray containing all input 2D numpy.ndarrays.
    (Each 2 rows is one data curve, and each column is data point)
    If input arrays have different number of points,
    fills missing values with 'nan'.

    args -- 2d ndarrays
    """
    # CONST
    # each curve consist of 2 rows: time-row and amplitude-row
    # and have the shape == (1, 2, N),
    # where N is the number of maximum points among all curves
    # if the curve have n points and N > n:
    # then the last (N - n) values of time-row and amp-row
    # is filled with NaN
    rows_in_curve = 2

    data_type = args[0].dtype

    # TODO: handle arrays with odd row number (one time row for several value rows)

    max_points = 0
    for idx in range(len(args)):
        if args[idx].shape[2] > max_points:
            max_points = args[idx].shape[2]

    curves_count = 0
    for idx in range(len(args)):
        curves_count += args[idx].shape[0]

    # print("max_points = {}".format(max_points))
    data = np.ndarray(shape=(curves_count, 2, max_points), dtype=data_type, order='C')

    @jit(nopython=True, nogil=True)
    def fill_data(data, *args):
        curve_idx = 0
        for arr_idx in range(len(args)):
            arr = args[arr_idx]
            for sub_cur in range(arr.shape[0]):
                for axis in range(2):
                    for idx in range(arr.shape[2]):
                        data[curve_idx][axis][idx] = arr[sub_cur][axis][idx]
                    for idx in range(arr.shape[2], max_points):
                        data[curve_idx][axis][idx] = np.nan
                curve_idx += 1

    fill_data(data, *args)

    return data


# ==========================================================================
# --------   ROW-ORDERED ARRAY   -------------------------------------------

@print_func_time
@guvectorize([(float64[:], float64, float64, float64[:])], '(n),(),()->(n)', target='cuda')
def mult_del_row_order_vectorized_cuda(data, multiplyer, delay, res):
    for i in range(data.shape[0]):
        res[i] = data[i] * multiplyer - delay


@print_func_time
@guvectorize([(float64[:], float64, float64, float64[:])], '(n),(),()->(n)', target='cpu')
def mult_del_row_order_vectorized_cpu(data, multiplyer, delay, res):
    for i in range(data.shape[0]):
        res[i] = data[i] * multiplyer - delay


@print_func_time
@jit(nopython=True, nogil=True)
def mult_del_row_order_jit_nogil(data, multiplier, delay):
    row_number = data.shape[0]
    point_number = data.shape[1]
    for row_idx in np.arange(row_number):
        for point_idx in np.arange(point_number):
            data[row_idx][point_idx] = (data[row_idx][point_idx] *
                                        multiplier[row_idx] -
                                        delay[row_idx])


@print_func_time
@jit(nopython=True)
def mult_del_row_order_jit(data, multiplier, delay):
    row_number = data.shape[0]
    point_number = data.shape[1]
    for row_idx in np.arange(row_number):
        for point_idx in np.arange(point_number):
            data[row_idx][point_idx] = (data[row_idx][point_idx] *
                                        multiplier[row_idx] -
                                        delay[row_idx])


# # @print_func_time
# @cuda.jit('void(float64[:,:], float64[:], float64[:])')
# def mult_del_row_order_cuda(data, multiplier, delay):
#     row_number = data.shape[0]
#     point_number = data.shape[1]
#     for row_idx in range(row_number):
#         for point_idx in range(point_number):
#             data[row_idx][point_idx] = (data[row_idx][point_idx] *
#                                         multiplier[row_idx] -
#                                         delay[row_idx])


@print_func_time
@jit(nopython=True, nogil=True, parallel=True)
def mult_del_row_order_parallel(data, multiplier, delay):
    row_number = data.shape[0]
    point_number = data.shape[1]
    for row_idx in np.arange(row_number):
        for point_idx in np.arange(point_number):
            data[row_idx][point_idx] = (data[row_idx][point_idx] *
                                        multiplier[row_idx] -
                                        delay[row_idx])


@print_func_time
def mult_del_row_order_std(data, multiplier, delay):
    row_number = data.shape[0]
    point_number = data.shape[1]
    print("row_number = {}    point_number = {}".format(row_number, point_number))
    for row_idx in np.arange(row_number):
        for point_idx in np.arange(point_number):
            data[row_idx][point_idx] = (data[row_idx][point_idx] *
                                        multiplier[row_idx] -
                                        delay[row_idx])

# ==========================================================================
# --------   CLASSIC-ORDERED ARRAY   ---------------------------------------


@print_func_time
@jit(nopython=True, nogil=True, parallel=True)
def mult_del_classic_order_parallel(data, multiplier, delay):
    col_number = data.shape[1]
    row_number = data.shape[0]
    for row_idx in np.arange(row_number):
        for col_idx in np.arange(col_number):
            data[row_idx][col_idx] = (data[row_idx][col_idx] *
                                      multiplier[col_idx] -
                                      delay[col_idx])


@print_func_time
@jit(nopython=True)
def mult_del_classic_order_jit(data, multiplier, delay):
    col_number = data.shape[1]
    row_number = data.shape[0]
    for row_idx in np.arange(row_number):
        for col_idx in np.arange(col_number):
            data[row_idx][col_idx] = (data[row_idx][col_idx] *
                                      multiplier[col_idx] -
                                      delay[col_idx])


@print_func_time
def mult_del_classic_order_std(data, multiplier, delay):
    col_number = data.shape[1]
    row_number = data.shape[0]
    for row_idx in np.arange(row_number):
        for col_idx in np.arange(col_number):
            data[row_idx][col_idx] = (data[row_idx][col_idx] *
                                        multiplier[col_idx] -
                                        delay[col_idx])


# ===============================================================================
# --------   3D - ROW-ORDERED ARRAY   -------------------------------------------
# ===============================================================================


# @print_func_time
# @guvectorize([(float64[:], float64, float64, float64[:])], '(n),(),()->(n)', target='cuda')
# def mult_del_3d_vectorized_cuda(data, multiplyer, delay, res):
#     for i in range(data.shape[0]):
#         res[i] = data[i] * multiplyer - delay


@print_func_time
@vectorize([float64(float64, float64, float64)],
           target='cpu', nopython=True)
def mult_del_3d_single_vector(data, multiplier, delay):
    return data * multiplier - delay


@print_func_time
@guvectorize([(float64[:, :, :], float64[:, :], float64[:, :], float64[:, :, :])],
             '(c,a,n),(c,a),(c,a)->(c,a,n)',
             target='cpu', nopython=True)
def mult_del_3d_vectorized_cpu(data, multiplier, delay, res):
    for curve in range(data.shape[0]):
        for axis in range(data.shape[1]):
            res[curve][axis] = data[curve][axis] * multiplier[curve][axis] - delay[curve][axis]


@vectorize([float64(float64, float64, float64)], target='cpu', nopython=True)
def mult_del_single_vectorized_cpu(data, multiplier, delay):
    return data * multiplier - delay


@guvectorize([(float64[:], float64, float64, float64[:])],
             '(n),(),()->(n)',
             target='cpu', nopython=True)
def mult_del_guvectorized_for_slice(data, multiplier, delay, res):
    for idx in range(data.shape[0]):
        res[idx] = data[idx] * multiplier - delay


@print_func_time
@jit(nopython=True)
def mult_del_3d_jit_of_vector(data, multiplier, delay):
    for curve in range(data.shape[0]):
        for axis in range(data.shape[1]):
            data[curve][axis] = mult_del_single_vectorized_cpu(data[curve][axis],
                                                               multiplier[curve][axis],
                                                               delay[curve][axis])


@print_func_time
@jit(nopython=True)
def mult_del_3d_jit(data, multiplier, delay):
    curves = data.shape[0]
    points = data.shape[2]
    for cur in range(curves):
        for axis in range(2):
            for point_idx in range(points):
                data[cur][axis][point_idx] = (data[cur][axis][point_idx] *
                                              multiplier[cur][axis] -
                                              delay[cur][axis])


# =================================================================================================
# -------------           MAIN           ----------------------------------------------------------
# -------------------------------------------------------------------------------------------------
@print_func_time
@jit(nopython=True, nogil=True)
def mult_del_3d_jit_v2(data, multiplier, delay):
    curves = data.shape[0]
    points = data.shape[2]
    for cur in range(curves):
        for axis in range(2):
            for point_idx in range(points):
                data[cur][axis][point_idx] = (data[cur][axis][point_idx] *
                                              multiplier[cur][axis] -
                                              delay[cur][axis])
# -------------------------------------------------------------------------------------------------
# =================================================================================================


@print_func_time
@jit(nopython=True, nogil=True, parallel=True)
def mult_del_3d_parallel(data, multiplier, delay):
    curves = data.shape[0]
    points = data.shape[2]
    time_axis = 0
    amp_axis = 1
    for cur in range(curves):
        for point_idx in range(points):
            data[cur][time_axis][point_idx] = (data[cur][0][point_idx] *
                                               multiplier[cur][time_axis] -
                                               delay[cur][time_axis])

            data[cur][amp_axis][point_idx] = (data[cur][1][point_idx] *
                                              multiplier[cur][amp_axis] -
                                              delay[cur][amp_axis])


@print_func_time
def mult_del_3d_std(data, multiplier, delay):
    curves = data.shape[0]
    points = data.shape[2]
    time_axis = 0
    amp_axis = 1
    for cur in range(curves):
        for point_idx in range(points):
            data[cur][time_axis][point_idx] = (data[cur][0][point_idx] *
                                               multiplier[cur][time_axis] -
                                               delay[cur][time_axis])

            data[cur][amp_axis][point_idx] = (data[cur][1][point_idx] *
                                              multiplier[cur][amp_axis] -
                                              delay[cur][amp_axis])

# =================================================================================================
# -------------------------------------------------------------------------------------------------
# **********     TESTS     ************************************************************************
# -------------------------------------------------------------------------------------------------
# =================================================================================================


def test_numba(number=1000000):
    np_data = np.zeros(shape=(2, number), dtype=np.float64, order='C')
    for idx in range(number):
        np_data[0, idx] = idx
        np_data[1, idx] = random.random() * 1000

    print("==========================================================================")
    print("--------   ROW-ORDERED ARRAY   -------------------------------------------")
    print("..........................................................................")
    print("Data.shape = {}".format(np_data.shape))
    print("Min_y = {}     Max_y = {}".format(min(np_data[:, 1]), max(np_data[:, 1])))
    print()
    print(np_data)
    print()

    mult = np.zeros((2,), dtype=np_data.dtype, order='C')
    mult[0] = 1E+6
    mult[1] = 1.13768
    delay = np.zeros((2,), dtype=np_data.dtype, order='C')
    delay[0] = 3543137
    delay[1] = 437.513

    # WORKING COPY of data
    out_tmp = np.zeros_like(np_data)
    # tmp_vector = np_data.copy()
    # tmp_vector_out = np.zeros_like(tmp_vector)
    # tmp_njit = np_data.copy()
    # tmp_nogil = np_data.copy()

    print("Multiplier = {}".format(mult))
    print("Delay      = {}".format(delay))
    print("============================================")

    print("Standard function:")
    std_output = np_data.copy()
    mult_del_row_order_std(std_output, mult, delay)
    repeats = 10
    print()
    print(std_output)
    print()

    print("Jitted function:")
    for _ in range(repeats):
        tmp_std = np_data.copy()
        mult_del_row_order_jit(tmp_std, mult, delay)
    # print("Data is {}".format("OK" if np.isclose(std_output, tmp_std).all() else "bad!!"))

    # tmp_std = np_data.copy()
    # mult_del_row_order_jit(tmp_std, mult, delay)

    # print("Data is {}".format("OK" if np.isclose(std_output, tmp_std).all() else "bad!!"))
    print()

    print("NO-GIL function:")
    for _ in range(repeats):
        tmp_std = np_data.copy()
        mult_del_row_order_jit_nogil(tmp_std, mult, delay)
    # print("Data is {}".format("OK" if np.isclose(std_output, tmp_std).all() else "bad!!"))

    # tmp_std = np_data.copy()
    # mult_del_row_order_jit_nogil(tmp_std, mult, delay)

    # print("Data is {}".format("OK" if np.isclose(std_output, tmp_std).all() else "bad!!"))
    print()

    print("Parallel no-gil function:")
    for _ in range(repeats):
        tmp_std = np_data.copy()
        mult_del_row_order_parallel(tmp_std, mult, delay)
    # print("Data is {}".format("OK" if np.isclose(std_output, tmp_std).all() else "bad!!"))

    # tmp_std = np_data.copy()
    # mult_del_row_order_parallel(tmp_std, mult, delay)

    # print("Data is {}".format("OK" if np.isclose(std_output, tmp_std).all() else "bad!!"))

    # print()
    # print("Pause 5 seconds ~~~~~~~~")
    # print()
    # time.sleep(5)

    print("Guvectorized (CPU) function:")
    for _ in range(repeats):
        tmp_std = np_data.copy()
        out_tmp = np.zeros_like(np_data)
        mult_del_row_order_vectorized_cpu(tmp_std, mult, delay, out_tmp)
        # print("Data is {}".format("OK" if np.isclose(out_tmp, std_output).all() else "bad!!"))

    print("Guvectorized (CPU) no out buffer function:")
    for _ in range(repeats):
        tmp_std = np_data.copy()
        mult_del_row_order_vectorized_cpu(tmp_std, mult, delay, tmp_std)
        # print("Data is {}".format("OK" if np.isclose(tmp_std, std_output).all() else "bad!!"))

    print("Guvectorized (CUDA) no out buffer function:")
    for _ in range(repeats):
        tmp_std = np_data.copy()
        mult_del_row_order_vectorized_cuda(tmp_std, mult, delay, tmp_std)
        # print("Data is {}".format("OK" if np.isclose(tmp_std, std_output).all() else "bad!!"))

    # print("CUDA function:")
    # threadsperblock = 32
    # blockspergrid = (tmp_std.size + (threadsperblock - 1)) // threadsperblock
    # for _ in range(2):
    #     tmp_std = np_data.copy()
    #     # mult_del_row_order_jit_nogil(tmp_std, mult, delay)
    #     mult_del_row_order_cuda[threadsperblock, blockspergrid](tmp_std, mult, delay)
    #     print("Data is {}".format("OK" if np.isclose(tmp_std, std_output).all() else "bad!!"))
    # print(tmp_std)
    # print()
    # print(std_output)

    # print("Data is close for {} from {} elements".format(np.isclose(out_tmp, std_output).sum(),
    #                                                      std_output.shape[0] * std_output.shape[1]))
    # print(out_tmp)
    # print()
    # print(tmp_std)

    # tmp_std = np_data.copy()
    # out_tmp = np.zeros_like(np_data)
    # mult_del_row_order_vectorized(tmp_std, mult, delay, out_tmp)

    # print("Data is {}".format("OK" if np.isclose(out_tmp, std_output).all() else "bad!!"))
    # print("Data is close for {} from {} elements".format(np.isclose(out_tmp, std_output).sum(),
    #                                                      std_output.shape[0] * std_output.shape[1]))
    print()

    # for idx in range(30):
    #     print("{x1}\t{y1}\n{x2}\t{y2}\n".format(x1=std_output[0, idx], y1=std_output[1, idx],
    #                                             x2=out_tmp[0, idx], y2=out_tmp[1, idx]))

    # ==========================================================================
    # --------   CLASSIC COLUMN-ORDERED ARRAY   --------------------------------
    # ..........................................................................

    @print_func_time
    @jit(nopython=True)
    def mult_del_col_order_jit(data, multiplier, delay):
        col_number = data.shape[1]
        row_number = data.shape[0]
        for col_idx in np.arange(col_number):
            for row_idx in np.arange(row_number):
                data[col_idx][row_idx] = (data[col_idx][row_idx] *
                                          multiplier[col_idx] -
                                          delay[col_idx])

    @print_func_time
    def mult_del_trans_and_vec(data, multiplier, delay, res):
        @guvectorize([(float64[:], float64, float64, float64[:])], '(n),(),()->(n)')
        def mult_del_row_order_vectorized(data, multiplyer, delay, res):
            for i in range(data.shape[0]):
                res[i] = data[i] * multiplyer - delay

        out_data = np.transpose(data)
        res = np.transpose(res)
        mult_del_row_order_vectorized(out_data, multiplier, delay, res)
        return np.transpose(res)

    classic_data = np.zeros(shape=(number, 2), dtype=np.float64, order='C')
    for idx in range(number):
        classic_data[idx, 0] = idx
        classic_data[idx, 1] = random.random() * 1000

    # print("Data.shape = {}".format(classic_data.shape))
    # print("Min_y = {}     Max_y = {}".format(min(classic_data[:, 1]), max(classic_data[:, 1])))
    # print()
    # print(classic_data)
    # print()
    # cx = classic_data.copy()

# =================================================================================================
# -------------------------------------------------------------------------------------------------
# **********     ARRAY FUNCTION     ***************************************************************
# -------------------------------------------------------------------------------------------------
# =================================================================================================


def make_array(rows, columns, np_type=np.float64):
    np_data = np.zeros(shape=(rows, columns), dtype=np_type, order='C')
    for idx in range(rows):
        for col in range(0, columns, 2):
            np_data[idx, col] = idx
            np_data[idx, col + 1] = random.random() * 1000
    return np_data


def make_2d_multiplier(elements, np_type=np.float64):
    np_data = np.zeros(shape=(elements // 2, 2), dtype=np_type, order='C')
    for idx in range(elements // 2):
        np_data[idx, 0] = 1E6
        np_data[idx, 1] = random.random() * 33
    return np_data


def make_1d_multiplier(elements, np_type=np.float64):
    np_data = np.zeros(shape=(elements,), dtype=np_type, order='C')
    for idx in range(0, elements, 2):
        np_data[idx] = 1E6
        np_data[idx + 1] = random.random() * 33
    return np_data


def make_2d_delay(elements, np_type=np.float64):
    np_data = np.zeros(shape=(elements // 2, 2), dtype=np_type, order='C')
    for idx in range(elements // 2):
        np_data[idx, 0] = random.random() * 100
        np_data[idx, 1] = random.random() * 100
    return np_data


def make_1d_delay(elements, np_type=np.float64):
    np_data = np.zeros(shape=(elements,), dtype=np_type, order='C')
    for idx in range(0, elements, 2):
        np_data[idx] = random.random() * 100
        np_data[idx + 1] = random.random() * 100
    return np_data


def make_row_ordered_array(rows, points, np_type=np.float64):
    np_data = np.zeros(shape=(rows, points), dtype=np_type, order='C')
    for idx in range(points):
        for row in range(0, rows, 2):
            np_data[row, idx] = idx
            np_data[row + 1, idx] = random.random() * 1000
    return np_data


def make_row_ordered_array_single_time_row(rows, points, np_type=np.float64):
    np_data = np.zeros(shape=(rows, points), dtype=np_type, order='C')
    for idx in range(points):
        np_data[0, idx] = idx
    for idx in range(points):
        for row in range(1, rows):
            np_data[row, idx] = random.random() * 1000
    return np_data


def make_row_ordered_3d_array(rows, points, np_type=np.float64):
    np_data = np.zeros(shape=(rows // 2, 2, points), dtype=np_type, order='C')
    for curve_idx in range(rows // 2):
        for idx in range(points):
            np_data[curve_idx, 0, idx] = idx
            np_data[curve_idx, 1, idx] = random.random() * 1000
    return np_data


def make_1d_array(points, np_type=np.float64):
    np_data = np.zeros(shape=(points, ), dtype=np_type, order='C')
    for idx in range(points):
        np_data[idx] = random.random() * 1000
    return np_data


def convert_3d_to_2d_row_ordered(arr):
    curves = arr.shape[0]
    axes = arr.shape[1]
    points = arr.shape[2]
    list_1d_arr_str = list()
    for cur in range(curves):
        list_1d_arr_str.append("arr[{}]".format(cur))
    str_args = ", ".join(list_1d_arr_str)
    command = "align_and_append_row_ordered_ndarray(" + str_args + ")"
    return eval(command)


def convert_2d_to_1d_row_ordered(arr):
    curves = arr.shape[0]
    axes = arr.shape[1]
    np_data = np.zeros(shape=(curves * axes,), dtype=arr.dtype, order='C')
    for cur in range(curves):
        for ax in range(axes):
            np_data[cur * axes + ax] = arr[cur][ax]
    return np_data


def speed_test_row_order_2d_array():
    # test_numba(10000000)

    @print_func_time
    def rotate_array(arr):
        return np.transpose(arr)

    # np_data = np.zeros(shape=(2, 1000000000), dtype=np.float64, order='C')
    # print(np_data.shape)
    x1 = make_row_ordered_array(rows=4,  points=900000)
    x2 = make_row_ordered_array(rows=2,  points=1200000)
    x3 = make_row_ordered_array(rows=12, points=1000000)

    data = None
    for _ in range(1):
        start = time.time_ns()
        data = align_and_append_row_ordered_ndarray(x1, x2, x3)
        end = time.time_ns()
        print('[*] Runtime of the align_and_append function: {:.15f} seconds.'.format((end - start) / 1.0E+9))

    # x1_rotated = x1.copy()
    # x2_rotated = x2.copy()
    # x3_rotated = x3.copy()
    # print("x1_rotated.shape = {}".format(x1_rotated.shape))
    # x1_rotated = np.transpose(x1_rotated)
    # print("x1_rotated.shape = {}".format(x1_rotated.shape))
    # x2_rotated = np.transpose(x2_rotated)
    # x3_rotated = np.transpose(x3_rotated)
    # print("x1_rotated.shape = {}".format(x1_rotated.shape))
    # signals = None
    # for _ in range(1):
    #     start = time.time_ns()
    #     signals = SignalsData(x1_rotated)
    #     signals.add_curves(x2_rotated)
    #     signals.add_curves(x3_rotated)
    #     end = time.time_ns()
    #     print('[*] Runtime of the SignalsData creation is:   {:.15f} seconds.'.format((end - start) / 1.0E+9))
    # print("SignalsData.shape = {}".format(signals.get_array().shape))

    # print(data)

    repeats = 10

    mult = make_1d_multiplier(data.shape[0])
    delay = make_1d_delay(data.shape[0])

    print()
    print("Data ({}):".format(data.shape))
    print()
    print("Multiplier ({}):".format(mult.shape))
    print(mult)
    print()
    print("Delay ({}):".format(delay.shape))
    print(delay)

    # ==========================================================

    std_output = data.copy()
    mult_del_row_order_std(std_output, mult, delay)
    print()
    # print(std_output)

    print()
    print("Jitted function ~~~~~")
    for _ in range(repeats):
        tmp_data = data.copy()
        mult_del_row_order_jit(tmp_data, mult, delay)
        # print()
        # print(np.isclose(tmp_data, std_output))

    print()
    print("Jitted no-gil function ~~~~~")
    for _ in range(repeats):
        tmp_data = data.copy()
        mult_del_row_order_jit_nogil(tmp_data, mult, delay)
        # print()
        # print(np.isclose(tmp_data, std_output))

    print()
    print("Jitted no-gil parallel function ~~~~~")
    for _ in range(repeats):
        tmp_data = data.copy()
        mult_del_row_order_parallel(tmp_data, mult, delay)
        # print()
        # print(np.isclose(tmp_data, std_output))

    print()
    print("GUVectorize (CPU) function ~~~~~")
    print("Multiplyer shape = {}".format(mult.view()))
    print()
    print("Delay shape = {}".format(delay.view()))
    print()
    print("New data shape = {}".format(tmp_data.shape))
    for _ in range(repeats):
        tmp_data = data.copy()
        mult_del_row_order_vectorized_cpu(tmp_data, mult, delay, tmp_data)
        # print()
        # print(np.isclose(tmp_data, std_output))

    # print()
    # print("GUVectorize (CUDA) function ~~~~~")
    # for _ in range(repeats):
    #     tmp_data = data.copy()
    #     mult_del_row_order_vectorized_cuda(tmp_data, mult, delay, tmp_data)
    #     # print()
    #     # print(np.isclose(tmp_data, std_output))


def speed_test_row_order_3d_array():
    # np_data = np.zeros(shape=(2, 1000000000), dtype=np.float64, order='C')
    # print(np_data.shape)
    x1 = make_row_ordered_3d_array(rows=4, points=900000)
    x2 = make_row_ordered_3d_array(rows=2, points=1200000)
    x3 = make_row_ordered_3d_array(rows=12, points=1000000)

    print(x1.shape)
    print(x2.shape)
    print(x3.shape)

    data = None
    for _ in range(5):
        start = time.time_ns()
        data = align_and_append_row_ordered_3d_array(x1, x2, x3)
        end = time.time_ns()
        print('[*] Runtime of the align_and_append function: {:.15f} seconds.'.format((end - start) / 1.0E+9))

    print()
    data2 = None
    for _ in range(5):
        start = time.time_ns()
        data2 = SignalsDataEnhanced.align_and_append_3d_arrays(x1, x2, x3)
        end = time.time_ns()
        print('[*] Runtime of the copy-to-aligned-array function: {:.15f} seconds.'.format((end - start) / 1.0E+9))

    print()
    print(np.isclose(data, data2))
    print()

    mult = make_2d_multiplier(data.shape[0] * 2)
    delay = make_2d_delay(data.shape[0] * 2)

    print()
    print("Multiplier ({}):".format(mult.shape))
    print(mult)
    print()
    print("Delay ({}):".format(delay.shape))
    print(delay)

    repeats = 10

    # ======================================
    # ---   TESTS   ------------------------
    # ======================================

    std_output = data.copy()
    print("------------   STD OUTPUT   --------------------")
    # mult_del_3d_std(std_output, mult, delay)
    # print(std_output)
    print("----------------------------------------")

    print()
    print("Jitted function ~~~~~")
    for _ in range(repeats):
        std_output = data.copy()
        mult_del_3d_jit(std_output, mult, delay)
    # data_is_ok = np.isclose(std_output[:, :, :900000], std_output[:, :, :900000]).all()
    # print("Data is {}".format("Ok" if data_is_ok else "Bad!!"))
    print()

    print()
    print("Jitted function v2 ~~~~~")
    tmp_data = None
    for _ in range(repeats):
        tmp_data = data.copy()
        mult_del_3d_jit_v2(tmp_data, mult, delay)
    data_is_ok = np.isclose(tmp_data[:, :, :900000], std_output[:, :, :900000]).all()
    print("Data is {}".format("Ok" if data_is_ok else "Bad!!"))
    print()

    print()
    print("GUVectorize (CPU) function ~~~~~")
    for _ in range(repeats):
        tmp_data = data.copy()
        mult_del_3d_vectorized_cpu(tmp_data, mult, delay, tmp_data)
    data_is_ok = np.isclose(tmp_data[:, :, :900000], std_output[:, :, :900000]).all()
    print("Data is {}".format("Ok" if data_is_ok else "Bad!!"))
    print()

    print()
    print("Jitted loop of vectorized (CPU) function ~~~~~")
    for _ in range(repeats):
        tmp_data = data.copy()
        mult_del_3d_jit_of_vector(tmp_data, mult, delay)
    data_is_ok = np.isclose(tmp_data[:, :, :900000], std_output[:, :, :900000]).all()
    print("Data is {}".format("Ok" if data_is_ok else "Bad!!"))
    print()

    print()
    print("Single vector for 3D array ~~~~~")
    tmp_data = None
    mult_3d = mult[:, :, np.newaxis]
    delay_3d = delay[:, :, np.newaxis]
    for _ in range(repeats):
        tmp_data = data.copy()
        tmp_data = mult_del_3d_single_vector(tmp_data, mult_3d, delay_3d)
    # data_is_ok = np.isclose(tmp_data[:, :, :900000], std_output[:, :, :900000]).all()
    # print("Data is {}".format("Ok" if data_is_ok else "Bad!!"))
    print()

    print()
    print(" SLICES 1D ~~~~~")
    tmp_data = None
    for _ in range(repeats):
        tmp_data = data.copy()
        start = time.time_ns()
        for curve_idx in range(data.shape[0]):
            for axis_idx in range(data.shape[1]):
                mult_del_guvectorized_for_slice(tmp_data[curve_idx][axis_idx],
                                                mult[curve_idx][axis_idx],
                                                delay[curve_idx][axis_idx],
                                                tmp_data[curve_idx][axis_idx])
        end = time.time_ns()
        print('[*] Runtime of the SignalsDataEnhanced(data): {:.15f} seconds.'.format((end - start) / 1.0E+9))
    data_is_ok = np.isclose(tmp_data[:, :, :900000], std_output[:, :, :900000]).all()
    print("Data is {}".format("Ok" if data_is_ok else "Bad!!"))
    print()


    print("------------------------------------------")
    print("Convert to 2d")

    tmp_2d = convert_3d_to_2d_row_ordered(data)
    mult_1d = convert_2d_to_1d_row_ordered(mult)
    delay_1d = convert_2d_to_1d_row_ordered(delay)
    print("Multiplyer shape = {}".format(mult_1d.view()))
    print()
    print("Delay shape = {}".format(delay_1d.view()))
    print()
    print("New data shape = {}".format(tmp_2d.shape))
    print()
    print("GUVectorize 2D (CPU) function ~~~~~")
    for _ in range(repeats):
        tmp_data = tmp_2d.copy()
        mult_del_row_order_vectorized_cpu(tmp_data, mult_1d, delay_1d, tmp_data)


if __name__ == "__main__":

    @print_func_time
    def rotate_array(arr):
        return np.transpose(arr)

    # print()
    # print("========================================================================================")
    # print("-----------     2D ARRAY     -----------------------------------------------------------")
    # print("========================================================================================")
    # print()
    #
    # speed_test_row_order_2d_array()

    # print()
    print("========================================================================================")
    print("-----------     3D ARRAY     -----------------------------------------------------------")
    print("========================================================================================")
    print()
    speed_test_row_order_3d_array()

    print()
    print("========================================================================================")
    print("-----------     SIGNALS_DATA_ENHANCED     ----------------------------------------------")
    print("========================================================================================")
    print()

    # print("Current memory usage = {} bytes".format(process.memory_info().rss))
    # print()
    # x1 = make_row_ordered_3d_array(rows=8, points=900000)
    # x2 = make_row_ordered_3d_array(rows=4, points=1200000)
    #
    # data = None
    # for _ in range(1):
    #     start = time.time_ns()
    #     data = align_and_append_row_ordered_3d_array(x1, x2)
    #     end = time.time_ns()
    #     print('[*] Runtime of the align_and_append function: {:.15f} seconds.'.format((end - start) / 1.0E+9))
    #
    # print("Data shape = {}".format(data.shape))
    # # print(data)
    # print()
    # print("Current memory usage = {} bytes".format(process.memory_info().rss))
    #
    # for _ in range(10):
    #     tmp = data.copy()
    #     start = time.time_ns()
    #     signals = SignalsDataEnhanced(tmp)
    #     end = time.time_ns()
    #     print('[*] Runtime of the SignalsDataEnhanced(data): {:.15f} seconds.'.format((end - start) / 1.0E+9))
    #     print()
    #
    # print("SignalsDataEnhanced.data.shape = {}".format(signals.data.shape))
    # # (self, from_array=None, labels=None,
    # # units=None, time_units=None, dtype=np.float64):

    print()
    print("========================================================================================")
    print("-----------     SIGNALS_DATA_ENHANCED   FROM 2D  --------------------------------------")
    print("========================================================================================")
    print()


    x1 = make_row_ordered_array(rows=4, points=900000)
    x2 = make_row_ordered_array(rows=2, points=1200000)
    x3 = make_row_ordered_array_single_time_row(rows=7, points=1000000)

    print(x3)
    print()
    print()
    print()

    print("Data shape = {}; {}; {}".format(x1.shape, x2.shape, x3.shape))
    # print(data)
    print()
    print("Current memory usage = {} bytes".format(process.memory_info().rss))

    signals = None
    for _ in range(10):
        tmp = x1.copy()
        start = time.time_ns()
        signals = SignalsDataEnhanced(tmp)
        end = time.time_ns()
        print('[*] Runtime of the SignalsDataEnhanced(data): {:.15f} seconds.'.format((end - start) / 1.0E+9))
        print()

    print("Data shape = {}".format(signals.data.shape))

    start = time.time_ns()
    signals.add_from_array(x2)
    end = time.time_ns()
    print('[*] Runtime of the SignalsDataEnhanced.append(x2): {:.15f} seconds.'.format((end - start) / 1.0E+9))
    print()
    print("Data shape = {}".format(signals.data.shape))

    start = time.time_ns()
    signals.add_from_array(x3, force_single_time_row=True)
    end = time.time_ns()
    print('[*] Runtime of the SignalsDataEnhanced.append(x3): {:.15f} seconds.'.format((end - start) / 1.0E+9))
    print()
    print("Data shape = {}".format(signals.data.shape))

    print()
    for curve in range(3, 9):
        print("Curve[{}]".format(curve))
        print("X        Y")  # 4 5
        print("{}       {}".format(signals.data[curve][0][0], signals.data[curve][1][0]))
        print("{}       {}".format(signals.data[curve][0][1], signals.data[curve][1][1]))
        print("{}       {}".format(signals.data[curve][0][2], signals.data[curve][1][2]))
        print()

    print("=================================")
    print(signals.get_labels())
    print()
    print(signals.get_units())
    print()
    print()