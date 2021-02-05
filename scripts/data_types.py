# Python 3.6
"""
Maintainer: Shpakov Konstantin
Link: https://github.com/shpakovkv/SignalProcess
"""

import numpy as np
from numba import jit, guvectorize, float64, float32, int32, int64

# =======================================================================
# -----     CONST     ---------------------------------------------------
# =======================================================================
DEBUG = 0
VERBOSE = 0


# =======================================================================
# -----     CLASSES     -------------------------------------------------
# =======================================================================
class SignalsData:

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
        elif from_array is not None:
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

        self.data = align_and_append_2d_arrays(self.data,
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
            self.data = align_and_append_3d_arrays(self.data, input_data, dtype=self.dtype)
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

    def get_array_to_print(self, curves_list=None):
        # DOTO: test SignalsData save
        list_of_2d_arr = list()
        for curve in self.data:
            list_of_2d_arr.append(curve)
        return align_and_append_ndarray(*list_of_2d_arr)

    def get_x(self, curve_idx=-1, curve_label=None):
        curve_idx = self._check_and_get_index(curve_idx, curve_label)
        return self.data[curve_idx][0]

    def get_y(self, curve_idx=-1, curve_label=None):
        curve_idx = self._check_and_get_index(curve_idx, curve_label)
        return self.data[curve_idx][1]

    def get_curve(self, idx=-1, label=None):
        idx = self._check_and_get_index(idx, label)
        return self.data[idx]

    def get_labels(self):
        """Returns a list of the curves labels
        (copy of the self.labels list).

        :return: labels (copy)
        :rtype: list
        """
        return list(self.labels)

    def get_units(self):
        """Returns a list of the curves units
        (copy of the self.units list).

        :return: units (copy)
        :rtype: list
        """
        return list(self.units)

    def get_curve_units(self, idx=-1, label=None):
        idx = self._check_and_get_index(idx, label)
        return self.units[idx]

    def get_curve_label(self, idx):
        idx = self._check_and_get_index(idx=idx)
        return self.labels[idx]

    def get_curve_idx(self, label):
        if label in self.labels:
            return self.labels.index(label)
        return None

    def _check_and_get_index(self, idx=-1, label=None):
        if label is not None:
            assert label in self.labels, \
                "No curve with label '{}' found.".format(label)
            idx = self.get_curve_idx(label)
        assert 0 <= idx < self.cnt_curves, \
            "Curve index [{}] is out of bounds.".format(idx)
        return idx

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
        self._time_units = s

    def is_empty(self):
        """Check if there are no curves (no data) in this structure.

        :return: True if there are no curves, else False
        :rtype: bool
        """
        return True if self.cnt_curves == 0 else False


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

    print()
    print("EMPTY DATA")
    print(data)
    for cur in range(curves):
        cur_row = cur * axes_number
        fill_2d_array(arr_2d[0: 1, :], data[cur_row: cur_row + 1, :])

    print()
    print("DATA WITH TIME")
    print(data)

    time_rows = 0
    for row in range(1, arr_2d.shape[0]):
        fill_2d_array(arr_2d[row: row + 1, :], data[row + time_rows: row + time_rows + 1, :])
        print("Copy from [{} : {}]  to  [{} : {}]".format(row, row + 1, row + time_rows, row + time_rows + 1))
        if row % (axes_number - 1) == 0:
            time_rows += 1

    print()
    print("DATA FILLED")
    print(data)

    # fill_2d_array_and_multiply_time_jit(time, arr_2d[1:], axes_number, data)
    return data


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


class SinglePeak:
    """Peak object. Contains information on one peak point.
    """
    def __init__(self, time=None, value=None, index=None,
                 sqr_l=0, sqr_r=0):
        """
        :param time: time (X) value of peak point
        :param value: amplitude (Y) value of peak point
        :param index: index of peak point (in SingleCurve data)
        :param sqr_l: 'square factor' of the left edge of the peak:
                      the closer the both sqr_l and sqr_r values
                      to '1', the higher the probability
                      that this is an erroneous peak
        :param sqr_r: same as sqr_r
        """
        self.time = time
        self.val = value
        self.idx = index
        self.sqr_l = sqr_l
        self.sqr_r = sqr_r

    def invert(self):
        if self.val is not None:
            self.val = -self.val

    def get_time_val(self):
        return [self.time, self.val]

    def set_time_val_idx(self, data):
        if len(data) != 3:
            raise ValueError("Wrong number of values to unpack. "
                             "3 expected, " + str(len(data)) +
                             " given.")
        self.time = data[0]
        self.val = data[1]
        self.idx = data[2]

    def set_data_full(self, data):
        count = 5
        if len(data) != count:
            raise ValueError("Wrong number of values to unpack. "
                             "{} expected, {} given."
                             "".format(count, len(data)))
        self.time = data[0]
        self.val = data[1]
        self.idx = data[2]
        self.sqr_l = data[3]
        self.sqr_r = data[4]

    def get_time_val_idx(self):
        return [self.time, self.val, self.idx]

    def get_data_full(self):
        return [self.time, self.val, self.idx, self.sqr_l, self.sqr_r]

    xy = property(get_time_val, doc="Get [time, value] of peak.")
    data = property(get_time_val_idx, set_time_val_idx,
                    doc="Get/set [time, value, index] of peak.")
    data_full = property(get_data_full, set_data_full,
                         doc="Get/set [time, value, index, "
                             "sqr_l, sqr_r] of peak.")


# =======================================================================
# -----     FUNCTIONS     -----------------------------------------------
# =======================================================================
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
    data = np.empty(shape=(max_rows, 0), dtype=float, order='F')
    for arr in args:
        miss_rows = max_rows - arr.shape[0]
        nan_arr = np.empty(shape=(miss_rows, arr.shape[1]),
                           dtype=float, order='F')
        nan_arr[:] = np.nan
        # nan_arr[:] = np.NAN

        aligned_arr = np.append(arr, nan_arr, axis=0)
        data = np.append(data, aligned_arr, axis=1)
    if DEBUG:
        print("aligned array shape = {}".format(data.shape))
    return data
