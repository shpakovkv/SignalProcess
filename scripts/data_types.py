# Python 3.6
"""
Maintainer: Shpakov Konstantin
Link: https://github.com/shpakovkv/SignalProcess
"""

import numpy as np


# =======================================================================
# -----     CONST     ---------------------------------------------------
# =======================================================================
DEBUG = 0
VERBOSE = 0


# =======================================================================
# -----     CLASSES     -------------------------------------------------
# =======================================================================
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
            self.add_curves(input_data, labels, units, time_units)

    def append(self, input_data):
        """Appends new points from ndarray to existing curves in self.
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

        :return: None
        :rtype: None
        """
        self.check_ndarray(input_data)
        if input_data.shape[1] % 2 != 0:
            multiple_x_columns = False
        else:
            multiple_x_columns = True

        for idx in range(0, self.count):
            if multiple_x_columns:
                self.curves[idx].append(input_data[:, idx * 2],
                                        input_data[:, idx * 2 + 1])
            else:
                self.curves[idx].append(input_data[:, 0],
                                        input_data[:, idx + 1])

    def add_curves(self, input_data, labels=None, units=None, time_unit=None):
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
        new_data = np.array(input_data, dtype=float, order='F')
        self.check_ndarray(new_data)
        self.check_labels(new_data, labels, units)
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
                current_label = "Curve[{}]".format(new_idx)
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

    def check_ndarray(self, data_ndarray):
        """
        Checks the correctness of input data_ndarray.
        Raises exception if data_ndarray check fails.

        :param data_ndarray: ndarray with data to add to a SignalsData instance
        :type data_ndarray: np.ndarray

        :return: None
        :rtype: None
        """
        # CHECK INPUT DATA

        if np.ndim(data_ndarray) != 2:
            raise ValueError("Input array must have 2 dimensions.")
        if data_ndarray.shape[1] % 2 != 0:
            if VERBOSE:
                print("The input array has an odd number of columns! "
                      "The first column is considered as X column, "
                      "and the rest as Y columns.")

    def check_labels(self, data_ndarray, new_labels=None, new_units=None):
        """
        Checks the correctness of input new labels and units.
        Raises exception if check fails.

        :param data_ndarray: ndarray with data to add to a SignalsData instance
        :param new_labels: the list of labels for the new curves
        :param new_units: he list of units for the new curves

        :return: None
        :rtype: None
        """

        if data_ndarray.shape[1] % 2 != 0:
            # multiple_X_columns = False
            curves_count = data_ndarray.shape[1] - 1
        else:
            # multiple_X_columns = True
            curves_count = data_ndarray.shape[1] // 2

        if new_labels is not None:
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
        if new_units is not None:
            if not isinstance(new_labels, list):
                raise TypeError("The variable 'units' must be "
                                "an instance of the list class.")
            if len(new_labels) != curves_count:
                raise IndexError("The number of curves ({}) (the pair "
                                 "of time-value columns) in the appending "
                                 "data and tne number of new units ({}) "
                                 "must be the same."
                                 "".format(curves_count, len(new_labels)))

    # def get_array(self):
    #     """Returns all curves data as united 2D array
    #     short curve arrays are supplemented with
    #     required amount of rows (filled with 'nan')
    #
    #     return -- 2d ndarray
    #     """
    #     list_of_2d_arr = [self.curves[idx].data for
    #                       idx in sorted(self.idx_to_label.keys())]
    #     if DEBUG:
    #         print("len(lest of 2D arr) = {}".format(len(list_of_2d_arr)))
    #     return align_and_append_ndarray(*list_of_2d_arr)

    def get_array(self, curves_list=None):
        """Returns selected curves data as 2D array
        short curve arrays are supplemented with
        required amount of rows (filled with 'nan').

        Selects all curves if curves_list=None.

        :param curves_list: None or list of curves to be exported as 2D array
        :return: 2d ndarray
        """

        # check for default value
        if curves_list is None:
            curves_list = self.idx_to_label.keys()
        else:
            # check inputs
            for idx in curves_list:
                assert idx in self.idx_to_label.keys(), ("Curve index ({}) is out of bounds ({})."
                                                         "".format(idx, self.idx_to_label.keys()))

        list_of_2d_arr = [self.curves[idx].data for
                          idx in sorted(curves_list)]
        if DEBUG:
            print("len(lest of 2D arr) = {}".format(len(list_of_2d_arr)))
        return align_and_append_ndarray(*list_of_2d_arr)

    def by_label(self, label):
        # returns SingleCurve by name
        return self.curves[self.label_to_idx[label]]

    def get_label(self, idx):
        # return label of the SingleCurve by index
        for key, value in self.label_to_idx.items():
            if value == idx:
                return key

    def get_idx(self, label):
        # returns index of the SingleCurve by label
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
