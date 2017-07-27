import numpy as np
import os
import re
from matplotlib import pyplot as plt
import SignalProcess as sp
import PeakProcess as pp
import wfm_reader_lite as wfm


class SingleCurve:
    def __init__(self, in_x, in_y):
        # INPUT DATA CHECK
        x = in_x
        y = in_y
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError("Input time and value arrays must be instances of numpy.ndarray class.")
        if len(x) != len(y):
            raise IndexError("Input time and value arrays must have same length.")
        self.points = len(x)

        if np.ndim(x) == 1:                     # check if X array has 2 dimensions
            x = np.expand_dims(in_x, axis=1)    # add dimension to the array
        if np.ndim(y) == 1:                     # check if Y array has 2 dimensions
            y = np.expand_dims(in_y, axis=1)    # add dimension to the array

        if x.shape[1] != 1 or y.shape[1] != 1:  # check if X and Y arrays have 1 column
            raise ValueError("Input time and value arrays must have 1 column and any number of rows.")

        # CONVERT TO NDARRAY
        self.data = np.append(x, y, axis=1)

    def get_x(self):
        return self.data[:, 0]

    def get_y(self):
        return self.data[:, 1]


class SignalsData:
    def __init__(self, input_data=None):
        # EMPTY INSTANCE
        self.count = 0      # number of curves
        self.curves = []    # list of curves data (SingleCurve instances)

        # FILL WITH VALUES
        if input_data:
            self.append(input_data)

    def check_input(self, data):
        # CHECK INPUT DATA
        if np.ndim(data) != 2:
            raise ValueError("Input array must have 2 dimensions.")
        if data.shape[1] % 2 != 0:
            raise IndexError("Input array must have even number of columns.")

    def append(self, input_data):
        # appends new SingleCurves to the self.curves list

        data = np.array(input_data, dtype=float, order='F') # convert input data to numpy.ndarray
        self.check_input(data)                              # check inputs
        self.count += data.shape[1] // 2                    # update the number of curves
        for curve_idx in range(0, data.shape[1], 2):
            self.curves.append(SingleCurve(data[:,curve_idx], data[:, curve_idx + 1]))  # add new SingleCurve

    def get_array(self):
        # return all curves data as united 2D array
        # short curve arrays are supplemented with required amount of rows (filled with 'nan')
        return align_and_append_ndarray(*[curve.data for curve in self.curves])



def align_and_append_ndarray(*args):
    # returns 2D numpy.ndarray containing all input 2D numpy.ndarrays
    # if input arrays have different number of rows, fills missing values with 'nan'

    # CHECK TYPE & LENGTH
    for arr in args:
        if not isinstance(arr, np.ndarray):
            raise TypeError("Input arrays must be instances of numpy.ndarray class.")
        if np.ndim(arr) != 2:
            raise ValueError("Input arrays must have 2 dimensions.")

    # ALIGN & APPEND
    max_len = max([arr.shape[0] for arr in args])                   # output array's rows number == max rows number
    data = np.empty(shape=(max_len, 0), dtype=float, order='F')     # empty 2-dim array
    for arr in args:
        miss_rows = max_len - arr.shape[0]                                          # number of missing rows
        cols = arr.shape[1]
        nan_arr = np.empty(shape=(miss_rows, arr.shape[1]), dtype=float, order='F') # array with missing rows...
        nan_arr *= np.nan                                                           # ...filled with 'nan'
        aligned_arr = np.append(arr, nan_arr, axis=0)                               # fill missing row with nans
        data = np.append(data, aligned_arr, axis=1)                                 # append arr to data
    return data


if __name__ == '__main__':
    # ===========================================================================================================
    # -----     MAIN     -----------------------------
    # ===========================================================================================================
    # hmo_data = np.genfromtxt('data/ERG0027.CSV', delimiter=',', skip_header=0)
    # LeCroy_data = np.genfromtxt('data/LeCroy_00027.CSV', delimiter=',', skip_header=0)
    # print("Data loaded")
    # multiplier = [1E+9, 0.697, 1E+9, 4.554, 1E+9, 0.25]
    # delay = [80.3, 0, 367.1, 0, 0, 0]
    # sp.multiplier_and_delay(LeCroy_data, multiplier, delay)
    # print("Correction completed")
    #
    # U = LeCroy_data[:,1]
    # t = LeCroy_data[:,0]
    # print("Time step = " + str(t[1] - t[0]))
    #
    # u_smoothed = pp.smooth_voltage(t, U)
    #
    # plt.plot(t, U, '-b')
    # plt.plot(t,u_smoothed, '-r')
    # front = pp.find_voltage_front(t, u_smoothed)
    # plt.plot([front[0]], [front[1]], '*g')
    # print(front)
    # plt.show()
    # print("Plot completed")

    # # ==========================================================================================
    # x = [[[44, 1156], [178, 2478]], [[46, 6744], [101, 4004], [181, 1231]], [[1, 1986], [98, 1022], [304, 1241]]]
    # peak_data, peak_map = pp.group_peaks(x, 5)
    # c = 0
    # for curve in peak_data:
    #     print "Waveform #00" + str(c)
    #     c += 1
    #     for peak in curve:
    #         print peak
    #     print

    # GET FOLDERs PATHs
    path_dict = dict()
    path_list = list()
    osc_list = ["DPO7054", "HMO3004", "TDS2024C", "LeCroy"]

    # path_dict["DPO7054"] = ("/media/shpakovkv/6ADA8899DA886365/WORK/2017/2017 05 12-19 ERG" +
    #                         "/2017 05 13 ERG Output final/DPO7054")
    # path_dict["HMO3004"] = ("/media/shpakovkv/6ADA8899DA886365/WORK/2017/2017 05 12-19 ERG/" +
    #                         "2017 05 13 ERG Output final/HMO3004")
    #
    # path_dict["TDS2024C"] = ("/media/shpakovkv/6ADA8899DA886365/WORK/2017/2017 05 12-19 ERG/" +
    #                          "2017 05 13 ERG Output final/TDS2024C")
    #
    # path_dict["LeCroy"] = ("/media/shpakovkv/6ADA8899DA886365/WORK/2017/2017 05 12-19 ERG/" +
    #                        "2017 05 13 ERG Output final/LeCroy")

    path_dict["DPO7054"] = "G:\\WORK\\2017 05 13-19 ERG\\2017 05 13 ERG Input united\\2017 05 13 DPO7054 united_CSV"
    path_dict["HMO3004"] = "G:\\WORK\\2017 05 13-19 ERG\\2017 05 13 ERG Input united\\2017 05 13 HMO3004 united_CSV"
    path_dict["TDS2024C"] = "G:\\WORK\\2017 05 13-19 ERG\\2017 05 13 ERG Input united\\2017 05 13 TDS2024C united_CSV"
    path_dict["LeCroy"] = "G:\\WORK\\2017 05 13-19 ERG\\2017 05 13 ERG Input united\\2017 05 13 LeCroy united_CSV"

    save_to_folder = "G:\\WORK\\2017 05 13-19 ERG\\2017 05 13 ERG Output final"

    file_dict = dict()
    multiplier = dict()
    delay = dict()
    cols_label = dict()

    # GET MULTIPLIERs & DELAYs
    multiplier["DPO7054"] = [1e9, 1.216, 1e9, 0.991, 1e9, 1, 1e9, 0.994]
    delay["DPO7054"] = [134.7, 0, 117.7, 0, 121.7, 0, 131.7, 0]
    cols_label["DPO7054"] = ["Time1", "PMTD1", "Time2", "PMTD2", "Time3", "PMTD3", "Time4", "PMTD4"]

    multiplier["HMO3004"] = [1e9, 0.272, 1e9, 0.951, 1e9, 1.138, 1e9, 1.592]
    delay["HMO3004"] = [87, 0, 124, 0, 142, 0, 124, 0]

    multiplier["TDS2024C"] = [1e9, 1.261, 1e9, 1.094, 1e9, 1.222, 1e9, 1.59]
    delay["TDS2024C"] = [126.5, 0, 111.5, 0, 104.5, 0, 129.5, 0]

    multiplier["LeCroy"] = [1e9, 0.697, 1e9, 4.554, 1e9, 0.25]
    delay["LeCroy"] = [80.3, 0, 367.1, 0, 0, 0]

    voltage_pair_idx = 12    # zero-based index of voltage pair (time-value columns)

    # GET FILE GROUPS
    for key in path_dict:   # unsorted order
        file_dict[key] = sp.get_file_list_by_ext(path_dict[key], ".CSV", sort=False)

    shoots_count = len(file_dict["DPO7054"])
    for key in path_dict:
        if shoot_number != len(file_dict[key]):
            raise IndexError("The number of .csv files in osc-directories must be the same.")

    # READ CSV
    for number in range(shoots_count):      # zero-based index of shoot
        data = SignalsData()
        for osc in osc_list:    # sorted order
            print "reading " + file_dict[osc][number]
            temp = np.genfromtxt(file_dict[osc][number], delimiter=",")         # read data
            temp = sp.multiplier_and_delay_arr(temp, multiplier[osc], delay[osc])   # corr data
            data.append(temp)                                                   # add data to array
        delta_time, voltage_level = pp.find_voltage_front(data.curves[voltage_pair_idx].get_x(),
                                                          data.curves[voltage_pair_idx].get_y,
                                                          level=0.2, is_positive=False)
        if delta_time:
            data = pp.timeline_corr(data, delta_time)
        table = data.get_array()
        print "Table shape = (" + str(table.shape[0]) + ", " + str(table.shape[1]) + ")"
        # print table[len(table) - 1,:]

        # for idx in range(0, data.count, 1):
        #     plt.plot(data.curves[idx].get_x(), data.curves[idx].get_y())
        #     print("Curve #0" + str(idx + 1))
        #     plt.show()

        # for idx in range(0, table.shape[1], 2):
        #     plt.plot(table[:, idx], table[:, idx + 1])
        #     print("Curve #0" + str(idx // 2 + 1))
        #     plt.show()

        file_name = "ERG_" + sp.get_name_from_group_of_files([file_dict["DPO7054"][number]], 4, 3) + ".csv"
        print "Saving file \"" + file_name + "\" ..."
        save_to = os.path.join(save_to_folder, file_name)
        sp.save_ndarray_csv(save_to, table)
        print "Done!"
        print

    # MULTIPLIER & DELAY

    # TIME CORR

    # SAVE UNITED FINAL DATA

    # FIND PEAKS

    # data = np.genfromtxt(file_name, delimiter=",")
    # print os.path.basename(file_name) + "    " + str(data.shape)
