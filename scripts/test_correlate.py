import numpy as np
from matplotlib import pyplot as plt
from analysis import correlation_func_2d
from plotter import find_nearest_idx


def convolve_2d(curve1, curve2):
    """ Returns the convolve of a signal (curve1) with
    another signal (curve2) as a function of delay (2D ndarray).
    The input array structure:
    [type][point]
    where type is the type of column - time (0) or value (1)
    point is the index of time-value pair in the array.

    The signals shape must be equal.
    The time step of both signals must be the same.

    The length of output signal is curve1.shape[1]

    :param curve1: 2D ndarray with time column and value column
    :type curve1: np.ndarray
    :param curve2: 2D ndarray with time column and value column
    :type curve2: np.ndarray
    :return: 2D ndarray with time column and value column
    :rtype: np.ndarray
    """
    assert curve1.ndim == curve2.ndim, \
        "Curves have different number of dimensions: {} and {}" \
        "".format(curve1.ndim, curve2.ndim)
    assert curve1.ndim == 2, \
        "The curve1 has the number of dimensions ({}) not as expected ({})." \
        "".format(curve1.ndim, 2)
    assert curve2.ndim == 2, \
        "The curve2 has the number of dimensions ({}) not as expected ({})." \
        "".format(curve2.ndim, 2)
    # assert curve1.shape[1] == curve2.shape[1], \
    #     "The curves have different number of points: {} and {}" \
    #     "".format(curve1.shape[1], curve2.shape[1])

    time_step = np.ndarray(shape=(2,), dtype=np.float64)
    time_step[0] = (curve1[0, -1] - curve1[0, 0]) / (curve1.shape[1] - 1)
    time_step[1] = (curve2[0, -1] - curve2[0, 0]) / (curve2.shape[1] - 1)
    tolerance = time_step.min() / 1e6
    assert np.isclose(time_step[0], time_step[1], atol=tolerance), \
        "Curve1 and curve2 have different time step: {} and {}. " \
        "The difference exceeds tolerance {}" \
        "".format(time_step[0], time_step[1], tolerance)

    # get correlation
    # res = np.correlate(curve1[1], curve2[1], mode='full')
    res = np.convolve(curve1[1], curve2[1], mode='full')

    # add time column
    # always symmetric, always odd length
    time_col = np.arange(- (res.shape[0] // 2), res.shape[0] // 2 + 1, dtype=np.float64)
    time_col *= time_step[0]

    # make 2D array [time/val][point]
    res = np.stack((time_col, res), axis=0)
    return res


def correlate_part(curve1, left1, right1, curve2, left2, right2):
    start1, stop1, start2, stop2 = 0, 0, 0, 0
    if left1 != right1:
        start1 = find_nearest_idx(curve1[0], left1, side='right')
        stop1 = find_nearest_idx(curve1[0], right1, side='left')
    else:
        start1 = 0
        stop1 = curve1.shape[1]

    if left2 != right2:
        start2 = find_nearest_idx(curve2[0], left2, side='right')
        stop2 = find_nearest_idx(curve2[0], right2, side='left')
    else:
        start2 = 0
        stop2 = curve2.shape[1]

    corr_curve = correlation_func_2d(curve1[:, start1: stop1], curve2[:, start2: stop2])
    return corr_curve


def correlation_func_2d_new(curve1, curve2):
    """ Returns the correlation of a signal (curve1) with
    another signal (curve2) as a function of delay (2D ndarray).
    The input array structure:
    [type][point]
    where type is the type of column - time (0) or value (1)
    point is the index of time-value pair in the array.

    The signals shape must be equal.
    The time step of both signals must be the same.

    The length of output signal is curve1.shape[1]

    :param curve1: 2D ndarray with time column and value column
    :type curve1: np.ndarray
    :param curve2: 2D ndarray with time column and value column
    :type curve2: np.ndarray
    :return: 2D ndarray with time column and value column
    :rtype: np.ndarray
    """
    assert curve1.ndim == curve2.ndim, \
        "Curves have different number of dimensions: {} and {}" \
        "".format(curve1.ndim, curve2.ndim)
    assert curve1.ndim == 2, \
        "The curve1 has the number of dimensions ({}) not as expected ({})." \
        "".format(curve1.ndim, 2)
    assert curve2.ndim == 2, \
        "The curve2 has the number of dimensions ({}) not as expected ({})." \
        "".format(curve2.ndim, 2)
    # assert curve1.shape[1] == curve2.shape[1], \
    #     "The curves have different number of points: {} and {}" \
    #     "".format(curve1.shape[1], curve2.shape[1])

    time_step = np.ndarray(shape=(2,), dtype=np.float64)
    time_step[0] = (curve1[0, -1] - curve1[0, 0]) / (curve1.shape[1] - 1)
    time_step[1] = (curve2[0, -1] - curve2[0, 0]) / (curve2.shape[1] - 1)
    tolerance = time_step.min() / 1e6
    assert np.isclose(time_step[0], time_step[1], atol=tolerance), \
        "Curve1 and curve2 have different time step: {} and {}. " \
        "The difference exceeds tolerance {}" \
        "".format(time_step[0], time_step[1], tolerance)

    # normalize
    y1_norm = curve1[1].copy()
    y1_norm /= abs(max(np.nanmax(curve1[1]), abs(np.nanmin(curve1[1]))))
    y2_norm = curve2[1].copy()
    y2_norm /= abs(max(np.nanmax(curve2[1]), abs(np.nanmin(curve2[1]))))
    print("curve 1 norm max = {};   min = {}".format(np.nanmax(y1_norm), np.nanmin(y1_norm)))
    print("curve 2 norm max = {};   min = {}".format(np.nanmax(y2_norm), np.nanmin(y2_norm)))

    # get correlation
    res = np.correlate(y1_norm, y2_norm, mode='full')
    print("res max = {};   min = {}".format(np.nanmax(res), np.nanmin(res)))
    plt.plot(y1_norm)
    plt.show()
    plt.plot(y2_norm)
    plt.show()
    plt.plot(res)
    plt.show()

    # add time column
    # always symmetric, always odd length
    time_col = np.arange(- (res.shape[0] // 2), res.shape[0] // 2 + 1, dtype=np.float64)
    time_col *= time_step[0]

    # make 2D array [time/val][point]
    res = np.stack((time_col, res), axis=0)
    return res


def get_square_pulse(length=1000, start=200, width=300, height=1.0, show=False):
    x_axis = np.arange(0, length, dtype=np.float64)
    y_axis = np.zeros(shape=(length, ), dtype=np.float64)
    for idx in range(start, start + width):
        y_axis[idx] = height
    data = np.stack((x_axis, y_axis), axis=0)
    if show:
        print(x_axis.shape)
        print(y_axis.shape)
        print(data.shape)
        print(data)
        plt.plot(data[0], data[1])
        plt.show()
    return data


def get_half_triangle_pulse(length=1000, start=200, width=300, height=1.0, show=False):
    x_axis = np.arange(0, length, dtype=np.float64)
    y_axis = np.zeros(shape=(length,), dtype=np.float64)
    for idx in range(width):
        y_axis[idx + start] = height * (1 - idx / width)
    data = np.stack((x_axis, y_axis), axis=0)
    if show:
        print(x_axis.shape)
        print(y_axis.shape)
        print(data.shape)
        print(data)
        plt.plot(data[0], data[1])
        plt.show()
    return data


def plot_correlate_plot(list_of_curves, list_of_results, curves_labels=None, results_labels=None):
    if curves_labels is not None:
        assert len(list_of_curves) == len(curves_labels), \
            "The number of curves ({}) and the number of curve " \
            "labels ({}) are not equal.".format(len(list_of_curves), len(curves_labels))
    if results_labels is not None:
        assert len(list_of_results) == len(results_labels), \
            "The number of curves ({}) and the number of curve " \
            "labels ({}) are not equal.".format(len(list_of_results), len(results_labels))
    fig, axes = plt.subplots(2, 1)
    for idx, curve in enumerate(list_of_curves):
        if curves_labels is not None:
            axes[0].plot(curve[0], curve[1], label=curves_labels[idx])
        else:
            axes[0].plot(curve[0], curve[1], label="Curve{}".format(idx))
    for idx, result_curve in enumerate(list_of_results):
        if results_labels is not None:
            axes[1].plot(result_curve[0], result_curve[1], label=results_labels[idx])
        else:
            axes[1].plot(result_curve[0], result_curve[1], label="Curve{}".format(idx))
    axes[0].legend()
    axes[1].legend()
    plt.show()


def test_equal_length():
    square_pulse = get_square_pulse(length=400, start=50, width=300, height=1.0, show=False)
    half_triangle_pulse = get_half_triangle_pulse(length=1000, start=200, width=300, height=1.0, show=False)
    correlation_curve = correlation_func_2d(square_pulse, half_triangle_pulse)
    plot_correlate_plot((square_pulse, half_triangle_pulse), (correlation_curve,))


def compare_correlate_and_convolve():
    square_pulse = get_square_pulse(length=400, start=50, width=300, height=1.0, show=False)
    half_triangle_pulse = get_half_triangle_pulse(length=1000, start=200, width=300, height=1.0, show=False)
    correlation_curve = correlation_func_2d(square_pulse, half_triangle_pulse)
    # correlation_curve = correlation_func_2d(half_triangle_pulse, square_pulse)

    convolve_curve = convolve_2d(square_pulse, half_triangle_pulse)
    print("Square pulse length = {}".format(square_pulse.shape[0]))
    print("Half triangle pulse length = {}".format(half_triangle_pulse.shape[0]))
    print("Correlation data shape = {}".format(correlation_curve.shape))
    print("Convolve data shape = {}".format(convolve_curve.shape))
    plot_correlate_plot((square_pulse, half_triangle_pulse), (correlation_curve, convolve_curve))


def compare_correlate_xy_and_yx():
    square_pulse = get_square_pulse(length=400, start=50, width=300, height=1.0, show=False)
    half_triangle_pulse = get_half_triangle_pulse(length=1000, start=200, width=300, height=1.0, show=False)
    correlation_xy = correlation_func_2d(square_pulse, half_triangle_pulse)
    correlation_yx = correlation_func_2d(half_triangle_pulse, square_pulse)

    print("Square pulse length = {}".format(square_pulse.shape[0]))
    print("Correlation XY shape = {}".format(correlation_xy.shape))
    print("Convolve YX shape = {}".format(correlation_yx.shape))
    plot_correlate_plot((square_pulse, half_triangle_pulse),
                        (correlation_xy, correlation_yx),
                        curves_labels=("Square pulse", "Half-triangle pulse"),
                        results_labels=("Correlation_XY", "Correlation_YX"))


def test_auto_square():
    square_pulse = get_square_pulse(length=400, start=50, width=300, height=1.0, show=False)
    square_pulse2 = get_square_pulse(length=399, start=50, width=300, height=1.0, show=False)
    correlation_curve = correlation_func_2d(square_pulse, square_pulse2)
    plot_correlate_plot((square_pulse, square_pulse2), (correlation_curve,))


def test_correlate_part():
    left1 = 0
    right1 = 400
    left2 = 199
    right2 = 400
    square_pulse = get_square_pulse(length=400, start=50, width=300, height=1.0, show=False)
    half_triangle_pulse = get_half_triangle_pulse(length=1000, start=200, width=300, height=1.0, show=False)
    correlate_curve = correlate_part(square_pulse, left1, right1, half_triangle_pulse, left2, right2)
    plot_correlate_plot((square_pulse[:, left1: right1], half_triangle_pulse[:, left2: right2]),
                        (correlate_curve,),
                        curves_labels=("Square pulse", "Half-triangle pulse"),
                        results_labels=("Correlation",))


def get_test_signal_1(show=False):
    length = 4000
    width = 300
    height = 1.0

    start = 500
    gap = 300
    num = 3

    x_axis = np.arange(0, length, dtype=np.float64)
    y_axis = np.zeros(shape=(length,), dtype=np.float64)
    while num > 0:
        for idx in range(width):
            y_axis[idx + start] = height * (1 - idx / width)
        start += width
        start += gap
        num -= 1
        gap *= 2
    data = np.stack((x_axis, y_axis), axis=0)
    if show:
        print(x_axis.shape)
        print(y_axis.shape)
        print(data.shape)
        print(data)
        plt.plot(data[0], data[1])
        plt.show()
    return data


def get_test_signal_2(show=False):
    length = 15000
    width = 300
    height_base = 1.0

    start = 500
    gap_base = 300
    num = 3
    obj_to_plot = num

    x_axis = np.arange(0, length, dtype=np.float64)
    y_axis = np.zeros(shape=(length,), dtype=np.float64)

    # Base triangles
    gap = gap_base
    while obj_to_plot > 0:
        for idx in range(width):
            y_axis[idx + start] = height_base * (1 - idx / width)
        start += width
        start += gap
        obj_to_plot -= 1
        gap *= 2

    # Squares
    start += gap_base * 10
    obj_to_plot = num
    gap = gap_base
    height = height_base * 4
    while obj_to_plot > 0:
        for idx in range(width):
            y_axis[idx + start] = height
        start += width
        start += gap
        obj_to_plot -= 1
        gap *= 2

    # big triangles
    start += gap_base * 10
    obj_to_plot = num
    gap = gap_base
    height = height_base * 2
    while obj_to_plot > 0:
        for idx in range(width):
            y_axis[idx + start] = height * (1 - idx / width)
        start += width
        start += gap
        obj_to_plot -= 1
        gap *= 2

    data = np.stack((x_axis, y_axis), axis=0)
    if show:
        print(x_axis.shape)
        print(y_axis.shape)
        print(data.shape)
        print(data)
        plt.plot(data[0], data[1])
        plt.show()
    return data


def get_test_signal_3(show=False):
    length = 15000
    width = 300
    height_base = 1.0

    start = 500
    gap_base = 300
    num = 3
    obj_to_plot = num

    x_axis = np.arange(0, length, dtype=np.float64)
    y_axis = np.zeros(shape=(length,), dtype=np.float64)

    # Squares go first
    obj_to_plot = num
    gap = gap_base
    height = height_base * 4
    while obj_to_plot > 0:
        for idx in range(width):
            y_axis[idx + start] = height
        start += width
        start += gap
        obj_to_plot -= 1
        gap *= 2

    # Base triangles
    start += gap_base * 10
    obj_to_plot = num
    gap = gap_base
    while obj_to_plot > 0:
        for idx in range(width):
            y_axis[idx + start] = height_base * (1 - idx / width)
        start += width
        start += gap
        obj_to_plot -= 1
        gap *= 2

    # big triangles
    start += gap_base * 10
    obj_to_plot = num
    gap = gap_base
    height = height_base * 2
    while obj_to_plot > 0:
        for idx in range(width):
            y_axis[idx + start] = height * (1 - idx / width)
        start += width
        start += gap
        obj_to_plot -= 1
        gap *= 2

    data = np.stack((x_axis, y_axis), axis=0)
    if show:
        print(x_axis.shape)
        print(y_axis.shape)
        print(data.shape)
        print(data)
        plt.plot(data[0], data[1])
        plt.show()
    return data


def test_signal_1_and_2():
    signal1 = get_test_signal_1()
    signal2 = get_test_signal_2()
    correlate_curve = correlation_func_2d(signal1, signal2)
    print("Signal 1 shape = {},  [{} to {}]".format(signal1.shape, signal1[0, 0], signal1[0, -1]))
    print("Signal 2 shape = {},  [{} to {}]".format(signal2.shape, signal2[0, 0], signal2[0, -1]))
    print("Correlate curve shape = {},  [{} to {}]".format(correlate_curve.shape, correlate_curve[0, 0], correlate_curve[0, -1]))

    plot_correlate_plot((signal2, signal1),
                        (correlate_curve,),
                        curves_labels=("Signal2", "Signal1"),
                        results_labels=("Correlation",))


def test_signal_1_and_3():
    signal1 = get_test_signal_1()
    signal3 = get_test_signal_3()
    signal1[0, :] += 6000.0
    correlate_curve = correlation_func_2d(signal1, signal3)
    print("Signal 1 shape = {},  [{} to {}]".format(signal1.shape, signal1[0, 0], signal1[0, -1]))
    print("Signal 2 shape = {},  [{} to {}]".format(signal3.shape, signal3[0, 0], signal3[0, -1]))
    print("Correlate curve shape = {},  [{} to {}]".format(correlate_curve.shape, correlate_curve[0, 0], correlate_curve[0, -1]))

    plot_correlate_plot((signal3, signal1),
                        (correlate_curve,),
                        curves_labels=("Signal2", "Signal1"),
                        results_labels=("Correlation",))


if __name__ == "__main__":
    # test_auto_square()
    # test_equal_length()
    # compare_correlate_and_convolve()
    # compare_correlate_xy_and_yx()
    # test_correlate_part()
    test_signal_1_and_3()

