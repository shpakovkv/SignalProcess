"""
Python 2.7

wfm2read(filename, datapoints, step = 1, startind = 1)
Optional input arguments:
read data points startind:step:datapoints from the wfm file.
if datapoints is omitted, all data are read,
if step is omitted, step=1.
If startind omitted, startind=1
"""

from __future__ import print_function, with_statement
import re
import numpy


def warning(s):
    print("Warning!\n" + s)


class BinaryReadEOFException(Exception):  # EOF error for fread function
    def __init__(self):
        pass

    def __str__(self):
        return 'Not enought bytes in file. EOF reached while reading.'


def fread(file_obj, count, data_type, b_order, skip=0):
    """
    binary file read function
    file      - reference (file pointer)
    count     - number of values if file to be processed
    data_type - type of values to be read
    b_order   - bytes order of file
    skip      - number of values to be skip every 1 read value
                useful for reading array of values with specific step
    """

    type_len = {  # length in bytes for all supported types
        'int8': 1,
        'uint8': 1,
        'int16': 2,
        'uint16': 2,
        'int32': 4,
        'uint32': 4,
        'int64': 8,
        'uint64': 8,
        'float32': 4,
        'double': 8,
        'str': 1}

    # data type for bytes-to-value convert function
    type_dict = {
        'int8': 'i',
        'uint8': 'u',
        'int16': 'i',
        'uint16': 'u',
        'int32': 'i',
        'uint32': 'u',
        'int64': 'i',
        'uint64': 'u',
        'float32': 'f',
        'double': 'f',
        'str': 'u'}
    # for char:
    #   1) converts bytes to uint8,
    #   2) converts uint8 to char using ASCII table

    # print("bytes to read = {0}".format(type_len[data_type] * count))
    if skip < 0:  # check input parameters
        skip = 0
    if skip > count:
        skip = count

    units_to_read = int(float(count) / (1 + skip))

    bytes_read = bytes()
    for _ in range(0, units_to_read):
        bytes_read += file_obj.read(type_len[data_type])
        file_obj.seek(type_len[data_type] * skip, 1)

    # print("bytes read = {0}".format(bytes_read))
    # check end of file
    if len(bytes_read) != type_len[data_type] * count:
        raise BinaryReadEOFException

    numpy_type = (b_order + type_dict[data_type] +
                  str(type_len[data_type]))
    # print("numpy dtype = " + numpy_type)
    result = numpy.ndarray(shape=(count,), dtype=numpy_type,
                           buffer=bytes_read)

    # string output:
    if data_type == 'str':
        result_str = ''
        for value in result:
            if value == 0:  # trim right all after first Null
                break
            result_str += chr(value)
        return result_str

    # single number output:
    if result.size == 1:
        return result[0]

    # ndarray output:
    return result


def read_wfm_group(group_of_files, start_index=0, number_of_points=-1,
                   read_step=1, silent_mode=True):
    # reads a number of wfm files, unites columns to 1 table
    # returns data as 2-dimensional ndarray
    t, y, info, over_i, under_i = read_wfm(group_of_files[0])
    data = numpy.c_[t, y]
    for i in range(1, len(group_of_files)):
        t, y, info, over_i, under_i = \
            read_wfm(group_of_files[i],
                     start_index=start_index,
                     number_of_points=number_of_points,
                     read_step=read_step,
                     silent_mode=silent_mode)
        data = numpy.c_[data, t, y]
    return data


# ====================================================================
# ------    WFM READER FUNCTION---------------------------------------
#  ===================================================================
def read_wfm(filename, start_index=0, number_of_points=-1,
             read_step=1, silent_mode=True):

    file_info = {}  # dict with file parameters
    output = list()
    with open(filename, "rb") as fid:
        # BYTES ORDER (2 bytes always)
        # 0F0F == 'little' little-endian order
        # else == 'big'     big-endian order
        current_bytes = fid.read(2)
        if current_bytes == b'\x0f\x0f':
            # py_byte_order = 'little'
            byteorder = '<'
        else:
            # py_byte_order = 'big'
            byteorder = '>'

        # VERSION (8 bytes always)
        # version looks like ":WFM#xxx" where xxx - 3-digits int number
        file_info['version_string'] = fread(fid, 8, 'str', byteorder)
        # print("File version = {0}".format(file_info['version_string']))
        file_info['version_number'] = \
            int(re.search(r"([^:WFM#])+",
                          file_info['version_string']).group())
        if file_info['version_number'] > 3 and not silent_mode:
            print('WFM2read:HigherVersionNumber\n'
                  'wfm2read has only been tested with WFM fid versions <= 3')

        # FILE INFO
        fid.seek(5, 1)
        file_info['num_bytes_per_point'] = fread(fid, 1, 'uint8', byteorder)
        file_info['byte_offset_to_beginning_of_curve_buffer'] = \
            fread(fid, 1, 'uint32', byteorder)

        if file_info['version_number'] >= 2:
            fid.seek(148, 1)
        else:
            fid.seek(146, 1)

        # EXPLICIT DIMENSIONS 1
        file_info['ed1_dim_scale'] = fread(fid, 1, 'double', byteorder)
        file_info['ed1_dim_offset'] = fread(fid, 1, 'double', byteorder)
        fid.seek(56, 1)
        file_info['ed1_format'] = fread(fid, 4, 'int8', byteorder)

        if file_info['version_number'] >= 3:
            fid.seek(244, 1)
        else:
            fid.seek(236, 1)

        # IMPLICIT DIMENSION 1
        file_info['id1_dim_scale'] = fread(fid, 1, 'double', byteorder)
        file_info['id1_dim_offset'] = fread(fid, 1, 'double', byteorder)

        if file_info['version_number'] >= 3:
            fid.seek(318, 1)
        else:
            fid.seek(310, 1)

        file_info['data_start_offset'] = fread(fid, 1, 'uint32', byteorder)
        file_info['postcharge_start_offset'] = fread(fid, 1,
                                                     'uint32', byteorder)
        file_info['postcharge_stop_offset'] = fread(fid, 1,
                                                    'uint32', byteorder)
        file_info['end_of_curve_buffer_offset'] = fread(fid, 1,
                                                        'uint32', byteorder)

        # choose correct DATA FORMAT of the curve buffer data
        format_dict = {0: 'int16',
                       1: 'int32',
                       2: 'uint32',
                       3: 'uint64',
                       4: 'float32',
                       5: 'float64',
                       6: 'uint8',
                       7: 'int8'}

        if file_info['ed1_format'][0] not in format_dict.keys():
            raise ValueError("Invalid data format or error in file.")

        if ((file_info['version_number'] < 3) and
                (file_info['ed1_format'] == 6 or
                 file_info['ed1_format'] == 7)):
            raise ValueError("Data format " +
                             format_dict[file_info['ed1_format']] +
                             "is not compatible with wfm version \"" +
                             file_info['version_string'] +
                             "\"!\nInvalid data format or error in file.")

        curve_data_format = format_dict[file_info['ed1_format'][0]]
        file_info['data_format_str'] = curve_data_format

        # RESETS FILE CURSOR to the beginning of the curve buffer
        offset = (file_info['byte_offset_to_beginning_of_curve_buffer']
                  + file_info['data_start_offset']
                  + start_index * file_info['num_bytes_per_point'])
        fid.seek(offset)

        # GETS NUMBER OF ALL POINTS wrote in file
        data_points_all = int((file_info['postcharge_start_offset'] -
                               file_info['data_start_offset']) /
                              file_info['num_bytes_per_point'])

        # calc POINTS TO BE READ
        data_points_to_read = data_points_all - start_index
        if number_of_points < 1 or not isinstance(number_of_points, int):
            # set to maximum number of data points
            # which can be securely read from the file,
            # using startind and step parameters:
            number_of_points = int(data_points_to_read / read_step)
            if not silent_mode:
                warning("WFM2read:data_pointsPosInt \"data_points\" "
                        "input parameter must be a positive integer.\n" +
                        "Setting data_points = " + str(number_of_points) +
                        ".")

        # maximum number of data points
        # which can be securely read from the frame in the file,
        # using startind and step parameters:
        data_points_to_read = int(data_points_to_read / read_step)

        # if more data_points are requested than provided in the file
        if number_of_points > data_points_to_read and not silent_mode:
            message = ("WFM2read:inconsistent_params\n"
                       "The requested combination of input parameters \n" +
                       "data_points, read_step and start_index "
                       "would require at least " +
                       str(number_of_points * read_step + start_index) +
                       " data points in " + filename +
                       "\nThe actual number of data points "
                       "in the trace is only " +
                       str(data_points_all) + " .\n")
            warning(message)
        else:
            data_points_to_read = number_of_points

        # READ DATA values from curve buffer
        # t - Time array (continuous uniform increasing sequence)
        data_t = numpy.zeros((data_points_to_read, 1), dtype='<f8')
        for i in range(0, data_points_to_read):
            tic = (i + 1) * read_step
            data_t[i] = (file_info['id1_dim_offset'] +
                         file_info['id1_dim_scale'] * (tic + start_index))

        raw_data = fread(fid, data_points_to_read,
                         curve_data_format, byteorder, read_step - 1)
        data_y = (raw_data * file_info['ed1_dim_scale'] +
                  file_info['ed1_dim_offset'])

        # handling RAW DATA over- and underranged values
        if file_info['ed1_format'][0] < 4 or file_info['ed1_format'][0] > 5:
            # integer type
            file_info['raw_data_upper_bound'] = \
                numpy.iinfo(numpy.dtype(raw_data[0])).max
            file_info['raw_data_lower_bound'] = \
                numpy.iinfo(numpy.dtype(raw_data[0])).min + 1
        else:
            # float type
            file_info['raw_data_upper_bound'] = \
                numpy.finfo(numpy.dtype(raw_data[0])).max
            file_info['raw_data_lower_bound'] = \
                numpy.finfo(numpy.dtype(raw_data[0])).min

        # handling DATA over- and underranged values
        file_info['data_upper_bound'] = (file_info['ed1_dim_offset'] +
                                         file_info['ed1_dim_scale'] *
                                         file_info['raw_data_upper_bound'])

        file_info['data_lower_bound'] = (file_info['ed1_dim_offset'] +
                                         file_info['ed1_dim_scale'] *
                                         file_info['raw_data_lower_bound'])

        over_val_ind = numpy.array(raw_data >=
                                   file_info['raw_data_upper_bound'])
        under_val_ind = numpy.array(raw_data <=
                                    file_info['raw_data_lower_bound'])

        file_info['y_resolution'] = file_info['ed1_dim_scale']
        file_info['t_resolution'] = 1 / file_info['id1_dim_scale']
        file_info['t_step'] = file_info['id1_dim_scale']
        file_info['points_count'] = data_points_to_read

        # print warning if there are wrong values
        # because they are lying outside
        # the AD converter digitization window:
        if any(over_val_ind) and not silent_mode:
            warning('WFM2read:OverRangeValues\nThere are  ' +
                    str(sum(over_val_ind)) +
                    ' over range value(s) in file ' + filename)
        if any(under_val_ind) and not silent_mode:
            warning('WFM2read:UnderRangeValues\nThere are ' +
                    str(sum(under_val_ind)) +
                    ' under range value(s) in file ' + filename)
        output.append(data_t)
        output.append(data_y)
        output.append(file_info)
        output.append(over_val_ind)
        output.append(under_val_ind)

        # print("data_y max = {0}".format(numpy.amax(raw_data)))
        # print()
        # print("data_y min = {0}".format(numpy.amin(raw_data)))

    return output

# ====================================================================
# --------      MAIN      --------------------------------------------
# ====================================================================
if __name__ == "__main__":
    import sys
    import os

    args = sys.argv[1:]
    params = {}  # input parameters dict

    key_list = (
        '-d',  # input dir path
        '-u',  # setup filename
        '-o',  # output file name
        '-i',  # input file names (Do not change index of this parameter)
        '-g',  # number of files in groupe (one shot)
        '-b',  # sorted by (num-first/ch-first) (num/ch)
        '-t'  # save to dir
    )
    long_keys = (
        '--dir-path',
        '--setup-file',
        '--output-file',
        '--input-files',  # Do not change index of this parameter
        '--grouped-by',
        '--sorted-by',
        '--save-to'
    )
    # some keys should not be used together:
    #                (u, i); (u, o); (u, g); (o, g); (i, g)
    key_conflicts = ((1, 3), (1, 2), (1, 4), (2, 4), (3, 4))

    # checks input parameters and gets values
    for arg_idx in range(len(args)):
        match = re.match(r'(--?[^=]+)(?:[="\' ]*)([^="\']*)', args[arg_idx])
        assert match, "Error! Invalid syntax at:\"" + args[arg_idx] + "\""
        key = match.group(1)
        val = match.group(2)
        # print("{} : {}".format(key, val))
        # key = re.match(r'--?[\w]+', args[arg_idx])
        # val = re.search(r'(?:=["\' ]*)([^="]+)', args[arg_idx]).group(1)
        if key == key_list[3] or key == long_keys[3]:
            # handle input file list at the end of the string
            assert len(args) > arg_idx + 1, ("Specify file name(s) "
                                             "after '" + key + "'.")
            params[key_list[3]] = [args[arg_idx + 1:]]
            break
        if key in long_keys:
            key_idx = long_keys.index(key)
        elif key in key_list:
            key_idx = key_list.index(key)
        else:
            raise Exception('Error! Unexpected parameter \'' + key + '\'.')
        params[key_list[key_idx]] = val

    # check parameters conflicts
    for k1, k2 in key_conflicts:
        ex_keys = params.keys()
        assert not (key_list[k1] in ex_keys and key_list[k2] in ex_keys), \
            ('Error! The parameters \'' + key_list[k1] + '\'(\'' +
             long_keys[k1] + '\') and \'' + key_list[k2] + '\'(\'' +
             long_keys[k2] + '\') can not be used together.')

    # assign parameters values
    file_list = params.get('-i', [])
    setup_file = params.get('-u', False)
    save_as = params.get('-o', '')
    path = params.get('-d', '')
    group_size = params.get('-g', 0)
    sorted_by = params.get('b', 'num')
    save_to = params.get('-t', '')

    if path:
        # path = os.path.abspath(path)
        assert os.path.isdir(path), ("Error! Can not find dir '" +
                                     path + "'.")
    if save_to:
        # save_to = os.path.abspath(save_to)
        if not os.path.isdir(save_to):
            os.mkdir(save_to)

    assert len(file_list) or setup_file or group_size and path, \
        "Error! No input files specified!"

    if setup_file:
        '''
        import xml
        '''
        pass
    elif group_size:
        '''
        Dir with files was input.
        '''
        pass
    else:
        ''' 
        One group of files was input.
        file_list == [[file1.wfm, file2.wfm, etc.]] 
        List with 1 group of files, that corresponds to one shot. 
        Thus the output of the program will be 1 file.
        '''
        for idx, fname in enumerate(file_list[0]):
            file_list[0][idx] = os.path.join(path, fname)
            assert os.path.isfile(file_list[0][idx]), ("Error! Can not find "
                                                       "file '" + fname + "'.")
        # if save_as:
        #     save_as = os.path.join(save_to, save_as)
        # elif save_to:
        #     save_as = os.path.basename(file_list[0][0])[:-4] + '.csv'
        #     save_as = os.path.join(save_to, save_as)
        if not save_as:
            save_as = file_list[0][0]
        if save_as.lower().endswith(".wfm"):
            save_as = save_as[:-4]
        if not save_as.lower().endswith(".csv"):
            save_as = save_as + '.csv'
        save_as = os.path.join(save_to, save_as)
        save_as = [save_as]

    for idx, group in enumerate(file_list):
        print('Reading files: ', end='')
        print(', '.join(group))
        data = read_wfm_group(group)
        numpy.savetxt(save_as[idx], data, delimiter=",")
        print('Saved as: \'{}\''.format(save_as[idx]))
