import itertools

import numpy as np
import scipy

from spectrum_fundamentals.fragments import compute_ion_masses
from spectrum_fundamentals.mod_string import parse_modstrings
from spectrum_fundamentals.charge import indices_to_one_hot


def normalize_intensities(x: np.ndarray) -> np.ndarray:
    """This function normalizes the given intensity array of shape (num_seq, num_peaks)."""
    return np.transpose(np.transpose(x) / np.max(x, 1))


def generate_new_matrix_v2(np_matrix, i_from_replace_value, i_to_replace_value, number_at_the_same_time: int = 2):
    """Generates new matrix v2."""
    """
    >>> a0 = np.array([[1,1,1,1]])
    >>> x, m = generate_new_matrix_v2(a0, 2, 21)
    >>> x
    array([[1, 1, 1, 1]])
    >>> m
    array([0])
    >>> a1 = np.array([[1,1,2,1]])
    >>> x, m = generate_new_matrix_v2(a1, 2, 21)
    >>> x
    array([[ 1,  1,  2,  1],
           [ 1,  1, 21,  1]])
    >>> m
    array([1])
    >>> a2 = np.array([[1,2,2,1]])
    >>> x, m = generate_new_matrix_v2(a2, 2, 21)
    >>> x
    array([[ 1,  2,  2,  1],
           [ 1, 21,  2,  1],
           [ 1,  2, 21,  1],
           [ 1, 21, 21,  1]])
    >>> m
    array([3])
    >>> a2 = np.array([[1,2,2,1]])
    >>> x, m = generate_new_matrix_v2(a2, 2, 21, 1)
    >>> x
    array([[ 1,  2,  2,  1],
           [ 1, 21,  2,  1],
           [ 1,  2, 21,  1]])
    >>> m
    array([2])
    >>> a3 = np.array([[2,2,2,1]])
    >>> x, m = generate_new_matrix_v2(a3, 2, 21)
    >>> x
    array([[ 2,  2,  2,  1],
           [21,  2,  2,  1],
           [ 2, 21,  2,  1],
           [ 2,  2, 21,  1],
           [21, 21,  2,  1],
           [21,  2, 21,  1],
           [ 2, 21, 21,  1]])
    >>> m
    array([6])
    >>> a4 = np.array([[2,2,2,2]])
    >>> x, m = generate_new_matrix_v2(a4, 2, 21)
    >>> x
    array([[ 2,  2,  2,  2],
           [21,  2,  2,  2],
           [ 2, 21,  2,  2],
           [ 2,  2, 21,  2],
           [ 2,  2,  2, 21],
           [21, 21,  2,  2],
           [21,  2, 21,  2],
           [21,  2,  2, 21],
           [ 2, 21, 21,  2],
           [ 2, 21,  2, 21],
           [ 2,  2, 21, 21]])
    >>> m
    array([10])
    >>> a = np.array([a0[0], a1[0],a2[0],a3[0],a4[0]])
    >>> x, m = generate_new_matrix_v2(a, 2, 21)
    >>> x
    array([[ 1,  1,  1,  1],
           [ 1,  1,  2,  1],
           [ 1,  2,  2,  1],
           [ 2,  2,  2,  1],
           [ 2,  2,  2,  2],
           [ 1,  1, 21,  1],
           [ 1, 21,  2,  1],
           [ 1,  2, 21,  1],
           [ 1, 21, 21,  1],
           [21,  2,  2,  1],
           [ 2, 21,  2,  1],
           [ 2,  2, 21,  1],
           [21, 21,  2,  1],
           [21,  2, 21,  1],
           [ 2, 21, 21,  1],
           [21,  2,  2,  2],
           [ 2, 21,  2,  2],
           [ 2,  2, 21,  2],
           [ 2,  2,  2, 21],
           [21, 21,  2,  2],
           [21,  2, 21,  2],
           [21,  2,  2, 21],
           [ 2, 21, 21,  2],
           [ 2, 21,  2, 21],
           [ 2,  2, 21, 21]])
    >>> m
    array([ 0,  1,  3,  6, 10])
    """
    # logging.info("start - concatenation of sequence integers")
    # print("shape {}".format(np_matrix.shape))
    v_num_to_be_replaced = np.sum(np_matrix == i_from_replace_value, 1)
    dic_row_multiplier = {}
    # lets hash number of possible additions to be faster
    for num in np.unique(v_num_to_be_replaced):
        buf = 0
        for i in range(0, number_at_the_same_time + 1):
            buf += scipy.special.comb(num, i)
        dic_row_multiplier[num] = buf

    dim_x_v = [int(dic_row_multiplier[i]) for i in v_num_to_be_replaced]
    dim_x = np.sum(dim_x_v)
    # logging.info("Number of new combinations: {}".format(dim_x))

    x = np.zeros((int(dim_x), np.shape(np_matrix)[1]))
    # print("original shape {}".format(x.shape))
    x[: np.shape(np_matrix)[0], : np.shape(np_matrix)[1]] = np_matrix

    position_x = int(np.shape(np_matrix)[0])
    for i, j in enumerate(dim_x_v):
        j = j - 1  # -1 because we need identify lines
        if j != 0:
            x[position_x : position_x + j] = np_matrix[i]
            for k in range(1, number_at_the_same_time + 1):
                pos = np.where(np_matrix[i] == i_from_replace_value)[0]
                index = np.fromiter(
                    itertools.chain.from_iterable(itertools.combinations(pos, k)), int
                )  # will return tuple for combinatiosn if k =2
                # index_reshaped = index.reshape((-1, k))
                a1, a2 = index.reshape((-1, k)).shape
                # repeats = a1
                if index.size > 0:
                    row_pos = np.repeat(np.arange(a1), a2)
                    x[np.add(row_pos, position_x), index] = i_to_replace_value
                    position_x += int(row_pos.size / a2)

    # logging.info("done - concatenation of sequence integers")
    return (x.astype(np.int32), np.array(dim_x_v) - 1)
