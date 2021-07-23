from . import __constants__ as C
import numpy as np
import itertools
import scipy

from fundamentals.fragments import compute_ion_masses
from fundamentals.mod_string import parse_modstrings
from fundamentals.charge import indices_to_one_hot

def normalize_intensities(x):
    """
    This function normalizes the given intensity array of shape (num_seq, num_peaks)

    """
    return np.transpose(np.transpose(x)/np.max(x, 1))

def generate_newMatrix_v2(
    npMatrix, iFromReplaceValue, iToReplaceValue, numberAtTheSameTime=2
):
    """
    >>> a0 = np.array([[1,1,1,1]])
    >>> X, m = generate_newMatrix_v2(a0, 2, 21)
    >>> X
    array([[1, 1, 1, 1]])
    >>> m
    array([0])
    >>> a1 = np.array([[1,1,2,1]])
    >>> X, m = generate_newMatrix_v2(a1, 2, 21)
    >>> X
    array([[ 1,  1,  2,  1],
           [ 1,  1, 21,  1]])
    >>> m
    array([1])
    >>> a2 = np.array([[1,2,2,1]])
    >>> X, m = generate_newMatrix_v2(a2, 2, 21)
    >>> X
    array([[ 1,  2,  2,  1],
           [ 1, 21,  2,  1],
           [ 1,  2, 21,  1],
           [ 1, 21, 21,  1]])
    >>> m
    array([3])
    >>> a2 = np.array([[1,2,2,1]])
    >>> X, m = generate_newMatrix_v2(a2, 2, 21, 1)
    >>> X
    array([[ 1,  2,  2,  1],
           [ 1, 21,  2,  1],
           [ 1,  2, 21,  1]])
    >>> m
    array([2])
    >>> a3 = np.array([[2,2,2,1]])
    >>> X, m = generate_newMatrix_v2(a3, 2, 21)
    >>> X
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
    >>> X, m = generate_newMatrix_v2(a4, 2, 21)
    >>> X
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
    >>> X, m = generate_newMatrix_v2(a, 2, 21)
    >>> X
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
    # print("shape {}".format(npMatrix.shape))
    vNumToBeRelaced = np.sum(npMatrix == iFromReplaceValue, 1)
    dicRowMultiplier = {}
    # lets hash number of possible additions to be faster
    for num in np.unique(vNumToBeRelaced):
        buf = 0
        for i in range(0, numberAtTheSameTime + 1):
            buf += scipy.special.comb(num, i)
        dicRowMultiplier[num] = buf

    dim_x_v = [int(dicRowMultiplier[i]) for i in vNumToBeRelaced]
    dim_x = np.sum(dim_x_v)
    # logging.info("Number of new combinations: {}".format(dim_x))

    X = np.zeros((int(dim_x), np.shape(npMatrix)[1]))
    # print("original shape {}".format(X.shape))
    X[: np.shape(npMatrix)[0], : np.shape(npMatrix)[1]] = npMatrix

    position_x = int(np.shape(npMatrix)[0])
    for i, j in enumerate(dim_x_v):
        j = j - 1  # -1 because we need identify lines
        if j != 0:
            X[position_x: position_x + j] = npMatrix[i]
            for k in range(1, numberAtTheSameTime + 1):
                pos = np.where(npMatrix[i] == iFromReplaceValue)[0]
                index = np.fromiter(
                    itertools.chain.from_iterable(
                        itertools.combinations(pos, k)), int
                )  # will return tuple for combinatiosn if k =2
                index_reshaped = index.reshape((-1, k))
                a1, a2 = index.reshape((-1, k)).shape
                repeats = a1
                if index.size > 0:
                    row_pos = np.repeat(np.arange(a1), a2)
                    X[np.add(row_pos, position_x), index] = iToReplaceValue
                    position_x += int(row_pos.size / a2)

    # logging.info("done - concatenation of sequence integers")
    return (X.astype(np.int32), np.array(dim_x_v) - 1)
