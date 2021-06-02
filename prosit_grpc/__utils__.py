from . import __constants__ as C
import numpy as np
import itertools
import scipy
import difflib

def compute_ion_masses(seq_int, charge_onehot):
    """ 
    Collects an integer sequence e.g. [1,2,3] with charge 2 and returns array with 174 positions for ion masses. 
    Invalid masses are set to -1
    charge_one is a onehot representation of charge with 6 elems for charges 1 to 6
    """
    charge = list(charge_onehot).index(1) + 1
    if not (charge in (1, 2, 3, 4, 5, 6) and len(charge_onehot) == 6):
        print("[ERROR] One-hot-enconded Charge is not in valid range 1 to 6")
        return

    if not len(seq_int) == C.SEQ_LEN:
        print("[ERROR] Sequence length {} is not desired length of {}".format(
            len(seq_int), C.SEQ_LEN))
        return

    l = list(seq_int).index(0) if 0 in seq_int else C.SEQ_LEN
    masses = np.ones((C.SEQ_LEN-1)*2*3, dtype=np.float32)*-1
    mass_b = 0
    mass_y = 0
    j = 0  # iterate over masses

    # Iterate over sequence, sequence should have length 30
    for i in range(l-1):  # only 29 possible ios
        j = i*6  # index for masses array at position

        # MASS FOR Y IONS
        # print("Addded", C.VEC_MZ[seq_int[l-1-i]])
        mass_y += C.VEC_MZ[seq_int[l-1-i]]

        # Compute charge +1
        masses[j] = (mass_y + 1*C.MASSES["PROTON"] +
                     C.MASSES["C_TERMINUS"] + C.MASSES["H"])/1.0
        # Compute charge +2
        masses[j+1] = (mass_y + 2*C.MASSES["PROTON"] + C.MASSES["C_TERMINUS"] +
                       C.MASSES["H"])/2.0 if charge >= 2 else -1.0
        # Compute charge +3
        masses[j+2] = (mass_y + 3*C.MASSES["PROTON"] + C.MASSES["C_TERMINUS"] +
                       C.MASSES["H"])/3.0 if charge >= 3.0 else -1.0

        # MASS FOR B IONS
        mass_b += C.VEC_MZ[seq_int[i]]

        # Compute charge +1
        masses[j+3] = (mass_b + 1*C.MASSES["PROTON"] +
                       C.MASSES["N_TERMINUS"] - C.MASSES["H"])/1.0
        # Compute charge +2
        masses[j+4] = (mass_b + 2*C.MASSES["PROTON"] + C.MASSES["N_TERMINUS"] -
                       C.MASSES["H"])/2.0 if charge >= 2 else -1.0
        # Compute charge +3
        masses[j+5] = (mass_b + 3*C.MASSES["PROTON"] + C.MASSES["N_TERMINUS"] -
                       C.MASSES["H"])/3.0 if charge >= 3.0 else -1.0

    return masses


def normalize_intensities(x):
    """
    This function normalizes the given intensity array of shape (num_seq, num_peaks)

    """
    return np.transpose(np.transpose(x)/np.max(x, 1))


def parse_modstrings(sequences, alphabet, translate=False, filter=False):
    """
    :param sequences: List of strings
    :param ALPHABET: dictionary where the keys correspond to all possible 'Elements' that can occur in the string
    :param translate: boolean to determine if the Elements should be translated to the corresponding values of ALPHABET
    :return: generator that yields a list of sequence 'Elements' or the translated sequence "Elements"
    """
    import re
    from itertools import repeat

    def split_modstring(sequence, r_pattern):
        sequence = sequence.replace('M(ox)','M(U:35)')
        sequence = sequence.replace('C','C(U:4)')
        split_seq = r_pattern.findall(sequence)
        if "".join(split_seq) == sequence:
            if translate:
                return [alphabet[aa] for aa in split_seq]
            elif not translate:
                return split_seq
        elif filter:
            return [0]
        else:
            not_parsable_elements = "".join([li[2] for li in difflib.ndiff(sequence, "".join(split_seq)) if li[0] == '-'])
            raise ValueError(f"The element(s) [{not_parsable_elements}] "
                             f"in the sequence [{sequence}] could not be parsed")

    pattern = sorted(alphabet, key=len, reverse=True)
    pattern = [re.escape(i) for i in pattern]
    regex_pattern = re.compile("|".join(pattern))
    return map(split_modstring, sequences, repeat(regex_pattern))

def indices_to_one_hot(data, nb_classes):
    """
    Convert an iterable of indices to one-hot encoded labels.
    :param data: charge, int between 1 and 6
    """
    targets = np.array([data-1])  # -1 for 0 indexing
    return np.int_((np.eye(nb_classes)[targets])).tolist()[0]

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
