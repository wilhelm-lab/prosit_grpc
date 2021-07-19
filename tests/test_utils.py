from prosit_grpc import __utils__ as U
from fundamentals import constants as C
import numpy as np
import pytest


def test_generate_newMatrix_v2():
    a0 = np.array([[1, 1, 1, 1]])
    matrix, m = U.generate_newMatrix_v2(a0, 2, 21)
    assert np.array_equal(matrix, [[1, 1, 1, 1]])
    assert np.array_equal(m, [0])

    a1 = np.array([[1, 1, 2, 1]])
    matrix, m = U.generate_newMatrix_v2(a1, 2, 21)
    assert np.array_equal(matrix, [[1,  1,  2,  1],
                                   [1,  1, 21,  1]])
    assert np.array_equal(m, [1])

    a2 = np.array([[1, 2, 2, 1]])
    matrix, m = U.generate_newMatrix_v2(a2, 2, 21)
    assert np.array_equal(matrix,  [[1,  2,  2,  1],
                                    [1, 21,  2,  1],
                                    [1,  2, 21,  1],
                                    [1, 21, 21,  1]])
    assert np.array_equal(m, [3])

    a2 = np.array([[1, 2, 2, 1]])
    matrix, m = U.generate_newMatrix_v2(a2, 2, 21, 1)

    assert np.array_equal(matrix, [[1,  2, 2, 1],
                                   [1, 21, 2, 1],
                                   [1, 2, 21, 1]])
    assert np.array_equal(m, [2])

    a3 = np.array([[2, 2, 2, 1]])
    matrix, m = U.generate_newMatrix_v2(a3, 2, 21)
    assert np.array_equal(matrix, [[2, 2, 2, 1],
                                   [21, 2, 2, 1],
                                   [2, 21, 2, 1],
                                   [2, 2, 21, 1],
                                   [21, 21, 2, 1],
                                   [21, 2, 21, 1],
                                   [2, 21, 21, 1]])
    assert np.array_equal(m, [6])

    a4 = np.array([[2, 2, 2, 2]])
    matrix, m = U.generate_newMatrix_v2(a4, 2, 21)
    assert np.array_equal(matrix, [[2, 2, 2, 2],
                                   [21, 2, 2, 2],
                                   [2, 21, 2, 2],
                                   [2, 2, 21, 2],
                                   [2, 2, 2, 21],
                                   [21, 21, 2, 2],
                                   [21, 2, 21, 2],
                                   [21, 2, 2, 21],
                                   [2, 21, 21, 2],
                                   [2, 21, 2, 21],
                                   [2, 2, 21, 21]])
    assert np.array_equal(m, [10])

    a = np.array([a0[0], a1[0], a2[0], a3[0], a4[0]])
    matrix, m = U.generate_newMatrix_v2(a, 2, 21)

    assert np.array_equal(matrix, [[1, 1, 1, 1],
                                   [1, 1, 2, 1],
                                   [1, 2, 2, 1],
                                   [2, 2, 2, 1],
                                   [2, 2, 2, 2],
                                   [1, 1, 21, 1],
                                   [1, 21, 2, 1],
                                   [1, 2, 21, 1],
                                   [1, 21, 21, 1],
                                   [21, 2, 2, 1],
                                   [2, 21, 2, 1],
                                   [2, 2, 21, 1],
                                   [21, 21, 2, 1],
                                   [21, 2, 21, 1],
                                   [2, 21, 21, 1],
                                   [21, 2, 2, 2],
                                   [2, 21, 2, 2],
                                   [2, 2, 21, 2],
                                   [2, 2, 2, 21],
                                   [21, 21, 2, 2],
                                   [21, 2, 21, 2],
                                   [21, 2, 2, 21],
                                   [2, 21, 21, 2],
                                   [2, 21, 2, 21],
                                   [2, 2, 21, 21]])
    assert np.array_equal(m, [0, 1, 3, 6, 10])

def test_parse_modstrings():
    valid_seq = "AC(U:4)C(U:4)CDEFGHIKLMNM(U:35)PQRSTVWYM(U:35)STY"
    invalid_seq = "testing"
    assert "".join((list(U.parse_modstrings([valid_seq], alphabet=C.ALPHABET)))[0]) == valid_seq

    with pytest.raises(ValueError):
        list(U.parse_modstrings([invalid_seq], alphabet=C.ALPHABET))
