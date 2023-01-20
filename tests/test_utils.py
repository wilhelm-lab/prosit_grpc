import numpy as np
import pytest
import spectrum_fundamentals.constants as c

from prosit_grpc import __utils__ as u


def test_generate_new_matrix_v2():
    """Test generate_new_matrix_v2."""
    a0 = np.array([[1, 1, 1, 1]])
    matrix, m = u.generate_new_matrix_v2(a0, 2, 21)
    assert np.array_equal(matrix, [[1, 1, 1, 1]])
    assert np.array_equal(m, [0])

    a1 = np.array([[1, 1, 2, 1]])
    matrix, m = u.generate_new_matrix_v2(a1, 2, 21)
    assert np.array_equal(matrix, [[1, 1, 2, 1], [1, 1, 21, 1]])
    assert np.array_equal(m, [1])

    a2 = np.array([[1, 2, 2, 1]])
    matrix, m = u.generate_new_matrix_v2(a2, 2, 21)
    assert np.array_equal(matrix, [[1, 2, 2, 1], [1, 21, 2, 1], [1, 2, 21, 1], [1, 21, 21, 1]])
    assert np.array_equal(m, [3])

    a2 = np.array([[1, 2, 2, 1]])
    matrix, m = u.generate_new_matrix_v2(a2, 2, 21, 1)

    assert np.array_equal(matrix, [[1, 2, 2, 1], [1, 21, 2, 1], [1, 2, 21, 1]])
    assert np.array_equal(m, [2])

    a3 = np.array([[2, 2, 2, 1]])
    matrix, m = u.generate_new_matrix_v2(a3, 2, 21)
    assert np.array_equal(
        matrix,
        [[2, 2, 2, 1], [21, 2, 2, 1], [2, 21, 2, 1], [2, 2, 21, 1], [21, 21, 2, 1], [21, 2, 21, 1], [2, 21, 21, 1]],
    )
    assert np.array_equal(m, [6])

    a4 = np.array([[2, 2, 2, 2]])
    matrix, m = u.generate_new_matrix_v2(a4, 2, 21)
    assert np.array_equal(
        matrix,
        [
            [2, 2, 2, 2],
            [21, 2, 2, 2],
            [2, 21, 2, 2],
            [2, 2, 21, 2],
            [2, 2, 2, 21],
            [21, 21, 2, 2],
            [21, 2, 21, 2],
            [21, 2, 2, 21],
            [2, 21, 21, 2],
            [2, 21, 2, 21],
            [2, 2, 21, 21],
        ],
    )
    assert np.array_equal(m, [10])

    a = np.array([a0[0], a1[0], a2[0], a3[0], a4[0]])
    matrix, m = u.generate_new_matrix_v2(a, 2, 21)

    assert np.array_equal(
        matrix,
        [
            [1, 1, 1, 1],
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
            [2, 2, 21, 21],
        ],
    )
    assert np.array_equal(m, [0, 1, 3, 6, 10])


def test_parse_modstrings():
    """Test parse_modstrings."""
    valid_seq = "AC[UNIMOD:4]C[UNIMOD:4]CDEFGHIKLMNM[UNIMOD:35]PQRSTVWYM[UNIMOD:35]STY"
    invalid_seq = "testing"
    assert "".join((list(u.parse_modstrings([valid_seq], alphabet=c.ALPHABET)))[0]) == valid_seq

    with pytest.raises(ValueError):
        list(u.parse_modstrings([invalid_seq], alphabet=c.ALPHABET))
