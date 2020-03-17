from prosit_grpc import __utils__ as U
import numpy as np

def test_generate_newMatrix_v2():
    a0 = np.array([[1,1,1,1]])
    matrix, m = U.generate_newMatrix_v2(a0, 2, 21)
    assert np.array_equal(matrix, [[1, 1, 1, 1]])
    assert np.array_equal(m, [0])
    
    a1 = np.array([[1,1,2,1]])
    matrix, m = U.generate_newMatrix_v2(a1, 2, 21)
    assert np.array_equal(matrix, [[ 1,  1,  2,  1],
                                   [ 1,  1, 21,  1]])
    assert np.array_equal(m, [1])


    a2 = np.array([[1,2,2,1]])
    matrix, m = U.generate_newMatrix_v2(a2, 2, 21)
    assert np.array_equal(matrix,  [[ 1,  2,  2,  1],
                                    [ 1, 21,  2,  1],
                                    [ 1,  2, 21,  1],
                                    [ 1, 21, 21,  1]])
    assert np.array_equal(m, [3])

    a2 = np.array([[1, 2, 2, 1]])
    matrix, m = U.generate_newMatrix_v2(a2, 2, 21, 1)

    assert np.array_equal(matrix,[[1,  2, 2, 1],
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