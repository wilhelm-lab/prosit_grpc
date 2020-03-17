"""
>>> a2 = np.array([[1,2,2,1]])
>>> X, m = generate_newMatrix_v2(a2, 2, 21, 1)
>>> X
array([[ 1r  2,  2,  1],
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

from prosit_grpc import __utils__ as U
import numpy as np

def test_generate_newMatrix_v2():
    a0 = np.array([[1,1,1,1]])
    X, m = U.generate_newMatrix_v2(a0, 2, 21)
    assert np.array_equal(X, [[1, 1, 1, 1]])
    assert np.array_equal(m, [0])
    
    a1 = np.array([[1,1,2,1]])
    X, m = U.generate_newMatrix_v2(a1, 2, 21)
    assert np.array_equal(X, [[ 1,  1,  2,  1],
                              [ 1,  1, 21,  1]])
    assert np.array_equal(m, [1])


    a2 = np.array([[1,2,2,1]])
    X, m = U.generate_newMatrix_v2(a2, 2, 21)
    assert np.array_equal(X,  [[ 1,  2,  2,  1],
                               [ 1, 21,  2,  1],
                               [ 1,  2, 21,  1],
                               [ 1, 21, 21,  1]])
    assert np.array_equal(m, [3])

