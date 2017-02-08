"""
Cython optimized pairwise distance function on numpy matrix (3xN).
"""

from __future__ import print_function
import numpy as np
cimport numpy as np

# data type of our arrays
DTYPE = np.float
ctypedef np.float_t DTYPE_t

cdef inline DTYPE_t float_max(DTYPE_t a, DTYPE_t b): return a if a >= b else b

cdef extern from "math.h":
    double sqrt(double m)

cimport cython

@cython.wraparound(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
def pw_dist(np.ndarray[DTYPE_t, ndim=2] xyz_array):
    """
    Returns the maximum pairwise distance between all elements
    of the array.
    """

    assert xyz_array.shape[1] == 3
    assert xyz_array.dtype == DTYPE

    # var type declarations
    cdef int M, i, j
    cdef DTYPE_t max_d, d
    cdef DTYPE_t _x, _y, _z, _xx, _yy, _zz

    M = xyz_array.shape[0]

    max_d = 0.0
    for i in range(M):
        _x, _y, _z = xyz_array[i,0], xyz_array[i,1], xyz_array[i,2]
        for j in xrange(i+1, M):
            _xx, _yy, _zz = xyz_array[j,0], xyz_array[j,1], xyz_array[j,2]
            d = (_x - _xx)**2 + (_y - _yy)**2 + (_z - _zz)**2
            max_d = float_max(max_d, d)
    return sqrt(max_d)
