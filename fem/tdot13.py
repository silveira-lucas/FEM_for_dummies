import numpy as np
from numba import jit, prange

#%% 

@jit(nopython=True)
def tdot13(A, B):
    C = np.zeros((B.shape[1], B.shape[2]), dtype=np.float64)
    for l in range(A.shape[0]):
        for i in range(B.shape[1]):
            for j in range(B.shape[2]):
                C[i, j] += A[l]*B[l, i, j]
    return C