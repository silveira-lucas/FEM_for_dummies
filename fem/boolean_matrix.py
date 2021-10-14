import numpy as np
import scipy.sparse as ss
from numba import jit

#%%

@jit(nopython=True)
def boolean_matrix_j(a, b):
    # Get matrix sparsity
    n = 0
    for i in range(a.size):
        for j in range(b.size):
            if (a[i]==b[j]):
                n += 1
    # Allocating memory
    data = np.ones((n,), dtype=np.int32)
    idx = np.zeros((n,), dtype=np.int32)
    jdx = np.zeros((n,), dtype=np.int32)
    # Constructing idx and jdx
    n = 0
    for i in range(a.size):
        for j in range(b.size):
            if (a[i]==b[j]):
                idx[n] = i
                jdx[n] = j
                n += 1
    return data, idx, jdx

def boolean_matrix(a, b):
    data, idx, jdx = boolean_matrix_j(a, b)
    L = ss.csc_matrix((data, (idx, jdx)), shape=(a.size, b.size), dtype=np.float64)
    return L