import numpy as np
import scipy.linalg as sl
import scipy.sparse.linalg as ssl

#%%

def freqsh(M, K, n_m=20):
    # Eigenvalue problem
    vals, vecs = ssl.eigsh(A=M, k=n_m, M=K)
    # Frequency in hertz
    f_n = 1./(2.*np.pi) * np.sqrt(1./vals)
    # Reorder the eigeinvalues and eigenvectors
    idx = f_n.argsort()
    f_n = f_n[idx]
    phi = vecs[:, idx]
    # Normilise the eigenvectors
    for i_m in range(phi.shape[1]):
        # Normalise the magnitude
        phi[:, i_m] = phi[:, i_m] / sl.norm(phi[:, i_m])
        # Normalise the sign
        idx = np.argmax(np.abs(phi[:, i_m]))
        sign = phi[idx, i_m]/np.abs(phi[idx, i_m])
        phi[:, i_m] = sign*phi[:, i_m];
    return f_n, phi


def freqsh_v1(M, K, n_m=20, phi_ref=None):
    # Eigenvalue problem
    vals, vecs = ssl.eigsh(A=M, k=n_m, M=K)

    # Frequency in hertz
    f_n = 1./(2.*np.pi) * np.sqrt(1./vals)

    # Reorder the eigeinvalues and eigenvectors
    idx = f_n.argsort()
    f_n = f_n[idx]
    phi = vecs[:, idx]

    # Reference mode shapes
    if phi_ref is None:
        phi_ref = phi

    # Normilise the eigenvectors
    for i_m in range(phi.shape[1]):
        # Normalise the magnitude
        phi[:, i_m] = phi[:, i_m] / sl.norm(phi[:, i_m])
    
    # Modal assurance criterion
    A = np.zeros((n_m, n_m), dtype=np.float64)
    for i in range(n_m):
        for j in range(n_m):
            A[i, j] = sl.norm(phi[:, i].conj()@phi_ref[:, j])
            
    # Reorder the eigenvalues and eigenvectors based on MAC
    idx = np.array([A[i, :].argmax() for i in range(n_m)], dtype=np.int32)        
    f_n = f_n[idx]
    phi = phi[:, idx]

    # Check correlation
    cf = np.array([A[i, idx[i]] for i in range(n_m)], dtype=np.float64)
    if (cf >= 0.95).all():
        correlated = True
    else:
        correlated = False
    
    # Normalise the sign
    for i_m in range(n_m):
        idx = np.argmax(np.abs(phi_ref[:, i_m]))
        sign_1 = phi_ref[idx, i_m]/np.abs(phi_ref[idx, i_m])
        sign_2 = phi[idx, i_m]/np.abs(phi[idx, i_m])
        sign = sign_1*sign_2
        phi[:, i_m] = sign*phi[:, i_m];
    
    return f_n, phi, A, correlated

















def freqshm(M, K, n_m=20):
    # Eigenvalue problem
    vals, vecs = ssl.eigsh(A=M, k=n_m, M=K, maxiter=M.shape[0]*1.e3)
    # Frequency in hertz
    f_n = 1./(2.*np.pi) * np.sqrt(1./vals)
    # Reorder the eigeinvalues and eigenvectors
    idx = f_n.argsort()
    f_n = f_n[idx]
    phi = vecs[:, idx]
    # Normilise the eigenvectors
    for i_m in range(phi.shape[1]):
        # Normalise the magnitude
        phi[:, i_m] = phi[:, i_m] / np.sqrt(phi[:, i_m].T@M@phi[:, i_m])
        # Normalise the sign
        idx = np.argmax(np.abs(phi[:, i_m]))
        sign = phi[idx, i_m]/np.abs(phi[idx, i_m])
        phi[:, i_m] = sign*phi[:, i_m];
    return f_n, phi

# def freqsm(M, K, n_m=20):
#     # Eigenvalue problem
#     vals, vecs = ssl.eigsh(A=M, k=n_m, M=K)
#     # Frequency in hertz
#     f_n = 1./(2.*np.pi) * np.sqrt(1./vals)
#     # Reorder the eigeinvalues and eigenvectors
#     idx = f_n.argsort()
#     f_n = f_n[idx]
#     phi = vecs[:, idx]
#     # Normilise the eigenvectors
#     for i_m in range(phi.shape[1]):
#         phi[:, i_m] = phi[:, i_m] / np.sqrt(phi[:, i_m].T@M@phi[:, i_m])
#     return f_n, phi



def freqs(M, K, n_m=20):
    # Eigenvalue problem
    vals, vecs = ssl.eigs(A=M, k=n_m, M=K)
    # Frequency in hertz
    f_n = 1./(2.*np.pi) * np.sqrt(1./vals)
    # Reorder the eigeinvalues and eigenvectors
    idx = f_n.argsort()
    f_n = f_n[idx]
    phi = vecs[:, idx]
    # Normilise the eigenvectors
    for i_m in range(phi.shape[1]):
        phi[:, i_m] = phi[:, i_m] / sl.norm(phi[:, i_m])
    return f_n, phi

def freqh(M, K):
    # Eigenvalue problem
    vals, vecs = sl.eigh(a=M, b=K)
    # Frequency in hertz
    f_n = 1./(2.*np.pi) * np.sqrt(1./vals)
    # Reorder the eigeinvalues and eigenvectors
    idx = f_n.argsort()
    f_n = f_n[idx]
    phi = vecs[:, idx]
    # Normilise the eigenvectors
    for i_m in range(phi.shape[1]):
        phi[:, i_m] = phi[:, i_m] / sl.norm(phi[:, i_m])
    return f_n, phi




#%%



