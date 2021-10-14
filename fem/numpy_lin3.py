import numpy as np

#%% 

def lin3(th, r_i, xi, t_i):
    
    # h
    h = np.zeros((3, ), dtype=np.float64)
    h[0] = (1/2)*xi*(xi - 1)
    h[1] = -(xi - 1)*(xi + 1)
    h[2] = (1/2)*xi*(xi + 1)
    

    # Ne
    Ne = np.zeros((2, 2*h.shape[0]), dtype=np.float64)
    for i in range(h.shape[0]):
        Ne[0, 2*i] = h[i]
        Ne[1, 2*i+1] = h[i]

    # dhdxi
    dhdxi = np.zeros((3, ), dtype=np.float64)
    dhdxi[0] = (1/2)*(2*xi - 1)
    dhdxi[1] = -2*xi
    dhdxi[2] = (1/2)*(2*xi + 1)
    

    # Je
    Je = r_i.T @ dhdxi


    # dl
    dl = np.linalg.norm(Je)
    t = h@t_i
    dg = Ne.T@t * th*dl 

    return dg
