import numpy as np
from numba import jit
from .tdot13 import tdot13

#%% 

@jit(nopython=True)
def quad9(C, rho, th, r_i, r_xi, u_e):
    '''
    Calculates the integrands of the mass matrix, stiffness matrix and internal
    elastic force vector at a given position r_xi, in natural coordinates. This
    function was created automatically by the symbolic script sympy_quad9.py
    
    Parameters
    ----------
    C : numpy.ndarray[:, :], dtype=numpy.float64
        Material elastisticity tensor, in Voigt notation [Pa]
    rho : float
        Mateiral density [kg/m**3]
    th : float
        Object thickness [m]
    r_i : numpy.ndarray[:, :], dtype=numpy.float64
        Physical coordinates of the elemement nodes [m]
    r_xi : numpy.ndarray[:], dtype=numpy.float64
        Natural coordinates of the Gaussian Point where the integrands are
        evaluated
    u_e : numpy.ndarray[:, :], dtype=numpy.float64
        Displacement of the element nodes
    
    Returns
    -------
    dM : numpy.ndarray[:, :], dtype=numpy.float64
        Mass matrix integrand at the Gaussian point
    df_int : numpy.ndarray[:], dtype=numpy.float64
        Internal elastic force vector integrand at the Gaussian point
    dK_geo : numpy.ndarray[:, :], dtype=numpy.float64
        Geometric stiffness matrix integrand at the Gaussian point
    dK_mat : numpy.ndarray[:, :], dtype=numpy.float64
        Geometric stiffness matrix integrand at the Gaussian point
    '''
    
    # Unpacking input variables
    u_0 = u_e[0]
    v_0 = u_e[1]
    u_1 = u_e[2]
    v_1 = u_e[3]
    u_2 = u_e[4]
    v_2 = u_e[5]
    u_3 = u_e[6]
    v_3 = u_e[7]
    u_4 = u_e[8]
    v_4 = u_e[9]
    u_5 = u_e[10]
    v_5 = u_e[11]
    u_6 = u_e[12]
    v_6 = u_e[13]
    u_7 = u_e[14]
    v_7 = u_e[15]
    u_8 = u_e[16]
    v_8 = u_e[17]
    
    xi = r_xi[0]
    eta = r_xi[1]
    
    # h
    h = np.zeros((9, ), dtype=np.float64)
    h[0] = (1/4)*eta*xi*(eta - 1)*(xi - 1)
    h[1] = (1/4)*eta*xi*(eta - 1)*(xi + 1)
    h[2] = (1/4)*eta*xi*(eta + 1)*(xi + 1)
    h[3] = (1/4)*eta*xi*(eta + 1)*(xi - 1)
    h[4] = -1/2*eta*(eta - 1)*(xi - 1)*(xi + 1)
    h[5] = -1/2*xi*(eta - 1)*(eta + 1)*(xi + 1)
    h[6] = -1/2*eta*(eta + 1)*(xi - 1)*(xi + 1)
    h[7] = -1/2*xi*(eta - 1)*(eta + 1)*(xi - 1)
    h[8] = (eta - 1)*(eta + 1)*(xi - 1)*(xi + 1)
    
    # dhdxi
    dhdxi = np.zeros((9, ), dtype=np.float64)
    dhdxi[0] = (1/4)*eta*(eta - 1)*(2*xi - 1)
    dhdxi[1] = (1/4)*eta*(eta - 1)*(2*xi + 1)
    dhdxi[2] = (1/4)*eta*(eta + 1)*(2*xi + 1)
    dhdxi[3] = (1/4)*eta*(eta + 1)*(2*xi - 1)
    dhdxi[4] = -eta*xi*(eta - 1)
    dhdxi[5] = -1/2*(eta - 1)*(eta + 1)*(2*xi + 1)
    dhdxi[6] = -eta*xi*(eta + 1)
    dhdxi[7] = -1/2*(eta - 1)*(eta + 1)*(2*xi - 1)
    dhdxi[8] = 2*xi*(eta - 1)*(eta + 1)
    
    # dhdeta
    dhdeta = np.zeros((9, ), dtype=np.float64)
    dhdeta[0] = (1/4)*xi*(2*eta - 1)*(xi - 1)
    dhdeta[1] = (1/4)*xi*(2*eta - 1)*(xi + 1)
    dhdeta[2] = (1/4)*xi*(2*eta + 1)*(xi + 1)
    dhdeta[3] = (1/4)*xi*(2*eta + 1)*(xi - 1)
    dhdeta[4] = -1/2*(2*eta - 1)*(xi - 1)*(xi + 1)
    dhdeta[5] = -eta*xi*(xi + 1)
    dhdeta[6] = -1/2*(2*eta + 1)*(xi - 1)*(xi + 1)
    dhdeta[7] = -eta*xi*(xi - 1)
    dhdeta[8] = 2*eta*(xi - 1)*(xi + 1)
    

    # Ne
    Ne = np.zeros((2, 2*h.shape[0]), dtype=np.float64)
    for i in range(h.shape[0]):
        Ne[0, 2*i] = h[i]
        Ne[1, 2*i+1] = h[i]


    # DN
    DN = np.zeros((h.shape[0], 2), dtype=np.float64)
    for i in range(h.shape[0]):
        DN[i, 0] = dhdxi[i]
        DN[i, 1] = dhdeta[i]
    
    Je = r_i.T @ DN
    Jem1 = np.linalg.inv(Je)
    
    GN = DN @ Jem1
    dhdX = np.zeros(h.shape[0], dtype=np.float64)
    dhdY = np.zeros(h.shape[0], dtype=np.float64)
    for i in range(h.shape[0]):
        dhdX[i] = GN[i, 0]
        dhdY[i] = GN[i, 1]    

    # E
    E = np.zeros((3, ), dtype=np.float64)
    E[0] = dhdX[0]*u_0 + dhdX[1]*u_1 + dhdX[2]*u_2 + dhdX[3]*u_3 + dhdX[4]*u_4 + dhdX[5]*u_5 + dhdX[6]*u_6 + dhdX[7]*u_7 + dhdX[8]*u_8 + (1/2)*(dhdX[0]*u_0 + dhdX[1]*u_1 + dhdX[2]*u_2 + dhdX[3]*u_3 + dhdX[4]*u_4 + dhdX[5]*u_5 + dhdX[6]*u_6 + dhdX[7]*u_7 + dhdX[8]*u_8)**2 + (1/2)*(dhdX[0]*v_0 + dhdX[1]*v_1 + dhdX[2]*v_2 + dhdX[3]*v_3 + dhdX[4]*v_4 + dhdX[5]*v_5 + dhdX[6]*v_6 + dhdX[7]*v_7 + dhdX[8]*v_8)**2
    E[1] = dhdY[0]*v_0 + dhdY[1]*v_1 + dhdY[2]*v_2 + dhdY[3]*v_3 + dhdY[4]*v_4 + dhdY[5]*v_5 + dhdY[6]*v_6 + dhdY[7]*v_7 + dhdY[8]*v_8 + (1/2)*(dhdY[0]*u_0 + dhdY[1]*u_1 + dhdY[2]*u_2 + dhdY[3]*u_3 + dhdY[4]*u_4 + dhdY[5]*u_5 + dhdY[6]*u_6 + dhdY[7]*u_7 + dhdY[8]*u_8)**2 + (1/2)*(dhdY[0]*v_0 + dhdY[1]*v_1 + dhdY[2]*v_2 + dhdY[3]*v_3 + dhdY[4]*v_4 + dhdY[5]*v_5 + dhdY[6]*v_6 + dhdY[7]*v_7 + dhdY[8]*v_8)**2
    E[2] = dhdX[0]*v_0 + dhdX[1]*v_1 + dhdX[2]*v_2 + dhdX[3]*v_3 + dhdX[4]*v_4 + dhdX[5]*v_5 + dhdX[6]*v_6 + dhdX[7]*v_7 + dhdX[8]*v_8 + dhdY[0]*u_0 + dhdY[1]*u_1 + dhdY[2]*u_2 + dhdY[3]*u_3 + dhdY[4]*u_4 + dhdY[5]*u_5 + dhdY[6]*u_6 + dhdY[7]*u_7 + dhdY[8]*u_8 + (dhdX[0]*u_0 + dhdX[1]*u_1 + dhdX[2]*u_2 + dhdX[3]*u_3 + dhdX[4]*u_4 + dhdX[5]*u_5 + dhdX[6]*u_6 + dhdX[7]*u_7 + dhdX[8]*u_8)*(dhdY[0]*u_0 + dhdY[1]*u_1 + dhdY[2]*u_2 + dhdY[3]*u_3 + dhdY[4]*u_4 + dhdY[5]*u_5 + dhdY[6]*u_6 + dhdY[7]*u_7 + dhdY[8]*u_8) + (dhdX[0]*v_0 + dhdX[1]*v_1 + dhdX[2]*v_2 + dhdX[3]*v_3 + dhdX[4]*v_4 + dhdX[5]*v_5 + dhdX[6]*v_6 + dhdX[7]*v_7 + dhdX[8]*v_8)*(dhdY[0]*v_0 + dhdY[1]*v_1 + dhdY[2]*v_2 + dhdY[3]*v_3 + dhdY[4]*v_4 + dhdY[5]*v_5 + dhdY[6]*v_6 + dhdY[7]*v_7 + dhdY[8]*v_8)
    
    Be = np.zeros((3, 18), dtype=np.float64)
    Be[0, 0] = dhdX[0]*(dhdX[0]*u_0 + dhdX[1]*u_1 + dhdX[2]*u_2 + dhdX[3]*u_3 + dhdX[4]*u_4 + dhdX[5]*u_5 + dhdX[6]*u_6 + dhdX[7]*u_7 + dhdX[8]*u_8) + dhdX[0]
    Be[0, 1] = dhdX[0]*(dhdX[0]*v_0 + dhdX[1]*v_1 + dhdX[2]*v_2 + dhdX[3]*v_3 + dhdX[4]*v_4 + dhdX[5]*v_5 + dhdX[6]*v_6 + dhdX[7]*v_7 + dhdX[8]*v_8)
    Be[0, 2] = dhdX[1]*(dhdX[0]*u_0 + dhdX[1]*u_1 + dhdX[2]*u_2 + dhdX[3]*u_3 + dhdX[4]*u_4 + dhdX[5]*u_5 + dhdX[6]*u_6 + dhdX[7]*u_7 + dhdX[8]*u_8) + dhdX[1]
    Be[0, 3] = dhdX[1]*(dhdX[0]*v_0 + dhdX[1]*v_1 + dhdX[2]*v_2 + dhdX[3]*v_3 + dhdX[4]*v_4 + dhdX[5]*v_5 + dhdX[6]*v_6 + dhdX[7]*v_7 + dhdX[8]*v_8)
    Be[0, 4] = dhdX[2]*(dhdX[0]*u_0 + dhdX[1]*u_1 + dhdX[2]*u_2 + dhdX[3]*u_3 + dhdX[4]*u_4 + dhdX[5]*u_5 + dhdX[6]*u_6 + dhdX[7]*u_7 + dhdX[8]*u_8) + dhdX[2]
    Be[0, 5] = dhdX[2]*(dhdX[0]*v_0 + dhdX[1]*v_1 + dhdX[2]*v_2 + dhdX[3]*v_3 + dhdX[4]*v_4 + dhdX[5]*v_5 + dhdX[6]*v_6 + dhdX[7]*v_7 + dhdX[8]*v_8)
    Be[0, 6] = dhdX[3]*(dhdX[0]*u_0 + dhdX[1]*u_1 + dhdX[2]*u_2 + dhdX[3]*u_3 + dhdX[4]*u_4 + dhdX[5]*u_5 + dhdX[6]*u_6 + dhdX[7]*u_7 + dhdX[8]*u_8) + dhdX[3]
    Be[0, 7] = dhdX[3]*(dhdX[0]*v_0 + dhdX[1]*v_1 + dhdX[2]*v_2 + dhdX[3]*v_3 + dhdX[4]*v_4 + dhdX[5]*v_5 + dhdX[6]*v_6 + dhdX[7]*v_7 + dhdX[8]*v_8)
    Be[0, 8] = dhdX[4]*(dhdX[0]*u_0 + dhdX[1]*u_1 + dhdX[2]*u_2 + dhdX[3]*u_3 + dhdX[4]*u_4 + dhdX[5]*u_5 + dhdX[6]*u_6 + dhdX[7]*u_7 + dhdX[8]*u_8) + dhdX[4]
    Be[0, 9] = dhdX[4]*(dhdX[0]*v_0 + dhdX[1]*v_1 + dhdX[2]*v_2 + dhdX[3]*v_3 + dhdX[4]*v_4 + dhdX[5]*v_5 + dhdX[6]*v_6 + dhdX[7]*v_7 + dhdX[8]*v_8)
    Be[0, 10] = dhdX[5]*(dhdX[0]*u_0 + dhdX[1]*u_1 + dhdX[2]*u_2 + dhdX[3]*u_3 + dhdX[4]*u_4 + dhdX[5]*u_5 + dhdX[6]*u_6 + dhdX[7]*u_7 + dhdX[8]*u_8) + dhdX[5]
    Be[0, 11] = dhdX[5]*(dhdX[0]*v_0 + dhdX[1]*v_1 + dhdX[2]*v_2 + dhdX[3]*v_3 + dhdX[4]*v_4 + dhdX[5]*v_5 + dhdX[6]*v_6 + dhdX[7]*v_7 + dhdX[8]*v_8)
    Be[0, 12] = dhdX[6]*(dhdX[0]*u_0 + dhdX[1]*u_1 + dhdX[2]*u_2 + dhdX[3]*u_3 + dhdX[4]*u_4 + dhdX[5]*u_5 + dhdX[6]*u_6 + dhdX[7]*u_7 + dhdX[8]*u_8) + dhdX[6]
    Be[0, 13] = dhdX[6]*(dhdX[0]*v_0 + dhdX[1]*v_1 + dhdX[2]*v_2 + dhdX[3]*v_3 + dhdX[4]*v_4 + dhdX[5]*v_5 + dhdX[6]*v_6 + dhdX[7]*v_7 + dhdX[8]*v_8)
    Be[0, 14] = dhdX[7]*(dhdX[0]*u_0 + dhdX[1]*u_1 + dhdX[2]*u_2 + dhdX[3]*u_3 + dhdX[4]*u_4 + dhdX[5]*u_5 + dhdX[6]*u_6 + dhdX[7]*u_7 + dhdX[8]*u_8) + dhdX[7]
    Be[0, 15] = dhdX[7]*(dhdX[0]*v_0 + dhdX[1]*v_1 + dhdX[2]*v_2 + dhdX[3]*v_3 + dhdX[4]*v_4 + dhdX[5]*v_5 + dhdX[6]*v_6 + dhdX[7]*v_7 + dhdX[8]*v_8)
    Be[0, 16] = dhdX[8]*(dhdX[0]*u_0 + dhdX[1]*u_1 + dhdX[2]*u_2 + dhdX[3]*u_3 + dhdX[4]*u_4 + dhdX[5]*u_5 + dhdX[6]*u_6 + dhdX[7]*u_7 + dhdX[8]*u_8) + dhdX[8]
    Be[0, 17] = dhdX[8]*(dhdX[0]*v_0 + dhdX[1]*v_1 + dhdX[2]*v_2 + dhdX[3]*v_3 + dhdX[4]*v_4 + dhdX[5]*v_5 + dhdX[6]*v_6 + dhdX[7]*v_7 + dhdX[8]*v_8)
    Be[1, 0] = dhdY[0]*(dhdY[0]*u_0 + dhdY[1]*u_1 + dhdY[2]*u_2 + dhdY[3]*u_3 + dhdY[4]*u_4 + dhdY[5]*u_5 + dhdY[6]*u_6 + dhdY[7]*u_7 + dhdY[8]*u_8)
    Be[1, 1] = dhdY[0]*(dhdY[0]*v_0 + dhdY[1]*v_1 + dhdY[2]*v_2 + dhdY[3]*v_3 + dhdY[4]*v_4 + dhdY[5]*v_5 + dhdY[6]*v_6 + dhdY[7]*v_7 + dhdY[8]*v_8) + dhdY[0]
    Be[1, 2] = dhdY[1]*(dhdY[0]*u_0 + dhdY[1]*u_1 + dhdY[2]*u_2 + dhdY[3]*u_3 + dhdY[4]*u_4 + dhdY[5]*u_5 + dhdY[6]*u_6 + dhdY[7]*u_7 + dhdY[8]*u_8)
    Be[1, 3] = dhdY[1]*(dhdY[0]*v_0 + dhdY[1]*v_1 + dhdY[2]*v_2 + dhdY[3]*v_3 + dhdY[4]*v_4 + dhdY[5]*v_5 + dhdY[6]*v_6 + dhdY[7]*v_7 + dhdY[8]*v_8) + dhdY[1]
    Be[1, 4] = dhdY[2]*(dhdY[0]*u_0 + dhdY[1]*u_1 + dhdY[2]*u_2 + dhdY[3]*u_3 + dhdY[4]*u_4 + dhdY[5]*u_5 + dhdY[6]*u_6 + dhdY[7]*u_7 + dhdY[8]*u_8)
    Be[1, 5] = dhdY[2]*(dhdY[0]*v_0 + dhdY[1]*v_1 + dhdY[2]*v_2 + dhdY[3]*v_3 + dhdY[4]*v_4 + dhdY[5]*v_5 + dhdY[6]*v_6 + dhdY[7]*v_7 + dhdY[8]*v_8) + dhdY[2]
    Be[1, 6] = dhdY[3]*(dhdY[0]*u_0 + dhdY[1]*u_1 + dhdY[2]*u_2 + dhdY[3]*u_3 + dhdY[4]*u_4 + dhdY[5]*u_5 + dhdY[6]*u_6 + dhdY[7]*u_7 + dhdY[8]*u_8)
    Be[1, 7] = dhdY[3]*(dhdY[0]*v_0 + dhdY[1]*v_1 + dhdY[2]*v_2 + dhdY[3]*v_3 + dhdY[4]*v_4 + dhdY[5]*v_5 + dhdY[6]*v_6 + dhdY[7]*v_7 + dhdY[8]*v_8) + dhdY[3]
    Be[1, 8] = dhdY[4]*(dhdY[0]*u_0 + dhdY[1]*u_1 + dhdY[2]*u_2 + dhdY[3]*u_3 + dhdY[4]*u_4 + dhdY[5]*u_5 + dhdY[6]*u_6 + dhdY[7]*u_7 + dhdY[8]*u_8)
    Be[1, 9] = dhdY[4]*(dhdY[0]*v_0 + dhdY[1]*v_1 + dhdY[2]*v_2 + dhdY[3]*v_3 + dhdY[4]*v_4 + dhdY[5]*v_5 + dhdY[6]*v_6 + dhdY[7]*v_7 + dhdY[8]*v_8) + dhdY[4]
    Be[1, 10] = dhdY[5]*(dhdY[0]*u_0 + dhdY[1]*u_1 + dhdY[2]*u_2 + dhdY[3]*u_3 + dhdY[4]*u_4 + dhdY[5]*u_5 + dhdY[6]*u_6 + dhdY[7]*u_7 + dhdY[8]*u_8)
    Be[1, 11] = dhdY[5]*(dhdY[0]*v_0 + dhdY[1]*v_1 + dhdY[2]*v_2 + dhdY[3]*v_3 + dhdY[4]*v_4 + dhdY[5]*v_5 + dhdY[6]*v_6 + dhdY[7]*v_7 + dhdY[8]*v_8) + dhdY[5]
    Be[1, 12] = dhdY[6]*(dhdY[0]*u_0 + dhdY[1]*u_1 + dhdY[2]*u_2 + dhdY[3]*u_3 + dhdY[4]*u_4 + dhdY[5]*u_5 + dhdY[6]*u_6 + dhdY[7]*u_7 + dhdY[8]*u_8)
    Be[1, 13] = dhdY[6]*(dhdY[0]*v_0 + dhdY[1]*v_1 + dhdY[2]*v_2 + dhdY[3]*v_3 + dhdY[4]*v_4 + dhdY[5]*v_5 + dhdY[6]*v_6 + dhdY[7]*v_7 + dhdY[8]*v_8) + dhdY[6]
    Be[1, 14] = dhdY[7]*(dhdY[0]*u_0 + dhdY[1]*u_1 + dhdY[2]*u_2 + dhdY[3]*u_3 + dhdY[4]*u_4 + dhdY[5]*u_5 + dhdY[6]*u_6 + dhdY[7]*u_7 + dhdY[8]*u_8)
    Be[1, 15] = dhdY[7]*(dhdY[0]*v_0 + dhdY[1]*v_1 + dhdY[2]*v_2 + dhdY[3]*v_3 + dhdY[4]*v_4 + dhdY[5]*v_5 + dhdY[6]*v_6 + dhdY[7]*v_7 + dhdY[8]*v_8) + dhdY[7]
    Be[1, 16] = dhdY[8]*(dhdY[0]*u_0 + dhdY[1]*u_1 + dhdY[2]*u_2 + dhdY[3]*u_3 + dhdY[4]*u_4 + dhdY[5]*u_5 + dhdY[6]*u_6 + dhdY[7]*u_7 + dhdY[8]*u_8)
    Be[1, 17] = dhdY[8]*(dhdY[0]*v_0 + dhdY[1]*v_1 + dhdY[2]*v_2 + dhdY[3]*v_3 + dhdY[4]*v_4 + dhdY[5]*v_5 + dhdY[6]*v_6 + dhdY[7]*v_7 + dhdY[8]*v_8) + dhdY[8]
    Be[2, 0] = dhdX[0]*(dhdY[0]*u_0 + dhdY[1]*u_1 + dhdY[2]*u_2 + dhdY[3]*u_3 + dhdY[4]*u_4 + dhdY[5]*u_5 + dhdY[6]*u_6 + dhdY[7]*u_7 + dhdY[8]*u_8) + dhdY[0]*(dhdX[0]*u_0 + dhdX[1]*u_1 + dhdX[2]*u_2 + dhdX[3]*u_3 + dhdX[4]*u_4 + dhdX[5]*u_5 + dhdX[6]*u_6 + dhdX[7]*u_7 + dhdX[8]*u_8) + dhdY[0]
    Be[2, 1] = dhdX[0]*(dhdY[0]*v_0 + dhdY[1]*v_1 + dhdY[2]*v_2 + dhdY[3]*v_3 + dhdY[4]*v_4 + dhdY[5]*v_5 + dhdY[6]*v_6 + dhdY[7]*v_7 + dhdY[8]*v_8) + dhdX[0] + dhdY[0]*(dhdX[0]*v_0 + dhdX[1]*v_1 + dhdX[2]*v_2 + dhdX[3]*v_3 + dhdX[4]*v_4 + dhdX[5]*v_5 + dhdX[6]*v_6 + dhdX[7]*v_7 + dhdX[8]*v_8)
    Be[2, 2] = dhdX[1]*(dhdY[0]*u_0 + dhdY[1]*u_1 + dhdY[2]*u_2 + dhdY[3]*u_3 + dhdY[4]*u_4 + dhdY[5]*u_5 + dhdY[6]*u_6 + dhdY[7]*u_7 + dhdY[8]*u_8) + dhdY[1]*(dhdX[0]*u_0 + dhdX[1]*u_1 + dhdX[2]*u_2 + dhdX[3]*u_3 + dhdX[4]*u_4 + dhdX[5]*u_5 + dhdX[6]*u_6 + dhdX[7]*u_7 + dhdX[8]*u_8) + dhdY[1]
    Be[2, 3] = dhdX[1]*(dhdY[0]*v_0 + dhdY[1]*v_1 + dhdY[2]*v_2 + dhdY[3]*v_3 + dhdY[4]*v_4 + dhdY[5]*v_5 + dhdY[6]*v_6 + dhdY[7]*v_7 + dhdY[8]*v_8) + dhdX[1] + dhdY[1]*(dhdX[0]*v_0 + dhdX[1]*v_1 + dhdX[2]*v_2 + dhdX[3]*v_3 + dhdX[4]*v_4 + dhdX[5]*v_5 + dhdX[6]*v_6 + dhdX[7]*v_7 + dhdX[8]*v_8)
    Be[2, 4] = dhdX[2]*(dhdY[0]*u_0 + dhdY[1]*u_1 + dhdY[2]*u_2 + dhdY[3]*u_3 + dhdY[4]*u_4 + dhdY[5]*u_5 + dhdY[6]*u_6 + dhdY[7]*u_7 + dhdY[8]*u_8) + dhdY[2]*(dhdX[0]*u_0 + dhdX[1]*u_1 + dhdX[2]*u_2 + dhdX[3]*u_3 + dhdX[4]*u_4 + dhdX[5]*u_5 + dhdX[6]*u_6 + dhdX[7]*u_7 + dhdX[8]*u_8) + dhdY[2]
    Be[2, 5] = dhdX[2]*(dhdY[0]*v_0 + dhdY[1]*v_1 + dhdY[2]*v_2 + dhdY[3]*v_3 + dhdY[4]*v_4 + dhdY[5]*v_5 + dhdY[6]*v_6 + dhdY[7]*v_7 + dhdY[8]*v_8) + dhdX[2] + dhdY[2]*(dhdX[0]*v_0 + dhdX[1]*v_1 + dhdX[2]*v_2 + dhdX[3]*v_3 + dhdX[4]*v_4 + dhdX[5]*v_5 + dhdX[6]*v_6 + dhdX[7]*v_7 + dhdX[8]*v_8)
    Be[2, 6] = dhdX[3]*(dhdY[0]*u_0 + dhdY[1]*u_1 + dhdY[2]*u_2 + dhdY[3]*u_3 + dhdY[4]*u_4 + dhdY[5]*u_5 + dhdY[6]*u_6 + dhdY[7]*u_7 + dhdY[8]*u_8) + dhdY[3]*(dhdX[0]*u_0 + dhdX[1]*u_1 + dhdX[2]*u_2 + dhdX[3]*u_3 + dhdX[4]*u_4 + dhdX[5]*u_5 + dhdX[6]*u_6 + dhdX[7]*u_7 + dhdX[8]*u_8) + dhdY[3]
    Be[2, 7] = dhdX[3]*(dhdY[0]*v_0 + dhdY[1]*v_1 + dhdY[2]*v_2 + dhdY[3]*v_3 + dhdY[4]*v_4 + dhdY[5]*v_5 + dhdY[6]*v_6 + dhdY[7]*v_7 + dhdY[8]*v_8) + dhdX[3] + dhdY[3]*(dhdX[0]*v_0 + dhdX[1]*v_1 + dhdX[2]*v_2 + dhdX[3]*v_3 + dhdX[4]*v_4 + dhdX[5]*v_5 + dhdX[6]*v_6 + dhdX[7]*v_7 + dhdX[8]*v_8)
    Be[2, 8] = dhdX[4]*(dhdY[0]*u_0 + dhdY[1]*u_1 + dhdY[2]*u_2 + dhdY[3]*u_3 + dhdY[4]*u_4 + dhdY[5]*u_5 + dhdY[6]*u_6 + dhdY[7]*u_7 + dhdY[8]*u_8) + dhdY[4]*(dhdX[0]*u_0 + dhdX[1]*u_1 + dhdX[2]*u_2 + dhdX[3]*u_3 + dhdX[4]*u_4 + dhdX[5]*u_5 + dhdX[6]*u_6 + dhdX[7]*u_7 + dhdX[8]*u_8) + dhdY[4]
    Be[2, 9] = dhdX[4]*(dhdY[0]*v_0 + dhdY[1]*v_1 + dhdY[2]*v_2 + dhdY[3]*v_3 + dhdY[4]*v_4 + dhdY[5]*v_5 + dhdY[6]*v_6 + dhdY[7]*v_7 + dhdY[8]*v_8) + dhdX[4] + dhdY[4]*(dhdX[0]*v_0 + dhdX[1]*v_1 + dhdX[2]*v_2 + dhdX[3]*v_3 + dhdX[4]*v_4 + dhdX[5]*v_5 + dhdX[6]*v_6 + dhdX[7]*v_7 + dhdX[8]*v_8)
    Be[2, 10] = dhdX[5]*(dhdY[0]*u_0 + dhdY[1]*u_1 + dhdY[2]*u_2 + dhdY[3]*u_3 + dhdY[4]*u_4 + dhdY[5]*u_5 + dhdY[6]*u_6 + dhdY[7]*u_7 + dhdY[8]*u_8) + dhdY[5]*(dhdX[0]*u_0 + dhdX[1]*u_1 + dhdX[2]*u_2 + dhdX[3]*u_3 + dhdX[4]*u_4 + dhdX[5]*u_5 + dhdX[6]*u_6 + dhdX[7]*u_7 + dhdX[8]*u_8) + dhdY[5]
    Be[2, 11] = dhdX[5]*(dhdY[0]*v_0 + dhdY[1]*v_1 + dhdY[2]*v_2 + dhdY[3]*v_3 + dhdY[4]*v_4 + dhdY[5]*v_5 + dhdY[6]*v_6 + dhdY[7]*v_7 + dhdY[8]*v_8) + dhdX[5] + dhdY[5]*(dhdX[0]*v_0 + dhdX[1]*v_1 + dhdX[2]*v_2 + dhdX[3]*v_3 + dhdX[4]*v_4 + dhdX[5]*v_5 + dhdX[6]*v_6 + dhdX[7]*v_7 + dhdX[8]*v_8)
    Be[2, 12] = dhdX[6]*(dhdY[0]*u_0 + dhdY[1]*u_1 + dhdY[2]*u_2 + dhdY[3]*u_3 + dhdY[4]*u_4 + dhdY[5]*u_5 + dhdY[6]*u_6 + dhdY[7]*u_7 + dhdY[8]*u_8) + dhdY[6]*(dhdX[0]*u_0 + dhdX[1]*u_1 + dhdX[2]*u_2 + dhdX[3]*u_3 + dhdX[4]*u_4 + dhdX[5]*u_5 + dhdX[6]*u_6 + dhdX[7]*u_7 + dhdX[8]*u_8) + dhdY[6]
    Be[2, 13] = dhdX[6]*(dhdY[0]*v_0 + dhdY[1]*v_1 + dhdY[2]*v_2 + dhdY[3]*v_3 + dhdY[4]*v_4 + dhdY[5]*v_5 + dhdY[6]*v_6 + dhdY[7]*v_7 + dhdY[8]*v_8) + dhdX[6] + dhdY[6]*(dhdX[0]*v_0 + dhdX[1]*v_1 + dhdX[2]*v_2 + dhdX[3]*v_3 + dhdX[4]*v_4 + dhdX[5]*v_5 + dhdX[6]*v_6 + dhdX[7]*v_7 + dhdX[8]*v_8)
    Be[2, 14] = dhdX[7]*(dhdY[0]*u_0 + dhdY[1]*u_1 + dhdY[2]*u_2 + dhdY[3]*u_3 + dhdY[4]*u_4 + dhdY[5]*u_5 + dhdY[6]*u_6 + dhdY[7]*u_7 + dhdY[8]*u_8) + dhdY[7]*(dhdX[0]*u_0 + dhdX[1]*u_1 + dhdX[2]*u_2 + dhdX[3]*u_3 + dhdX[4]*u_4 + dhdX[5]*u_5 + dhdX[6]*u_6 + dhdX[7]*u_7 + dhdX[8]*u_8) + dhdY[7]
    Be[2, 15] = dhdX[7]*(dhdY[0]*v_0 + dhdY[1]*v_1 + dhdY[2]*v_2 + dhdY[3]*v_3 + dhdY[4]*v_4 + dhdY[5]*v_5 + dhdY[6]*v_6 + dhdY[7]*v_7 + dhdY[8]*v_8) + dhdX[7] + dhdY[7]*(dhdX[0]*v_0 + dhdX[1]*v_1 + dhdX[2]*v_2 + dhdX[3]*v_3 + dhdX[4]*v_4 + dhdX[5]*v_5 + dhdX[6]*v_6 + dhdX[7]*v_7 + dhdX[8]*v_8)
    Be[2, 16] = dhdX[8]*(dhdY[0]*u_0 + dhdY[1]*u_1 + dhdY[2]*u_2 + dhdY[3]*u_3 + dhdY[4]*u_4 + dhdY[5]*u_5 + dhdY[6]*u_6 + dhdY[7]*u_7 + dhdY[8]*u_8) + dhdY[8]*(dhdX[0]*u_0 + dhdX[1]*u_1 + dhdX[2]*u_2 + dhdX[3]*u_3 + dhdX[4]*u_4 + dhdX[5]*u_5 + dhdX[6]*u_6 + dhdX[7]*u_7 + dhdX[8]*u_8) + dhdY[8]
    Be[2, 17] = dhdX[8]*(dhdY[0]*v_0 + dhdY[1]*v_1 + dhdY[2]*v_2 + dhdY[3]*v_3 + dhdY[4]*v_4 + dhdY[5]*v_5 + dhdY[6]*v_6 + dhdY[7]*v_7 + dhdY[8]*v_8) + dhdX[8] + dhdY[8]*(dhdX[0]*v_0 + dhdX[1]*v_1 + dhdX[2]*v_2 + dhdX[3]*v_3 + dhdX[4]*v_4 + dhdX[5]*v_5 + dhdX[6]*v_6 + dhdX[7]*v_7 + dhdX[8]*v_8)
    
    dBedu = np.zeros((3, 18, 18), dtype=np.float64)
    dBedu[0, 0, 0] = dhdX[0]**2
    dBedu[0, 0, 2] = dhdX[0]*dhdX[1]
    dBedu[0, 0, 4] = dhdX[0]*dhdX[2]
    dBedu[0, 0, 6] = dhdX[0]*dhdX[3]
    dBedu[0, 0, 8] = dhdX[0]*dhdX[4]
    dBedu[0, 0, 10] = dhdX[0]*dhdX[5]
    dBedu[0, 0, 12] = dhdX[0]*dhdX[6]
    dBedu[0, 0, 14] = dhdX[0]*dhdX[7]
    dBedu[0, 0, 16] = dhdX[0]*dhdX[8]
    dBedu[0, 1, 1] = dhdX[0]**2
    dBedu[0, 1, 3] = dhdX[0]*dhdX[1]
    dBedu[0, 1, 5] = dhdX[0]*dhdX[2]
    dBedu[0, 1, 7] = dhdX[0]*dhdX[3]
    dBedu[0, 1, 9] = dhdX[0]*dhdX[4]
    dBedu[0, 1, 11] = dhdX[0]*dhdX[5]
    dBedu[0, 1, 13] = dhdX[0]*dhdX[6]
    dBedu[0, 1, 15] = dhdX[0]*dhdX[7]
    dBedu[0, 1, 17] = dhdX[0]*dhdX[8]
    dBedu[0, 2, 0] = dhdX[0]*dhdX[1]
    dBedu[0, 2, 2] = dhdX[1]**2
    dBedu[0, 2, 4] = dhdX[1]*dhdX[2]
    dBedu[0, 2, 6] = dhdX[1]*dhdX[3]
    dBedu[0, 2, 8] = dhdX[1]*dhdX[4]
    dBedu[0, 2, 10] = dhdX[1]*dhdX[5]
    dBedu[0, 2, 12] = dhdX[1]*dhdX[6]
    dBedu[0, 2, 14] = dhdX[1]*dhdX[7]
    dBedu[0, 2, 16] = dhdX[1]*dhdX[8]
    dBedu[0, 3, 1] = dhdX[0]*dhdX[1]
    dBedu[0, 3, 3] = dhdX[1]**2
    dBedu[0, 3, 5] = dhdX[1]*dhdX[2]
    dBedu[0, 3, 7] = dhdX[1]*dhdX[3]
    dBedu[0, 3, 9] = dhdX[1]*dhdX[4]
    dBedu[0, 3, 11] = dhdX[1]*dhdX[5]
    dBedu[0, 3, 13] = dhdX[1]*dhdX[6]
    dBedu[0, 3, 15] = dhdX[1]*dhdX[7]
    dBedu[0, 3, 17] = dhdX[1]*dhdX[8]
    dBedu[0, 4, 0] = dhdX[0]*dhdX[2]
    dBedu[0, 4, 2] = dhdX[1]*dhdX[2]
    dBedu[0, 4, 4] = dhdX[2]**2
    dBedu[0, 4, 6] = dhdX[2]*dhdX[3]
    dBedu[0, 4, 8] = dhdX[2]*dhdX[4]
    dBedu[0, 4, 10] = dhdX[2]*dhdX[5]
    dBedu[0, 4, 12] = dhdX[2]*dhdX[6]
    dBedu[0, 4, 14] = dhdX[2]*dhdX[7]
    dBedu[0, 4, 16] = dhdX[2]*dhdX[8]
    dBedu[0, 5, 1] = dhdX[0]*dhdX[2]
    dBedu[0, 5, 3] = dhdX[1]*dhdX[2]
    dBedu[0, 5, 5] = dhdX[2]**2
    dBedu[0, 5, 7] = dhdX[2]*dhdX[3]
    dBedu[0, 5, 9] = dhdX[2]*dhdX[4]
    dBedu[0, 5, 11] = dhdX[2]*dhdX[5]
    dBedu[0, 5, 13] = dhdX[2]*dhdX[6]
    dBedu[0, 5, 15] = dhdX[2]*dhdX[7]
    dBedu[0, 5, 17] = dhdX[2]*dhdX[8]
    dBedu[0, 6, 0] = dhdX[0]*dhdX[3]
    dBedu[0, 6, 2] = dhdX[1]*dhdX[3]
    dBedu[0, 6, 4] = dhdX[2]*dhdX[3]
    dBedu[0, 6, 6] = dhdX[3]**2
    dBedu[0, 6, 8] = dhdX[3]*dhdX[4]
    dBedu[0, 6, 10] = dhdX[3]*dhdX[5]
    dBedu[0, 6, 12] = dhdX[3]*dhdX[6]
    dBedu[0, 6, 14] = dhdX[3]*dhdX[7]
    dBedu[0, 6, 16] = dhdX[3]*dhdX[8]
    dBedu[0, 7, 1] = dhdX[0]*dhdX[3]
    dBedu[0, 7, 3] = dhdX[1]*dhdX[3]
    dBedu[0, 7, 5] = dhdX[2]*dhdX[3]
    dBedu[0, 7, 7] = dhdX[3]**2
    dBedu[0, 7, 9] = dhdX[3]*dhdX[4]
    dBedu[0, 7, 11] = dhdX[3]*dhdX[5]
    dBedu[0, 7, 13] = dhdX[3]*dhdX[6]
    dBedu[0, 7, 15] = dhdX[3]*dhdX[7]
    dBedu[0, 7, 17] = dhdX[3]*dhdX[8]
    dBedu[0, 8, 0] = dhdX[0]*dhdX[4]
    dBedu[0, 8, 2] = dhdX[1]*dhdX[4]
    dBedu[0, 8, 4] = dhdX[2]*dhdX[4]
    dBedu[0, 8, 6] = dhdX[3]*dhdX[4]
    dBedu[0, 8, 8] = dhdX[4]**2
    dBedu[0, 8, 10] = dhdX[4]*dhdX[5]
    dBedu[0, 8, 12] = dhdX[4]*dhdX[6]
    dBedu[0, 8, 14] = dhdX[4]*dhdX[7]
    dBedu[0, 8, 16] = dhdX[4]*dhdX[8]
    dBedu[0, 9, 1] = dhdX[0]*dhdX[4]
    dBedu[0, 9, 3] = dhdX[1]*dhdX[4]
    dBedu[0, 9, 5] = dhdX[2]*dhdX[4]
    dBedu[0, 9, 7] = dhdX[3]*dhdX[4]
    dBedu[0, 9, 9] = dhdX[4]**2
    dBedu[0, 9, 11] = dhdX[4]*dhdX[5]
    dBedu[0, 9, 13] = dhdX[4]*dhdX[6]
    dBedu[0, 9, 15] = dhdX[4]*dhdX[7]
    dBedu[0, 9, 17] = dhdX[4]*dhdX[8]
    dBedu[0, 10, 0] = dhdX[0]*dhdX[5]
    dBedu[0, 10, 2] = dhdX[1]*dhdX[5]
    dBedu[0, 10, 4] = dhdX[2]*dhdX[5]
    dBedu[0, 10, 6] = dhdX[3]*dhdX[5]
    dBedu[0, 10, 8] = dhdX[4]*dhdX[5]
    dBedu[0, 10, 10] = dhdX[5]**2
    dBedu[0, 10, 12] = dhdX[5]*dhdX[6]
    dBedu[0, 10, 14] = dhdX[5]*dhdX[7]
    dBedu[0, 10, 16] = dhdX[5]*dhdX[8]
    dBedu[0, 11, 1] = dhdX[0]*dhdX[5]
    dBedu[0, 11, 3] = dhdX[1]*dhdX[5]
    dBedu[0, 11, 5] = dhdX[2]*dhdX[5]
    dBedu[0, 11, 7] = dhdX[3]*dhdX[5]
    dBedu[0, 11, 9] = dhdX[4]*dhdX[5]
    dBedu[0, 11, 11] = dhdX[5]**2
    dBedu[0, 11, 13] = dhdX[5]*dhdX[6]
    dBedu[0, 11, 15] = dhdX[5]*dhdX[7]
    dBedu[0, 11, 17] = dhdX[5]*dhdX[8]
    dBedu[0, 12, 0] = dhdX[0]*dhdX[6]
    dBedu[0, 12, 2] = dhdX[1]*dhdX[6]
    dBedu[0, 12, 4] = dhdX[2]*dhdX[6]
    dBedu[0, 12, 6] = dhdX[3]*dhdX[6]
    dBedu[0, 12, 8] = dhdX[4]*dhdX[6]
    dBedu[0, 12, 10] = dhdX[5]*dhdX[6]
    dBedu[0, 12, 12] = dhdX[6]**2
    dBedu[0, 12, 14] = dhdX[6]*dhdX[7]
    dBedu[0, 12, 16] = dhdX[6]*dhdX[8]
    dBedu[0, 13, 1] = dhdX[0]*dhdX[6]
    dBedu[0, 13, 3] = dhdX[1]*dhdX[6]
    dBedu[0, 13, 5] = dhdX[2]*dhdX[6]
    dBedu[0, 13, 7] = dhdX[3]*dhdX[6]
    dBedu[0, 13, 9] = dhdX[4]*dhdX[6]
    dBedu[0, 13, 11] = dhdX[5]*dhdX[6]
    dBedu[0, 13, 13] = dhdX[6]**2
    dBedu[0, 13, 15] = dhdX[6]*dhdX[7]
    dBedu[0, 13, 17] = dhdX[6]*dhdX[8]
    dBedu[0, 14, 0] = dhdX[0]*dhdX[7]
    dBedu[0, 14, 2] = dhdX[1]*dhdX[7]
    dBedu[0, 14, 4] = dhdX[2]*dhdX[7]
    dBedu[0, 14, 6] = dhdX[3]*dhdX[7]
    dBedu[0, 14, 8] = dhdX[4]*dhdX[7]
    dBedu[0, 14, 10] = dhdX[5]*dhdX[7]
    dBedu[0, 14, 12] = dhdX[6]*dhdX[7]
    dBedu[0, 14, 14] = dhdX[7]**2
    dBedu[0, 14, 16] = dhdX[7]*dhdX[8]
    dBedu[0, 15, 1] = dhdX[0]*dhdX[7]
    dBedu[0, 15, 3] = dhdX[1]*dhdX[7]
    dBedu[0, 15, 5] = dhdX[2]*dhdX[7]
    dBedu[0, 15, 7] = dhdX[3]*dhdX[7]
    dBedu[0, 15, 9] = dhdX[4]*dhdX[7]
    dBedu[0, 15, 11] = dhdX[5]*dhdX[7]
    dBedu[0, 15, 13] = dhdX[6]*dhdX[7]
    dBedu[0, 15, 15] = dhdX[7]**2
    dBedu[0, 15, 17] = dhdX[7]*dhdX[8]
    dBedu[0, 16, 0] = dhdX[0]*dhdX[8]
    dBedu[0, 16, 2] = dhdX[1]*dhdX[8]
    dBedu[0, 16, 4] = dhdX[2]*dhdX[8]
    dBedu[0, 16, 6] = dhdX[3]*dhdX[8]
    dBedu[0, 16, 8] = dhdX[4]*dhdX[8]
    dBedu[0, 16, 10] = dhdX[5]*dhdX[8]
    dBedu[0, 16, 12] = dhdX[6]*dhdX[8]
    dBedu[0, 16, 14] = dhdX[7]*dhdX[8]
    dBedu[0, 16, 16] = dhdX[8]**2
    dBedu[0, 17, 1] = dhdX[0]*dhdX[8]
    dBedu[0, 17, 3] = dhdX[1]*dhdX[8]
    dBedu[0, 17, 5] = dhdX[2]*dhdX[8]
    dBedu[0, 17, 7] = dhdX[3]*dhdX[8]
    dBedu[0, 17, 9] = dhdX[4]*dhdX[8]
    dBedu[0, 17, 11] = dhdX[5]*dhdX[8]
    dBedu[0, 17, 13] = dhdX[6]*dhdX[8]
    dBedu[0, 17, 15] = dhdX[7]*dhdX[8]
    dBedu[0, 17, 17] = dhdX[8]**2
    dBedu[1, 0, 0] = dhdY[0]**2
    dBedu[1, 0, 2] = dhdY[0]*dhdY[1]
    dBedu[1, 0, 4] = dhdY[0]*dhdY[2]
    dBedu[1, 0, 6] = dhdY[0]*dhdY[3]
    dBedu[1, 0, 8] = dhdY[0]*dhdY[4]
    dBedu[1, 0, 10] = dhdY[0]*dhdY[5]
    dBedu[1, 0, 12] = dhdY[0]*dhdY[6]
    dBedu[1, 0, 14] = dhdY[0]*dhdY[7]
    dBedu[1, 0, 16] = dhdY[0]*dhdY[8]
    dBedu[1, 1, 1] = dhdY[0]**2
    dBedu[1, 1, 3] = dhdY[0]*dhdY[1]
    dBedu[1, 1, 5] = dhdY[0]*dhdY[2]
    dBedu[1, 1, 7] = dhdY[0]*dhdY[3]
    dBedu[1, 1, 9] = dhdY[0]*dhdY[4]
    dBedu[1, 1, 11] = dhdY[0]*dhdY[5]
    dBedu[1, 1, 13] = dhdY[0]*dhdY[6]
    dBedu[1, 1, 15] = dhdY[0]*dhdY[7]
    dBedu[1, 1, 17] = dhdY[0]*dhdY[8]
    dBedu[1, 2, 0] = dhdY[0]*dhdY[1]
    dBedu[1, 2, 2] = dhdY[1]**2
    dBedu[1, 2, 4] = dhdY[1]*dhdY[2]
    dBedu[1, 2, 6] = dhdY[1]*dhdY[3]
    dBedu[1, 2, 8] = dhdY[1]*dhdY[4]
    dBedu[1, 2, 10] = dhdY[1]*dhdY[5]
    dBedu[1, 2, 12] = dhdY[1]*dhdY[6]
    dBedu[1, 2, 14] = dhdY[1]*dhdY[7]
    dBedu[1, 2, 16] = dhdY[1]*dhdY[8]
    dBedu[1, 3, 1] = dhdY[0]*dhdY[1]
    dBedu[1, 3, 3] = dhdY[1]**2
    dBedu[1, 3, 5] = dhdY[1]*dhdY[2]
    dBedu[1, 3, 7] = dhdY[1]*dhdY[3]
    dBedu[1, 3, 9] = dhdY[1]*dhdY[4]
    dBedu[1, 3, 11] = dhdY[1]*dhdY[5]
    dBedu[1, 3, 13] = dhdY[1]*dhdY[6]
    dBedu[1, 3, 15] = dhdY[1]*dhdY[7]
    dBedu[1, 3, 17] = dhdY[1]*dhdY[8]
    dBedu[1, 4, 0] = dhdY[0]*dhdY[2]
    dBedu[1, 4, 2] = dhdY[1]*dhdY[2]
    dBedu[1, 4, 4] = dhdY[2]**2
    dBedu[1, 4, 6] = dhdY[2]*dhdY[3]
    dBedu[1, 4, 8] = dhdY[2]*dhdY[4]
    dBedu[1, 4, 10] = dhdY[2]*dhdY[5]
    dBedu[1, 4, 12] = dhdY[2]*dhdY[6]
    dBedu[1, 4, 14] = dhdY[2]*dhdY[7]
    dBedu[1, 4, 16] = dhdY[2]*dhdY[8]
    dBedu[1, 5, 1] = dhdY[0]*dhdY[2]
    dBedu[1, 5, 3] = dhdY[1]*dhdY[2]
    dBedu[1, 5, 5] = dhdY[2]**2
    dBedu[1, 5, 7] = dhdY[2]*dhdY[3]
    dBedu[1, 5, 9] = dhdY[2]*dhdY[4]
    dBedu[1, 5, 11] = dhdY[2]*dhdY[5]
    dBedu[1, 5, 13] = dhdY[2]*dhdY[6]
    dBedu[1, 5, 15] = dhdY[2]*dhdY[7]
    dBedu[1, 5, 17] = dhdY[2]*dhdY[8]
    dBedu[1, 6, 0] = dhdY[0]*dhdY[3]
    dBedu[1, 6, 2] = dhdY[1]*dhdY[3]
    dBedu[1, 6, 4] = dhdY[2]*dhdY[3]
    dBedu[1, 6, 6] = dhdY[3]**2
    dBedu[1, 6, 8] = dhdY[3]*dhdY[4]
    dBedu[1, 6, 10] = dhdY[3]*dhdY[5]
    dBedu[1, 6, 12] = dhdY[3]*dhdY[6]
    dBedu[1, 6, 14] = dhdY[3]*dhdY[7]
    dBedu[1, 6, 16] = dhdY[3]*dhdY[8]
    dBedu[1, 7, 1] = dhdY[0]*dhdY[3]
    dBedu[1, 7, 3] = dhdY[1]*dhdY[3]
    dBedu[1, 7, 5] = dhdY[2]*dhdY[3]
    dBedu[1, 7, 7] = dhdY[3]**2
    dBedu[1, 7, 9] = dhdY[3]*dhdY[4]
    dBedu[1, 7, 11] = dhdY[3]*dhdY[5]
    dBedu[1, 7, 13] = dhdY[3]*dhdY[6]
    dBedu[1, 7, 15] = dhdY[3]*dhdY[7]
    dBedu[1, 7, 17] = dhdY[3]*dhdY[8]
    dBedu[1, 8, 0] = dhdY[0]*dhdY[4]
    dBedu[1, 8, 2] = dhdY[1]*dhdY[4]
    dBedu[1, 8, 4] = dhdY[2]*dhdY[4]
    dBedu[1, 8, 6] = dhdY[3]*dhdY[4]
    dBedu[1, 8, 8] = dhdY[4]**2
    dBedu[1, 8, 10] = dhdY[4]*dhdY[5]
    dBedu[1, 8, 12] = dhdY[4]*dhdY[6]
    dBedu[1, 8, 14] = dhdY[4]*dhdY[7]
    dBedu[1, 8, 16] = dhdY[4]*dhdY[8]
    dBedu[1, 9, 1] = dhdY[0]*dhdY[4]
    dBedu[1, 9, 3] = dhdY[1]*dhdY[4]
    dBedu[1, 9, 5] = dhdY[2]*dhdY[4]
    dBedu[1, 9, 7] = dhdY[3]*dhdY[4]
    dBedu[1, 9, 9] = dhdY[4]**2
    dBedu[1, 9, 11] = dhdY[4]*dhdY[5]
    dBedu[1, 9, 13] = dhdY[4]*dhdY[6]
    dBedu[1, 9, 15] = dhdY[4]*dhdY[7]
    dBedu[1, 9, 17] = dhdY[4]*dhdY[8]
    dBedu[1, 10, 0] = dhdY[0]*dhdY[5]
    dBedu[1, 10, 2] = dhdY[1]*dhdY[5]
    dBedu[1, 10, 4] = dhdY[2]*dhdY[5]
    dBedu[1, 10, 6] = dhdY[3]*dhdY[5]
    dBedu[1, 10, 8] = dhdY[4]*dhdY[5]
    dBedu[1, 10, 10] = dhdY[5]**2
    dBedu[1, 10, 12] = dhdY[5]*dhdY[6]
    dBedu[1, 10, 14] = dhdY[5]*dhdY[7]
    dBedu[1, 10, 16] = dhdY[5]*dhdY[8]
    dBedu[1, 11, 1] = dhdY[0]*dhdY[5]
    dBedu[1, 11, 3] = dhdY[1]*dhdY[5]
    dBedu[1, 11, 5] = dhdY[2]*dhdY[5]
    dBedu[1, 11, 7] = dhdY[3]*dhdY[5]
    dBedu[1, 11, 9] = dhdY[4]*dhdY[5]
    dBedu[1, 11, 11] = dhdY[5]**2
    dBedu[1, 11, 13] = dhdY[5]*dhdY[6]
    dBedu[1, 11, 15] = dhdY[5]*dhdY[7]
    dBedu[1, 11, 17] = dhdY[5]*dhdY[8]
    dBedu[1, 12, 0] = dhdY[0]*dhdY[6]
    dBedu[1, 12, 2] = dhdY[1]*dhdY[6]
    dBedu[1, 12, 4] = dhdY[2]*dhdY[6]
    dBedu[1, 12, 6] = dhdY[3]*dhdY[6]
    dBedu[1, 12, 8] = dhdY[4]*dhdY[6]
    dBedu[1, 12, 10] = dhdY[5]*dhdY[6]
    dBedu[1, 12, 12] = dhdY[6]**2
    dBedu[1, 12, 14] = dhdY[6]*dhdY[7]
    dBedu[1, 12, 16] = dhdY[6]*dhdY[8]
    dBedu[1, 13, 1] = dhdY[0]*dhdY[6]
    dBedu[1, 13, 3] = dhdY[1]*dhdY[6]
    dBedu[1, 13, 5] = dhdY[2]*dhdY[6]
    dBedu[1, 13, 7] = dhdY[3]*dhdY[6]
    dBedu[1, 13, 9] = dhdY[4]*dhdY[6]
    dBedu[1, 13, 11] = dhdY[5]*dhdY[6]
    dBedu[1, 13, 13] = dhdY[6]**2
    dBedu[1, 13, 15] = dhdY[6]*dhdY[7]
    dBedu[1, 13, 17] = dhdY[6]*dhdY[8]
    dBedu[1, 14, 0] = dhdY[0]*dhdY[7]
    dBedu[1, 14, 2] = dhdY[1]*dhdY[7]
    dBedu[1, 14, 4] = dhdY[2]*dhdY[7]
    dBedu[1, 14, 6] = dhdY[3]*dhdY[7]
    dBedu[1, 14, 8] = dhdY[4]*dhdY[7]
    dBedu[1, 14, 10] = dhdY[5]*dhdY[7]
    dBedu[1, 14, 12] = dhdY[6]*dhdY[7]
    dBedu[1, 14, 14] = dhdY[7]**2
    dBedu[1, 14, 16] = dhdY[7]*dhdY[8]
    dBedu[1, 15, 1] = dhdY[0]*dhdY[7]
    dBedu[1, 15, 3] = dhdY[1]*dhdY[7]
    dBedu[1, 15, 5] = dhdY[2]*dhdY[7]
    dBedu[1, 15, 7] = dhdY[3]*dhdY[7]
    dBedu[1, 15, 9] = dhdY[4]*dhdY[7]
    dBedu[1, 15, 11] = dhdY[5]*dhdY[7]
    dBedu[1, 15, 13] = dhdY[6]*dhdY[7]
    dBedu[1, 15, 15] = dhdY[7]**2
    dBedu[1, 15, 17] = dhdY[7]*dhdY[8]
    dBedu[1, 16, 0] = dhdY[0]*dhdY[8]
    dBedu[1, 16, 2] = dhdY[1]*dhdY[8]
    dBedu[1, 16, 4] = dhdY[2]*dhdY[8]
    dBedu[1, 16, 6] = dhdY[3]*dhdY[8]
    dBedu[1, 16, 8] = dhdY[4]*dhdY[8]
    dBedu[1, 16, 10] = dhdY[5]*dhdY[8]
    dBedu[1, 16, 12] = dhdY[6]*dhdY[8]
    dBedu[1, 16, 14] = dhdY[7]*dhdY[8]
    dBedu[1, 16, 16] = dhdY[8]**2
    dBedu[1, 17, 1] = dhdY[0]*dhdY[8]
    dBedu[1, 17, 3] = dhdY[1]*dhdY[8]
    dBedu[1, 17, 5] = dhdY[2]*dhdY[8]
    dBedu[1, 17, 7] = dhdY[3]*dhdY[8]
    dBedu[1, 17, 9] = dhdY[4]*dhdY[8]
    dBedu[1, 17, 11] = dhdY[5]*dhdY[8]
    dBedu[1, 17, 13] = dhdY[6]*dhdY[8]
    dBedu[1, 17, 15] = dhdY[7]*dhdY[8]
    dBedu[1, 17, 17] = dhdY[8]**2
    dBedu[2, 0, 0] = 2*dhdX[0]*dhdY[0]
    dBedu[2, 0, 2] = dhdX[0]*dhdY[1] + dhdX[1]*dhdY[0]
    dBedu[2, 0, 4] = dhdX[0]*dhdY[2] + dhdX[2]*dhdY[0]
    dBedu[2, 0, 6] = dhdX[0]*dhdY[3] + dhdX[3]*dhdY[0]
    dBedu[2, 0, 8] = dhdX[0]*dhdY[4] + dhdX[4]*dhdY[0]
    dBedu[2, 0, 10] = dhdX[0]*dhdY[5] + dhdX[5]*dhdY[0]
    dBedu[2, 0, 12] = dhdX[0]*dhdY[6] + dhdX[6]*dhdY[0]
    dBedu[2, 0, 14] = dhdX[0]*dhdY[7] + dhdX[7]*dhdY[0]
    dBedu[2, 0, 16] = dhdX[0]*dhdY[8] + dhdX[8]*dhdY[0]
    dBedu[2, 1, 1] = 2*dhdX[0]*dhdY[0]
    dBedu[2, 1, 3] = dhdX[0]*dhdY[1] + dhdX[1]*dhdY[0]
    dBedu[2, 1, 5] = dhdX[0]*dhdY[2] + dhdX[2]*dhdY[0]
    dBedu[2, 1, 7] = dhdX[0]*dhdY[3] + dhdX[3]*dhdY[0]
    dBedu[2, 1, 9] = dhdX[0]*dhdY[4] + dhdX[4]*dhdY[0]
    dBedu[2, 1, 11] = dhdX[0]*dhdY[5] + dhdX[5]*dhdY[0]
    dBedu[2, 1, 13] = dhdX[0]*dhdY[6] + dhdX[6]*dhdY[0]
    dBedu[2, 1, 15] = dhdX[0]*dhdY[7] + dhdX[7]*dhdY[0]
    dBedu[2, 1, 17] = dhdX[0]*dhdY[8] + dhdX[8]*dhdY[0]
    dBedu[2, 2, 0] = dhdX[0]*dhdY[1] + dhdX[1]*dhdY[0]
    dBedu[2, 2, 2] = 2*dhdX[1]*dhdY[1]
    dBedu[2, 2, 4] = dhdX[1]*dhdY[2] + dhdX[2]*dhdY[1]
    dBedu[2, 2, 6] = dhdX[1]*dhdY[3] + dhdX[3]*dhdY[1]
    dBedu[2, 2, 8] = dhdX[1]*dhdY[4] + dhdX[4]*dhdY[1]
    dBedu[2, 2, 10] = dhdX[1]*dhdY[5] + dhdX[5]*dhdY[1]
    dBedu[2, 2, 12] = dhdX[1]*dhdY[6] + dhdX[6]*dhdY[1]
    dBedu[2, 2, 14] = dhdX[1]*dhdY[7] + dhdX[7]*dhdY[1]
    dBedu[2, 2, 16] = dhdX[1]*dhdY[8] + dhdX[8]*dhdY[1]
    dBedu[2, 3, 1] = dhdX[0]*dhdY[1] + dhdX[1]*dhdY[0]
    dBedu[2, 3, 3] = 2*dhdX[1]*dhdY[1]
    dBedu[2, 3, 5] = dhdX[1]*dhdY[2] + dhdX[2]*dhdY[1]
    dBedu[2, 3, 7] = dhdX[1]*dhdY[3] + dhdX[3]*dhdY[1]
    dBedu[2, 3, 9] = dhdX[1]*dhdY[4] + dhdX[4]*dhdY[1]
    dBedu[2, 3, 11] = dhdX[1]*dhdY[5] + dhdX[5]*dhdY[1]
    dBedu[2, 3, 13] = dhdX[1]*dhdY[6] + dhdX[6]*dhdY[1]
    dBedu[2, 3, 15] = dhdX[1]*dhdY[7] + dhdX[7]*dhdY[1]
    dBedu[2, 3, 17] = dhdX[1]*dhdY[8] + dhdX[8]*dhdY[1]
    dBedu[2, 4, 0] = dhdX[0]*dhdY[2] + dhdX[2]*dhdY[0]
    dBedu[2, 4, 2] = dhdX[1]*dhdY[2] + dhdX[2]*dhdY[1]
    dBedu[2, 4, 4] = 2*dhdX[2]*dhdY[2]
    dBedu[2, 4, 6] = dhdX[2]*dhdY[3] + dhdX[3]*dhdY[2]
    dBedu[2, 4, 8] = dhdX[2]*dhdY[4] + dhdX[4]*dhdY[2]
    dBedu[2, 4, 10] = dhdX[2]*dhdY[5] + dhdX[5]*dhdY[2]
    dBedu[2, 4, 12] = dhdX[2]*dhdY[6] + dhdX[6]*dhdY[2]
    dBedu[2, 4, 14] = dhdX[2]*dhdY[7] + dhdX[7]*dhdY[2]
    dBedu[2, 4, 16] = dhdX[2]*dhdY[8] + dhdX[8]*dhdY[2]
    dBedu[2, 5, 1] = dhdX[0]*dhdY[2] + dhdX[2]*dhdY[0]
    dBedu[2, 5, 3] = dhdX[1]*dhdY[2] + dhdX[2]*dhdY[1]
    dBedu[2, 5, 5] = 2*dhdX[2]*dhdY[2]
    dBedu[2, 5, 7] = dhdX[2]*dhdY[3] + dhdX[3]*dhdY[2]
    dBedu[2, 5, 9] = dhdX[2]*dhdY[4] + dhdX[4]*dhdY[2]
    dBedu[2, 5, 11] = dhdX[2]*dhdY[5] + dhdX[5]*dhdY[2]
    dBedu[2, 5, 13] = dhdX[2]*dhdY[6] + dhdX[6]*dhdY[2]
    dBedu[2, 5, 15] = dhdX[2]*dhdY[7] + dhdX[7]*dhdY[2]
    dBedu[2, 5, 17] = dhdX[2]*dhdY[8] + dhdX[8]*dhdY[2]
    dBedu[2, 6, 0] = dhdX[0]*dhdY[3] + dhdX[3]*dhdY[0]
    dBedu[2, 6, 2] = dhdX[1]*dhdY[3] + dhdX[3]*dhdY[1]
    dBedu[2, 6, 4] = dhdX[2]*dhdY[3] + dhdX[3]*dhdY[2]
    dBedu[2, 6, 6] = 2*dhdX[3]*dhdY[3]
    dBedu[2, 6, 8] = dhdX[3]*dhdY[4] + dhdX[4]*dhdY[3]
    dBedu[2, 6, 10] = dhdX[3]*dhdY[5] + dhdX[5]*dhdY[3]
    dBedu[2, 6, 12] = dhdX[3]*dhdY[6] + dhdX[6]*dhdY[3]
    dBedu[2, 6, 14] = dhdX[3]*dhdY[7] + dhdX[7]*dhdY[3]
    dBedu[2, 6, 16] = dhdX[3]*dhdY[8] + dhdX[8]*dhdY[3]
    dBedu[2, 7, 1] = dhdX[0]*dhdY[3] + dhdX[3]*dhdY[0]
    dBedu[2, 7, 3] = dhdX[1]*dhdY[3] + dhdX[3]*dhdY[1]
    dBedu[2, 7, 5] = dhdX[2]*dhdY[3] + dhdX[3]*dhdY[2]
    dBedu[2, 7, 7] = 2*dhdX[3]*dhdY[3]
    dBedu[2, 7, 9] = dhdX[3]*dhdY[4] + dhdX[4]*dhdY[3]
    dBedu[2, 7, 11] = dhdX[3]*dhdY[5] + dhdX[5]*dhdY[3]
    dBedu[2, 7, 13] = dhdX[3]*dhdY[6] + dhdX[6]*dhdY[3]
    dBedu[2, 7, 15] = dhdX[3]*dhdY[7] + dhdX[7]*dhdY[3]
    dBedu[2, 7, 17] = dhdX[3]*dhdY[8] + dhdX[8]*dhdY[3]
    dBedu[2, 8, 0] = dhdX[0]*dhdY[4] + dhdX[4]*dhdY[0]
    dBedu[2, 8, 2] = dhdX[1]*dhdY[4] + dhdX[4]*dhdY[1]
    dBedu[2, 8, 4] = dhdX[2]*dhdY[4] + dhdX[4]*dhdY[2]
    dBedu[2, 8, 6] = dhdX[3]*dhdY[4] + dhdX[4]*dhdY[3]
    dBedu[2, 8, 8] = 2*dhdX[4]*dhdY[4]
    dBedu[2, 8, 10] = dhdX[4]*dhdY[5] + dhdX[5]*dhdY[4]
    dBedu[2, 8, 12] = dhdX[4]*dhdY[6] + dhdX[6]*dhdY[4]
    dBedu[2, 8, 14] = dhdX[4]*dhdY[7] + dhdX[7]*dhdY[4]
    dBedu[2, 8, 16] = dhdX[4]*dhdY[8] + dhdX[8]*dhdY[4]
    dBedu[2, 9, 1] = dhdX[0]*dhdY[4] + dhdX[4]*dhdY[0]
    dBedu[2, 9, 3] = dhdX[1]*dhdY[4] + dhdX[4]*dhdY[1]
    dBedu[2, 9, 5] = dhdX[2]*dhdY[4] + dhdX[4]*dhdY[2]
    dBedu[2, 9, 7] = dhdX[3]*dhdY[4] + dhdX[4]*dhdY[3]
    dBedu[2, 9, 9] = 2*dhdX[4]*dhdY[4]
    dBedu[2, 9, 11] = dhdX[4]*dhdY[5] + dhdX[5]*dhdY[4]
    dBedu[2, 9, 13] = dhdX[4]*dhdY[6] + dhdX[6]*dhdY[4]
    dBedu[2, 9, 15] = dhdX[4]*dhdY[7] + dhdX[7]*dhdY[4]
    dBedu[2, 9, 17] = dhdX[4]*dhdY[8] + dhdX[8]*dhdY[4]
    dBedu[2, 10, 0] = dhdX[0]*dhdY[5] + dhdX[5]*dhdY[0]
    dBedu[2, 10, 2] = dhdX[1]*dhdY[5] + dhdX[5]*dhdY[1]
    dBedu[2, 10, 4] = dhdX[2]*dhdY[5] + dhdX[5]*dhdY[2]
    dBedu[2, 10, 6] = dhdX[3]*dhdY[5] + dhdX[5]*dhdY[3]
    dBedu[2, 10, 8] = dhdX[4]*dhdY[5] + dhdX[5]*dhdY[4]
    dBedu[2, 10, 10] = 2*dhdX[5]*dhdY[5]
    dBedu[2, 10, 12] = dhdX[5]*dhdY[6] + dhdX[6]*dhdY[5]
    dBedu[2, 10, 14] = dhdX[5]*dhdY[7] + dhdX[7]*dhdY[5]
    dBedu[2, 10, 16] = dhdX[5]*dhdY[8] + dhdX[8]*dhdY[5]
    dBedu[2, 11, 1] = dhdX[0]*dhdY[5] + dhdX[5]*dhdY[0]
    dBedu[2, 11, 3] = dhdX[1]*dhdY[5] + dhdX[5]*dhdY[1]
    dBedu[2, 11, 5] = dhdX[2]*dhdY[5] + dhdX[5]*dhdY[2]
    dBedu[2, 11, 7] = dhdX[3]*dhdY[5] + dhdX[5]*dhdY[3]
    dBedu[2, 11, 9] = dhdX[4]*dhdY[5] + dhdX[5]*dhdY[4]
    dBedu[2, 11, 11] = 2*dhdX[5]*dhdY[5]
    dBedu[2, 11, 13] = dhdX[5]*dhdY[6] + dhdX[6]*dhdY[5]
    dBedu[2, 11, 15] = dhdX[5]*dhdY[7] + dhdX[7]*dhdY[5]
    dBedu[2, 11, 17] = dhdX[5]*dhdY[8] + dhdX[8]*dhdY[5]
    dBedu[2, 12, 0] = dhdX[0]*dhdY[6] + dhdX[6]*dhdY[0]
    dBedu[2, 12, 2] = dhdX[1]*dhdY[6] + dhdX[6]*dhdY[1]
    dBedu[2, 12, 4] = dhdX[2]*dhdY[6] + dhdX[6]*dhdY[2]
    dBedu[2, 12, 6] = dhdX[3]*dhdY[6] + dhdX[6]*dhdY[3]
    dBedu[2, 12, 8] = dhdX[4]*dhdY[6] + dhdX[6]*dhdY[4]
    dBedu[2, 12, 10] = dhdX[5]*dhdY[6] + dhdX[6]*dhdY[5]
    dBedu[2, 12, 12] = 2*dhdX[6]*dhdY[6]
    dBedu[2, 12, 14] = dhdX[6]*dhdY[7] + dhdX[7]*dhdY[6]
    dBedu[2, 12, 16] = dhdX[6]*dhdY[8] + dhdX[8]*dhdY[6]
    dBedu[2, 13, 1] = dhdX[0]*dhdY[6] + dhdX[6]*dhdY[0]
    dBedu[2, 13, 3] = dhdX[1]*dhdY[6] + dhdX[6]*dhdY[1]
    dBedu[2, 13, 5] = dhdX[2]*dhdY[6] + dhdX[6]*dhdY[2]
    dBedu[2, 13, 7] = dhdX[3]*dhdY[6] + dhdX[6]*dhdY[3]
    dBedu[2, 13, 9] = dhdX[4]*dhdY[6] + dhdX[6]*dhdY[4]
    dBedu[2, 13, 11] = dhdX[5]*dhdY[6] + dhdX[6]*dhdY[5]
    dBedu[2, 13, 13] = 2*dhdX[6]*dhdY[6]
    dBedu[2, 13, 15] = dhdX[6]*dhdY[7] + dhdX[7]*dhdY[6]
    dBedu[2, 13, 17] = dhdX[6]*dhdY[8] + dhdX[8]*dhdY[6]
    dBedu[2, 14, 0] = dhdX[0]*dhdY[7] + dhdX[7]*dhdY[0]
    dBedu[2, 14, 2] = dhdX[1]*dhdY[7] + dhdX[7]*dhdY[1]
    dBedu[2, 14, 4] = dhdX[2]*dhdY[7] + dhdX[7]*dhdY[2]
    dBedu[2, 14, 6] = dhdX[3]*dhdY[7] + dhdX[7]*dhdY[3]
    dBedu[2, 14, 8] = dhdX[4]*dhdY[7] + dhdX[7]*dhdY[4]
    dBedu[2, 14, 10] = dhdX[5]*dhdY[7] + dhdX[7]*dhdY[5]
    dBedu[2, 14, 12] = dhdX[6]*dhdY[7] + dhdX[7]*dhdY[6]
    dBedu[2, 14, 14] = 2*dhdX[7]*dhdY[7]
    dBedu[2, 14, 16] = dhdX[7]*dhdY[8] + dhdX[8]*dhdY[7]
    dBedu[2, 15, 1] = dhdX[0]*dhdY[7] + dhdX[7]*dhdY[0]
    dBedu[2, 15, 3] = dhdX[1]*dhdY[7] + dhdX[7]*dhdY[1]
    dBedu[2, 15, 5] = dhdX[2]*dhdY[7] + dhdX[7]*dhdY[2]
    dBedu[2, 15, 7] = dhdX[3]*dhdY[7] + dhdX[7]*dhdY[3]
    dBedu[2, 15, 9] = dhdX[4]*dhdY[7] + dhdX[7]*dhdY[4]
    dBedu[2, 15, 11] = dhdX[5]*dhdY[7] + dhdX[7]*dhdY[5]
    dBedu[2, 15, 13] = dhdX[6]*dhdY[7] + dhdX[7]*dhdY[6]
    dBedu[2, 15, 15] = 2*dhdX[7]*dhdY[7]
    dBedu[2, 15, 17] = dhdX[7]*dhdY[8] + dhdX[8]*dhdY[7]
    dBedu[2, 16, 0] = dhdX[0]*dhdY[8] + dhdX[8]*dhdY[0]
    dBedu[2, 16, 2] = dhdX[1]*dhdY[8] + dhdX[8]*dhdY[1]
    dBedu[2, 16, 4] = dhdX[2]*dhdY[8] + dhdX[8]*dhdY[2]
    dBedu[2, 16, 6] = dhdX[3]*dhdY[8] + dhdX[8]*dhdY[3]
    dBedu[2, 16, 8] = dhdX[4]*dhdY[8] + dhdX[8]*dhdY[4]
    dBedu[2, 16, 10] = dhdX[5]*dhdY[8] + dhdX[8]*dhdY[5]
    dBedu[2, 16, 12] = dhdX[6]*dhdY[8] + dhdX[8]*dhdY[6]
    dBedu[2, 16, 14] = dhdX[7]*dhdY[8] + dhdX[8]*dhdY[7]
    dBedu[2, 16, 16] = 2*dhdX[8]*dhdY[8]
    dBedu[2, 17, 1] = dhdX[0]*dhdY[8] + dhdX[8]*dhdY[0]
    dBedu[2, 17, 3] = dhdX[1]*dhdY[8] + dhdX[8]*dhdY[1]
    dBedu[2, 17, 5] = dhdX[2]*dhdY[8] + dhdX[8]*dhdY[2]
    dBedu[2, 17, 7] = dhdX[3]*dhdY[8] + dhdX[8]*dhdY[3]
    dBedu[2, 17, 9] = dhdX[4]*dhdY[8] + dhdX[8]*dhdY[4]
    dBedu[2, 17, 11] = dhdX[5]*dhdY[8] + dhdX[8]*dhdY[5]
    dBedu[2, 17, 13] = dhdX[6]*dhdY[8] + dhdX[8]*dhdY[6]
    dBedu[2, 17, 15] = dhdX[7]*dhdY[8] + dhdX[8]*dhdY[7]
    dBedu[2, 17, 17] = 2*dhdX[8]*dhdY[8]
    

    det_Je = np.linalg.det(Je)
    dM = rho*(Ne.T@Ne) * th*det_Je
    
    S = C@E
    df_int = (Be.T@S) * th*det_Je
    dK_geo = tdot13(S, dBedu) * th*det_Je
    dK_mat = (Be.T@C@Be) * th*det_Je

    return dM, df_int, dK_geo, dK_mat
