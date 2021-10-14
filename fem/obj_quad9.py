import numpy as np
import scipy.linalg as sl
import scipy.sparse as ss
import scipy.sparse.linalg as ssl
import matplotlib.pyplot as plt
from copy import deepcopy
from numpy.polynomial.legendre import leggauss
from numba import jit, prange

from .numpy_quad9 import quad9
from .numpy_lin3 import lin3
from .boolean_matrix import boolean_matrix
#%%

@jit(nopython=True, parallel=True)
def intf_jitv(u, p, w, coord, elm, dof, Cm, rho, th):
    '''
    Performs the bulk of calculation for the function Quad9Obj.intf. The
    function is written in procedural programming  with the intent of being
    precompiled with numba.jit for increased performance. The Quad9Obj instance
    attributes are passed as parameters.

    Parameters
    ----------
    u : numpy.array((n_dofs, ), dtype=numpy.float64)
        Nodal displacements [m]. Default value is zero.
    p : numpy.array((leg_order, ), dtype=numpy.float64)
        Roots of the Gauss-Legendre polynomials
    w : np.array((leg_order, ), dtype=numpy.float64)
        Weight of the Gauss-Legendre polynomials
    coord : numpy.ndarray((n_nodes, 2), dtype=numpy.float64)
            Nodal coordinates     
    elm : numpy.ndarray((n_elms, 9), dtype=numpy.int32)
          List of nodes in each element. The nodes order is important.
    dof : numpy.ndarray((n_nodes, n_dpn), dtype=numpy.int32)
          List of degrees of freedom per node, unconstrained problem
    Cm : numpy.ndarray((3, 3), dtype=numpy.float64
         Material elastisticity tensor, in Voigt notation [Pa]
    rho : float
          Material density [kg/m**3]
    th : float
         Object thickness [m]

    Returns
    -------
    F : numpy.ndarray((n_dofs, ), dtype=np.float64)
        Global internal elastic force vector, unconstrained
    m : numpy.ndarray((:, ), dtype=np.float64)
        List of coefficients of the mass matrix.
    m : numpy.ndarray((:, ), dtype=np.float64)
        List of coefficients of the stiffness matrix.
    idx : numpy.ndarray((:, ), dtype=np.int32)
        First index of the coefficients in 'm'
    jdx : numpy.ndarray((:, ), dtype=np.int32)
        Second index of the coefficients in 'm'
    '''
    
    # Number of degress of freedom
    n_dofs = u.shape[0]
    # Number of elements
    n_elms = elm.shape[0]
    # Number of degrees of freedom per element
    n_dpe = 18
    
    # Legendre polynomial order
    leg_order = p.shape[0]
    
    # Memory allocation
    F = np.zeros((n_dofs, ), dtype=np.float64)
    m = np.zeros((n_elms, n_dpe**2), dtype=np.float64)
    k = np.zeros((n_elms, n_dpe**2), dtype=np.float64)
    idx = np.zeros((n_elms, n_dpe**2), dtype=np.int32)
    jdx = np.zeros((n_elms, n_dpe**2), dtype=np.int32)
    
    # Element-wise loop (in parallel)
    for i_e in prange(n_elms):
        # Element nodal coordinates
        r_i = coord[elm[i_e, :], :]
        # Element degrees of freedom 
        q_e = dof[elm[i_e, :]].flatten()
        # Element nodal displacement
        u_e = u[q_e]
        # Eleemnt mass matrix, stiffness matrix and elastic force vector
        Me = np.zeros((q_e.shape[0], q_e.shape[0]), dtype=np.float64)
        Ke = np.zeros((q_e.shape[0], q_e.shape[0]), dtype=np.float64)
        Fe = np.zeros((q_e.shape[0], ), dtype=np.float64)

        # Gaussian integration
        for i in range(leg_order):
            for j in range(leg_order):
                # Natural coordinates
                r_xi = np.array([p[i], p[j]], dtype=np.float64)
                # Internal forces integrands
                dMe, dFe, dKe_geo, dKe_mat = quad9(Cm, rho, th, r_i, r_xi, u_e)
                # Element mass and stiffness matrices and stiffness force vector
                Me += w[i]*w[j] * dMe
                Ke += w[i]*w[j] * (dKe_geo + dKe_mat)
                Fe += w[i]*w[j] * dFe
        
        # Coefficients and indices of the global mass and stiffness matrices
        m[i_e, :] = Me.copy().flatten()
        k[i_e, :] = Ke.copy().flatten()
        idx[i_e, :] = np.array([i for i in q_e for j in q_e], dtype=np.int32)
        jdx[i_e, :] = np.array([j for i in q_e for j in q_e], dtype=np.int32)
        
        # Stiffness force vector
        F[q_e] = F[q_e] + Fe
        
    return F, m, k, idx, jdx


#%%

class Quad9Obj(object):
    '''
    Class containing the attributes and methods relative to 2D FEM objects
    discretised with quadrialateral elemtns with nine nodes.
    
    Attributes
    ----------
    coord : numpy.ndarray((n_nodes, 2), dtype=numpy.float64)
            Nodal coordinates     
    elm : numpy.ndarray((n_elms, 9), dtype=numpy.int32)
          List of nodes in each element. The nodes order is important.
    lin : numpy.ndarray((:, 3), dtype=numpy.int32)
          List of nodes in each boundary element. The nodes order is important.
    
    n_nodes : int
              Total number of nodes
    n_elms : int
             Total number of elements
    n_dpn : int
            Number of degrees of freedom per node
    n_dpe : int
            Number of degrees of freedom per eleemnt
    n_dofs : int
             Total number of degrees of freedom of the unconstrained problem
    dof : numpy.ndarray((n_nodes, n_dpn), dtype=numpy.int32)
          List of degrees of freedom per node, unconstrained problem
    q_idx : numpy.ndarray((n_dofs, ), dtype=np.int32)
            Sequential list of degrees of freedom
    actv : numpy.ndarray((n_dofs, n_dpn), dtype=bool)
           List of status of the degrees of freedom: True for active DOFs and
           False for constrained DOFs
    E : float
        Material Young's modulus [Pa]
    rho : float
          Material density [kg/m**3]
    nu : float
         Material Poisson's ratio
    th : float
         Object thickness [m]
    Cm : numpy.ndarray((3, 3), dtype=numpy.float64
         Material elastisticity tensor, in Voigt notation [Pa]

    N : scipy.sparse.csc_matrix((n_dofs, :), dtype=np.float64)
        Constraint matrix, i.e. N is such that
        Mc = N.T @ M @ N
        Kc = N.T @ K @ N
        Fc = N.T @ F
        where M, K and F are the unconstrained mass matrix, stiffness matrix
        and elastic force vectors respectively and Mc, Kc and Fc their
        constrained counterpart
    T : numpy.ndarray((:, :, 2), dtype=numpy.float64)
        External force density. The first index indicates the line element
        number, the second the node number, and the third the force direction    
    '''
    
    def __init__(self):
        '''
        Creates new instances of the class. No variable value initilisation is
        performed, the variables initialisation will be delegated to a
        classmethod or performed direclty in the running script.

        Returns
        -------
        None.
        '''
        
        self.coord = []
        self.elm = []
        self.lin = []
        
        self.n_nodes = []
        self.n_elms = []

        self.n_dpn = []
        self.n_dpe = []
        self.n_dofs = []
        
        self.dof = []
        self.q_idx = []
        self.actv = []
        
        self.E = []
        self.rho = []
        self.nu = []
        self.th = []
        
        self.Cm = []
        
        self.N = []
        self.T = []

    def intf(self, u=None, leg_order=4):
        '''
        Integrates numerically and arranges the mass matrix, stiffness matrix
        and internal elastic force vector. The bulk of calculation is performed
        by calling the function 'intf_jitv'. The instance attributes are passed
        as attributes for the function 'intf_jitv' in such a way it can be
        precompiled for greater performance. The arrays of values returned from
        the 'intf_jitv' are then arranged in sparse format matrices.
        

        Parameters
        ----------
        u : np.array((n_dofs, ), dtype=numpy.float64)
            Nodal displacements [m]. Default value is zero.
        leg_order : int
            Gaussian quadrature order. The default value is 4.

        Returns
        -------
        M : scipy.sparse.csc_matrix((n_dofs, n_dofs), dtype=np.float64)
            Global mass matrix, unconstrained, sparse
        F : numpy.ndarray((n_dofs, ), dtype=np.float64)
            Global internal elastic force vector, unconstrained
        K : scipy.sparse.csc_matrix((n_dofs, n_dofs), dtype=np.float64)
            Global mass matrix, unconstrained, sparse
        '''
        
        if (u is None):
            u = np.zeros((self.n_dofs, ), dtype=np.float64)
        
        # Gauss-Legendre roots and weights
        p, w = leggauss(leg_order)
        
        # Numerical integration
        F, me, ke, idx, jdx = intf_jitv(u, p, w, self.coord, self.elm, self.dof, self.Cm, self.rho, self.th)
        
        # Sparse assembly
        M = ss.csc_matrix((me.flatten(), (idx.flatten(), jdx.flatten())), shape=(self.n_dofs, self.n_dofs))
        K = ss.csc_matrix((ke.flatten(), (idx.flatten(), jdx.flatten())), shape=(self.n_dofs, self.n_dofs))
        
        return M, F, K
    
    def extf(self, leg_order=4):
        '''
        Integrates numerically and arranges the external boundary force vector.

        Parameters
        ----------
        leg_order : int
            Gaussian quadrature order. The default value is 4.

        Returns
        -------
        G : numpy.ndarray((n_dofs, ), dtype=np.float64)
            External boundary force vector
        '''

        # Gauss-Legendre roots and weights
        p, w = leggauss(leg_order)

        # Global external boundary force vector
        G = np.zeros((self.n_dofs, ), dtype=float)
        
        # Numerical integration (Gaussian quadrature)
        for i_e in range(self.lin.shape[0]):
            # Line element nodal coordinates
            q_e = self.dof[self.lin[i_e, :], :].flatten()
            # Line element degrees of freedom 
            r_i = self.coord[self.lin[i_e, :], :]
            # Line element external force density
            t_i = self.T[i_e, :, :]
            # Line element external boundary force vector
            Ge = np.zeros((q_e.shape[0], ), dtype=np.float64)
            # Gaussian integration
            for j in range(leg_order):
                dge = lin3(self.th, r_i, p[j], t_i)
                Ge += w[j]*dge
            
            # Global external boundary force vector assembly
            G[q_e] = G[q_e] + Ge
        
        return G
    
    def fixed_constraint(self, actv=None):
        '''
        Applies fixed boundary conditions constraints. Other boundary
        conditions haven't been implemented yet. The function returns no
        output, rather it initialises the instance attribute N.

        Parameters
        ----------
        actv : numpy.ndarray((n_dofs, n_dpn), dtype=bool)
               List of status of the degrees of freedom: True for active DOFs
               and False for constrained DOFs

        Returns
        -------
        None.
        '''
        
        if (actv is not None):        
            self.actv = actv

        q_bool = self.actv.flatten()
        
        self.q_c = self.q_idx[q_bool]
        self.N = boolean_matrix(self.q_idx, self.q_c)
    
    def uplt(self, u, u_0=False):
        '''
        Generates a plot of the deformed geometry.

        Parameters
        ----------
        u : np.array((n_dofs, ), dtype=numpy.float64)
            Nodal displacements [m]. Default value is zero.
        u_0 : bool, optional
            If True, the undeformed geometry is plotted beneath the deformed
            one. The default value is False.

        Returns
        -------
        None.
        '''
        
        # Element boundaries
        lin = np.zeros((4*self.n_elms, 3), dtype=np.int32)
        for i_e in range(self.n_elms):
            lin[4*i_e+0, :] = self.elm[i_e, [0, 4, 1]]
            lin[4*i_e+1, :] = self.elm[i_e, [1, 5, 2]]
            lin[4*i_e+2, :] = self.elm[i_e, [2, 6, 3]]
            lin[4*i_e+3, :] = self.elm[i_e, [3, 7, 0]]
        
        # Deformed and undeformed coordinates
        x = np.zeros(self.coord.shape, dtype=np.float64)
        x_0 = np.zeros(self.coord.shape, dtype=np.float64)
        for i_n in range(self.n_nodes):
            xdof = self.dof[i_n, 0]
            ydof = self.dof[i_n, 1]
            
            x[i_n, 0] = self.coord[i_n, 0] + u[xdof]
            x[i_n, 1] = self.coord[i_n, 1] + u[ydof]
            x_0[i_n, 0] = self.coord[i_n, 0]
            x_0[i_n, 1] = self.coord[i_n, 1]
        
        # Plotting
        fig = plt.figure()
        if u_0:
            for i_l in range(lin.shape[0]):
                df_0 = x_0[lin[i_l, :], :]
                plt.plot(df_0[:, 0], df_0[:, 1], 'r-')
        for i_l in range(lin.shape[0]):
            df = x[lin[i_l, :], :]
            plt.plot(df[:, 0], df[:, 1], 'b-')
        plt.axis('equal')
        plt.tight_layout()    

    def copy(self):
        '''
        Generates a deep-copy of the instance

        Returns
        -------
        Quad9Obj object
        '''
        return deepcopy(self)
    