import numpy as np
import scipy.linalg as sl
import scipy.sparse.linalg as ssl

#%%

def newmark(ode, t, y0, dy0, gamma=0.5, beta=0.25, epsilon=1e-5):
    '''
    Newmark direct integration method, based on the algorithm described on
    Rixen's Mechanical Vibrations. 

    Parameters
    ----------
    ode : function
          System of differential equations of the form
          (M, C, K, p) = ode(t, q, q_dot)
    t : np.array[:], dtype=np.float64
        Time steps.
    y0 : np.array[:], dtype=np.float64
         Intial displacement condition
    dy0 : np.array[:], dtype=np.float64
          Initial velocity conditon
    gamma : flaot, optional
            Newmark gamma parameter. The default is 0.5.
    beta : float, optional
           Newmark beta parameter. The default is 0.25.
    epsilon : float, optional
              Newthon-Raphson iterations relative tolerance. The default is
              1e-5.

    Returns
    -------
    q : np.array[:, :], dtype=np.float64
        Displacement.
    q_dot : np.array[:, :], dtype=np.float64
            Velocity.
    q_ddot : np.array[:, :], dtype=np.float64
             Aceleration.
    '''
    
    q = np.zeros((t.size, y0.size), dtype=float)
    q_dot = np.zeros((t.size, y0.size), dtype=float)
    q_ddot = np.zeros((t.size, y0.size), dtype=float)
    
    # Initial conditions
    q[0, :] = y0
    q_dot[0, :] = dy0
    
    # Linearised ode
    M, C, K, p = ode(t[0], q[0, :], q_dot[0, :])
    q_ddot[0, :] = ssl.spsolve(M, p - C@q_dot[0, :] - K@q[0, :])
    
    # Time integration
    h = np.diff(t)[0]    
    for i_t in range(len(t)-1):
        # Prediction
        q_ddot[i_t+1, :] = np.zeros(q_ddot[i_t, :].shape, dtype=float)
        q_dot[i_t+1, :] = q_dot[i_t, :] + (1-gamma)*h*q_ddot[i_t, :]
        q[i_t+1, :] = q[i_t, :] + h*q_dot[i_t, :] + (0.5-beta)*h**2*q_ddot[i_t, :]
    
        # Newton-Raphson iterations        
        cond = True
        while cond:
            # Ode
            M, C, K, p = ode(t[i_t+1], q[i_t+1, :], q_dot[i_t+1, :])
            r = M@q_ddot[i_t+1, :] + C@q_dot[i_t+1, :] + K@q[i_t+1, :] - p
        
            # Convergence check
            val1 = epsilon * sl.norm(C@q_dot[i_t+1, :]+K@q[i_t+1, :]-p)
            val2 = sl.norm(r)
            if val2 < val1:
                cond = False
            else:
                # Correction
                S = K + gamma/(beta*h)*C + 1./(beta*h**2)*M
                delta_q = - ssl.spsolve(S, r)
                q[i_t+1, :] = q[i_t+1, :] + delta_q
                q_dot[i_t+1, :] = q_dot[i_t+1, :] + gamma/(beta*h)*delta_q
                q_ddot[i_t+1, :] = q_ddot[i_t+1, :] + 1/(beta*h**2)*delta_q
    
    return q, q_dot, q_ddot