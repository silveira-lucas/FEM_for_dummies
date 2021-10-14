import sympy as sym
from my_printer import MyPrinter
code_gen = MyPrinter().doprint

#%%

export_code = True

#%% 

n_nodes = 9

xi, eta = sym.symbols('xi, eta')

# Nodal natural coordinates
xi_i = sym.Matrix([-1, 1, 1, -1, 0, 1, 0, -1, 0])
eta_i = sym.Matrix([-1, -1, 1, 1, -1, 0, 1, 0, 0])

# Nodal phisical coordinates
X_i = sym.Matrix(sym.symbols('X_:9'))
Y_i = sym.Matrix(sym.symbols('Y_:9'))

# Nodal phisical coordinates vector in Voigt notation
X_e = sym.zeros((X_i.shape[0]+Y_i.shape[0]), 1)
for i in range(X_i.shape[0]):
    X_e[2*i] = X_i[i]
    X_e[2*i+1] = Y_i[i]

# Material properties and thickness
E, nu, rho, h = sym.symbols('E, nu, rho, h')

# Shape functions
A = sym.zeros(X_i.shape[0], X_i.shape[0])
for i in range(X_i.shape[0]):
    A[i, 0] = 1
    A[i, 1] = xi_i[i]
    A[i, 2] = eta_i[i]
    A[i, 3] = xi_i[i]*eta_i[i]
    A[i, 4] = xi_i[i]**2
    A[i, 5] = eta_i[i]**2
    A[i, 6] = xi_i[i]**2*eta_i[i]
    A[i, 7] = xi_i[i]*eta_i[i]**2
    A[i, 8] = xi_i[i]**2*eta_i[i]**2
Am1 = A.inv()
#
p = sym.Matrix([1, xi, eta, xi*eta, xi**2, eta**2, xi**2*eta, xi*eta**2, xi**2*eta**2])
h_i = sym.zeros(xi_i.shape[0], 1)
for i in range(xi_i.shape[0]):
    ci = Am1 @ sym.eye(xi_i.shape[0])[:, i]
    h_i[i] = (ci.T@p)[0].simplify().factor()

# Shape functions derivatives on natural coordinates
dhdxi = sym.zeros(h_i.shape[0], 1)
dhdeta = sym.zeros(h_i.shape[0], 1)
for i in range(h_i.shape[0]):
    dhdxi[i] = h_i[i].diff(xi).expand().simplify().factor()
    dhdeta[i] = h_i[i].diff(eta).expand().simplify().factor()

# Phisical coordinates as function of nodal coordinates and shape functions
X = (h_i.T@X_i)[0].expand().simplify()
Y = (h_i.T@Y_i)[0].expand().simplify()

# Jacobian matrix
Je = sym.Matrix([[X.diff(xi), X.diff(eta)], 
                 [Y.diff(xi), Y.diff(eta)]])

#%%

'''
In the first part, the shape fuctions, shape functions derivatives and the
element Jacobian matrix are obtained as fuction of the natural coordinates. For
simple elements such as a three node triangle, the deformation gradient and the
Gree-Lagrange strain tensor can be expressed directly as a function of the
natural coordinates with ease; for more complex elements the expressions become
larger and is simpler to express them in terms of mode shapes.
'''
#
h = h_i.copy()
del X, Y, h_i
X, Y = sym.symbols('X, Y')

# Mode shapes
h_i = sym.Matrix(sym.symbols('h_:9'))
for i in range(h_i.shape[0]):
    h_i[i] = sym.Function('h_%i'%i)(X, Y)

# Nodal displacements
u_i = sym.Matrix(sym.symbols('u_:9'))
v_i = sym.Matrix(sym.symbols('v_:9'))

# Displacement functions
u = (h_i.T @ u_i)[0]
v = (h_i.T @ v_i)[0]

# Position vector in natural coordinates
xi_vec = sym.Matrix([xi, eta])
# Position vector in physical coordinates
X_vec = sym.Matrix([X, Y])

# Displacemnt vector as a fuction of nodal displacements and shape functions
u_vec = sym.Matrix([u, v])

# Displacement derivative
dudX = sym.zeros(2, 2)
for i in range(dudX.shape[0]):
    for j in range(dudX.shape[1]):
        dudX[i, j] = u_vec[i].diff(X_vec[j])

# Green-Lagrange strain tensor
E = sym.Rational(1, 2)*(dudX + dudX.T + dudX.T@dudX)
# Green-Lagrange strain tensor in Voigt notation
E_vec = sym.Matrix([E[0, 0], E[1, 1], 2*E[0, 1]])

# Nodal displacement in Voigt notation
u_e = sym.zeros(2*u_i.shape[0], 1)
for i in range(u_i.shape[0]):
    u_e[2*i] = u_i[i]
    u_e[2*i+1] = v_i[i]

# B matrix
Be = sym.zeros(E_vec.shape[0], u_e.shape[0])
for i in range(E_vec.shape[0]):
    for j in range(u_e.shape[0]):
        Be[i, j] = (E_vec[i]).diff(u_e[j])

# B derivative tensor
dBedue = sym.MutableDenseNDimArray.zeros(Be.shape[0], Be.shape[1], u_e.shape[0])
for i in range(dBedue.shape[0]):
    for j in range(dBedue.shape[1]):
        for k in range(dBedue.shape[2]):
            dBedue[i, j, k] = Be[i, j].diff(u_e[k])

#%%

# Dictionairy for better printing
h_dict = {}
for i in range(h_i.shape[0]):
    h_dict.update({h_i[i]: sym.symbols('h[%i]'%(i))})
    h_dict.update({h_i[i].diff(X): sym.symbols('dhdX[%i]'%(i))})
    h_dict.update({h_i[i].diff(Y): sym.symbols('dhdY[%i]'%(i))})


for i in range(E_vec.shape[0]):
    E_vec[i] = E_vec[i].xreplace(h_dict)

for i in range(Be.shape[0]):
    for j in range(Be.shape[1]):
        Be[i, j] = Be[i, j].xreplace(h_dict)

for i in range(dBedue.shape[0]):
    for j in range(dBedue.shape[1]):
        for k in range(dBedue.shape[2]):
            dBedue[i, j, k] = dBedue[i, j, k].xreplace(h_dict)

#%%

ident = '    '

text_1 = '''
    # Ne
    Ne = np.zeros((2, 2*h.shape[0]), dtype=np.float64)
    for i in range(h.shape[0]):
        Ne[0, 2*i] = h[i]
        Ne[1, 2*i+1] = h[i]
'''

text_2 = '''
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
'''

text_3 = '''
    det_Je = np.linalg.det(Je)
    dM = rho*(Ne.T@Ne) * th*det_Je
    
    S = C@E
    df_int = (Be.T@S) * th*det_Je
    dK_geo = tdot13(S, dBedu) * th*det_Je
    dK_mat = (Be.T@C@Be) * th*det_Je
'''


#%%


if (export_code):
    print('Exporting code')
    
    with open('numpy_quad9_v1.py', mode='w+') as f:
        
        print('import numpy as np', file=f)
        print('from numba import jit', file=f)        
        print('from .tdot13 import tdot13', file=f)
        print('\n#%% \n', file =f)
        
        print('@jit(nopython=True)', file=f)
        print('def quad9(C, rho, th, r_i, r_xi, u_e):', file=f)
        print(ident, file=f)
        
        print(ident+'# Unpacking input variables', file=f)
        for i in range(n_nodes):
            print(ident+'u_%i = u_e[%i]'%(i, 2*i), file=f)
            print(ident+'v_%i = u_e[%i]'%(i, 2*i+1), file=f)
        print(ident, file=f)
        print(ident+'xi = r_xi[0]', file=f)
        print(ident+'eta = r_xi[1]', file=f)
        print(ident, file=f)
        
        print(ident+'# h', file=f)
        print(ident+'h = np.zeros((%i, ), dtype=np.float64)' %(h.shape[0]), file=f)
        for i in range(h.shape[0]):
            if (code_gen(h[i])!='0'):
                text = code_gen(h[i]).replace('numpy.', 'np.')
                print(ident+'h[%i] = %s' %(i, text), file=f)
        print(ident, file=f)
        
        print(text_1, file=f)
        
        print(ident+'# dhdxi', file=f)
        print(ident+'dhdxi = np.zeros((%i, ), dtype=np.float64)' %(dhdxi.shape[0]), file=f)
        for i in range(dhdxi.shape[0]):
            if (code_gen(dhdxi[i])!='0'):
                text = code_gen(dhdxi[i]).replace('numpy.', 'np.')
                print(ident+'dhdxi[%i] = %s' %(i, text), file=f)
        print(ident, file=f)
        
        print(ident+'# dhdeta', file=f)
        print(ident+'dhdeta = np.zeros((%i, ), dtype=np.float64)' %(dhdeta.shape[0]), file=f)
        for i in range(dhdeta.shape[0]):
            if (code_gen(dhdeta[i])!='0'):
                text = code_gen(dhdeta[i]).replace('numpy.', 'np.')
                print(ident+'dhdeta[%i] = %s' %(i, text), file=f)
        print(ident, file=f)
        
        print(text_2, file=f)

        print(ident+'# E', file=f)
        print(ident+'E = np.zeros((%i, ), dtype=np.float64)' %(E_vec.shape[0]), file=f)
        for i in range(E_vec.shape[0]):
            if (code_gen(E_vec[i])!='0'):
                text = code_gen(E_vec[i]).replace('numpy.', 'np.')
                print(ident+'E[%i] = %s' %(i, text), file=f)
        print(ident, file=f)

        print(ident+'Be = np.zeros((%i, %i), dtype=np.float64)' %(Be.shape[0], Be.shape[1]), file=f)
        for i in range(Be.shape[0]):
            for j in range(Be.shape[1]):
                if (code_gen(Be[i, j])!='0'):
                    text = code_gen(Be[i, j]).replace('numpy.', 'np.')
                    print(ident+'Be[%i, %i] = %s' %(i, j, text), file=f)
        print(ident, file=f)
        
        print(ident+'dBedu = np.zeros((%i, %i, %i), dtype=np.float64)' %(dBedue.shape[0], dBedue.shape[1], dBedue.shape[2]), file=f)
        for i in range(dBedue.shape[0]):
            for j in range(dBedue.shape[1]):
                for k in range(dBedue.shape[2]):
                    if (code_gen(dBedue[i, j, k])!='0'):
                        text = code_gen(dBedue[i, j, k]).replace('numpy.', 'np.')
                        print(ident+'dBedu[%i, %i, %i] = %s' %(i, j, k, text), file=f)
        print(ident, file=f)
        
        print(text_3, file=f)
                
        print(ident+'return dM, df_int, dK_geo, dK_mat', file=f)
