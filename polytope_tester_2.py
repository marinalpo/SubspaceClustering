# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 14:32:11 2020

@author: jared
"""

import numpy as np
import sympy as sp
import cvxpy as cvx

import polytope as pt
import aux2 as aux


x = sp.symarray("x", 3)

#p = x[0]**4 + x[1]**4 + x[2]**4 

p = 1 + x[0]**4 * x[1]**2 + x[0]**2 * x[1]**4

#p = -4*x[0]**3 * x[1]**4 + 2*x[0]**4 * x[1]**3 + 5*x[0]**6 * x[1]**8 - 2*x[1]**7 * x[0]**7 + 2 * x[0]**8 * x[1]**6
P = sp.Poly(p)

#th = sp.symarray('th',3)
#y = sp.symarray('y',2)
#p = (y[0]-th[0])**2 + (y[1]-th[1])**2 - th[2]
#
#P = sp.Poly(p, *th)

b = np.array(P.coeffs());
A_pre = np.array(P.monoms());
A = np.ones((A_pre.shape[1] + 1, A_pre.shape[0]), dtype = int)
A[1:,:] = A_pre.T

support = np.array(pt.interior(A, strict = False))

half_support = [tuple(v // 2) for v in support if not (v % 2).any()]
#n = A.shape[1]
#d = P.degree() // 2
#size = aux.binomial(n + 2*d, n)
