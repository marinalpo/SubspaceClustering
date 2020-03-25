# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 16:20:05 2020

@author: Jared
"""
#Trying to rebuild the code I had at RSL
#In quarantine
import numpy as np
import sympy as sp
import cvxpy as cvx

import polytope
import aux2 as aux
import datetime


#t0 = datetime.now()
#modify input and init variables


x = sp.symarray("x", 3)

#p = x[0]**4 + x[1]**4 + x[2]**4 

#p = 1 + x[0]**4 * x[1]**2 + x[0]**2 * x[1]**4

p = -4*x[0]**3 * x[1]**4 + 2*x[0]**4 * x[1]**3 + 5*x[0]**6 * x[1]**8 - 2*x[1]**7 * x[0]**7 + 2 * x[0]**8 * x[1]**6
P = sp.Poly(p)

#th = sp.symarray('th',3)
#y = sp.symarray('y',2)
#p = (y[0]-th[0])**2 + (y[1]-th[1])**2 - th[2]
#
#P = sp.Poly(p, *th)

b = np.array(P.coeffs());
A_pre = np.array(P.monoms());


#add in constant term?
if [0,0] not in A_pre:
    A_pre= np.append(A_pre, np.zeros([1, A_pre.shape[1]]), axis = 0)
    print(A_pre)
    b = np.append(b, 0)

A = np.ones((A_pre.shape[1] + 1, A_pre.shape[0]), dtype = int)
A[1:,:] = A_pre.T





# gamma = cvx.Variable()

   
support = np.array(polytope.interior(A, strict = False))
half_support = [tuple(v // 2) for v in support if v.any() and not (v % 2).any()]
if [0,0] not in half_support:
    half_support = [(0,0)] + half_support

#moment variables and matrix
y = cvx.Variable(len(support))
C = cvx.Variable((len(half_support),len(half_support)), PSD = True)
coeffs = {tuple(e): 0 for e in support}

for i in range(A.shape[1]):
    coeffs[tuple(A[1:,i])] += b[i]
#create lookup table: vector -> index
lookup = {half_support[i] : i for i in range(len(half_support))}
constraints = []
coeff_ref = np.zeros([len(support),1])
p = 0
for v,c in coeffs.items():
    if not any(v):
        #constant term gets special treatment
        # constraints.append(C[0,0] == coeffs[v] + gamma)
        constraints += [C[0,0] == y[p], y[p]==1]
        coeff_ref[p] = c
        p += 1
        continue
    #list all (indices of) pairs in half_support, that add up to v
    l = []
    for u in half_support:
        diff = tuple(v[i] - u[i] for i in range(len(v)))
        if diff in half_support:
            l.append((lookup[u],lookup[diff]))
    print(v, l)
    new_cons = [cvx.Zero(C[i,j] - y[p]) for i,j in l]
    constraints += new_cons
    coeff_ref[p] = c
    #constraints.append(cvx.Zero(cvx.sum([C[i,j] for i,j in l]) - cvx.expressions.constants.Constant(c)))
    p += 1
#define the problem

objective = y @ coeff_ref

prob = cvx.Problem(cvx.Minimize(objective),constraints)
prob.solve(verbose=0)
#prob_sos_sparse = cvx.Problem(cvx.Minimize(gamma),constraints)
#prob_sos = prob_sos_sparse
