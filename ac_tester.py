# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 16:04:18 2020

@author: Jared
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 12:40:55 2020

@author: Jared
"""

#In quarantine
import numpy as np
import sympy as sp
import cvxpy as cvx
from utils_reduced import varietySample, RwHoptCond, extract_monom
import matplotlib.pyplot as plt
from methods_ac import *

np.random.seed(42)
RwHopt = RwHoptCond(1, 0.97, 1) 

Nx = 2
Nth = 2


x = sp.symarray("x", Nx)
th = sp.symarray('th',Nth)



#Pt = []


eps_true = 0     #Noise level
eps_test = 0.05
#circle
R2 = 1;
V = x[0]**2 + x[1]**2 - R2;
count_max = 20

X1 = varietySample(V, x, count_max, R2, 0)
X2 = varietySample(V, x, count_max, R2, 0)

X2 = X2 + np.array([[2], [3]])
X = np.append(X1, X2, axis = 1)

plt.figure(3)  # Plot predicted data
#plt.subplot(2,1,1)
plt.scatter(X[0, :], X[1, :])


p = [(x[0]-th[0])**2 + (x[1]-th[1])**2 - 1]


pt = p + [th[0] - 1, 4-th[0], th[1]-1, 3-th[1]]

lt = extract_monom([x,th], pt)

mult = [2]
delta = 1e-4
sout = AC_CVXPY(X, eps_test, [x,th], p, mult, RwHopt)

M0 = sout["M"][0]
M1 = sout["M"][1]

#p = (x[0]-th[0])**2 + (x[1]-th[1])**2  + (th[3]*x[2] - th[2])**2
#p = (x[0]-th[0])**2 + (x[1]-th[1])**2  + (x[2] - th[2])**2 - th[3]**2 + (x[0] - th[0] + x[1] - th[1])**2

#p = x[1]**2 - (x[0]**3 + th[0]*x[0] + th[1])



#p = (x[0] - th[1]*x[1] + x[2]*th[2])**4 + (x[1]*th[1])**2 - th[3]
# P = sp.Poly(p, *th)

#M_out = ACmomentConstraint(P, [x, th])

# fb = M_out["fb"]