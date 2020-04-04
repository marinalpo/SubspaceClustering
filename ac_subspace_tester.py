# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 16:04:18 2020

@author: Jared
"""

import numpy as np
import sympy as sp
import cvxpy as cvx
from utils_reduced import RwHoptCond, createNormalVectors, generatePoints
import matplotlib.pyplot as plt
from methods_ac import Model, AC_manager

np.random.seed(42)

# generate data in lines
Ns = 4  # Number of subspaces
D = 2  # Number of dimensions
Npoints = Ns*(np.arange(20)+1)  # Number of points per dimension
k = 5
Np = Npoints[k]
normals = createNormalVectors(D, Ns)
num_points = np.hstack([np.ones((1,Ns-1))*np.round(Np/Ns), np.ones((1,1))*(Np-(Ns-1)*np.round(Np/Ns))])
eps = 0

X, ss_ind = generatePoints(normals, num_points[0].astype(int), eps, 5)

plt.figure(3)  # Plot predicted data
# plt.subplot(2,1,1)
plt.scatter(X[0, :], X[1, :])

#Form the model
RwHopt = RwHoptCond(10, 0.97, 1, 1e-1)

Nx = 2
Nth = 2

x = sp.symarray("x", Nx)
th = sp.symarray('th', Nth)



V_line = x[0]*th[0] + x[1]*th[1]
line = Model(x, th)
line.add_eq(V_line)
line.add_ineq(th[0]**2 + th[1]**2 - 1)
#box_bounds = [th[0] + 0.5, 4 - th[0], th[1] + 0.5, 4 - th[1]]
# box_bounds = [th[0] + 0.25, 4 - th[0], th[1] + 0.5, 4 - th[1]]
# mult = 2
# circle.add_ineq(box_bounds)
# circle.add_eq(V_circle)

ac = AC_manager(x, th)
ac.add_model(line, Ns)

#Run the classifier
eps_test = 0.05
cvx_classify = ac.generate_SDP(X, eps_test)
cvx_result = ac.solve_SDP(cvx_classify, RwHopt)

#view results
rank1ness = cvx_result["rank1ness"]
S = cvx_result["S"]
TH = cvx_result["TH"]
M = cvx_result["M"]
Mth = cvx_result["Mth"]

print("Done!")
#p = [(x[0] - th[0]) ** 2 + (x[1] - th[1]) ** 2 - 1]

#pt = p + [th[0] - 1, 4 - th[0], th[1] - 1, 3 - th[1]]

#mom = ACmomentConstraint(pt, [x,th])

# sout = AC_CVXPY(X, eps_test, [x, th], p, mult, RwHopt)
#
# M0 = sout["M"][0]
# M1 = sout["M"][1]