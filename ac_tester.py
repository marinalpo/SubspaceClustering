# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 16:04:18 2020

@author: Jared
"""

import numpy as np
import sympy as sp
import cvxpy as cvx
from utils_reduced import varietySample, RwHoptCond, extract_monom
import matplotlib.pyplot as plt
from methods_ac import Model, AC_manager

np.random.seed(42)
RwHopt = RwHoptCond(30, 0.98, 1, 1e-2)

Nx = 2
Nth = 2

x = sp.symarray("x", Nx)
th = sp.symarray('th', Nth)

# Pt = []


eps_true = 0.005  # Noise level
eps_test = 0.05
# circle
R2 = 1;
V = x[0] ** 2 + x[1] ** 2 - R2;
count_max = 20

X1 = varietySample(V, x, count_max, R2, 0)
X2 = varietySample(V, x, count_max, R2, 0)

X2 = X2 + np.array([[2], [3]])
X = np.append(X1, X2, axis=1)



V_circle = (x[0] - th[0]) ** 2 + (x[1] - th[1]) ** 2 - 1
circle = Model(x, th)
box_bounds = [th[0] + 0.25, 3 - th[0], th[1] + 0.25, 4 - th[1]]
# box_bounds = [th[0] + 0.25, 4 - th[0], th[1] + 0.5, 4 - th[1]]
mult = 2
circle.add_ineq(box_bounds)
circle.add_eq(V_circle)

# circ1 = Model(x, th)
# circ1.add_eq(V_circle)
# circ1.add_ineq([th[0] + 0.25, 1 - th[0], th[1] + 0.25, 1 - th[1]])
#
# circ2 = Model(x, th)
# circ2.add_eq(V_circle)
# circ2.add_ineq([th[0] - 1.5, 4 - th[0], th[1] - 2.5, 4 - th[1]])

#cvx_moment = circle.generate_moment(mult)
#cvx_classify = circle.generate_classify(X, eps_test, cvx_moment["Mth"])
ac = AC_manager(x, th)
ac.add_model(circle, mult)
# ac.add_model(circ1, 1)
# ac.add_model(circ2, 1)

tau = [3, 1]

cvx_classify = ac.generate_SDP(X, eps_test)
#cvx_result = ac.solve_SDP_rstar(cvx_classify)
cvx_result = ac.solve_SDP(cvx_classify, RwHopt, tau)
#cvx_result = ac.run_SDP(X, eps_test, RwHopt)


#view results
rank1ness = cvx_result["rank1ness"]
S = cvx_result["S"]
TH = cvx_result["TH"]
M = cvx_result["M"]
Mth = cvx_result["Mth"]

#p = [(x[0] - th[0]) ** 2 + (x[1] - th[1]) ** 2 - 1]

#pt = p + [th[0] - 1, 4 - th[0], th[1] - 1, 3 - th[1]]

#mom = ACmomentConstraint(pt, [x,th])

# sout = AC_CVXPY(X, eps_test, [x, th], p, mult, RwHopt)
#
# M0 = sout["M"][0]
# M1 = sout["M"][1]

plt.figure(3)  # Plot predicted data
# plt.subplot(2,1,1)
st = S[0, :] > S[1, :]
plt.scatter(X[0, :], X[1, :])
plt.scatter(X[0, st], X[1, st], marker='o', color="blue")
plt.scatter(TH[0][0], TH[0][1], marker='x', color="blue")
# plt.scatter(X[0, ~st], X[1, ~st], marker='o', color="red")
# plt.scatter(TH[1][0], TH[1][1], marker='x', color="red")
plt.show()

print("Done!")
