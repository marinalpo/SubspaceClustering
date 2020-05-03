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
import matplotlib.patches as patches
from methods_ac import Model, AC_manager
import cvxpy as cp

np.random.seed(42)
RwHopt = RwHoptCond(1, 0.98, 1, 1e-3)

Nx = 2
Nth = 2

#center of circle
cx = 2
cy = 3

x = sp.symarray("x", Nx)
th = sp.symarray('th', Nth)

# Pt = []


eps_true = 0.05  # Noise level
eps_test = 0.1
# circle
R2 = 1;
V = x[0] ** 2 + x[1] ** 2 - R2;
count_max = 25

X1 = varietySample(V, x, count_max, R2, 0)
X2 = varietySample(V, x, count_max, R2, 0)

X2 = X2 + np.array([[2], [3]])
X = np.append(X1, X2, axis=1)

ac = AC_manager(x, th)

V_circle = (x[0] - th[0]) ** 2 + (x[1] - th[1]) ** 2 - 1
# circle = Model(x, th)
# # box_bounds = [th[0] + 0.25, 3 - th[0], th[1] + 0.25, 4 - th[1]]
# # box_bounds = [th[0] + 0.25, 4 - th[0], th[1] + 0.5, 4 - th[1]]
# mult = 2
# circle.add_ineq(box_bounds)
# circle.add_eq(V_circle)
# ac.add_model(circle, mult)


"""Generate the model instances"""
# w1 = 0.25
# w2 = 0.25
# w1 = 1e-4
# w2 = 1e-4
w1 = 1
w2 = 1
box1 = [[0-w1, 0+w1], [0-w1, 0+w1]]
box2 = [[cx-w2, cx+w2], [cy-w2, cy+w2]]


circ1 = Model(x, th)
circ1.add_eq(V_circle)


circ1.add_ineq([-th[0] + box1[0][0], -box1[0][1] + th[0], -th[1] +box1[1][0], -box1[1][1] + th[1]])
# circ1.add_eq([th[0], th[1]])
circ2 = Model(x, th)
circ2.add_eq(V_circle)
# circ2.add_ineq([th[0] - 1.5, 4 - th[0], th[1] - 2.5, 4 - th[1]])
# circ2.add_eq([th[0]-2, th[1]-3])
# circ2.add_ineq([th[0] - 1.75, 2.25 - th[0], th[1] - 2.75, 3.25 - th[1]])

circ2.add_ineq([-th[0] + box2[0][0], -box2[0][1] + th[0], -th[1] +box2[1][0], -box2[1][1] + th[1]])
# circ2.add_ineq([th[0] - box2[0][0], box2[0][1] - th[0], th[1] -box2[1][0], box2[1][1] - th[1]])
ac.add_model(circ1, 1)
ac.add_model(circ2, 1)

tau = [1, 1]

cvx_moment = ac.moment_SDP()

prob0 = cp.Problem(cp.Minimize(0), cvx_moment["C"])
sol0 = prob0.solve(solver=cp.MOSEK, verbose=False, save_file='circle_classify0.task.gz')

cvx_classify = ac.classify_SDP(X, eps_test, cvx_moment)


# cvx_classify = ac.generate_SDP(X, eps_test)
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

# fig, ax = plt.figure(3)  # Plot predicted data
fig, ax = plt.subplots(1)
# plt.subplot(2,1,1)
st = S[0, :] > S[1, :]
plt.scatter(X[0, :], X[1, :])

plt.scatter(X[0, st], X[1, st], marker='o', color="blue")
plt.scatter(TH[0][0], TH[0][1], marker='x', color="blue")

# box1walkx = [box1[0][0], box1[0][1], box1[0][1], box1[0][0], box1[0][0]]
# box1walky = [box1[1][0], box1[1][0], box1[1][1], box1[1][1], box1[1][0]]
# plt.plot()
rect1 = patches.Rectangle((-w1, -w1), 2*w1, 2*w1, fill=False, color="blue")
ax.add_patch(rect1)

plt.scatter(X[0, ~st], X[1, ~st], marker='o', color="red")
plt.scatter(TH[1][0], TH[1][1], marker='x', color="red")
rect2 = patches.Rectangle((cx-w2, cy-w2), 2*w2, 2*w2, fill=False, color="red")
ax.add_patch(rect2)

minrank =np.min(rank1ness[cvx_result["iter"], :])

plt.title("Circle classification (min rank1ness={:1.3f})".format(minrank))

plt.show()

print("Done!")
