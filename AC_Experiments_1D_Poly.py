import numpy as np
import matplotlib.pyplot as plt
from tempfile import TemporaryFile
from utils import *
from AC_CVXPY import *
from SSC_CVXPY_Full import *
from SSC_CVXPY_Cheng import *
from SSC_CVXPY_cdc_new import *

np.random.seed(1)

# Parameters Configuration
Npoly = 2 # Number of polynomials
Np = 7  # Constant Np points per polynomial
num_points = (Np * np.ones(Npoly)).astype(int)  # Number of points per polynomial
cmax = 3  # Absolute maximum value of the polynomials coefficients
degmax = 2  # Maximum value of the polynomials degree
n = 2  # Veronesse degree
eps = 5  # Noise bound
sq_size = 10  # Horizontal sampling window
RwHopt = RwHoptCond(10, 0.97, 0)  # Conditions for reweighted heuristics
delta = 0.1
method_name = ['Full', 'Cheng', 'CDC New']
method = 2  # 0 - Full, 1 - Cheng, 2 - CDC New


# Data Creation and Ground Truth Plot
alpha = np.arange(degmax + 1)  # Powers of polynomials
C = createPolynomials(2, Npoly, alpha, cmax)
(Xp, ss_ind) = generatePointsPoly(C, alpha, num_points, eps, sq_size)
V = dataLift1D(Xp, n)

# Plot Ground Truth
Xline = generatePointsLines(C, alpha, sq_size)
plt.figure(1)
plotLinesAndPoints(Xline, Xp, Npoly, num_points)

# Problem Solving
print('METHOD:', method_name[method])
print(Npoly, 'polynomials and ', Np, 'points (',num_points[0], 'per subspace )')
if method == 0:
    R, S, runtime, rank1ness = SSC_CVXPY_Full(V, eps, Npoly, RwHopt, delta)
if method == 1:
    R, S, runtime, rank1ness = SSC_CVXPY_Cheng(V, eps, Npoly, RwHopt, delta)
if method == 2:
    R, S, runtime, rank1ness = SSC_CVXPY_cdc_new(V, eps, Npoly, RwHopt, delta)


# Print and Plot Results
ss_ind_pred = np.transpose(S.argmax(axis=0))
print('ss_ind_pred:', ss_ind_pred)
print('R:', R)

Cpred = dataUnlift1D(R)

# plotLinesAndPoints(Xline, Xp, Npoly, num_points):
# Inputs: - Xline: 2x(Npoly*pointsxpoly) matrix with the points of the polynomials
#         - Xp: 2xN matrix with the N points
#         - Npoly: Number of polynomials
#         - num_points: Array containing number of points per polynomial
c = ['r', 'b', 'g', 'm', 'c', 'k', 'y']
s = Xline.shape
pointsxpoly = int(s[1] / Npoly)
for p in range(Npoly):
    ini = pointsxpoly * p
    fin = pointsxpoly * (p + 1)
    xline = Xline[0, ini:fin]
    yline = Xline[1, ini:fin]
    ini = np.sum(num_points[0:p])
    fin = np.sum(num_points[0:(p + 1)])
    x = Xp[0, ini:fin]
    y = Xp[1, ini:fin]
    plt.plot(xline, yline, color=c[p])
    plt.scatter(x[:], y[:], c=c[p], edgecolors='k')
plt.title('Ground Truth')
plt.show()

xline = np.arange(-sq_size,sq_size, 0.01)
coef = np.zeros((Npoly, degmax))
for p in range(Npoly):
    coef = 3
Coef = - np.multiply((1/R[0,3]), [R[0,0], R[0,1], R[0,2]])
Coef2 = - np.multiply((1/R[1,3]), [R[1,0], R[1,1], R[1,2]])
Coef3 = - np.multiply((1/R[1,3]), [R[1,0]-(10*R[1,3]), R[1,1], R[1,2]])
Coef4 = - np.multiply((1/R[1,3]), [R[1,0]+(10*R[1,3]), R[1,1], R[1,2]])


yline = evaluatePoly(xline, Coef, A[0])
yline2 = evaluatePoly(xline, Coef2, A[0])
yline3 = evaluatePoly(xline, Coef3, A[0])
yline4 = evaluatePoly(xline, Coef4, A[0])

plt.figure(2)
plt.plot(xline[:], yline[:], color = 'r')
plt.plot(xline[:], yline2[:], color = 'b')
plt.plot(xline[:], yline3[:], color = 'k')
plt.plot(xline[:], yline4[:], color = 'k')
# plt.scatter(Xp[0, 0:7], Xp[1, 0:7], c='r')
# plt.scatter(Xp[0, 7:-1], Xp[1, 7:-1], c='b')
plt.scatter(Xp[0, :], Xp[1, :])
plt.show()


print('RESULTS\nS (first 5 columns):')
print(np.around(S[:,0:5], decimals=2))
print('Rank1ness of each iteration:', np.around(np.transpose(rank1ness), decimals=3))
print('Elapsed time [s]:', np.around(runtime, decimals=2))

