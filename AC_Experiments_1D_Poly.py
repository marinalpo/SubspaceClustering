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
Npoly = 2  # Number of polynomials
D = 2  # Number of dimensions ( D=2 -> p(x,y) )
Np = 7  # Constant Np points per polynomial
num_points = (Np * np.ones(Npoly)).astype(int)  # Number of points per polynomial
cmax = 5  # Absolute maximum value of the polynomials coefficients
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
C = createPolynomials(D, Npoly, alpha, cmax)
(Xp, ss_ind) = generatePointsPoly(C, alpha, num_points, eps, sq_size)
V = dataLift1D(Xp, n)

# Plot Ground Truth
Xp_line = generatePointsLines(C, alpha, sq_size)
plt.figure(1)
plotLinesAndPoints(Xp_line, Xp, Npoly, num_points, 'Ground Truth')

# Problem Solving
print('METHOD:', method_name[method])
print(Npoly, 'polynomials and ', np.sum(num_points), 'points (', Np, 'per subspace )')
if method == 0:
    R, S, runtime, rank1ness = SSC_CVXPY_Full(V, eps, Npoly, RwHopt, delta)
if method == 1:
    R, S, runtime, rank1ness = SSC_CVXPY_Cheng(V, eps, Npoly, RwHopt, delta)
if method == 2:
    R, S, runtime, rank1ness = SSC_CVXPY_cdc_new(V, eps, Npoly, RwHopt, delta)


# Print and Plot Results
ss_ind_pred = np.transpose(S.argmax(axis = 0))
print('ss_ind_pred:', ss_ind_pred)
print('R:',np.around(R, decimals = 1))
Cpred = dataUnlift1D(R, 0, degmax)
print('Cpred:', np.around(Cpred, decimals = 1))
Xp_line_pred = generatePointsLines(Cpred, alpha, sq_size)

Cpred_pluseps = dataUnlift1D(R, eps, degmax)
Xp_line_pluseps = generatePointsLines(Cpred_pluseps, alpha, sq_size)
Cpred_minuseps = dataUnlift1D(R, -eps, degmax)
Xp_line_minuseps = generatePointsLines(Cpred_minuseps, alpha, sq_size)

plt.figure(2)
plotLinesAndPointsPredicted(Xp_line_pred, Xp_line_pluseps, Xp_line_minuseps, Xp, Npoly, num_points, 'Predicted')
plt.show()






print('RESULTS\nS (first 5 columns):')
print(np.around(S[:,0:5], decimals=2))
print('Rank1ness of each:', np.around(np.transpose(rank1ness), decimals=3))
print('Elapsed time [s]:', np.around(runtime, decimals=2))

