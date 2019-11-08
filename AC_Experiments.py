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
D = 2  # Number of dimensions
Np = 15  # Constant Np points per polynomial
num_points = (Np * np.ones(Npoly)).astype(int)  # Number of points per polynomial
# Npoints = Ns*(np.arange(20)+1)  # Number of points per dimension
# k = 5
# Np = Npoints[k]
cmax = 10  # Absolute maximum value of the polynomials coefficients
degmax = 3  # Maximum value of the polynomials degree
eps = 0  # Noise bound
sq_size = 10  # Horizontal sampling window
RwHopt = RwHoptCond(2, 0.97, 0)  # Conditions for reweighted heuristics
delta = 0.1
method_name = ['Full', 'Cheng', 'CDC New']
method = 2  # 0 - Full, 1 - Cheng, 2 - CDC New


# Data Creation and Ground Truth Plot
(C, A) = createPolynomials(D, Npoly, degmax, cmax)
(Xp, ss_ind) = generatePointsPoly(C, A, num_points, eps, sq_size)
Xline = generatePointsLines(C, A, sq_size)

plt.figure(1)
plotLinesAndPoints(Xline, Xp, Npoly, num_points)
print('METHOD:', method_name[method])
print(D, 'dimensions, ', Npoly, 'polynomials and ', Np, 'points (',num_points[0][0].astype(int), 'per subspace )')

# Problem Solving
if method == 0:
    R, S, runtime, rank1ness = SSC_CVXPY_Full(Xp, eps, Npoly, RwHopt, delta)
if method == 1:
    R, S, runtime, rank1ness = SSC_CVXPY_Cheng(Xp, eps, Npoly, RwHopt, delta)
if method == 2:
    R, S, runtime, rank1ness = SSC_CVXPY_cdc_new(Xp, eps, Npoly, RwHopt, delta)


# Print and Plot Results
ss_ind_pred = np.transpose(S.argmax(axis=0))
normals_pred = np.transpose(np.vstack((R[0,:],R[1,:],R[2,:])))

print('RESULTS\nS (first 5 columns):')
print(np.around(S[:,0:5], decimals=2))
print('Rank1ness of each iteration:', np.around(np.transpose(rank1ness), decimals=3))
print('Elapsed time [s]:', np.around(runtime, decimals=2))

plt.figure(1)
plotNormalsAndPoints(normals,Xp,ss_ind,'Ground Truth',0)
plt.figure(2)
plotNormalsAndPoints(normals_pred,Xp,ss_ind_pred,'Predicted using ' + method_name[method], 1)