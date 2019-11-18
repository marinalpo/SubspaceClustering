import numpy as np
import matplotlib.pyplot as plt
from tempfile import TemporaryFile
from utils import *
from AC_CVXPY import *
from SSC_CVXPY_Full import *
from SSC_CVXPY_Cheng import *
from SSC_CVXPY_cdc_new import *
from scipy.special import comb
from math import pi

# Algebraic Clustering with fixed D=2 and degmax=2

np.random.seed(2)

# Parameters Configuration
Npoly = 2 # Number of polynomials
D = 2  # Number of dimensions ( D=2 -> p(x,y) )
degmax = 2  # Maximum value of the polynomials degree
cmax = 10  # Absolute maximum value of the polynomials coefficients

Np = 10  # EVEN. Constant Np points per polynomial
num_points = (Np * np.ones(Npoly)).astype(int)  # Number of points per polynomial
eps = 6 # Noise bound
sq_size = 10  # Horizontal sampling window

RwHopt = RwHoptCond(10, 0.97, 0)  # Conditions for reweighted heuristics
delta = 0.1
method_name = ['Full', 'Cheng', 'CDC New']
method = 2  # 0 - Full, 1 - Cheng, 2 - CDC New

# Data creation
alpha = exponent_nk(degmax, D)
D_lift = comb(D + degmax, degmax, exact=False)  # Length of the Veronesse map
coef = np.random.randint(-cmax, cmax, (2*Npoly, D_lift.astype(int)))
# coef = np.array([[ -3, 3, -4, 8, -5, 8]])

xPointsRange, typeRange, coef = findXPointsRange(coef, eps, Npoly)
xp = generateXpoints(xPointsRange, typeRange, Np, sq_size)
xp, yp, labels = generateYp(xp, coef, Np, eps, False, 0)
plt.figure(1)
plotGT(coef, labels, xp, yp, sq_size, eps, 'Ground Truth')



xp_flat = np.array(xp).flatten()
yp_flat = np.array(yp).flatten()
labels = np.array(labels).flatten()
points = np.vstack((xp_flat,yp_flat))

# print('Coefficients:\n', np.around(coef, decimals = 2))
# print('xrange:', xPointsRange)
# print('Points:\n', np.around(points, decimals = 2))
# print('Labels:', np.around(labels, decimals = 1))

# Generate Veronesse
points_torch = torch.from_numpy(points).long()
v, p = generate_veronese(points_torch, degmax)
v = v.numpy()
print('veronesse:\n', v)
print(v.shape)


# Problem Solving
plt.figure(2)
plotInput(xp, yp, sq_size)
print('METHOD:', method_name[method])
print(Npoly, 'polynomials and ', np.sum(num_points), 'points (', Np, 'per subspace )')
if method == 0:
    R, S, runtime, rank1ness = SSC_CVXPY_Full(v, eps, Npoly, RwHopt, delta)
if method == 1:
    R, S, runtime, rank1ness = SSC_CVXPY_Cheng(v, eps, Npoly, RwHopt, delta)
if method == 2:
    R, S, runtime, rank1ness = SSC_CVXPY_cdc_new(v, eps, Npoly, RwHopt, delta)


# Print and Plot Results
labels_pred = np.transpose(S.argmax(axis = 0))
print('labels_pred:', labels_pred)
print('R:',np.around(R, decimals = 1))

plt.figure(3)
plotPredictions(R, labels_pred, xp, yp, sq_size, eps, 'Predictions')
plt.show()
