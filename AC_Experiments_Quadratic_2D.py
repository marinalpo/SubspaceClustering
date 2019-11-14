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
Npoly = 4  # Number of polynomials
D = 2  # Number of dimensions ( D=2 -> p(x,y) )
degmax = 2  # Maximum value of the polynomials degree
cmax = 10  # Absolute maximum value of the polynomials coefficients


Np = 6  # EVEN. Constant Np points per polynomial
num_points = (Np * np.ones(Npoly)).astype(int)  # Number of points per polynomial
eps = 3 # Noise bound
sq_size = 5  # Horizontal sampling window

RwHopt = RwHoptCond(10, 0.97, 0)  # Conditions for reweighted heuristics
delta = 0.1
method_name = ['Full', 'Cheng', 'CDC New']
method = 2  # 0 - Full, 1 - Cheng, 2 - CDC New

# Data creation
alpha = exponent_nk(degmax, D)
D_lift = comb(D + degmax, degmax, exact=False)  # Length of the Veronesse map
coef = np.random.randint(-cmax, cmax, (2*Npoly, D_lift.astype(int)))
# coef = np.array([[-119, -27, -25, 9, 0, 25], [-1, 0, 0, 1, 1, 1]])  # Coefs. from two ellipses

xrange = findXRange(coef, 0)
xp, delPol = generateXpoints(xrange, Np, sq_size, True)
# Delete polynomials with no real roots
coef = np.delete(coef, delPol, 0)
coef = coef[0:Npoly, :]
xp = np.delete(xp, delPol, 0)
xp = xp[0:Npoly, :]
yp, labels = generateYp(xp, coef, sq_size, eps, False, 0)


plotGT(coef, xp, yp, sq_size, eps)
plt.show()




xp = np.array(xp).flatten()
yp = np.array(yp).flatten()
labels = np.array(labels).flatten()
points = np.vstack((xp,yp))

# print('points:\n', np.around(points, decimals = 1))
# print('labels:', np.around(labels, decimals = 1))

# Generate Veronesse
points_torch = torch.from_numpy(points).long()
v, p = generate_veronese(points_torch, degmax)
v = v.numpy()
# print('veronesse:\n', v)


