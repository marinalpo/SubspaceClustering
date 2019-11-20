# Many trials Algebraic Clustering with fixed D = 2 and degmax = 2

import numpy as np
import matplotlib.pyplot as plt
from utils import *
from methods import *
from scipy.special import comb
from sklearn import metrics
import sys

# Hide Warnings
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

# Parameters Configuration
Npoly = 2  # Number of polynomials
D = 2  # Number of dimensions ( D=2 -> p(x,y) )
degmax = 2  # Maximum value of the polynomials degree
cmax = 10  # Absolute maximum value of the polynomials coefficients

Np = 10  # EVEN. Constant Np points per polynomial
num_points = (Np * np.ones(Npoly)).astype(int)  # Number of points per polynomial
eps = 10 # Noise bound
sq_size = 10  # Horizontal sampling window

RwHopt = RwHoptCond(5, 0.97, 0)  # Conditions for reweighted heuristics
delta = 0.1
method_name = ['Full', 'Cheng', 'CDC New']
method = 2  # 0 - Full, 1 - Cheng, 2 - CDC New

trials = 10
alpha = exponent_nk(degmax, D)
D_lift = comb(D + degmax, degmax, exact = False)  # Length of the Veronesse map

rank1ness = np.zeros(trials)
runtime = np.zeros(trials)
score = np.zeros(trials)

print('METHOD:', method_name[method])
print(Npoly, 'polynomials and ', np.sum(num_points), 'points (', Np, 'per subspace )')

for t in range(trials):
    print('Trial: ', t)

    # Data creation
    coef = np.random.randint(-cmax, cmax, (4 * Npoly + 5, D_lift.astype(int)))
    xPointsRange, typeRange, coef = findXPointsRange(coef, eps, Npoly)
    xp = generateXpoints(xPointsRange, typeRange, Np, sq_size)
    xp, yp, labels = generateYp(xp, coef, Np, eps, False, 0)

    # Flatten and Stack Data
    xp = np.array(xp).flatten()
    yp = np.array(yp).flatten()
    labels_true = np.array(labels).flatten()
    points = np.vstack((xp, yp))

    # Generate Veronesse Map
    points_torch = torch.from_numpy(points).float()
    v, p = generate_veronese(points_torch, degmax)
    v = v.numpy()

    # Problem Solving
    if method == 0:
        R, S, runtime[t], rank1ness[t] = SSC_CVXPY_Full(v, eps, Npoly, RwHopt)
    if method == 1:
        R, S, runtime[t], rank1ness[t] = SSC_CVXPY_Cheng(v, eps, Npoly, RwHopt, delta)
    if method == 2:
        R, S, runtime[t], rank1ness[t] = SSC_CVXPY_cdc_new(v, eps, Npoly, RwHopt, delta)

    # Metrics Computation
    labels_pred = np.transpose(S.argmax(axis = 0))
    score[t] = metrics.adjusted_rand_score(labels_true, labels_pred)

print('Mean Rank1ness:', np.around(np.mean(rank1ness), decimals = 2))
print('Mean Runtime:', np.around(np.mean(runtime), decimals = 2))
print('Mean Score:', np.around(np.mean(score), decimals = 2))