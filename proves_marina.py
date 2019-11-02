# Experiments2_full.m amb SSC_CVXPY_Full
import numpy as np
import matplotlib.pyplot as plt
from tempfile import TemporaryFile
from utils import *
from SSC_CVXPY_Full import *

np.random.seed(0)

Ns = 3  # Number of subspaces
D = 2  # Number of dimensions
Npoints = Ns*(np.arange(20)+1)  # Number of points per dimension
N_tests = 8
RwHopt = RwHoptCond(10, 0.97, 0)  # Conditions for reweighted heuristics
delta = 0.1
eps = 0  # Noise bound

# Data dumps
Iterations = np.zeros([len(Npoints), N_tests])
Runtime = np.zeros([len(Npoints), N_tests])
Rank1ness = []

k = 6
t = 0
Np = Npoints[k]
normals = createNormalVectors(D, Ns)
num_points = np.hstack([np.ones((1,Ns-1))*np.round(Np/Ns), np.ones((1,1))*(Np-(Ns-1)*np.round(Np/Ns))])
print('\n', D, 'dimensions, ', Ns, 'subspaces and ', Np, 'points (',num_points[0][0].astype(int), 'per subspace )')
Xp, ss_ind = generatePoints(normals, num_points[0].astype(int), eps, 5)


#plotNormalsAndPoints(normals,Xp,ss_ind,'Ground Truth')

# Rank 1 on moment matrix
RwHopt.corner = 0
R, S, runtime, rank1ness = SSC_CVXPY_Full(Xp, eps, Ns, RwHopt, delta)
print('end function')
print('\n S:', np.around(S.value, decimals=2))
print('\n R1:', np.around(R[0].value, decimals=2))
print('\n R2:', np.around(R[1].value, decimals=2))
print('\n R3:', np.around(R[2].value, decimals=2))

a = (R[0][0].value)
b = (R[1][0].value)
c = (R[2][0].value)
normals_pred = np.transpose(np.vstack((a, b, c)))
# print('\n Rnova:', np.around(normals_pred, decimals=2))
# test = cp.sum(S, axis = 0)
# print('test:', test.value)
Iterations[k,t] = len(rank1ness)
Runtime[k,t] = runtime
Rank1ness.append(rank1ness)


ss = S.value
ss_ind_pred = np.transpose(ss.argmax(axis=0))
print(ss_ind_pred)
plotNormalsAndPoints(normals_pred,Xp,ss_ind_pred,'Predicted')