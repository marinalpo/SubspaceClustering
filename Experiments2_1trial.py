import numpy as np
import matplotlib.pyplot as plt
from tempfile import TemporaryFile
from utils import *
from SSC_CVXPY_Full import *
from SSC_CVXPY_Cheng import *
from SSC_CVXPY_cdc_new import *

np.random.seed(7)

# Parameters Configuration
Ns = 3  # Number of subspaces
D = 2  # Number of dimensions
Npoints = Ns*(np.arange(20)+1)  # Number of points per dimension
k = 8
Np = Npoints[k]
RwHopt = RwHoptCond(2, 0.97, 0)  # Conditions for reweighted heuristics
delta = 0.1
eps = 0.2  # Noise bound
method_name = ['Full', 'Cheng', 'CDC New']
method = 2  # 0 - Full, 1 - Cheng, 2 - CDC New

# Data Creation and Ground Truth Plot
normals = createNormalVectors(D, Ns)
num_points = np.hstack([np.ones((1,Ns-1))*np.round(Np/Ns), np.ones((1,1))*(Np-(Ns-1)*np.round(Np/Ns))])
Xp, ss_ind = generatePoints(normals, num_points[0].astype(int), eps, 5)
plt.figure(1)
plotNormalsAndPoints(normals,Xp,ss_ind,'Ground Truth',0)
print('METHOD:', method_name[method])
print(D, 'dimensions, ', Ns, 'subspaces and ', Np, 'points (',num_points[0][0].astype(int), 'per subspace )')

# Problem Solving
if method == 0:
    R, S, runtime, rank1ness = SSC_CVXPY_Full(Xp, eps, Ns, RwHopt, delta)
if method == 1:
    R, S, runtime, rank1ness = SSC_CVXPY_Cheng(Xp, eps, Ns, RwHopt, delta)
if method == 2:
    R, S, runtime, rank1ness = SSC_CVXPY_cdc_new(Xp, eps, Ns, RwHopt, delta)


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