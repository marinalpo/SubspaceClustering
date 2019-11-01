# Experiments2_full.m -----------------------------------------------------------------
# Exp 1: Fix Ns and D, run for different Np and record the running time and convergence
import numpy as np
import matplotlib.pyplot as plt
from tempfile import TemporaryFile
from utils import *
from SSC_CVXPY_Full import *

np.random.seed(2020)

Ns = 3  # Number of subspaces
D = 3  # Number of dimensions
Npoints = Ns*(np.arange(20)+1)  # Number of points per dimension
Npoints = Ns*(np.arange(2)+1)  # Number of points per dimension
N_tests = 8
N_tests = 1
RwHopt = RwHoptCond(20, 0.97, 0)  # Conditions for reweighted heuristics
delta = 0.1
eps = 0.1  # Noise bound

# Data dumps
Iterations = np.zeros([len(Npoints), N_tests])
Runtime = np.zeros([len(Npoints), N_tests])
Rank1ness = []

for k in range(len(Npoints)):
    for t in range(N_tests):
        Np = Npoints[k]
        print('Trial', t, 'with',Np, 'points')
        normals = createNormalVectors(D, Ns)
        num_points = np.hstack([np.ones((1,Ns-1))*np.round(Np/Ns), np.ones((1,1))*(Np-(Ns-1)*np.round(Np/Ns))])
        Xp, ss_ind = generatePoints(normals, num_points[0].astype(int), eps, 5)

        # Rank 1 on moment matrix
        RwHopt.corner = 0
        R, S, runtime, rank1ness = SSC_CVXPY_Full(Xp, eps, Ns, RwHopt, delta)
        print('\n S:', np.around(S.value, decimals=3))
        # test = cp.sum(S, axis = 0)
        # print('test:', test.value)
        Iterations[k,t] = len(rank1ness)
        Runtime[k,t] = runtime
        Rank1ness.append(rank1ness)

np.savez('/Users/marinaalonsopoal/PycharmProjects/SubspaceClustering/data_dump/experiments_2_dump_full.npz', name1 = Iterations, name2 = Runtime, name3 = Rank1ness, name4 = Ns, name5 = D, name6 = Npoints)

plt.plot(Npoints,np.mean(Runtime,axis=1))
plt.title('Runtime vs. Number of points')
plt.xlabel('Np')
plt.ylabel('time (s)')
plt.show()