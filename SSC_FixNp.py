# Experiments2_full.m -----------------------------------------------------------------
# Exp 1: Fix Ns and D, run for different Np and record the running time and convergence

import numpy as np
import matplotlib.pyplot as plt
from tempfile import TemporaryFile
from utils import *
from methods import *

np.random.seed(2020)

Ns = 3  # Number of subspaces
D = 3  # Number of dimensions
Npoints = Ns*(np.arange(10)+1)  # Number of points per dimension

trials = 2
RwHopt = RwHoptCond(20, 0.97, 0)  # Conditions for reweighted heuristics
delta = 0.1
eps = 0.1  # Noise bound
method_name = ['Full', 'Cheng', 'CDC New']
method = 2  # 0 - Full, 1 - Cheng, 2 - CDC New

# Data dumps
Runtime = np.zeros([len(Npoints), trials])
Rank1ness = []

#TODO: Add Score Computation

for k in range(len(Npoints)):
    for t in range(trials):
        Np = Npoints[k]
        print('Trial', t, 'with',Np, 'points')
        normals = createNormalVectors(D, Ns)
        num_points = np.hstack([np.ones((1,Ns-1))*np.round(Np/Ns), np.ones((1,1))*(Np-(Ns-1)*np.round(Np/Ns))])
        Xp, ss_ind = generatePoints(normals, num_points[0].astype(int), eps, 5)

        # Problem Solving
        if method == 0:
            R, S, runtime, rank1ness = SSC_CVXPY_Full(Xp, eps, Ns, RwHopt)
        if method == 1:
            R, S, runtime, rank1ness = SSC_CVXPY_Cheng(Xp, eps, Ns, RwHopt, delta)
        if method == 2:
            R, S, runtime, rank1ness = SSC_CVXPY_cdc_new(Xp, eps, Ns, RwHopt, delta)
        Runtime[k,t] = runtime
        Rank1ness.append(rank1ness)

np.savez('/Users/marinaalonsopoal/PycharmProjects/SubspaceClustering/data/dump/experiments_2_dump_full.npz', name2 = Runtime, name3 = Rank1ness, name4 = Ns, name5 = D, name6 = Npoints)

plt.plot(Npoints,np.mean(Runtime,axis=1))
plt.title('Runtime vs. Number of points')
plt.xlabel('Np')
plt.ylabel('time (s)')
plt.show()