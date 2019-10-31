# Experiments2_full.m -----------------------------------------------------------------
# Exp 1: Fix Ns and D, run for different Np and record the running time and convergence
import numpy as np
import cvxpy as cp
from time import time
from utils import *
from SSC_CVXPY_Full import *

np.random.seed(2020)

Ns = 3  # Number of subspaces
D = 3  # Number of dimensions
Npoints = Ns*(np.arange(20)+1)  # Number of points per dimension
N_tests = 8
RwHopt = RwHoptCond(20, 0.97, 0)  # Conditions for reweighted heuristics
delta = 0.1
eps = 0.1  # Noise bound

# Data dumps
iterations = np.zeros([len(Npoints), N_tests])
runtime = np.zeros([len(Npoints), N_tests])
rank1ness = np.zeros([len(Npoints), N_tests]) # MUST BE A CELL

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
        print(S)

# Rank 1 on corner
RwHopt.corner = 1

