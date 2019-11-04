import numpy as np
import cvxpy as cp
from time import time
from utils import *


def SSC_CVXPY_cdc_new(Xp, eps, Ns, RwHopt, delta):
    # Subspace Clustering using CVXPY, Naive implementation
    # Inputs: - Xp: DxNp matrix, each column is a point
    #         - eps: allowed distance from the subspace
    #         - Ns: number of subspaces
    #         - RwHopt: conditions for reweighted heuristic contained on an object that includes:
    #                   - maxIter: number of max iterations
    #                   - eigThres: threshold on the eigenvalue fraction for stopping procedure
    #                   - corner: rank-1 on corner (1) or on full matrix (0)
    #         - delta: noise factor on the identity at first iteration
    # Outputs: - R: tensor of length Ns, where each item is a (1+D)x(1+D) matrix with the subspace coordinates
    #          - S: NsxNp matrix, with labels for each point and subspace
    #          - runtime: runtime of the algorithm (excluding solution extraction)
    #          - rankness: rankness of every iteration
    # Define dimension and number of points

    # Define dimension and number of points
    [D, Np] = Xp.shape
    Nm = 1 + Ns * (Np + D)  # Size of the overall matrix

    # Define index entries
    ind_r = lambda i, d: (1 + i * D + d)
    ind_s = lambda i, j: (1 + Ns * D + j * Ns + i)

    # Create variables
    M = []  # size Ns x Np
    R = []  # size Ns x 1
    W = []  # size Ns x 1
    t = cp.Variable((Ns, 1))
    S = cp.Variable((Ns, Np))
    for i in range(Ns):
        R.append(cp.Variable(1+D))
        W.append(np.eye(1 + D) + delta*np.random.randn(1 + D, 1 + D))
        for j in range(Np):
            M[i].append(cp.Variable((2 + D, 2 + D)))

    C = []






    return R, S, runtime, rank1ness