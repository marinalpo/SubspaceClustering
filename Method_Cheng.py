import numpy as np
import cvxpy as cp
from time import time
from utils import *

def SSC_CVXPY_Cheng(Xp, eps, Ns, RwHopt, delta):
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
    [D, Np] = Xp.shape
    Nm = 1 + Ns * (Np + D)  # Size of the overall matrix

    # Define index entries
    ind_r = lambda i, d: (1 + i * D + d)
    ind_s = lambda i, j: (1 + Ns * D + j * Ns + i)

    # Create variables
    M = []
    for j in range(0, Np):  # For each point
        m = cp.Variable((1 + Ns * D + Ns, 1 + Ns * D + Ns), PSD=True)
        M.append(m)  # M[j] should be PSD

    R = cp.Variable((1 + Ns * D, 1 + Ns * D))
    C = []  # Constraints that are fixed through iterations

    for j in range(0, Np):  # For each point
        C.append(M[j][np.ix_(np.arange(0, 1 + Ns * D),
                             np.arange(0, 1 + Ns * D))] == R)  # Upper left submatrix of M_j should be
        C.append(cp.sum(M[j][0, ind_s(np.arange(0, Ns), 0)]) == 1)
        for i in range(0, Ns):
            C.append(M[j][0, ind_s(i, 0)] == M[j][ind_s(i, 0), ind_s(i, 0)])
            C.append(((M[j][ind_r(i, np.arange(0, D)), ind_s(i, 0)].T * Xp[:, j]) - eps * M[j][0, ind_s(i, 0)]) <= 0)
            C.append(((-M[j][ind_r(i, np.arange(0, D)), ind_s(i, 0)].T * Xp[:, j]) - eps * M[j][0, ind_s(i, 0)]) <= 0)
    C.append(R[0, 0] == 1)

    for i in range(0, Ns):  # For each subspace
        C.append(cp.trace(R[np.ix_(ind_r(i, np.arange(0, D)), ind_r(i, np.arange(0, D)))]) == 1)

    W = np.eye(1 + Ns * D) + delta * np.random.randn(1 + Ns * D, 1 + Ns * D)
    rank1ness = np.zeros(RwHopt.maxIter)
    bestR = R
    bestM = M

    tic = time()
    for iter in range(0, RwHopt.maxIter):
        print('   - R.H. iteration: ', iter)
        objective = cp.Minimize(cp.trace(W.T * R))
        prob = cp.Problem(objective, C)
        sol = prob.solve(solver=cp.MOSEK)
        val, vec = np.linalg.eig(R.value)
        [sortedval, sortedvec] = sortEigens(val, vec)
        rank1ness[iter] = sortedval[0] / np.sum(sortedval)
        W = np.matmul(np.matmul(sortedvec, np.diag(1 / (sortedval + np.exp(-5)))), sortedvec.T)

        if rank1ness[iter] > RwHopt.eigThres:
            iter = iter + 1  # To fill rank1ness vector
            break

    runtime = time() - tic

    S = np.zeros((Ns, Np))
    for j in range(0, Np):
        S[:, j] = M[j][1 + (Ns) * (D):, 0].value

    rank1ness = rank1ness[0:iter]

    R = np.zeros((Ns, D))
    for i in range(0, Ns):
        R[i, :] = M[0][ind_r(i, np.arange(0, D)), 0].value

    return R, S, runtime, rank1ness