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

    # Create variables
    R = []
    W = []
    M = []
    for i in range(0, Ns):
        w = np.eye(1 + D) + delta * np.random.randn(1 + D, 1 + D)
        W.append(w)
        R.append(cp.Variable((1 + D, 1 + D)))
        M1 = []
        for j in range(0, Np):
            M1.append(cp.Variable((D + 2, D + 2), PSD = True))
        M.append(M1)
    t = cp.Variable((Ns, 1))
    S = cp.Variable((Ns, Np))

    # Set Constraints
    C = []  # Constraints that are fixed through iterations
    Citer = []  # Constraints that change throughout iterations
    for i in range(Ns):  # For each subspace
        C.append(R[i][0,0] == 1)
        C.append(cp.trace(R[i]) == 2)  # Force normals to have norm 1
        Citer.append(t[i] == cp.trace(W[i].T*R[i]))
        for j in range(Np):  # For each point
            inds = np.arange(0, D + 1)
            C.append(M[i][j][np.ix_(inds,inds)] == R[i])  # Upper submatrix of M[i][j] should be R[i]
            C.append(M[i][j][0,-1] == M[i][j][-1,-1])  #s[i,j] and s[i,j]^2 should be equal
            inds = np.arange(1, D + 1)
            C.append(M[i][j][inds,-1].T * Xp[:,j] - eps*M[i][j][0,-1] <= 0)  # First inequality of abs. value
            C.append(- M[i][j][inds,-1].T * Xp[:,j] - eps*M[i][j][0,-1] <= 0)  # Second inequality of abs. value
            C.append(S[i,j] == M[i][j][-1,-1])  # Put s[i,j] in S matrix
    C.append(S.T*np.ones((Ns,1)) == np.ones((Np,1)))  # Sum of labels per point equals 1

    # Solve the problem using Reweighted Heuristics
    tic = time()
    rank1ness = np.zeros((RwHopt.maxIter, Ns))
    for iter in range(0, RwHopt.maxIter):
        print('   - R.H. iteration: ', iter)

        objective = cp.Minimize(cp.sum(t))
        constraints = C + Citer
        prob = cp.Problem(objective, constraints)
        sol = prob.solve(solver = cp.MOSEK)

        Citer = []  # Clean iteration constraints
        for i in range(Ns):
            val, vec = np.linalg.eig(R[i].value)
            [sortedval, sortedvec] = sortEigens(val, vec)
            rank1ness[iter,i] = sortedval[0] / np.sum(sortedval)
            W[i] = np.matmul(np.matmul(sortedvec, np.diag(1 / (sortedval + np.exp(-5)))), sortedvec.T)
            Citer.append(t[i] == cp.trace(W[i].T*R[i]))
        if np.min(rank1ness[iter,:]) > RwHopt.eigThres:
            iter = iter + 1  # To fill rank1ness vector
            break
    runtime = time() - tic

    # Extract Variables
    rank1ness = rank1ness[0:iter,:]
    S = S.value
    Rout = np.zeros((Ns, D))
    for i in range(0, Ns):
        Rout[i, :] = R[i][0, 1:].value

    return Rout, S, runtime, rank1ness