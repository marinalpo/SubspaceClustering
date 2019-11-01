import numpy as np
import cvxpy as cp
from time import time

def SSC_CVXPY_Full(Xp, eps, Ns, RwHopt, delta):
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
    Nc = 1 + Ns * D  # Size of the top left corner

    # Define index entries
    ind_r = lambda i, d: (1 + i * D + d)
    ind_s = lambda i, j: (1 + Ns * D + j * Ns + i)
    tic = time()

    # Create variables
    X = cp.Variable((Nm, Nm), PSD=True)  # X is PSD
    C = []  # Fixed constraints
    for i in range(Ns):  # For each subspace
        ind = ind_r(i, (np.arange(D)))
        C = C + [cp.trace(X[np.ix_(ind, ind)]) == 1]  # (8d) Subspace normals have norm 1
        for j in range(Np):  # For each subpoint
            C = C + [X[0, ind_s(i, j)] == X[ind_s(i, j), ind_s(i, j)]]  # (8b) s_ij and (s_ij)^2 are equal
            C = C + [X[ind, ind_s(i, j)].T * Xp[:, j] - eps * X[0, ind_s(i, j)] <= 0]  # (8a) First inequality of abs. value
            C = C + [
                - X[ind, ind_s(i, j)].T * Xp[:, j] - eps * X[0, ind_s(i, j)] <= 0]  # (8a) Second inequality of abs. value
            C = C + [cp.sum(X[0, ind_s(np.arange(Ns), j)]) == 1]  # (8c) Sum of labels for each point equals 1
    C = C + [X[0, 0] == 1]  # 1 on the upper left corner

    if RwHopt.corner:
        # W = np.eye(Nc) + delta*np.random.randn(Nc,Nc)
        W = np.eye(Nc)
    else:
        # W = np.eye(Nc) + delta*np.random.randn(Nm,Nm)
        W = np.eye(Nm)

    # Solve the problem with Reweighted Heuristics
    rank1ness = np.zeros([RwHopt.maxIter, 1])
    for iter in range(RwHopt.maxIter):
        print('   - reweighted h. iteration: ', iter)
        if RwHopt.corner:
            M = X[1:Nc, 1:Nc]
            obj = cp.Maximize(cp.trace(W.T * M))
            prob = cp.Problem(obj, C)
            sol = prob.solve(solver=cp.MOSEK, verbose=False)
            [val, vec] = np.linalg.eig(np.float(M.value))
        else:
            obj = cp.Maximize(cp.trace(W.T * X))
            prob = cp.Problem(obj, C)
            sol = prob.solve(solver=cp.MOSEK, verbose=False)
            [val, vec] = np.linalg.eig(np.array(X.value, dtype=np.float))
        idx = val.argsort()[::-1]
        sortedval = val[idx]
        sortedvec = vec[:, idx]
        rank1ness_curr = sortedval[0] / np.sum(sortedval)
        rank1ness[iter] = rank1ness_curr
        W = np.matmul(np.matmul(sortedvec, np.diag(1 / (sortedval + np.exp(-5)))), sortedvec.T)
        if np.min(rank1ness[iter]) > RwHopt.eigThres:
            break

    runtime = time() - tic
    s = X[0, ind_r(Ns - 1, D - 1) + 1:]
    S = cp.reshape(s, (Ns, Np))
    R = []
    for i in range(Ns):
        R.append(X[np.ix_(ind_r(i, (np.arange(0, D))), ind_r(i, (np.arange(0, D))))])
    return R, S, runtime, rank1ness

