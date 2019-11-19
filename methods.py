import numpy as np
import cvxpy as cp
from time import time
from utils import *


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
        # print('   - R.H. iteration: ', iter)
        if RwHopt.corner:  # Rank1 on corner
            M = X[1:Nc, 1:Nc]
            obj = cp.Minimize(cp.trace(W.T * M))
            prob = cp.Problem(obj, C)
            sol = prob.solve(solver=cp.MOSEK, verbose=False)
            [val, vec] = np.linalg.eig(np.float(M.value))
        else:  # Rank1 on full matrix
            obj = cp.Minimize(cp.trace(W.T * X))
            prob = cp.Problem(obj, C)
            sol = prob.solve(solver=cp.MOSEK, verbose=False)
            [val, vec] = np.linalg.eig(np.array(X.value, dtype=np.float))

        [sortedval, sortedvec] = sortEigens(val, vec)
        rank1ness_curr = sortedval[0] / np.sum(sortedval)
        rank1ness[iter] = rank1ness_curr
        W = np.matmul(np.matmul(sortedvec, np.diag(1 / (sortedval + np.exp(-5)))), sortedvec.T)

        if np.min(rank1ness[iter]) > RwHopt.eigThres:
            iter = iter + 1  # To fill rank1ness vector
            break

    runtime = time() - tic
    s = X[0, ind_r(Ns - 1, D - 1) + 1:]
    S = cp.reshape(s, (Ns, Np))
    S = S.value
    R = np.zeros((Ns, D))
    for i in range(Ns):
        R[i, :] = X[ind_r(i, np.arange(0, D)), 0].value

    return R, S, runtime, rank1ness



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
        # print('   - R.H. iteration: ', iter)
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
        # print('   - R.H. iteration: ', iter)
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