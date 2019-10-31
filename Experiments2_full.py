# Experiments2_full.m -----------------------------------------------------------------
# Exp 1: Fix Ns and D, run for different Np and record the running time and convergence
import numpy as np
import cvxpy as cp
from time import time
from utils import *

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

# for ...
k = 0
t = 1
Np = Npoints[k]
print('Trial', t, 'with',Np, 'points')
normals = createNormalVectors(D, Ns)
num_points = np.hstack([np.ones((1,Ns-1))*np.round(Np/Ns), np.ones((1,1))*(Np-(Ns-1)*np.round(Np/Ns))])
Xp, ss_ind = generatePoints(normals, num_points[0].astype(int), eps, 5)

# Rank 1 on moment matrix
RwHopt.corner = 0
# R, S, runtime, rank1ness = SSC_CVXPY_Full(Xp, eps, Ns, RwHopt, delta)
# FUNCTION begins ---------------------------------------------------------------------------------------

# Define dimension and number of points
[D, S] = Xp.shape
Nm = 1 + Ns*(Np + D)  # Size of the overall matrix
Nc = 1 + Ns*D  # Size of the top left corner

# Define index entries
ind_r = lambda i, d: (1+i*D+d)
ind_s = lambda i, j: (1+Ns*D+j*Ns+i)
tic = time()

# Create variables
X = cp.Variable((Nm, Nm), PSD=True)  # X is PSD
C = []  # Fixed constraints
for i in range(Ns):  # For each subspace
    ind = ind_r(i,(np.arange(D)))
    C = C + [cp.trace(X[np.ix_(ind,ind)]) == 1]  # (8d) Subspace normals have norm 1
    for j in range(Np):  # For each subpoint
        C = C + [ X[0,ind_s(i,j)] == X[ind_s(i,j),ind_s(i,j)] ]  # (8b) s_ij and (s_ij)^2 are equal
        C = C + [ X[ind,ind_s(i,j)].T*Xp[:,j] - eps*X[0,ind_s(i,j)] <= 0]  # (8a) First inequality of abs. value
        C = C + [ - X[ind,ind_s(i,j)].T*Xp[:,j] - eps*X[0,ind_s(i,j)] <= 0] # (8a) Second inequality of abs. value
        C = C + [cp.sum(X[0,ind_s(np.arange(Ns),j)]) == 1]  # (8c) Sum of labels for each point equals 1
C = C + [X[0,0]==1]  # 1 on the upper left corner

if RwHopt.corner:
    # W = np.eye(Nc) + delta*np.random.randn(Nc,Nc)
    W = np.eye(Nc)
else:
    # W = np.eye(Nc) + delta*np.random.randn(Nm,Nm)
    W = np.eye(Nm)

# Solve the problem with Reweighted Heuristics
rank1ness = np.zeros([RwHopt.maxIter,1])
bestX = X
for iter in range(RwHopt.maxIter):
    if RwHopt.corner:
        M = X[1:Nc, 1:Nc]
        obj = cp.Minimize(cp.trace(W.T*M))
        prob = cp.Problem(obj, C)
        sol = prob.solve(solver=cp.MOSEK, verbose=False)
        [Si, Ui] = np.linalg.eig(M.value)
    else:
        obj = cp.Minimize(cp.trace(W.T*X))
        prob = cp.Problem(obj, C)
        sol = prob.solve(solver=cp.MOSEK, verbose=False)
        # test = cp.sum(X[0,ind_s(np.arange(Ns),0)])
        # print('test:',test.value)
        [Si, Ui] = np.linalg.eig(X.value)

    sorted_eigs = -np.sort(-Si)
    print(np.round(sorted_eigs))
    rank1ness_curr = sorted_eigs[0]/np.sum(sorted_eigs)
    rank1ness[iter]=rank1ness_curr
    if np.max(rank1ness) == rank1ness[iter]:
        bestX = X.value
    if np.min(rank1ness[iter])>RwHopt.eigThres:
        break

runtime = time() - tic
s = X[ind_r(Ns,D):,0]
#S = cp.reshape(s,(Ns,Np))
#print(S.value)
R = ''
for i in range(Ns):
    a = 0


# FUNCTION ends -------------------------------------

# Rank 1 on corner
RwHopt.corner = 1

