# Experiments2_full.m -----------------------------------------------------------------
# Exp 1: Fix Ns and D, run for different Np and record the running time and convergence
import numpy as np
import cvxpy as cp
from utils import *
import time
np.random.seed(2020)

Ns = 3
D = 3
Npoints = Ns*(np.arange(20)+1)
N_tests = 8
RwHopt = RwHoptCond(20, 0.97, 0)
delta = 0.1
eps = 0.1

# Data dumps
iterations = np.zeros((len(Npoints), N_tests))
runtime = np.zeros((len(Npoints), N_tests))
rank1ness = np.zeros((len(Npoints), N_tests)) # MUST BE A CELL

# for ...
k = 0
t = 1
Np = Npoints[k]
print('Trial', t, 'with',Np, 'points')
normals = createNormalVectors(D, Ns)
num_points = np.hstack([np.ones((1,Ns-1))*np.round(Np/Ns), np.ones((1,1))*(Np-(Ns-1)*np.round(Np/Ns))])
Xp, ss_ind = generatePoints(normals, num_points[0].astype(int), eps, 5)



# FUNCTION begins ------------------------------------- SSC_Yalmip_Full(Xp, eps, Ns, RwHopt, delta)

# Define dimension and number of points
[D, S] = Xp.shape
Nm = 1 + Ns*(Np + D)  # Size of the overall matrix


# Define index entries
ind_r = lambda i, d: (1+(i-1)*D+d)
ind_s = lambda i, j: (1+Ns*D+(j-1)*Ns+i)
tic = time.time()

M = []

for j in range(0,Np):
	M.append(cp.Variable((1+Ns*D+Ns,1+Ns*D+Ns), PSD=True)) # M[j] should be psd

R = cp.Variable((1+Ns*D, 1+Ns*D)) # We don't know if R should be psd

C = [] # Constraints that are fixed through iterations

for j in range(0, Np):
	C.append(M[j][0:1+Ns*D, 0:1+Ns*D]==R) # Upper left submatrix of M_j should be
	C.append(cp.sum(M[j][np.arange(0,Ns), 0]) == 1)
	for i in range(0, Ns):
		C.append( M[j][0,ind_s(i,0)] == M[j][ind_s(i,0), ind_s(i,0)] )
		C.append(  ((M[j][ind_r(i,np.arange(0,D)), ind_s(i,0)].T * Xp[:,j]) - eps*M[j][0,ind_s(i,0)]) <= 0)
		C.append(  ((-M[j][ind_r(i,np.arange(0,D)), ind_s(i,0)].T * Xp[:,j]) - eps*M[j][0,ind_s(i,0)]) <= 0)


C.append(R[0,0] == 1)
print("ind_r(0,np.arange(0,D)) = " + str(ind_r(0,np.arange(0,D))))
print("ind_r(i, np.arange(0,D)) = " + str(ind_r(i, np.arange(0,D)))) 
for i in range(0, Ns):
	print("R[np.ix_(ind_r(0,np.arange(0,D))), np.ix_(ind_r(i, np.arange(0,D))) ] = " + str(R[np.ix_(ind_r(0,np.arange(0,D))), np.ix_(ind_r(i, np.arange(0,D))) ]))
	C.append(cp.trace( R[np.ix_(ind_r(0,np.arange(0,D))), np.ix_(ind_r(i, np.arange(0,D))) ]) == 1 )


W = np.eye(1+Ns*D) + delta*np.random.randn(1+Ns*D, 1+Ns*D)
rank1ness = np.zeros(RwHopt.maxIter)
bestR = R
bestM = M

objective = cp.trace(np.matmul(W.T, R))
prob = cp.Problem(objective, C)


for itera in range(0, RwHopt.maxIter):
	sol = prob.solve(solver=cp.MOSEK)
	Ui, Si = np.linalg.eig(R.value)
	sorted_eigs = np.sort(np.diagonal(Si))[::-1]
	rank1ness[itera] = sorted_eigs[0]/np.sum(sorted_eigs)








# Create variables
X = cp.Variable((Nm, Nm), PSD=True)  # X is PSD
C = []  # Fixed constraints
Citer = []  # Constraints that will change through iterations
for i in range(Ns):  # For each subspace
	ind = ind_r(i+1,(np.arange(D)+1))
	C = C + [cp.trace(X[np.ix_(ind,ind)]) == 1]
	for j in range(Np):  # For each subpoint
		C = C + [ X[1,ind_s(i,j)] == X[ind_s(i,j),ind_s(i,j)] ]  # s_ij and (s_ij)^2 are equal
		C = C + [ X[ind,ind_s(i,j)].T*Xp[:,j] - eps*X[1,ind_s(i,j)] <= 0]  # First inequality of abs. value
		C = C + [ - X[ind,ind_s(i,j)].T*Xp[:,j] - eps*X[1,ind_s(i,j)] <= 0] # Second inequality of abs. value
		C = C + [cp.sum(X[np.ix_(ind_r(np.arange(Ns),j))]) == 1]  # Sum of labels for each point equals 1
C = C + [X[0,0]==1]  # 1 on the upper left corner

##### TRADUIR options = sdpsettings('solver','mosek', 'verbose', 0, 'cachesolvers', 1);

if RwHopt.corner:	
	# W = np.eye(Nc) + delta*np.random.randn(Nc,Nc)
	W = np.eye(Nc)
else:
	# W = np.eye(Nc) + delta*np.random.randn(Nm,Nm)
	W = np.eye(Nm)

# Solve the problem
rank1ness = np.zeros([RwHopt.maxIter,1])
bestX = X
for iter in range(RwHopt.maxIter):
	if RwHopt.corner:
		M = X[1:Nc, 1:Nc]
		tr = cp.trace(W.T*M)
	else:
		tr = cp.trace(W.T*X)

### MISSING

runtime = time() - tic

# FUNCTION ends -------------------------------------

# Rank 1 on corner
RwHopt.corner = 1