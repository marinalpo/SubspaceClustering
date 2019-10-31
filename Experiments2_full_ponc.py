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
ind_r = lambda i, d: (1+i*D+d)
ind_s = lambda i, j: (1+Ns*D+j*Ns+i)
tic = time.time()

M = []

for j in range(0,Np):
	M.append(cp.Variable((1+Ns*D+Ns,1+Ns*D+Ns), PSD=True)) # M[j] should be psd

R = cp.Variable((1+Ns*D, 1+Ns*D)) 
print(R.shape)
C = [] # Constraints that are fixed through iterations

for j in range(0, Np):
	C.append(M[j][np.ix_(np.arange(0,1+Ns*D),np.arange(0,1+Ns*D))]==R) # Upper left submatrix of M_j should be
	C.append(cp.sum(M[j][0,ind_s(np.arange(0,Ns),0)]) == 1)
	for i in range(0, Ns):
		C.append( M[j][0,ind_s(i,0)] == M[j][ind_s(i,0), ind_s(i,0)] )
		C.append(  ((M[j][ind_r(i,np.arange(0,D)), ind_s(i,0)].T * Xp[:,j]) - eps*M[j][0,ind_s(i,0)]) <= 0)
		C.append(  ((-M[j][ind_r(i,np.arange(0,D)), ind_s(i,0)].T * Xp[:,j]) - eps*M[j][0,ind_s(i,0)]) <= 0)


C.append(R[0,0] == 1)
print("ind_r(0,np.arange(0,D)) = " + str(ind_r(0,np.arange(0,D))))
print("ind_r(i, np.arange(0,D)) = " + str(ind_r(i, np.arange(0,D)))) 
for i in range(0, Ns):
	# print("R[np.ix_(ind_r(0,np.arange(0,D))), np.ix_(ind_r(i, np.arange(0,D))) ] = " + str(R[np.ix_(ind_r(0,np.arange(0,D))), np.ix_(ind_r(i, np.arange(0,D))) ]))
	C.append(cp.trace( R[np.ix_(ind_r(i,np.arange(0,D)), ind_r(i, np.arange(0,D))) ]) == 1 )


W = np.eye(1+Ns*D) + delta*np.random.randn(1+Ns*D, 1+Ns*D)
rank1ness = np.zeros(RwHopt.maxIter)
bestR = R
bestM = M




for itera in range(0, RwHopt.maxIter):
	objective = cp.Minimize(cp.trace(W.T*R))
	prob = cp.Problem(objective, C)
	sol = prob.solve(solver=cp.MOSEK)
	val, vec = np.linalg.eig(R.value)
	idx = np.argsort(val)[::-1]
	sortedval = val[idx]
	sortedvec = vec[:, idx]
	
	rank1ness[itera] = sortedval[0]/np.sum(sortedval)
	W = np.matmul(np.matmul(sortedvec, np.diag(1/(sortedval+np.exp(-5)))), sortedvec.T)
	# W = sortedvec*np.diag(1/(sortedval+np.exp(-5)))*sortedvec.T
	
	if(max(rank1ness) == rank1ness[itera]):
		bestR = R.value
	
	if(rank1ness[itera] > RwHopt.eigThres):
		break


S = np.zeros((Ns, Np))
for j in range(0, Np):
	S[:,j] = M[j][1+(Ns)*(D):, 0].value

rank1ness = rank1ness[0:itera]

R = np.zeros((Ns,D))
for i in range(0, Ns):
	R[i,:] = M[0][ind_r(i, np.arange(0,D)),0].value
	print(M[0].value)

print(R)
print(S)


print(np.sum(S, axis=0))