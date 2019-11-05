import numpy as np
import cvxpy as cp
D = 2
Ns = 3
Np = 27
cols = 2
rows = 3
S = cp.Variable((3,5))
delta = 0.1

S = cp.Variable((3,5))
arr = [[S for i in range(cols)] for j in range(rows)]

r = cp.Variable((5, 5))
R = [r for i in range(Ns)]

m = cp.Variable((3, 3))
M = [r for i in range(Ns)]

C1 = R + M

C2 = R.append(M)

a = 3