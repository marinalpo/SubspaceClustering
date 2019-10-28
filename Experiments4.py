import numpy as np
from utils import *

D_list = np.arange(2,8 + 1)
Ns = 4
Np = 80
delta = 0.1
eps = 0.1
N_tests = 8
RwHopt = {
   'maxIter': [20],
   'eigThres': [0.97]
}

for D in D_list:
  for t in range(N_tests):
    print('Trial', t, 'with', D, 'dimensions')
    normals = createNormalVectors(D, Ns)
    num_points = np.hstack([np.ones((1,Ns-1))*np.round(Np/Ns), np.ones((1,1))*(Np-(Ns-1)*np.round(Np/Ns))])
    Xp, ss_ind = generatePoints(normals,num_points[0].astype(int),eps,5)

    """
    SparseCoL0 on CDC
    """
