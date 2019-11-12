import numpy as np
import cvxpy as cp
from utils import *

d = 2  # dimension
BS = 10  # Batch Size (length of x array)
n = 2  # degree of the polynomial, this will generate a moment matrix up to 2*n

Xp = np.random.randint(1, 10, size = (d, BS))
print('Xp:', Xp)

x = Xp[0,:]
V = np.zeros((n+2,BS))
for i in range(n + 1):
    V[i,:] = np.power(x,i)
V[-1,:] = Xp[1,:]
print('veronesse:', V)



# x = torch.from_numpy(x)
# x = torch.randint(1, 10, (d, BS))
# print(x)
# v, blu = generate_veronese(x, n)
# v = v.numpy()

# print('veronesse:', v)