import numpy as np
import cvxpy as cp
from utils import *

d = 2  # dimension
BS = 3  # Batch Size (length of x array)
n = 2  # degree of the polynomial, this will generate a moment matrix up to 2*n

x = np.random.randint(1, 5, size = (d, BS))
print(x.dtype)
x = torch.from_numpy(x)
#x = torch.rand([d, BS])

v, blu = generate_veronese(x, 1)
v = v.numpy()
print('veronesse:', v)