import numpy as np
import cvxpy as cp
from utils import *

def newton(f,Df,x0,epsilon,max_iter):
    """Approximate solution of f(x)=0 by Newton's method.

    Parameters
    ----------
    f : function
        Function for which we are searching for a solution f(x)=0.
    Df : function
        Derivative of f(x).
    x0 : number
        Initial guess for a solution f(x)=0.
    epsilon : number
        Stopping criteria is abs(f(x)) < epsilon.
    max_iter : integer
        Maximum number of iterations of Newton's method.

    Returns
    -------
    xn : number
        Implement Newton's method: compute the linear approximation
        of f(x) at xn and find x intercept by the formula
            x = xn - f(xn)/Df(xn)
        Continue until abs(f(xn)) < epsilon and return xn.
        If Df(xn) == 0, return None. If the number of iterations
        exceeds max_iter, then return None.

    Examples
    --------
    >>> f = lambda x: x**2 - x - 1
    >>> Df = lambda x: 2*x - 1
    >>> newton(f,Df,1,1e-8,10)
    Found solution after 5 iterations.
    1.618033988749989


    """
    xn = x0
    for n in range(0,max_iter):
        fxn = f(xn)
        if abs(fxn) < epsilon:
            print('Found solution after',n,'iterations.')
            return xn
        Dfxn = Df(xn)
        if Dfxn == 0:
            print('Zero derivative. No solution found.')
            return None
        xn = xn - fxn/Dfxn
    print('Exceeded maximum iterations. No solution found.')
    return None


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

coefs = np.random.rand(5)
print(coefs)

f = lambda x: x**2 +
Df = lambda x: 2*x
newton(f,Df,1,1e-8,10)

# x = torch.from_numpy(x)
# x = torch.randint(1, 10, (d, BS))
# print(x)
# v, blu = generate_veronese(x, n)
# v = v.numpy()

# print('veronesse:', v)