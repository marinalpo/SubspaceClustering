import numpy as np
import matplotlib.pyplot as plt
from tempfile import TemporaryFile
from utils import *
from AC_CVXPY import *
from SSC_CVXPY_Full import *
from SSC_CVXPY_Cheng import *
from SSC_CVXPY_cdc_new import *
from scipy.special import comb
import cmath
from math import pi

np.random.seed(1)

# Parameters Configuration
Npoly = 2  # Number of polynomials
D = 2  # Number of dimensions ( D=2 -> p(x,y) )
degmax = 2  # Maximum value of the polynomials degree
n = degmax  # Veronesse map degree
cmax = 10  # Absolute maximum value of the polynomials coefficients

Np = 4  # Constant Np points per polynomial
num_points = (Np * np.ones(Npoly)).astype(int)  # Number of points per polynomial
eps = 0.2  # Noise bound
sq_size = 10  # Horizontal sampling window

RwHopt = RwHoptCond(10, 0.97, 0)  # Conditions for reweighted heuristics
delta = 0.1
method_name = ['Full', 'Cheng', 'CDC New']
method = 2  # 0 - Full, 1 - Cheng, 2 - CDC New

# Data creation
alpha = exponent_nk(n, D)
D_lift = comb(D+n, n, exact=False)  # length of the veronesse map
# coefs = np.random.randint(-cmax, cmax, (Npoly, D_lift.astype(int)))
coefs = np.array([[-119, -27, -25, 9, 0, 25], [-119, -27, -25, 9, 0, 25]])  # Coefficients from an ellipse and a line
coefs = np.array([[-1, 0, 0, 1, 1, 1], [-1, 0, 0, 1, 1, 1]])  # Coefficients from an ellipse and a line
print('coefs:\n', coefs)

# Generate Points
Np = 100
xp1 = np.arange(-2, 2, 0.001)
xp2 = np.arange(-2, 2, 0.001)
Np = xp1.size
xp = np.vstack((xp1,xp2))
# Generate ys
for p in range(Npoly):
    a = coefs[p,5] * np.ones(Np)
    b = coefs[p,2] + coefs[p,4]*xp[p,:]
    c = coefs[p,0] + coefs[p,1]*xp[p,:] + coefs[p,3]*np.power(xp[p,:],2) - 0
    dis = np.power(b,2) - 4 * np.multiply(a, c)
    sol1 = (-b - np.sqrt(dis)) / (2 * a)
    sol2 = (-b + np.sqrt(dis)) / (2 * a)
xp = np.hstack((xp[0,:],xp[0,:]))
yp = np.hstack((sol1,sol2))
plt.plot(xp,yp, c='k')


Np = 10
# xp = np.random.uniform(-3, 9, (Npoly, Np))
xp1 = np.arange(-2, 2, 0.001)
xp2 = np.arange(-2, 2, 0.001)
xp = np.vstack((xp1,xp2))
Np = xp1.size
# Generate ys
for p in range(Npoly):
    a = coefs[p,5] * np.ones(Np)
    b = coefs[p,2] + coefs[p,4]*xp[p,:]
    c = coefs[p,0] + coefs[p,1]*xp[p,:] + coefs[p,3]*np.power(xp[p,:],2) - eps
    dis = np.power(b,2) - 4 * np.multiply(a, c)
    sol1 = (-b - np.sqrt(dis)) / (2 * a)
    sol2 = (-b + np.sqrt(dis)) / (2 * a)
xp = np.hstack((xp[0,:],xp[0,:]))
yp = np.hstack((sol1,sol2))
plt.plot(xp,yp, c='r', linestyle = ':')


Np = 10
# xp = np.random.uniform(-3, 9, (Npoly, Np))
xp1 = np.arange(-2, 2, 0.001)
xp2 = np.arange(-2, 2, 0.001)
xp = np.vstack((xp1,xp2))
Np = xp1.size
# Generate ys
for p in range(Npoly):
    a = coefs[p,5] * np.ones(Np)
    b = coefs[p,2] + coefs[p,4]*xp[p,:]
    c = coefs[p,0] + coefs[p,1]*xp[p,:] + coefs[p,3]*np.power(xp[p,:],2) + eps
    dis = np.power(b,2) - 4 * np.multiply(a, c)
    sol1 = (-b - np.sqrt(dis)) / (2 * a)
    sol2 = (-b + np.sqrt(dis)) / (2 * a)
xp = np.hstack((xp[0,:],xp[0,:]))
yp = np.hstack((sol1,sol2))
plt.plot(xp,yp,c='r', linestyle = ':')

Np = 100
xp1 = np.random.uniform(-2, 2, (1, Np))
xp = np.vstack((xp1,xp1))
Np = xp1.size
# Generate ys
for p in range(Npoly):
    a = coefs[p,5] * np.ones(Np)
    b = coefs[p,2] + coefs[p,4]*xp[p,:]
    c = coefs[p,0] + coefs[p,1]*xp[p,:] + coefs[p,3]*np.power(xp[p,:],2) + np.random.uniform(-eps,eps,Np)
    dis = np.power(b,2) - 4 * np.multiply(a, c)
    sol1 = (-b - np.sqrt(dis)) / (2 * a)
    sol2 = (-b + np.sqrt(dis)) / (2 * a)
xp = np.hstack((xp[0,:],xp[0,:]))
yp = np.hstack((sol1,sol2))
plt.scatter(xp,yp, c='g')




# x = torch.from_numpy(xp).long()
# v, p = generate_veronese(x, n)
#
# v = v.numpy()
# print('veronesse:\n', v)


u=2.     #x-position of the center
v=2    #y-position of the center
a=2.     #radius on the x-axis
b=1.    #radius on the y-axis

t = np.linspace(0, 2*pi, 100)
plt.plot( u+a*np.cos(t) , v+b*np.sin(t) , c='k')

a=2+eps     #radius on the x-axis
b=1+eps    #radius on the y-axis

t = np.linspace(0, 2*pi, 100)
plt.plot( u+a*np.cos(t) , v+b*np.sin(t), c='r', linestyle = ':')

a=2-eps     #radius on the x-axis
b=1-eps    #radius on the y-axis

t = np.linspace(0, 2*pi, 100)
plt.plot( u+a*np.cos(t) , v+b*np.sin(t), c='r', linestyle = ':')

a=2    #radius on the x-axis
b=1    #radius on the y-axis

t = np.linspace(0, 2*pi, Np)
plt.scatter( u+(a+np.random.uniform(-eps,eps,Np))*np.cos(t) , v+(b+np.random.uniform(-eps,eps,Np))*np.sin(t), c='b' )
plt.show()