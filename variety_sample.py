# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 11:17:44 2020

@author: jared
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from utils import varietySample

N = 2                       #Number of variables
x = sp.symarray("x", N)     #Variables in variety
eps_bound = 1              #Noise level

#circle
#R2 = 1;
#V = x[0]**2 + x[1]**2 - R2;

#equation 2.1
#V = x[0]**4 + x[1]**4 - 3* x[0]**2 - x[0] * x[1]**2 - x[1] + 1;
#R2 = 8;
#count_max =400
#elliptic curve
#noncompact
V = x[1]**2 - x[0]**3 + 3 * x[0]*2 - 1;
R2 = 60;

P = varietySample(V, x, count_max, R2, 0)



#eps_bound = 0.1
Pe= varietySample(V, x, count_max, R2, eps_bound)


plt.figure(3)  # Plot predicted data
#plt.subplot(2,1,1)
plt.scatter(P[0, :], P[1,:])
plt.title("Variety Sample (eps = 0)")
#plt.subplot(2,1,2)
plt.figure(4)  # Plot predicted data
plt.scatter(Pe[0, :], Pe[1,:])
plt.title("Variety Sample (eps = {})".format(eps_bound))
plt.show()
#u = Ur[2:, 1];
#v = Ur[2:, 2];