# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 12:40:55 2020

@author: Jared
"""

#In quarantine
import numpy as np
import sympy as sp
import cvxpy as cvx
from utils_reduced import ACmomentConstraint
import polytope
import aux2 as aux
import datetime

Nx = 3
Nth = 4


x = sp.symarray("x", Nx)
th = sp.symarray('th',Nth)


p = (x[0]-th[0])**2 + (x[1]-th[1])**2  + (x[2] - th[2])**2- th[3]**2
#p = (x[0]-th[0])**2 + (x[1]-th[1])**2  + (th[3]*x[2] - th[2])**2
#p = (x[0]-th[0])**2 + (x[1]-th[1])**2  + (x[2] - th[2])**2 - th[3]**2 + (x[0] - th[0] + x[1] - th[1])**2

#p = x[1]**2 - (x[0]**3 + th[0]*x[0] + th[1])



#p = (x[0] - th[1]*x[1] + x[2]*th[2])**4 + (x[1]*th[1])**2 - th[3]
P = sp.Poly(p, *th)

M_out = ACmomentConstraint(P, [x, th])

fb = M_out["fb"]