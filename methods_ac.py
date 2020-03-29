# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 14:11:54 2020

@author: Jared
"""

import numpy as np
import cvxpy as cp
import sympy as sp
from time import time
from utils_reduced import *

def AC_CVXPY(Xp, eps, var, P, mult): #, RwHopt, delta
    """ Subspace Clustering using Cheng Implementation
    :param Xp: DxNp matrix, each column is a point
    :param eps: Noise bound: allowed distance from the subspace
    :param var: sympy variables (x, th): x-data, th-parameters
    :param P: Polynomial classes in problem
    :param mult: multiplicity of each class in P
    :param RwHopt: conditions for reweighted heuristic contained on an object that includes:
                   - maxIter: number of max iterations
                   - eigThres: threshold on the eigenvalue fraction for stopping procedure
                   - corner: rank-1 on corner (1) or on full matrix (0)
    :param: delta: Noise factor on the identity at first iteration
    :return: R: tensor of length Ns, where each item is a (1+D)x(1+D) matrix with the subspace coordinates
    :return: S: NsxNp matrix, with labels for each point and subspace
    :return: runtime: runtime of the algorithm (excluding solution extraction)
    :return: rankness: rankness of every iteration
    """
    
    
    #This system will have cliques [1 parameter-monomials all data]
    #Will split up data later
    
    # Define dimension and number of points
    #x = var[0]
    #th = var[1]
    Mom_con = [ACmomentConstraint(p, var) for p in P]
    
    
    [D, Np] = Xp.shape


    Nclass = sum(mult)

    #number of monomials    
    Nm = [len(m["monom"]) for m in Mom_con]
    
    #Nm = 1 + Ns * (Np + D)  # Size of the overall matrix


    M = [] #moment matrix
    C = [] #constraints
    
    con_assign_pre = np.zeros([Np, 1])
    con_classify = []
    con_one = [] 
    con_mom = []
    con_bin = []
    con_assign = []
    count = 0
    
    
    #R = [None]*Nclass
    #S = [None]*Nclass
    #S = cp.Variable([Np, Nclass])
    
    class_sum = []
    for ic in range(len(P)):
        
        #Moment Constraints
        mom = Mom_con[ic]
                
        
        for im in range(mult[ic]):
            m = cp.Variable([Nm[ic] + Np, Nm[ic]+Np], PSD=True)
            M.append(m)
            
            #"corner" of moment matrix is 1
            ind_1 = len(mom["monom"])-1
            con_one.append(M[count][ind_1, ind_1] == 1)
            #r_curr = M[count][ind_1, :Nm[ic]]
            #R[count] = r_curr    
            
            #entries of moment matrix agree with each other
            #like (xy)(xy) == (x^2) (y^2)
            for monom, ind_agree in mom["cons"].items():                
                for iagree in range(len(ind_agree)-1):
                    iprev = ind_agree[iagree]
                    inext = ind_agree[iagree+1]
                    con_mom.append(M[count][iprev] == M[count][inext])
            
            
            #each point is properly classified (function evaluation)
            for ip in range(Np):                
                i = ip + Nm[ic]
                
                coeff_curr = mom["fb"](*Xp[:, ip])
                
                #f_curr = 0
                rs_curr = M[count][i, :Nm[ic]]
                s_curr  = M[count][i, ind_1]
                
                #add in the new variable
                #con_S += [S[ip, count]  == s_curr]
                
                
                f_curr = coeff_curr @ rs_curr
                
                #if count == 0:
                con_classify.append(-eps*s_curr <= f_curr)
                con_classify.append(f_curr <= eps*s_curr)
                
                if count == 0:
                    class_sum += [s_curr]
                else:
                    class_sum[ip] += s_curr
                
                #all classifications are binary
                con_bin += [M[count][i, ind_1] == M[count][i,i]]
            
            #prepare for new moment matrix
            count += 1

    #unique classification of points


    for ip in range(Np):
        con_assign += [cp.sum(class_sum[ip]) == 1]

    #con_assign = [class_sum == 1]
    
    
    #C = con_one + con_assign + con_mom + con_bin
    C = con_one + con_assign + con_mom + con_bin + con_classify

    objective = sum([cp.trace(mi) for mi in M])
    #objective = 0
    
    prob = cp.Problem(cp.Minimize(objective), C)
    sol = prob.solve(solver=cp.MOSEK, verbose = 1)


    return {"sol":sol, "M":[m.value for m in M], "mom":Mom_con}

    # # Define index entries
    # ind_r = lambda i, d: (1 + i * D + d)
    # ind_s = lambda i, j: (1 + Ns * D + j * Ns + i)

    # # Create variables
    # M = []
    # for j in range(0, Np):  # For each point
    #     m = cp.Variable((1 + Ns * D + Ns, 1 + Ns * D + Ns), PSD=True)
    #     M.append(m)  # M[j] should be PSD

    # R = cp.Variable((1 + Ns * D, 1 + Ns * D))
    # C = []  # Constraints that are fixed through iterations

    # for j in range(0, Np):  # For each point
    #     C.append(M[j][np.ix_(np.arange(0, 1 + Ns * D),
    #                          np.arange(0, 1 + Ns * D))] == R)  # Upper left submatrix of M_j should be
    #     C.append(cp.sum(M[j][0, ind_s(np.arange(0, Ns), 0)]) == 1)
    #     for i in range(0, Ns):
    #         C.append(M[j][0, ind_s(i, 0)] == M[j][ind_s(i, 0), ind_s(i, 0)])
    #         C.append(((M[j][ind_r(i, np.arange(0, D)), ind_s(i, 0)].T * Xp[:, j]) - eps * M[j][0, ind_s(i, 0)]) <= 0)
    #         C.append(((-M[j][ind_r(i, np.arange(0, D)), ind_s(i, 0)].T * Xp[:, j]) - eps * M[j][0, ind_s(i, 0)]) <= 0)
    # C.append(R[0, 0] == 1)

    # for i in range(0, Ns):  # For each subspace
    #     C.append(cp.trace(R[np.ix_(ind_r(i, np.arange(0, D)), ind_r(i, np.arange(0, D)))]) == 1)

    # W = np.eye(1 + Ns * D) + delta * np.random.randn(1 + Ns * D, 1 + Ns * D)
    # rank1ness = np.zeros(RwHopt.maxIter)

    # tic = time()
    # for iter in range(0, RwHopt.maxIter):
    #     print('   - R.H. iteration: ', iter)
    #     objective = cp.Minimize(cp.trace(W.T * R))
    #     prob = cp.Problem(objective, C)
    #     sol = prob.solve(solver=cp.MOSEK)
    #     val, vec = np.linalg.eig(R.value)
    #     [sortedval, sortedvec] = sortEigens(val, vec)
    #     rank1ness[iter] = sortedval[0] / np.sum(sortedval)
    #     W = np.matmul(np.matmul(sortedvec, np.diag(1 / (sortedval + np.exp(-5)))), sortedvec.T)

    #     if rank1ness[iter] > RwHopt.eigThres:
    #         iter = iter + 1  # To fill rank1ness vector
    #         break

    # runtime = time() - tic

    # S = np.zeros((Ns, Np))
    # for j in range(0, Np):
    #     S[:, j] = M[j][1 + (Ns) * (D):, 0].value

    # rank1ness = rank1ness[0:iter]

    # R = np.zeros((Ns, D))
    # for i in range(0, Ns):
    #     R[i, :] = M[0][ind_r(i, np.arange(0, D)), 0].value

    # return R, S, runtime, rank1ness